import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import {
	createServer as createHttpServer,
	type IncomingMessage,
	type ServerResponse,
} from "node:http";
import { randomUUID } from "node:crypto";
import { registerMcpBrowserTools } from "../adapters/mcp/index.js";
import { errorMessage } from "../errors.js";
import {
	createCliBrowserProvider,
	providerSupportsHeadless,
} from "./create-cli-provider.js";
import type { McpCliOptions } from "./parse-args.js";

const require = createRequire(fileURLToPath(import.meta.url));

/** Largest MCP request body the HTTP transport accepts. */
const MAX_HTTP_BODY_BYTES = 1_048_576;

function readPackageVersion(): string {
	try {
		const pkg = require("../../package.json") as { version?: string };
		return pkg.version ?? "0.0.0";
	} catch {
		return "0.0.0";
	}
}

/**
 * Start the stdio MCP server and keep the process alive until the transport
 * closes or the process receives SIGINT/SIGTERM.
 */
export async function startMcpStdioServer(
	options: McpCliOptions,
): Promise<Error | void> {
	const provider = createCliBrowserProvider({
		provider: options.provider,
		headless: options.headless,
	});
	if (provider instanceof Error) {
		return provider;
	}

	if (!options.headless && !providerSupportsHeadless(options.provider)) {
		process.stderr.write(
			`--headed is ignored for --provider ${options.provider}; that provider has no headed mode in this CLI.\n`,
		);
	}

	const server = new McpServer({
		name: "libretto-browser-tools",
		version: readPackageVersion(),
	});
	const toolkit = registerMcpBrowserTools(server, provider, {
		allowedDomains:
			options.allowedDomains.length > 0 ? options.allowedDomains : undefined,
		blockedDomains:
			options.blockedDomains.length > 0 ? options.blockedDomains : undefined,
	});

	let shuttingDown = false;
	async function shutdown(): Promise<void> {
		if (shuttingDown) return;
		shuttingDown = true;
		await server.close().catch(() => undefined);
		await toolkit.dispose();
	}

	process.once("SIGINT", () => {
		void shutdown().finally(() => process.exit(0));
	});
	process.once("SIGTERM", () => {
		void shutdown().finally(() => process.exit(0));
	});

	const transport = new StdioServerTransport();
	transport.onclose = () => {
		void shutdown().finally(() => {
			if (!process.exitCode) process.exit(0);
		});
	};
	await server.connect(transport);
}

/** An HTTP request the server rejects before it reaches the MCP transport. */
class HttpRequestError extends Error {
	constructor(
		readonly status: number,
		message: string,
	) {
		super(message);
	}
}

/**
 * Read a JSON request body, rejecting bodies above {@link MAX_HTTP_BODY_BYTES}.
 * The stream is drained either way so the connection stays usable.
 */
async function readHttpBody(req: IncomingMessage): Promise<unknown> {
	const chunks: Buffer[] = [];
	let size = 0;
	let tooLarge = false;
	for await (const chunk of req) {
		const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
		size += buffer.length;
		if (size > MAX_HTTP_BODY_BYTES) {
			tooLarge = true;
			continue;
		}
		chunks.push(buffer);
	}
	if (tooLarge) {
		throw new HttpRequestError(
			413,
			"Request body is larger than 1 MiB. Send a smaller MCP request.",
		);
	}
	if (chunks.length === 0) return undefined;
	try {
		return JSON.parse(Buffer.concat(chunks).toString("utf8")) as unknown;
	} catch {
		throw new HttpRequestError(
			400,
			"Request body is not valid JSON. Send a JSON-RPC MCP message.",
		);
	}
}

function writeHttpError(
	res: ServerResponse,
	status: number,
	message: string,
): void {
	if (!res.headersSent) {
		res.writeHead(status, { "content-type": "application/json" });
	}
	res.end(JSON.stringify({ error: message }));
}

function isInitializeRequest(body: unknown): boolean {
	return (
		typeof body === "object" &&
		body !== null &&
		"method" in body &&
		(body as { method?: unknown }).method === "initialize"
	);
}

type McpHttpSession = {
	transport: StreamableHTTPServerTransport;
	server: McpServer;
	toolkit: ReturnType<typeof registerMcpBrowserTools>;
};

/** A running Streamable HTTP server the caller owns and must close. */
export type McpHttpServerHandle = {
	/** The address MCP clients connect to, including the resolved port. */
	url: string;
	port: number;
	/** Closes the listener and every browser session it owns. */
	close(): Promise<void>;
};

/**
 * Start a stateful Streamable HTTP MCP server and return a handle to it.
 *
 * Each MCP session gets its own server, provider, and browser toolkit so
 * concurrent clients never share browser state. Sessions default to a throwaway
 * browser profile because several clients can be connected at once.
 */
export async function startMcpHttpServer(
	options: McpCliOptions,
): Promise<Error | McpHttpServerHandle> {
	const sessions = new Map<string, McpHttpSession>();

	async function closeSession(session: McpHttpSession): Promise<void> {
		await session.transport.close().catch(() => undefined);
		await session.server.close().catch(() => undefined);
		await session.toolkit.dispose().catch(() => undefined);
	}

	const httpServer = createHttpServer((req, res) => {
		void handleHttpRequest(req, res);
	});

	async function handleHttpRequest(
		req: IncomingMessage,
		res: ServerResponse,
	): Promise<void> {
		const pathname = new URL(req.url ?? "/", `http://${options.host}`).pathname;
		if (pathname !== options.path) {
			return writeHttpError(
				res,
				404,
				`Unknown path "${pathname}". Send MCP requests to "${options.path}".`,
			);
		}

		try {
			const header = req.headers["mcp-session-id"];
			const sessionId = typeof header === "string" ? header : undefined;
			const body = req.method === "POST" ? await readHttpBody(req) : undefined;
			let session = sessionId ? sessions.get(sessionId) : undefined;

			// An unknown ID must never fall through to session creation, or a client
			// could allocate a browser per bogus header.
			if (sessionId !== undefined && session === undefined) {
				return writeHttpError(
					res,
					404,
					`Unknown MCP session "${sessionId}". Start a new session with an initialize request that has no mcp-session-id header.`,
				);
			}

			if (session === undefined) {
				if (req.method !== "POST" || !isInitializeRequest(body)) {
					return writeHttpError(
						res,
						400,
						"Missing MCP session. Send an initialize request first, then pass the returned mcp-session-id header.",
					);
				}

				const provider = createCliBrowserProvider({
					provider: options.provider,
					headless: options.headless,
				});
				if (provider instanceof Error) {
					return writeHttpError(res, 500, provider.message);
				}

				const server = new McpServer({
					name: "libretto-browser-tools",
					version: readPackageVersion(),
				});
				const toolkit = registerMcpBrowserTools(server, provider, {
					allowedDomains:
						options.allowedDomains.length > 0
							? options.allowedDomains
							: undefined,
					blockedDomains:
						options.blockedDomains.length > 0
							? options.blockedDomains
							: undefined,
					// Concurrent HTTP clients cannot share one persistent Chromium
					// profile, so browser_open isolates sessions unless the caller asks
					// for a named profile.
					defaultAuthProfile: false,
				});
				const transport = new StreamableHTTPServerTransport({
					sessionIdGenerator: randomUUID,
				});
				transport.onclose = () => {
					if (transport.sessionId) sessions.delete(transport.sessionId);
					void toolkit.dispose();
				};
				await server.connect(transport);
				session = { transport, server, toolkit };

				await transport.handleRequest(req, res, body);
				// The transport assigns the ID while handling initialize.
				if (transport.sessionId) sessions.set(transport.sessionId, session);
				return;
			}

			await session.transport.handleRequest(req, res, body);
		} catch (error) {
			if (error instanceof HttpRequestError) {
				return writeHttpError(res, error.status, error.message);
			}
			process.stderr.write(`MCP HTTP request failed: ${errorMessage(error)}\n`);
			if (!res.headersSent) {
				writeHttpError(res, 500, "MCP request failed. Retry the request.");
			} else {
				res.end();
			}
		}
	}

	const listening = await new Promise<Error | null>((resolve) => {
		httpServer.once("error", (error: Error) => resolve(error));
		httpServer.listen(options.port, options.host, () => resolve(null));
	});
	if (listening) {
		return new Error(
			`Could not listen on ${options.host}:${options.port} (${errorMessage(listening)}). Pass a free --port, or stop the process already using it.`,
		);
	}

	const address = httpServer.address();
	const port =
		typeof address === "object" && address !== null ? address.port : options.port;

	let closed: Promise<void> | undefined;
	return {
		url: `http://${options.host}:${port}${options.path}`,
		port,
		close: () => {
			closed ??= (async () => {
				httpServer.closeAllConnections();
				await new Promise<void>((resolve) => httpServer.close(() => resolve()));
				const open = [...sessions.values()];
				sessions.clear();
				await Promise.all(open.map(closeSession));
			})();
			return closed;
		},
	};
}

/**
 * Start the Streamable HTTP server and keep the process alive until it receives
 * SIGINT/SIGTERM.
 */
export async function startMcpStreamableHttpServer(
	options: McpCliOptions,
): Promise<Error | void> {
	const started = await startMcpHttpServer(options);
	if (started instanceof Error) return started;

	if (!options.headless && !providerSupportsHeadless(options.provider)) {
		process.stderr.write(
			`--headed is ignored for --provider ${options.provider}; that provider has no headed mode in this CLI.\n`,
		);
	}
	process.stderr.write(
		`Libretto MCP Streamable HTTP listening on ${started.url}\n`,
	);

	const shutdown = () => {
		void started.close().finally(() => process.exit(0));
	};
	process.once("SIGINT", shutdown);
	process.once("SIGTERM", shutdown);
}

export async function startMcpServer(
	options: McpCliOptions,
): Promise<Error | void> {
	return options.transport === "streamable-http"
		? startMcpStreamableHttpServer(options)
		: startMcpStdioServer(options);
}
