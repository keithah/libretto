import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { createServer as createHttpServer, type IncomingMessage, type ServerResponse } from "node:http";
import { randomUUID } from "node:crypto";
import { registerMcpBrowserTools } from "../adapters/mcp/index.js";
import {
	createCliBrowserProvider,
	providerSupportsHeadless,
} from "./create-cli-provider.js";
import type { McpCliOptions } from "./parse-args.js";

const require = createRequire(fileURLToPath(import.meta.url));
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

async function readHttpBody(req: IncomingMessage): Promise<unknown> {
	const chunks: Buffer[] = [];
	let size = 0;
	let tooLarge = false;
	for await (const chunk of req) {
		const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
		size += buffer.length;
		if (size <= MAX_HTTP_BODY_BYTES) chunks.push(buffer);
		else tooLarge = true;
	}
	if (tooLarge) throw new HttpRequestError(413, "Request body exceeds the 1 MiB limit");
	return chunks.length ? JSON.parse(Buffer.concat(chunks).toString("utf8")) : undefined;
}

class HttpRequestError extends Error {
	constructor(readonly status: number, message: string) {
		super(message);
	}
}

function writeHttpError(res: ServerResponse, status: number, message: string): void {
	if (!res.headersSent) res.writeHead(status, { "content-type": "application/json" });
	res.end(JSON.stringify({ error: message }));
}

/** Start a stateful native Streamable HTTP MCP server. */
export async function startMcpStreamableHttpServer(
	options: McpCliOptions,
): Promise<Error | void> {
	const sessions = new Map<string, {
		transport: StreamableHTTPServerTransport;
		server: McpServer;
		toolkit: ReturnType<typeof registerMcpBrowserTools>;
	}>();
	const httpServer = createHttpServer(async (req, res) => {
		const pathname = new URL(req.url ?? "/", `http://${options.host}`).pathname;
		if (pathname !== options.path) return writeHttpError(res, 404, "Not found");
		try {
			const sessionId = typeof req.headers["mcp-session-id"] === "string" ? req.headers["mcp-session-id"] : undefined;
			let entry = sessionId ? sessions.get(sessionId) : undefined;
			const body = req.method === "POST" ? await readHttpBody(req) : undefined;
			if (sessionId && !entry) return writeHttpError(res, 404, "Invalid MCP session");
			if (!entry) {
				if (req.method !== "POST" || typeof body !== "object" || body === null || !("method" in body) || body.method !== "initialize") {
					return writeHttpError(res, 400, "Missing or invalid MCP session");
				}
				const provider = createCliBrowserProvider({ provider: options.provider, headless: options.headless });
				if (provider instanceof Error) return writeHttpError(res, 500, provider.message);
				const server = new McpServer({ name: "libretto-browser-tools", version: readPackageVersion() });
				const toolkit = registerMcpBrowserTools(server, provider, {
					allowedDomains: options.allowedDomains.length ? options.allowedDomains : undefined,
					blockedDomains: options.blockedDomains.length ? options.blockedDomains : undefined,
					defaultAuthProfile: false,
				});
				const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: randomUUID });
				transport.onclose = () => {
					if (transport.sessionId) sessions.delete(transport.sessionId);
					void toolkit.dispose();
				};
				await server.connect(transport);
				entry = { transport, server, toolkit };
			}
			await entry.transport.handleRequest(req, res, body);
			if (entry.transport.sessionId) sessions.set(entry.transport.sessionId, entry);
		} catch (error) {
			if (error instanceof SyntaxError) {
				return writeHttpError(res, 400, "Malformed JSON request body");
			}
			if (error instanceof HttpRequestError) {
				return writeHttpError(res, error.status, error.message);
			}
			console.error("MCP HTTP request failed:", error);
			if (!res.headersSent) writeHttpError(res, 500, "MCP request failed");
			else res.end();
		}
	});

	await new Promise<void>((resolve, reject) => {
		httpServer.once("error", reject);
		httpServer.listen(options.port, options.host, resolve);
	});
	process.stderr.write(`Libretto MCP Streamable HTTP listening on http://${options.host}:${options.port}${options.path}\n`);

	let shuttingDown = false;
	async function shutdown(): Promise<void> {
		if (shuttingDown) return;
		shuttingDown = true;
		httpServer.close();
		await Promise.all([...sessions.values()].map(async ({ transport, toolkit, server }) => {
			await transport.close().catch(() => undefined);
			await toolkit.dispose().catch(() => undefined);
			await server.close().catch(() => undefined);
		}));
		sessions.clear();
	}
	process.once("SIGINT", () => void shutdown().finally(() => process.exit(0)));
	process.once("SIGTERM", () => void shutdown().finally(() => process.exit(0)));
}

export async function startMcpServer(options: McpCliOptions): Promise<Error | void> {
	return options.transport === "streamable-http"
		? startMcpStreamableHttpServer(options)
		: startMcpStdioServer(options);
}
