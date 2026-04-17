import {
	calculateCost,
	createAssistantMessageEventStream,
	type Api,
	type AssistantMessage,
	type AssistantMessageEventStream,
	type Context,
	type Model,
	type SimpleStreamOptions,
} from "@mariozechner/pi-ai";
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import worker from "../../_worker.js";

const DEFAULT_CHATJIMMY_API_URL = "https://chatjimmy.ai/api/chat";
const DEFAULT_MODEL_ID = "llama3.1-8B";
const DEFAULT_CONTEXT_WINDOW = 32768;
const DEFAULT_MAX_TOKENS = 8192;
const ENV =
	(globalThis as { process?: { env?: Record<string, string | undefined> } })
		.process?.env || {};

type OpenAIToolCall = {
	id?: string;
	index?: number;
	type?: string;
	function?: {
		name?: string;
		arguments?: string;
	};
};

function readEnv(name: string): string | undefined {
	const value = ENV[name]?.trim();
	return value ? value : undefined;
}

function readNumberEnv(name: string, fallback: number): number {
	const value = readEnv(name);
	if (!value) return fallback;
	const parsed = Number.parseInt(value, 10);
	return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function parseModelIds(): string[] {
	const multi = readEnv("CHATJIMMY_MODELS");
	if (multi) {
		const ids = multi
			.split(",")
			.map((item) => item.trim())
			.filter(Boolean);
		if (ids.length > 0) return ids;
	}

	return [readEnv("CHATJIMMY_MODEL") || DEFAULT_MODEL_ID];
}

function formatModelName(id: string): string {
	return `ChatJimmy ${id}`;
}

function extractTextContent(
	content: string | Array<{ type: string; text?: string }>,
): string {
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return "";

	return content
		.filter(
			(part) => part && part.type === "text" && typeof part.text === "string",
		)
		.map((part) => part.text || "")
		.join("\n");
}

function convertMessages(model: Model<Api>, context: Context) {
	const messages: Array<Record<string, unknown>> = [];

	if (context.systemPrompt) {
		messages.push({ role: "system", content: context.systemPrompt });
	}

	for (const message of context.messages) {
		if (message.role === "user") {
			messages.push({
				role: "user",
				content: extractTextContent(message.content),
			});
			continue;
		}

		if (message.role === "assistant") {
			const text = message.content
				.filter((block: { type: string }) => block.type === "text")
				.map((block: { text: string }) => block.text)
				.join("\n");
			const toolCalls = message.content
				.filter((block: { type: string }) => block.type === "toolCall")
				.map(
					(block: {
						id: string;
						name: string;
						arguments: Record<string, unknown>;
					}) => ({
						id: block.id,
						type: "function",
						function: {
							name: block.name,
							arguments: JSON.stringify(block.arguments || {}),
						},
					}),
				);

			if (toolCalls.length > 0) {
				messages.push({
					role: "assistant",
					content: text || null,
					tool_calls: toolCalls,
				});
			} else {
				messages.push({ role: "assistant", content: text });
			}
			continue;
		}

		if (message.role === "toolResult") {
			messages.push({
				role: "tool",
				tool_call_id: message.toolCallId,
				name: message.toolName,
				content: extractTextContent(message.content),
			});
		}
	}

	return {
		model: model.id,
		stream: true,
		messages,
		tools: context.tools?.map(
			(tool: { name: string; description: string; parameters: unknown }) => ({
				type: "function",
				function: {
					name: tool.name,
					description: tool.description,
					parameters: tool.parameters,
				},
			}),
		),
	};
}

function createUsage(model: Model<Api>) {
	const usage = {
		input: 0,
		output: 0,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: 0,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
	calculateCost(model, usage);
	return usage;
}

function createOutputMessage(model: Model<Api>): AssistantMessage {
	return {
		role: "assistant",
		content: [],
		api: model.api,
		provider: model.provider,
		model: model.id,
		usage: createUsage(model),
		stopReason: "stop",
		timestamp: Date.now(),
	};
}

function tryParseToolArguments(text: string) {
	if (!text) return {};
	try {
		return JSON.parse(text);
	} catch (e) {
		return {};
	}
}

function finishOpenBlocks(
	stream: AssistantMessageEventStream,
	output: AssistantMessage,
	currentTextIndex: number | null,
	toolCallStates: Map<number, { contentIndex: number; partialArgs: string }>,
) {
	if (currentTextIndex != null) {
		const block = output.content[currentTextIndex];
		if (block?.type === "text") {
			stream.push({
				type: "text_end",
				contentIndex: currentTextIndex,
				content: block.text,
				partial: output,
			});
		}
	}

	const orderedStates = [...toolCallStates.entries()].sort(
		(a, b) => a[0] - b[0],
	);
	for (const [, state] of orderedStates) {
		const block = output.content[state.contentIndex];
		if (block?.type !== "toolCall") continue;
		block.arguments = tryParseToolArguments(state.partialArgs);
		stream.push({
			type: "toolcall_end",
			contentIndex: state.contentIndex,
			toolCall: block,
			partial: output,
		});
	}
	toolCallStates.clear();
}

function applyChunk(
	stream: AssistantMessageEventStream,
	output: AssistantMessage,
	payload: any,
	state: {
		currentTextIndex: number | null;
		toolCallStates: Map<number, { contentIndex: number; partialArgs: string }>;
		finished: boolean;
	},
) {
	if (payload.error?.message) {
		throw new Error(payload.error.message);
	}

	const choice = Array.isArray(payload.choices)
		? payload.choices[0]
		: undefined;
	if (!choice) return;

	const delta = choice.delta || {};

	if (typeof delta.content === "string" && delta.content.length > 0) {
		if (state.currentTextIndex == null) {
			output.content.push({ type: "text", text: delta.content });
			state.currentTextIndex = output.content.length - 1;
			stream.push({
				type: "text_start",
				contentIndex: state.currentTextIndex,
				partial: output,
			});
			stream.push({
				type: "text_delta",
				contentIndex: state.currentTextIndex,
				delta: delta.content,
				partial: output,
			});
		} else {
			const block = output.content[state.currentTextIndex];
			if (block?.type === "text") {
				block.text += delta.content;
				stream.push({
					type: "text_delta",
					contentIndex: state.currentTextIndex,
					delta: delta.content,
					partial: output,
				});
			}
		}
	}

	if (Array.isArray(delta.tool_calls)) {
		for (const toolCall of delta.tool_calls as OpenAIToolCall[]) {
			const toolIndex = typeof toolCall.index === "number" ? toolCall.index : 0;
			let toolState = state.toolCallStates.get(toolIndex);

			if (!toolState) {
				output.content.push({
					type: "toolCall",
					id: toolCall.id || `call_${toolIndex}`,
					name: toolCall.function?.name || "",
					arguments: {},
				});
				toolState = {
					contentIndex: output.content.length - 1,
					partialArgs: "",
				};
				state.toolCallStates.set(toolIndex, toolState);
				stream.push({
					type: "toolcall_start",
					contentIndex: toolState.contentIndex,
					partial: output,
				});
			}

			const block = output.content[toolState.contentIndex];
			if (block?.type !== "toolCall") continue;

			if (toolCall.id) block.id = toolCall.id;
			if (toolCall.function?.name) block.name = toolCall.function.name;

			const deltaArgs = toolCall.function?.arguments || "";
			if (deltaArgs) {
				toolState.partialArgs += deltaArgs;
				block.arguments = tryParseToolArguments(toolState.partialArgs);
				stream.push({
					type: "toolcall_delta",
					contentIndex: toolState.contentIndex,
					delta: deltaArgs,
					partial: output,
				});
			}
		}
	}

	if (choice.finish_reason) {
		finishOpenBlocks(
			stream,
			output,
			state.currentTextIndex,
			state.toolCallStates,
		);
		state.currentTextIndex = null;
		state.finished = true;

		if (choice.finish_reason === "tool_calls") {
			output.stopReason = "toolUse";
			stream.push({ type: "done", reason: "toolUse", message: output });
			return;
		}

		if (choice.finish_reason === "length") {
			output.stopReason = "length";
			stream.push({ type: "done", reason: "length", message: output });
			return;
		}

		output.stopReason = "stop";
		stream.push({ type: "done", reason: "stop", message: output });
	}
}

async function processSseResponse(
	response: Response,
	stream: AssistantMessageEventStream,
	output: AssistantMessage,
) {
	if (!response.body)
		throw new Error("ChatJimmy provider returned an empty stream");

	const decoder = new TextDecoder();
	const reader = response.body.getReader();
	let buffer = "";
	const state = {
		currentTextIndex: null as number | null,
		toolCallStates: new Map<
			number,
			{ contentIndex: number; partialArgs: string }
		>(),
		finished: false,
	};

	stream.push({ type: "start", partial: output });

	while (true) {
		const { value, done } = await reader.read();
		buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

		let boundary = buffer.indexOf("\n\n");
		while (boundary !== -1) {
			const eventBlock = buffer.slice(0, boundary);
			buffer = buffer.slice(boundary + 2);

			const dataLines = eventBlock
				.split(/\r?\n/)
				.filter((line) => line.startsWith("data:"))
				.map((line) => line.slice(5).trimStart());

			if (dataLines.length > 0) {
				const data = dataLines.join("\n");
				if (data === "[DONE]") {
					if (!state.finished) {
						finishOpenBlocks(
							stream,
							output,
							state.currentTextIndex,
							state.toolCallStates,
						);
						stream.push({ type: "done", reason: "stop", message: output });
						state.finished = true;
					}
					return;
				}

				applyChunk(stream, output, JSON.parse(data), state);
			}

			boundary = buffer.indexOf("\n\n");
		}

		if (done) break;
	}

	if (!state.finished) {
		finishOpenBlocks(
			stream,
			output,
			state.currentTextIndex,
			state.toolCallStates,
		);
		stream.push({ type: "done", reason: "stop", message: output });
	}
}

function streamChatJimmy(
	model: Model<Api>,
	context: Context,
	options?: SimpleStreamOptions,
): AssistantMessageEventStream {
	const stream = createAssistantMessageEventStream();

	(async () => {
		const output = createOutputMessage(model);
		try {
			const body = convertMessages(model, context);
			const request = new Request(
				"http://chatjimmy.local/v1/chat/completions",
				{
					method: "POST",
					headers: { "Content-Type": "application/json" },
					body: JSON.stringify(body),
				},
			);

			const response = await worker.fetch(request, {
				API_KEY: "",
				DEBUG: readEnv("CHATJIMMY_DEBUG") === "true" ? "true" : "false",
				CHATJIMMY_API_URL:
					readEnv("CHATJIMMY_API_URL") || DEFAULT_CHATJIMMY_API_URL,
			});

			if (
				!response.ok &&
				response.headers.get("content-type")?.includes("application/json")
			) {
				const error = await response.json();
				throw new Error(
					error?.error?.message ||
						`ChatJimmy request failed: ${response.status}`,
				);
			}

			await processSseResponse(response, stream, output);
			stream.end();
		} catch (error) {
			output.stopReason = options?.signal?.aborted ? "aborted" : "error";
			output.errorMessage =
				error instanceof Error ? error.message : String(error);
			stream.push({
				type: "error",
				reason: output.stopReason,
				error: output,
			});
			stream.end();
		}
	})();

	return stream;
}

export default function (pi: ExtensionAPI) {
	const modelIds = parseModelIds();
	const contextWindow = readNumberEnv(
		"CHATJIMMY_CONTEXT_WINDOW",
		DEFAULT_CONTEXT_WINDOW,
	);
	const maxTokens = readNumberEnv("CHATJIMMY_MAX_TOKENS", DEFAULT_MAX_TOKENS);
	const apiUrl = readEnv("CHATJIMMY_API_URL") || DEFAULT_CHATJIMMY_API_URL;

	pi.registerProvider("chatjimmy", {
		baseUrl: apiUrl,
		apiKey: "chatjimmy-local",
		api: "chatjimmy-api",
		models: modelIds.map((id) => ({
			id,
			name: formatModelName(id),
			reasoning: false,
			input: ["text"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow,
			maxTokens,
		})),
		streamSimple: streamChatJimmy,
	});
}
