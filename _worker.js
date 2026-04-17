/**
 * ChatJimmy OpenAI-Compatible API Proxy
 * Cloudflare Snippets / Workers — Pure JavaScript
 *
 * Env Variables:
 *   API_KEY  - Bearer token (空 = 不鉴权)
 *   DEBUG    - "true" 开启日志
 *   CHATJIMMY_API_URL   - override upstream ChatJimmy API URL
 *   CHATJIMMY_TOOL_MODE - basic|all|none (smart is accepted as a basic alias)
 */

import { jsonrepair } from "jsonrepair";

const CHATJIMMY_API_URL = "https://chatjimmy.ai/api/chat";
const DEFAULT_MODEL = "llama3.1-8B";
const DEFAULT_TOOL_MODE = "basic";
const BASIC_TOOL_ALLOWLIST = new Set(["read", "bash", "edit", "write"]);
const CHUNK_SIZE = 32;
const UPSTREAM_TIMEOUT_MS = 15000;

const CORS = {
	"Access-Control-Allow-Origin": "*",
	"Access-Control-Allow-Methods": "GET, POST, OPTIONS",
	"Access-Control-Allow-Headers": "Content-Type, Authorization",
};

// ✅ 新的（正确匹配 ChatJimmy 真实格式）
const THINK_RE = /<\|think\|>[\s\S]*?<\|\/think\|>/gi;
const STATS_TAG_RE = /<\|stats\|>[\s\S]*?<\|\/stats\|>/g;
const STATS_EXTRACT = /<\|stats\|>([\s\S]*?)<\|\/stats\|>/;
const TOOL_CALL_BLOCK_RE = /<tool_calls>[\s\S]*?<\/tool_calls>/gi;
const TOOL_CALL_EXTRACT = /<tool_calls>([\s\S]*?)<\/tool_calls>/i;

// ─── Utilities ────────────────────────────────────────────────────────────────

function generateId() {
	return "chatcmpl-" + Date.now();
}

function checkAuth(request, apiKey) {
	if (!apiKey) return true;
	const header = request.headers.get("Authorization") || "";
	const token = header.startsWith("Bearer ") ? header.slice(7) : header;
	return token === apiKey;
}

function jsonResp(data, status) {
	return new Response(JSON.stringify(data), {
		status: status || 200,
		headers: Object.assign({ "Content-Type": "application/json" }, CORS),
	});
}

function errResp(message, status, code) {
	return jsonResp(
		{
			error: {
				message: message,
				type: "invalid_request_error",
				code: code || "server_error",
			},
		},
		status || 500,
	);
}

// ★ 关键修复：流式请求的错误必须用 SSE 格式，否则 AI SDK 解析失败变 [undefined]
function sseErrResp(message) {
	const errChunk =
		"data: " +
		JSON.stringify({
			id: generateId(),
			object: "chat.completion.chunk",
			created: Math.floor(Date.now() / 1000),
			model: DEFAULT_MODEL,
			choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
			error: { message: message, type: "server_error" },
		}) +
		"\n\ndata: [DONE]\n\n";

	return new Response(errChunk, {
		status: 200, // ★ AI SDK 对 200 才会正常走 SSE 解析路径
		headers: Object.assign(
			{ "Content-Type": "text/event-stream", "Cache-Control": "no-cache" },
			CORS,
		),
	});
}

// ─── Content Processing ───────────────────────────────────────────────────────

function extractText(msg) {
	const content = msg.content;
	if (typeof content === "string") return { text: content, attachment: null };

	if (Array.isArray(content)) {
		const parts = content
			.filter(function (p) {
				return p && p.type === "text" && p.text;
			})
			.map(function (p) {
				return p.text;
			});

		if (parts.length <= 1) return { text: parts[0] || "", attachment: null };

		for (var i = 1; i < parts.length; i++) {
			var lines = parts[i].split("\n");
			if (lines.length >= 2 && lines[0].includes(".")) {
				var body = lines.slice(1).join("\n");
				return {
					text: parts[0],
					attachment: { name: lines[0], size: body.length, content: body },
				};
			}
		}
		return { text: parts[0], attachment: null };
	}
	return { text: "", attachment: null };
}

function stringifyContent(content) {
	if (typeof content === "string") return content;
	if (content == null) return "";

	if (Array.isArray(content)) {
		return content
			.map(function (part) {
				if (typeof part === "string") return part;
				if (part && part.type === "text" && typeof part.text === "string")
					return part.text;
				try {
					return JSON.stringify(part);
				} catch (e) {
					return String(part);
				}
			})
			.filter(Boolean)
			.join("\n");
	}

	try {
		return JSON.stringify(content);
	} catch (e) {
		return String(content);
	}
}

function normalizeToolMode(toolMode) {
	var mode = typeof toolMode === "string" ? toolMode.trim().toLowerCase() : "";
	if (!mode) mode = DEFAULT_TOOL_MODE;
	return mode === "smart" ? "basic" : mode;
}

function normalizeToolDefinitions(tools) {
	if (!Array.isArray(tools)) return [];

	return tools.reduce(function (result, tool) {
		if (!tool || tool.type !== "function" || !tool.function) return result;

		var name =
			typeof tool.function.name === "string" ? tool.function.name.trim() : "";
		if (!name) return result;

		result.push({
			name: name,
			description:
				typeof tool.function.description === "string"
					? tool.function.description
					: "",
			parameters:
				tool.function.parameters && typeof tool.function.parameters === "object"
					? tool.function.parameters
					: { type: "object", properties: {} },
		});
		return result;
	}, []);
}

function filterToolDefinitions(toolDefs, toolMode) {
	var mode = normalizeToolMode(toolMode);
	if (mode === "none") return [];
	if (mode === "all") return toolDefs;

	return toolDefs.filter(function (tool) {
		return BASIC_TOOL_ALLOWLIST.has(tool.name);
	});
}

function buildToolInstruction(toolDefs, toolChoice, parallelToolCalls) {
	if (!toolDefs.length) return "";

	var choiceLine = "Tool choice: auto. Call a tool only when it is necessary.";
	if (toolChoice === "none") {
		choiceLine = "Tool choice: none. Never call tools and answer directly.";
	} else if (toolChoice === "required") {
		choiceLine =
			"Tool choice: required. You must return at least one tool call before answering.";
	} else if (
		toolChoice &&
		toolChoice.type === "function" &&
		toolChoice.function &&
		typeof toolChoice.function.name === "string"
	) {
		choiceLine =
			"Tool choice: required. You must call only this tool: " +
			toolChoice.function.name +
			".";
	}

	return [
		"You support OpenAI-style tool use through a proxy shim.",
		choiceLine,
		parallelToolCalls === false
			? "Parallel tool calls are disabled. Return at most one tool call."
			: "You may return one or more tool calls when needed.",
		"Available tools JSON:",
		JSON.stringify(toolDefs),
		"If you need tools, respond with only this exact wrapper and valid JSON:",
		'<tool_calls>{"tool_calls":[{"id":"call_x","name":"tool_name","arguments":{}}]}</tool_calls>',
		"Never return an empty <tool_calls> block. If no tool is needed, answer normally.",
		"Do not wrap the tool call block in markdown.",
		"When tool results appear later in the conversation, use them to answer normally unless another tool call is necessary.",
	].join("\n");
}

function normalizeToolArguments(args) {
	if (args == null || args === "") return {};
	if (typeof args === "string") {
		try {
			return JSON.parse(args);
		} catch (e) {
			return args;
		}
	}
	return args;
}

function generateToolCallId(index) {
	return "call_" + Date.now() + "_" + index;
}

function serializeToolCallsForUpstream(toolCalls) {
	var payload = toolCalls.map(function (call, index) {
		var name =
			call && call.function && typeof call.function.name === "string"
				? call.function.name.trim()
				: "";
		if (!name)
			throw new Error(
				"assistant.tool_calls entries must include function.name",
			);

		return {
			id: call.id || generateToolCallId(index),
			type: "function",
			name: name,
			arguments: normalizeToolArguments(call.function.arguments),
		};
	});

	return (
		"<tool_calls>" + JSON.stringify({ tool_calls: payload }) + "</tool_calls>"
	);
}

function detectToolResultStatus(content) {
	var text = stringifyContent(content).trim();
	if (!text) return "empty";

	if (
		/(^|\b)(error|enoent|eacces|eperm|traceback|exception|failed)(\b|:)/i.test(
			text,
		) ||
		/no such file or directory|not found|permission denied|command exited with code|invalid /i.test(
			text,
		)
	) {
		return "error";
	}

	return "success";
}

function buildToolResultTranscript(msg) {
	var content = stringifyContent(msg.content).trim();
	var status = detectToolResultStatus(msg.content);
	var guidance =
		status === "error"
			? "The tool failed. Briefly explain the failure and either retry with corrected arguments or ask the user for the missing input."
			: "Use this result to continue helping the user.";

	return [
		"Tool result for your previous tool call. This is not a new user message.",
		"tool_call_id: " + (msg.tool_call_id || ""),
		msg.name ? "tool_name: " + msg.name : "",
		"tool_status: " + status,
		guidance,
		"tool_output:",
		content || "(empty tool output)",
	]
		.filter(Boolean)
		.join("\n");
}

function buildUpstreamMessage(msg) {
	if (!msg || typeof msg.role !== "string") {
		throw new Error("Every message must include a role");
	}

	if (
		msg.role === "assistant" &&
		Array.isArray(msg.tool_calls) &&
		msg.tool_calls.length
	) {
		var assistantText = extractText(msg).text;
		var assistantParts = [];
		if (assistantText) assistantParts.push(assistantText);
		assistantParts.push(serializeToolCallsForUpstream(msg.tool_calls));
		return { role: "assistant", content: assistantParts.join("\n\n") };
	}

	if (msg.role === "tool") {
		return { role: "user", content: buildToolResultTranscript(msg) };
	}

	var result = extractText(msg);
	return {
		role: msg.role,
		content: result.text,
		attachment: result.attachment,
	};
}

function stripCodeFences(text) {
	return text
		.replace(/^```(?:json)?\s*/i, "")
		.replace(/\s*```$/, "")
		.trim();
}

function repairToolCallJson(text) {
	if (typeof text !== "string") return null;

	var trimmed = stripCodeFences(text);
	if (!trimmed) return null;

	try {
		return JSON.parse(trimmed);
	} catch (e) {
		try {
			return JSON.parse(jsonrepair(trimmed));
		} catch (_) {
			return null;
		}
	}
}

function parseToolCalls(raw) {
	if (!raw) return null;

	var match = raw.match(TOOL_CALL_EXTRACT);
	if (!match) return null;

	var parsed;
	try {
		parsed = JSON.parse(match[1]);
	} catch (e) {
		parsed = repairToolCallJson(match[1]);
		if (!parsed) throw new Error("Model returned malformed <tool_calls> JSON");
	}

	if (!parsed || !Array.isArray(parsed.tool_calls)) {
		throw new Error("Model returned an invalid <tool_calls> block");
	}

	if (!parsed.tool_calls.length) {
		return null;
	}

	return parsed.tool_calls.map(function (call, index) {
		var name = typeof call.name === "string" ? call.name.trim() : "";
		if (!name) throw new Error("Model returned a tool call without a name");

		var args = call.arguments == null ? {} : call.arguments;
		return {
			id: call.id || generateToolCallId(index),
			type: "function",
			function: {
				name: name,
				arguments: typeof args === "string" ? args : JSON.stringify(args),
			},
		};
	});
}

function validateToolsRequest(body, normalizedToolDefs, toolDefs, toolMode) {
	if (body.tools != null && !Array.isArray(body.tools)) {
		return "tools field must be an array";
	}

	if (
		Array.isArray(body.tools) &&
		body.tools.length > 0 &&
		normalizedToolDefs.length === 0
	) {
		return "tools field must contain function definitions with non-empty names";
	}

	if (
		Array.isArray(body.tools) &&
		body.tools.length > 0 &&
		toolDefs.length === 0
	) {
		var mode = normalizeToolMode(toolMode);
		if (mode === "none") return "CHATJIMMY_TOOL_MODE=none disables all tools";
		return "No requested tools are allowed by CHATJIMMY_TOOL_MODE=" + mode;
	}

	if (
		body.parallel_tool_calls != null &&
		typeof body.parallel_tool_calls !== "boolean"
	) {
		return "parallel_tool_calls must be a boolean";
	}

	if (body.tool_choice == null) return null;

	if (typeof body.tool_choice === "string") {
		if (["auto", "none", "required"].indexOf(body.tool_choice) === -1) {
			return "tool_choice must be one of auto, none, required, or a function selector object";
		}
		if (body.tool_choice === "required" && toolDefs.length === 0) {
			return "tool_choice requires at least one tool definition";
		}
		return null;
	}

	if (
		!body.tool_choice ||
		body.tool_choice.type !== "function" ||
		!body.tool_choice.function ||
		typeof body.tool_choice.function.name !== "string" ||
		!body.tool_choice.function.name.trim()
	) {
		return "tool_choice function selector must include function.name";
	}

	if (toolDefs.length === 0) {
		return "tool_choice function selector requires tool definitions";
	}

	var allowed = toolDefs.some(function (tool) {
		return tool.name === body.tool_choice.function.name;
	});

	if (!allowed) {
		return (
			"tool_choice references an unknown or disallowed tool: " +
			body.tool_choice.function.name
		);
	}

	return null;
}

function ensureToolChoiceSatisfied(toolCalls, toolChoice, parallelToolCalls) {
	if (parallelToolCalls === false && toolCalls && toolCalls.length > 1) {
		throw new Error(
			"Model returned multiple tool calls while parallel_tool_calls is false",
		);
	}

	if (toolChoice == null || toolChoice === "auto") return;

	if (toolChoice === "none") {
		if (toolCalls && toolCalls.length) {
			throw new Error(
				"Model returned tool calls even though tool_choice was none",
			);
		}
		return;
	}

	if (toolChoice === "required") {
		if (!toolCalls || toolCalls.length === 0) {
			throw new Error(
				"Model did not return a tool call required by tool_choice",
			);
		}
		return;
	}

	if (
		toolChoice &&
		toolChoice.type === "function" &&
		toolChoice.function &&
		typeof toolChoice.function.name === "string"
	) {
		var requiredName = toolChoice.function.name;
		if (!toolCalls || toolCalls.length === 0) {
			throw new Error(
				"Model did not return the required tool call: " + requiredName,
			);
		}

		var invalid = toolCalls.find(function (call) {
			return !call || !call.function || call.function.name !== requiredName;
		});

		if (invalid) {
			var invalidName =
				invalid && invalid.function && invalid.function.name
					? invalid.function.name
					: "unknown";
			throw new Error("Model returned an unexpected tool call: " + invalidName);
		}
	}
}

// ─── Stats & Cleaning ─────────────────────────────────────────────────────────

function parseStats(content) {
	var m = content.match(STATS_EXTRACT);
	if (!m) return null;
	try {
		return JSON.parse(m[1]);
	} catch (e) {
		return null;
	}
}

function parseUsage(content) {
	var s = parseStats(content);
	if (s) {
		return {
			prompt_tokens: s.prefill_tokens || 0,
			completion_tokens: s.decode_tokens || 0,
			total_tokens: s.total_tokens || 0,
		};
	}
	var est = Math.floor(content.length / 4);
	return { prompt_tokens: est, completion_tokens: est, total_tokens: est * 2 };
}

function cleanContent(raw) {
	return raw
		.replace(THINK_RE, "")
		.replace(STATS_TAG_RE, "")
		.replace(TOOL_CALL_BLOCK_RE, "")
		.trim();
}

function chunkify(text, size) {
	size = size || CHUNK_SIZE;
	var chars = Array.from(text);
	var result = [];
	for (var i = 0; i < chars.length; i += size) {
		result.push(chars.slice(i, i + size).join(""));
	}
	return result;
}

// ─── ChatJimmy API ────────────────────────────────────────────────────────────

function makeAbortSignal(timeoutMs) {
	var controller = new AbortController();
	var timer = setTimeout(function () {
		controller.abort("upstream timeout");
	}, timeoutMs || UPSTREAM_TIMEOUT_MS);
	return {
		signal: controller.signal,
		clear: function () {
			clearTimeout(timer);
		},
	};
}

async function callChatJimmy(req, debug, apiUrl) {
	var body = JSON.stringify(req);
	if (debug) console.log("[ChatJimmy Request]", body);

	var abortCtl = makeAbortSignal(UPSTREAM_TIMEOUT_MS);
	var start = Date.now();
	var res;
	try {
		res = await fetch(apiUrl || CHATJIMMY_API_URL, {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
				Accept: "*/*",
				Origin: "https://chatjimmy.ai",
				Referer: "https://chatjimmy.ai/",
				"User-Agent":
					"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
			},
			body: body,
			signal: abortCtl.signal,
		});
	} finally {
		abortCtl.clear();
	}

	var upstreamLatency = Date.now() - start;

	if (!res.ok) {
		var errBody = "";
		try {
			errBody = await res.text();
		} catch (e) {}
		throw new Error(
			"ChatJimmy " + res.status + ": " + (errBody || res.statusText),
		);
	}

	var text = await res.text();
	if (debug) console.log("[ChatJimmy Response]", text);

	if (!text || text.trim() === "") {
		throw new Error("ChatJimmy returned empty response");
	}

	return { text: text, upstreamLatencyMs: upstreamLatency };
}

async function callChatJimmyStream(req, debug, apiUrl) {
	var body = JSON.stringify(req);
	if (debug) console.log("[ChatJimmy Request(stream)]", body);

	var abortCtl = makeAbortSignal(UPSTREAM_TIMEOUT_MS);
	var start = Date.now();
	var res;
	try {
		res = await fetch(apiUrl || CHATJIMMY_API_URL, {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
				Accept: "*/*",
				Origin: "https://chatjimmy.ai",
				Referer: "https://chatjimmy.ai/",
				"User-Agent":
					"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
			},
			body: body,
			signal: abortCtl.signal,
		});
	} finally {
		abortCtl.clear();
	}

	var upstreamLatency = Date.now() - start;

	if (!res.ok) {
		var errBody = "";
		try {
			errBody = await res.text();
		} catch (e) {}
		throw new Error(
			"ChatJimmy " + res.status + ": " + (errBody || res.statusText),
		);
	}

	if (!res.body) throw new Error("ChatJimmy stream body is empty");
	return { body: res.body, upstreamLatencyMs: upstreamLatency };
}

// ─── Build Request ────────────────────────────────────────────────────────────

function buildRequest(req, toolDefs) {
	var systemPrompt = "";
	var messages = [];
	var lastUserAttachment = null;
	var lastUserIdx = -1;
	var toolInstruction = buildToolInstruction(
		toolDefs || [],
		req.tool_choice,
		req.parallel_tool_calls,
	);

	(req.messages || []).forEach(function (msg, i) {
		if (msg.role === "system") {
			systemPrompt =
				typeof msg.content === "string" ? msg.content : extractText(msg).text;
			return;
		}
		var normalized = buildUpstreamMessage(msg);
		if (msg.role === "user" && normalized.attachment) {
			lastUserAttachment = normalized.attachment;
			lastUserIdx = i;
		}
		messages.push({ role: normalized.role, content: normalized.content });
	});

	var lastIdx = (req.messages || []).length - 1;
	var attachment = null;
	if (
		req.messages[lastIdx] &&
		req.messages[lastIdx].role === "user" &&
		lastUserIdx === lastIdx &&
		lastUserAttachment
	) {
		attachment = lastUserAttachment;
	}
	if (!attachment && req.attachment) attachment = req.attachment;

	if (toolInstruction) {
		systemPrompt = systemPrompt
			? systemPrompt + "\n\n" + toolInstruction
			: toolInstruction;
	}

	return {
		messages: messages,
		chatOptions: {
			selectedModel: req.model || DEFAULT_MODEL,
			systemPrompt: systemPrompt,
			topK: req.top_k || 8,
		},
		attachment: attachment,
	};
}

// ─── Response Builders ────────────────────────────────────────────────────────

function makeCompletion(raw, model, toolCalls) {
	var parsedToolCalls = toolCalls || parseToolCalls(raw);
	return {
		id: generateId(),
		object: "chat.completion",
		created: Math.floor(Date.now() / 1000),
		model: model,
		choices: [
			{
				index: 0,
				message: parsedToolCalls
					? { role: "assistant", content: null, tool_calls: parsedToolCalls }
					: { role: "assistant", content: cleanContent(raw) },
				finish_reason: parsedToolCalls ? "tool_calls" : "stop",
			},
		],
		usage: parseUsage(raw),
	};
}

function makePseudoStreamResponseFromRaw(
	raw,
	model,
	upstreamLatencyMs,
	startedAt,
) {
	var id = generateId();
	var created = Math.floor(Date.now() / 1000);
	var enc = new TextEncoder();
	var ts = new TransformStream();
	var writer = ts.writable.getWriter();
	var toolCalls = parseToolCalls(raw);

	function emit(delta, finish) {
		var payload = JSON.stringify({
			id: id,
			object: "chat.completion.chunk",
			created: created,
			model: model,
			choices: [{ index: 0, delta: delta, finish_reason: finish || null }],
		});
		return writer.write(enc.encode("data: " + payload + "\n\n"));
	}

	(async function () {
		try {
			await emit({ role: "assistant" });

			if (toolCalls && toolCalls.length) {
				for (var i = 0; i < toolCalls.length; i++) {
					var call = toolCalls[i];
					await emit({
						tool_calls: [
							{
								index: i,
								id: call.id,
								type: "function",
								function: {
									name: call.function.name,
									arguments: "",
								},
							},
						],
					});

					var argChunks = chunkify(call.function.arguments || "", CHUNK_SIZE);
					for (var j = 0; j < argChunks.length; j++) {
						await emit({
							tool_calls: [
								{
									index: i,
									function: { arguments: argChunks[j] },
								},
							],
						});
					}
				}

				await emit({}, "tool_calls");
			} else {
				var textChunks = chunkify(cleanContent(raw), CHUNK_SIZE);
				for (var k = 0; k < textChunks.length; k++) {
					if (textChunks[k]) await emit({ content: textChunks[k] });
				}
				await emit({}, "stop");
			}

			await writer.write(enc.encode("data: [DONE]\n\n"));
		} catch (e) {
			try {
				await writer.write(
					enc.encode(
						"data: " +
							JSON.stringify({ error: { message: String(e) } }) +
							"\n\n",
					),
				);
				await writer.write(enc.encode("data: [DONE]\n\n"));
			} catch (_) {}
		} finally {
			try {
				writer.close();
			} catch (_) {}
		}
	})();

	var headers = Object.assign(
		{
			"Content-Type": "text/event-stream",
			"Cache-Control": "no-cache",
			"X-Upstream-Latency-Ms": String(upstreamLatencyMs || 0),
			"X-Proxy-Start-Ms": String(Date.now() - (startedAt || Date.now())),
		},
		CORS,
	);

	return new Response(ts.readable, { headers: headers });
}

function makeStreamResponseFromUpstream(
	upstreamBody,
	model,
	upstreamLatencyMs,
	startedAt,
) {
	var id = generateId();
	var created = Math.floor(Date.now() / 1000);
	var enc = new TextEncoder();
	var dec = new TextDecoder();
	var ts = new TransformStream();
	var writer = ts.writable.getWriter();
	var reader = upstreamBody.getReader();

	function emit(delta, finish) {
		var payload = JSON.stringify({
			id: id,
			object: "chat.completion.chunk",
			created: created,
			model: model,
			choices: [{ index: 0, delta: delta, finish_reason: finish || null }],
		});
		return writer.write(enc.encode("data: " + payload + "\n\n"));
	}

	(async function () {
		var carry = "";
		try {
			await emit({ role: "assistant" });

			while (true) {
				var result = await reader.read();
				if (result.done) break;

				var text = dec.decode(result.value, { stream: true });
				if (!text) continue;

				carry += text;
				var safe = carry
					.replace(THINK_RE, "")
					.replace(STATS_TAG_RE, "")
					.replace(TOOL_CALL_BLOCK_RE, "");

				if (safe.length < CHUNK_SIZE) {
					carry = safe;
					continue;
				}

				var chunks = chunkify(safe, CHUNK_SIZE);
				carry = chunks.pop() || "";
				for (var i = 0; i < chunks.length; i++) {
					if (chunks[i]) await emit({ content: chunks[i] });
				}
			}

			carry += dec.decode();
			var tail = cleanContent(carry);
			if (tail) {
				var lastChunks = chunkify(tail, CHUNK_SIZE);
				for (var j = 0; j < lastChunks.length; j++) {
					if (lastChunks[j]) await emit({ content: lastChunks[j] });
				}
			}

			await emit({}, "stop");
			await writer.write(enc.encode("data: [DONE]\n\n"));
		} catch (e) {
			try {
				var errPayload =
					"data: " + JSON.stringify({ error: { message: String(e) } }) + "\n\n";
				await writer.write(enc.encode(errPayload));
				await writer.write(enc.encode("data: [DONE]\n\n"));
			} catch (_) {}
		} finally {
			try {
				reader.releaseLock();
			} catch (_) {}
			try {
				writer.close();
			} catch (_) {}
		}
	})();

	var headers = Object.assign(
		{
			"Content-Type": "text/event-stream",
			"Cache-Control": "no-cache",
			"X-Upstream-Latency-Ms": String(upstreamLatencyMs || 0),
			"X-Proxy-Start-Ms": String(Date.now() - (startedAt || Date.now())),
		},
		CORS,
	);

	return new Response(ts.readable, { headers: headers });
}

// ─── Route Handlers ───────────────────────────────────────────────────────────

function routeRoot() {
	return new Response(
		"<!DOCTYPE html><html><head><title>ChatJimmy API</title></head><body>" +
			"<h1>ChatJimmy API</h1><ul>" +
			"<li><code>POST /v1/chat/completions</code></li>" +
			"<li><code>GET  /v1/models</code></li>" +
			"<li><code>GET  /health</code> - 连接测试</li>" +
			"</ul></body></html>",
		{
			headers: Object.assign(
				{ "Content-Type": "text/html;charset=utf-8" },
				CORS,
			),
		},
	);
}

function routeModels() {
	return jsonResp({
		object: "list",
		data: [
			{
				id: DEFAULT_MODEL,
				object: "model",
				created: Math.floor(Date.now() / 1000),
				owned_by: "chatjimmy",
			},
		],
	});
}

// ★ 诊断端点：测试 ChatJimmy 是否可达
async function routeHealth(apiUrl) {
	var start = Date.now();
	try {
		var testReq = {
			messages: [{ role: "user", content: "hi" }],
			chatOptions: { selectedModel: DEFAULT_MODEL, systemPrompt: "", topK: 1 },
			attachment: null,
		};
		var ret = await callChatJimmy(testReq, false, apiUrl);
		return jsonResp({
			status: "ok",
			upstream: apiUrl || CHATJIMMY_API_URL,
			latency_ms: Date.now() - start,
			upstream_latency_ms: ret.upstreamLatencyMs,
			response_length: ret.text.length,
		});
	} catch (e) {
		return jsonResp(
			{
				status: "error",
				upstream: apiUrl || CHATJIMMY_API_URL,
				latency_ms: Date.now() - start,
				error: String(e),
			},
			502,
		);
	}
}

// ★ 核心修复：流式与非流式错误分开处理
async function routeChat(request, debug, apiUrl, toolMode) {
	if (request.method !== "POST") return errResp("Method not allowed", 405);

	var body;
	try {
		body = await request.json();
	} catch (e) {
		return errResp("Invalid JSON body", 400);
	}

	if (!body.messages || !Array.isArray(body.messages)) {
		return body.stream
			? sseErrResp("messages field is required and must be an array")
			: errResp("messages field is required", 400);
	}

	var model = body.model || DEFAULT_MODEL;
	var isStream = !!body.stream;
	var normalizedToolDefs = normalizeToolDefinitions(body.tools);
	var toolDefs = filterToolDefinitions(normalizedToolDefs, toolMode);
	var validationError = validateToolsRequest(
		body,
		normalizedToolDefs,
		toolDefs,
		toolMode,
	);

	if (validationError) {
		return isStream
			? sseErrResp(validationError)
			: errResp(validationError, 400, "invalid_request_error");
	}

	var jimmyReq;
	try {
		jimmyReq = buildRequest(body, toolDefs);
	} catch (e) {
		return isStream
			? sseErrResp(String(e))
			: errResp(String(e), 400, "invalid_request_error");
	}

	var hasTools = toolDefs.length > 0;

	var startedAt = Date.now();

	try {
		if (isStream && hasTools) {
			var bufferedRet = await callChatJimmy(jimmyReq, debug, apiUrl);
			ensureToolChoiceSatisfied(
				parseToolCalls(bufferedRet.text),
				body.tool_choice,
				body.parallel_tool_calls,
			);
			return makePseudoStreamResponseFromRaw(
				bufferedRet.text,
				model,
				bufferedRet.upstreamLatencyMs,
				startedAt,
			);
		}

		if (isStream) {
			var streamRet = await callChatJimmyStream(jimmyReq, debug, apiUrl);
			return makeStreamResponseFromUpstream(
				streamRet.body,
				model,
				streamRet.upstreamLatencyMs,
				startedAt,
			);
		}

		var ret = await callChatJimmy(jimmyReq, debug, apiUrl);
		var toolCalls = parseToolCalls(ret.text);
		ensureToolChoiceSatisfied(
			toolCalls,
			body.tool_choice,
			body.parallel_tool_calls,
		);
		var response = jsonResp(makeCompletion(ret.text, model, toolCalls));
		response.headers.set(
			"X-Upstream-Latency-Ms",
			String(ret.upstreamLatencyMs || 0),
		);
		response.headers.set("X-Total-Latency-Ms", String(Date.now() - startedAt));
		return response;
	} catch (e) {
		// ★ 流式错误用 SSE 格式，非流式用 JSON —— 这是 [undefined] 的根本修复
		return isStream
			? sseErrResp(String(e))
			: errResp(String(e), 502, "upstream_error");
	}
}

// ─── Main Export ──────────────────────────────────────────────────────────────

export default {
	async fetch(request, env) {
		var url = new URL(request.url);
		var path = url.pathname;
		var apiKey = (env && env.API_KEY) || "";
		var debug = !!(env && env.DEBUG === "true");
		var apiUrl = (env && env.CHATJIMMY_API_URL) || CHATJIMMY_API_URL;
		var toolMode = normalizeToolMode(env && env.CHATJIMMY_TOOL_MODE);

		if (request.method === "OPTIONS") {
			return new Response(null, { status: 204, headers: CORS });
		}

		if (path === "/" || path === "") return routeRoot();

		// /health 不需要鉴权，方便诊断
		if (path === "/health") return routeHealth(apiUrl);

		if (!checkAuth(request, apiKey)) {
			return errResp("Invalid API key", 401, "invalid_api_key");
		}

		if (path === "/v1/chat/completions")
			return routeChat(request, debug, apiUrl, toolMode);
		if (path === "/v1/models") return routeModels();

		return errResp("Not found", 404);
	},
};
