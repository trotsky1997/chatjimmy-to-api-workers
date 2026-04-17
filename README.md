# ChatJimmy Pi Provider

This repo now contains two pieces:

- `extensions/chatjimmy-provider/index.ts` - a Pi extension that calls
  ChatJimmy directly from your local Pi process
- `_worker.js` - an optional ChatJimmy-to-OpenAI-compatible Worker bridge
  for non-Pi clients

The Pi extension does not require a deployed Worker. It reuses the
bridge logic locally and sends requests from Pi straight to ChatJimmy.

## 1. Install the package in Pi

From this repo:

```bash
pi install .
```

Or for a quick one-off test without installing the package:

```bash
pi -e ./extensions/chatjimmy-provider/index.ts
```

## 2. Configure the provider

The extension talks to ChatJimmy directly and uses the public API URL by
default:

```bash
export CHATJIMMY_API_URL="https://chatjimmy.ai/api/chat"
```

Optional model metadata overrides:

```bash
export CHATJIMMY_MODEL="llama3.1-8B"
export CHATJIMMY_MODELS="llama3.1-8B,qwen2.5-coder"
export CHATJIMMY_CONTEXT_WINDOW="32768"
export CHATJIMMY_MAX_TOKENS="8192"
export CHATJIMMY_SYSTEM_PROMPT_MODE="compact"
```

Notes:

- `CHATJIMMY_MODELS` overrides `CHATJIMMY_MODEL` and accepts a
  comma-separated list.
- `CHATJIMMY_SYSTEM_PROMPT_MODE` defaults to `compact`, which replaces
  Pi's very large default coding prompt with a short ChatJimmy-friendly
  system prompt. This avoids the empty-response issue seen with the full
  prompt.
- Set `CHATJIMMY_SYSTEM_PROMPT_MODE=passthrough` if you want to send
  Pi's original system prompt unchanged.
- Set `CHATJIMMY_EXTRA_SYSTEM_PROMPT` to append custom instructions to
  the compact prompt.
- `_worker.js` is now optional. Keep it only if you also want an
  OpenAI-compatible HTTP bridge for other clients.

## 3. Use it in Pi

Start Pi and pick the model:

```bash
pi
```

Then select:

- provider: `chatjimmy`
- model: `chatjimmy/llama3.1-8B`

You can also launch directly from the CLI:

```bash
pi --model chatjimmy/llama3.1-8B
```

## Package layout

- `package.json` exposes the extension through the Pi package manifest
- `extensions/chatjimmy-provider/index.ts` registers the direct provider
- `_worker.js` remains available as an optional standalone bridge
