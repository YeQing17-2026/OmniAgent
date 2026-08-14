"""Thin proxy: OpenAI chat/completions → Responses API (http://127.0.0.1:6060/v1/responses).

Usage:
    python tests/proxy_server.py [--port 16060]

Exposes /v1/chat/completions and /v1/models on the given port,
forwarding to the Responses API backend.
"""

import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.request
import urllib.error

BACKEND_BASE = "http://127.0.0.1:6060"
API_KEY = "sk-api-12345689test"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("proxy")


def responses_to_chat(resp_json: dict) -> dict:
    """Convert Responses API response → chat completions format."""
    output = resp_json.get("output") or []
    content = ""
    tool_calls = []

    for item in output:
        itype = item.get("type", "")
        if itype == "message":
            for block in item.get("content", []):
                if block.get("type") == "output_text":
                    content += block.get("text", "")
        elif itype == "function_call":
            tool_calls.append({
                "id": item.get("call_id", item.get("id", "call_0")),
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": json.dumps(item.get("arguments", {}))
                    if isinstance(item.get("arguments"), dict)
                    else item.get("arguments", "{}"),
                },
            })

    message: dict = {"role": "assistant", "content": content or None}
    if tool_calls:
        message["tool_calls"] = tool_calls

    usage = resp_json.get("usage", {})
    return {
        "id": resp_json.get("id", "chatcmpl-0"),
        "object": "chat.completion",
        "created": resp_json.get("created_at") or 0,
        "model": resp_json.get("model", ""),
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": resp_json.get("stop_reason", "stop"),
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }


def chat_to_responses(req: dict) -> dict:
    """Convert chat completions request → Responses API format."""
    messages = req.get("messages", [])
    tools = req.get("tools", [])

    input_items = []
    system_parts = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            system_parts.append(content if isinstance(content, str) else "")
            continue

        if role == "tool":
            input_items.append({
                "type": "function_call_output",
                "call_id": msg.get("tool_call_id", "call_0"),
                "output": content if isinstance(content, str) else json.dumps(content),
            })
            continue

        if role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    args = fn.get("arguments", "{}")
                    try:
                        args_parsed = json.loads(args)
                    except Exception:
                        args_parsed = args
                    input_items.append({
                        "type": "function_call",
                        "call_id": tc.get("id", "call_0"),
                        "name": fn.get("name", ""),
                        "arguments": args_parsed,
                    })
                if content:
                    input_items.append({
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": content}],
                    })
            else:
                input_items.append({
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": content}] if content else [],
                })
            continue

        # user message
        if isinstance(content, str):
            blocks = [{"type": "input_text", "text": content}]
        elif isinstance(content, list):
            blocks = []
            for part in content:
                if isinstance(part, str):
                    blocks.append({"type": "input_text", "text": part})
                elif isinstance(part, dict):
                    if part.get("type") == "text":
                        blocks.append({"type": "input_text", "text": part.get("text", "")})
                    else:
                        blocks.append(part)
        else:
            blocks = []

        input_items.append({"type": "message", "role": role, "content": blocks})

    payload: dict = {
        "model": req.get("model", "gpt-5.4"),
        "input": input_items,
        "max_output_tokens": req.get("max_tokens", 4096),
        "temperature": req.get("temperature", 1.0),
    }

    if system_parts:
        payload["instructions"] = "\n\n".join(system_parts)

    if tools:
        api_tools = []
        for t in tools:
            fn = t.get("function", {})
            api_tools.append({
                "type": "function",
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
                "strict": False,
            })
        payload["tools"] = api_tools
        payload["tool_choice"] = req.get("tool_choice", "auto")

    return payload


class ProxyHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        log.info(fmt % args)

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            req = urllib.request.Request(
                f"{BACKEND_BASE}/v1/models",
                headers={"Authorization": f"Bearer {API_KEY}"},
            )
            try:
                with urllib.request.urlopen(req, timeout=10) as r:
                    body = r.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                self._error(500, str(e))
        else:
            self._error(404, "not found")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            req = json.loads(body)
        except Exception:
            self._error(400, "bad json")
            return

        if self.path.startswith("/v1/chat/completions"):
            self._handle_chat(req)
        else:
            self._error(404, "not found")

    def _handle_chat(self, req: dict):
        payload = chat_to_responses(req)
        data = json.dumps(payload).encode()

        api_req = urllib.request.Request(
            f"{BACKEND_BASE}/v1/responses",
            data=data,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(api_req, timeout=120) as r:
                resp_body = r.read()
            resp_json = json.loads(resp_body)
            chat_resp = responses_to_chat(resp_json)
            out = json.dumps(chat_resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out)
        except urllib.error.HTTPError as e:
            err = e.read().decode()
            log.error("backend error %s: %s", e.code, err)
            self._error(e.code, err)
        except Exception as e:
            log.error("proxy error: %s", e)
            self._error(500, str(e))

    def _error(self, code: int, msg: str):
        body = json.dumps({"error": {"message": msg}}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=16060)
    args = parser.parse_args()
    server = HTTPServer(("127.0.0.1", args.port), ProxyHandler)
    log.info("Proxy listening on http://127.0.0.1:%d", args.port)
    server.serve_forever()
