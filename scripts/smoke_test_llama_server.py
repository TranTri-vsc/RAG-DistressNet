#!/usr/bin/env python3
import argparse
import json
from urllib import request


def normalize_base_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    return base_url


def get_json(url: str, api_key: str) -> dict:
    req = request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    with request.urlopen(req, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url: str, payload: dict, api_key: str) -> dict:
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test a local llama.cpp OpenAI-compatible server")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1", help="Server base URL")
    parser.add_argument("--model", required=True, help="Model name exposed by the server")
    parser.add_argument("--api-key", default="EMPTY", help="API key for the local server")
    parser.add_argument(
        "--prompt",
        default="Reply with one sentence that confirms the model is running locally.",
        help="Prompt to send to the server",
    )
    args = parser.parse_args()

    base_url = normalize_base_url(args.base_url)
    models_payload = get_json(f"{base_url}/models", args.api_key)
    print("[INFO] /models response:")
    print(json.dumps(models_payload, indent=2))

    chat_payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "temperature": 0.0,
    }
    chat_response = post_json(f"{base_url}/chat/completions", chat_payload, args.api_key)
    message = chat_response["choices"][0]["message"]["content"]

    print()
    print("[INFO] chat/completions response:")
    print(message)


if __name__ == "__main__":
    main()
