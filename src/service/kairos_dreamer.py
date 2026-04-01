"""
KAIROS Dreamer - Memory consolidation service.

Receives batches of conversation memories from the Rust daemon via IPC,
extracts factual knowledge nodes using OpenAI's API, and returns structured
facts for persistence to the knowledge graph.

No torch/transformers dependencies - uses direct OpenAI API calls.
"""
import os
import sys
import json
import asyncio
from typing import List
import openai

# TCP port must match Rust's KAIROS_TCP_PORT (48765 on Windows)
IS_WINDOWS = os.name == "nt"
TCP_PORT = 48765
SOCKET_PATH = "/tmp/kairos_dreamer.sock"

SYSTEM_PROMPT = (
    "Extract factual assertions, preferences, and states into dense knowledge nodes. "
    "Ignore pleasantries. Return a JSON object with a 'compressed_nodes' array, "
    "where each node has: entity, relationship, target, context."
)

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "KairosConsolidation",
        "schema": {
            "type": "object",
            "properties": {
                "compressed_nodes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "entity": {"type": "string", "description": "The core subject"},
                            "relationship": {"type": "string", "description": "How it relates"},
                            "target": {"type": "string", "description": "The object of the relationship"},
                            "context": {"type": "string", "description": "Dense summary of the memory"}
                        },
                        "required": ["entity", "relationship", "target", "context"]
                    }
                }
            },
            "required": ["compressed_nodes"]
        }
    }
}


def format_memory(m: dict) -> str:
    """Format a memory row into a readable log line."""
    timestamp = m.get("created_at", m.get("timestamp", ""))
    role = m.get("role", "user")
    content = m.get("content", "")
    return f"[{timestamp}] {role}: {content}"


async def process_batch(rows: List[dict]) -> dict:
    """Process batch and return KairosBatchResponse-compatible dict."""
    if not rows:
        return {"source_ids": [], "facts": []}

    source_ids = [m.get("id") for m in rows if "id" in m]
    script = "\n".join(format_memory(m) for m in rows)

    try:
        # Use direct OpenAI API - no langchain/transformers/torch
        client = openai.OpenAI()

        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Raw logs:\n{script}"}
            ],
            response_format=RESPONSE_FORMAT,
            temperature=0.1,
        )

        content = response.choices[0].message.parsed
        facts = [
            {
                "entity": node.entity,
                "relationship": node.relationship,
                "target": node.target,
                "context": node.context,
            }
            for node in content.compressed_nodes
        ]

        return {"source_ids": source_ids, "facts": facts}

    except Exception as e:
        print(f"[KAIROS ERROR] {e}", file=sys.stderr)
        return {"source_ids": source_ids, "facts": []}


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    """Handle IPC connection from Rust daemon. Rust sends one JSON line + newline."""
    try:
        data = await reader.readline()
        if not data:
            return

        payload = json.loads(data.decode("utf-8").strip())
        # Rust KairosBatchRequest has 'request_id' and 'rows'
        rows = payload.get("rows", [])
        print(f"[KAIROS] Processing batch of {len(rows)} memories...")

        response = await process_batch(rows)

        # Send JSON response followed by newline (Rust reads with read_line)
        writer.write(json.dumps(response).encode("utf-8"))
        writer.write(b"\n")
        await writer.drain()
        print("[KAIROS] Response sent to Rust daemon.")
    except asyncio.IncompleteReadError:
        pass
    except Exception as e:
        print(f"[KAIROS DECODE ERROR] {e}", file=sys.stderr)
        writer.write(b'{"source_ids": [], "facts": []}\n')
        await writer.drain()
    finally:
        writer.close()
        await writer.wait_closed()


async def main():
    if IS_WINDOWS:
        server = await asyncio.start_server(handle_client, "127.0.0.1", TCP_PORT)
        print(f"[KAIROS] Listening on TCP {TCP_PORT} (Windows)")
    else:
        if os.path.exists(SOCKET_PATH):
            os.remove(SOCKET_PATH)
        server = await asyncio.start_unix_server(handle_client, path=SOCKET_PATH)
        print(f"[KAIROS] Listening on UDS {SOCKET_PATH}")

    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
