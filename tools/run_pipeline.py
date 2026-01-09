"""Async entrypoint to execute the Neo4j GraphRAG pipeline from a file payload."""

import asyncio
import json
import sys

from dotenv import load_dotenv
from neo4j_graphrag.experimental.pipeline.config.runner import PipelineRunner


async def main():
    load_dotenv(".env.local")
    config_path = "pipelines/kg_ingest.yaml"
    payload = {"splitter": {"text": "Hello graph"}}
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8") as handle:
            payload["splitter"]["text"] = handle.read()
    runner = PipelineRunner.from_config_file(config_path)
    result = await runner.run(payload)
    print(json.dumps({"ok": True, "stats": getattr(result, "stats", None)}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
