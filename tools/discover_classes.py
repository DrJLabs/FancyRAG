"""Locate key GraphRAG component implementations to prevent import drift."""

import importlib
import inspect
import pkgutil

TARGETS = {
    "FixedSizeSplitter": "splitter",
    "LLMEntityRelationExtractor": "extractor",
    "Neo4jWriter": "writer",
    "OpenAILLM": "llm",
    "OpenAIEmbeddings": "embedder",
}


def walk(prefix: str):
    """Yield module names under the given package prefix."""
    module = importlib.import_module(prefix)
    yield from (
        name for _, name, _ in pkgutil.walk_packages(module.__path__, prefix + ".")
    )


def resolve(mod_name: str, class_name: str):
    """Return fully-qualified reference if class exists in module."""
    try:
        module = importlib.import_module(mod_name)
        value = getattr(module, class_name, None)
        return f"{mod_name}.{class_name}" if inspect.isclass(value) else None
    except Exception:
        return None


def main():
    hits = {}
    for module_name in walk("neo4j_graphrag"):
        for class_name, key in TARGETS.items():
            if key in hits:
                continue
            fq_name = resolve(module_name, class_name)
            if fq_name:
                hits[key] = fq_name
    for key in TARGETS.values():
        print(f"{key}={hits.get(key, 'NOT_FOUND')}")


if __name__ == "__main__":
    main()
