from __future__ import annotations
from typing import Callable, Any

TOOL_REGISTRY: dict[str, dict] = {}

def register_tool(name: str, description: str):
    def decorator(fn: Callable):
        TOOL_REGISTRY[name] = {
            "name": name,
            "description": description,
            "fn": fn
        }
        return fn
    return decorator

def get_tool(name: str) -> Callable:
    if name not in TOOL_REGISTRY:
        raise ValueError(f"Tool {name} not found")
    return TOOL_REGISTRY[name]["fn"]

def list_tools() -> list[dict]:
    return [
        {"name": info["name"], "description": info["description"]}
        for info in TOOL_REGISTRY.values()
    ]

# Backward compatibility and initial registration
from tools.search import search as search_web
from tools.crawl import fetch_url
from tools.pdf import extract_pdf_url as extract_pdf

register_tool("search_web", "Search the web for current information and news.")(search_web)
register_tool("fetch_url", "Extract clean text content from a web URL.")(fetch_url)
register_tool("extract_pdf", "Download and extract text from a PDF URL.")(extract_pdf)
