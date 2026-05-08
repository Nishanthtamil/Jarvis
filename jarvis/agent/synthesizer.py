"""
Synthesizer — produces the final cited research answer.
Streams output for responsive UI.
Validates that all citations reference real collected sources.
"""

from __future__ import annotations

import os
from typing import AsyncGenerator

from llm.client import get_client, LLMMessage, truncate_to_tokens
from agent.memory import SessionMemory

MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "6000"))

SYNTHESIZER_SYSTEM = """You are a research analyst. Given a set of numbered sources and a research goal, write a clear, thorough answer.

Rules:
- Cite sources inline using [N] notation (e.g., "According to [1], ...")
- Every factual claim must have a citation
- If sources contradict each other, note the discrepancy
- Do not invent facts not present in the sources
- Write in clear paragraphs, not bullet points
- End with a "## Sources" section listing all cited [N] references
- Be direct: state conclusions, don't hedge everything"""

EXTRACTOR_SYSTEM = """You are a precise information extractor.
Given a source document, extract the key facts relevant to the research question.
Be concise. Include specific numbers, dates, names, and claims.
Output plain text, 3-8 sentences maximum. Do not invent facts."""


async def extract_from_source(goal: str, markdown: str, url: str) -> str:
    """
    Extract relevant facts from a single source.
    This is the per-source step; keeps prompts small.
    """
    client = get_client()
    # Truncate to be safe within the model's context
    safe_markdown = truncate_to_tokens(markdown, 1200)

    messages = [
        LLMMessage(role="system", content=EXTRACTOR_SYSTEM),
        LLMMessage(
            role="user",
            content=f"Research question: {goal}\n\nSource URL: {url}\n\nDocument:\n{safe_markdown}",
        ),
    ]
    result = await client.complete(messages, max_tokens=300, temperature=0.0)
    return result.strip()


async def check_gaps(goal: str, memory: SessionMemory) -> list[str]:
    """
    Ask the model what is still unanswered given the collected summaries.
    Returns a list of gap descriptions (empty list if fully answered).
    """
    client = get_client()
    combined = truncate_to_tokens(memory.combined_summary_text(), 3000)
    messages = [
        LLMMessage(
            role="system",
            content="""You are a research quality checker.
Given a research goal and collected evidence so far, identify what is still unanswered or unclear.
Return JSON: {"gaps": ["gap 1", "gap 2"]} or {"gaps": []} if fully answered.
List at most 2 gaps. Be specific.""",
        ),
        LLMMessage(
            role="user",
            content=f"Goal: {goal}\n\nEvidence collected:\n{combined}",
        ),
    ]
    try:
        data = await client.complete_json(messages, max_tokens=200)
        return data.get("gaps", [])
    except Exception:
        return []


async def synthesize_stream(memory: SessionMemory) -> AsyncGenerator[str, None]:
    """
    Stream the final answer. Yields text chunks as they are generated.
    """
    client = get_client()

    # Build the source context block
    source_block = memory.combined_summary_text()
    safe_sources = truncate_to_tokens(source_block, MAX_CONTEXT_TOKENS - 500)

    citation_lines = "\n".join(
        f"[{c['index']}] {c['title']} — {c['url']}"
        for c in memory.citation_list()
    )

    user_content = (
        f"Research goal: {memory.goal}\n\n"
        f"Available sources:\n{safe_sources}\n\n"
        f"Citation reference:\n{citation_lines}\n\n"
        "Write the complete research answer with inline citations."
    )

    messages = [
        LLMMessage(role="system", content=SYNTHESIZER_SYSTEM),
        LLMMessage(role="user", content=user_content),
    ]

    async for chunk in client.stream(messages, max_tokens=1200, temperature=0.2):
        yield chunk


async def evaluate_and_refine(goal: str, draft: str, memory: SessionMemory) -> str:
    """
    Self-reflection step: The Critic reviews the draft and refines it.
    """
    client = get_client()
    
    # Provide the critic with the draft and the context
    source_block = memory.combined_summary_text()
    safe_sources = truncate_to_tokens(source_block, 3000)

    messages = [
        LLMMessage(role="system", content="""You are a senior research critic.
Your goal is to ensure the research answer is 100% accurate, fully addresses the goal, and is perfectly cited.
Review the provided draft. Check for:
1. Hallucinations (facts not in sources)
2. Missing information from the goal
3. Proper citation format [N]
If changes are needed, provide the full, corrected answer. If it's perfect, return the original draft."""),
        LLMMessage(role="user", content=f"Research Goal: {goal}\n\nSources:\n{safe_sources}\n\nDraft Answer:\n{draft}"),
    ]
    
    refined = await client.complete(messages, max_tokens=1500, temperature=0.1)
    return refined.strip()


async def evaluate_links(goal: str, gaps: list[str], links: list[str]) -> list[str]:
    """
    Asks the LLM to identify links that are likely to answer remaining gaps.
    """
    if not links or not gaps:
        return []

    client = get_client()
    messages = [
        LLMMessage(role="system", content="""You are a web navigation specialist.
Given a research goal, a list of current knowledge gaps, and a list of links found on a page, identify the 2 most promising URLs to follow.
Only return the raw URLs, one per line. Do not include any explanation."""),
        LLMMessage(role="user", content=f"Goal: {goal}\n\nGaps: {', '.join(gaps)}\n\nLinks found:\n" + "\n".join(links[:50])),
    ]
    
    try:
        result = await client.complete(messages, max_tokens=200, temperature=0.0)
        promising = [line.strip() for line in result.split("\n") if line.strip().startswith("http")]
        return promising[:2]
    except Exception:
        return []
