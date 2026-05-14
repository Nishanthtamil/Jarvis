"""
Orchestrator — the main ReAct loop.
Emits progress events as async generator for streaming to the frontend.

Event schema:
  {"type": "status", "message": "..."}
  {"type": "search", "query": "...", "count": N}
  {"type": "fetch", "url": "...", "ok": bool}
  {"type": "summary", "url": "...", "text": "..."}
  {"type": "gap", "gaps": [...]}
  {"type": "answer_chunk", "text": "..."}
  {"type": "done", "citations": [...], "iterations": N}
  {"type": "error", "message": "..."}
"""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import AsyncGenerator

import re
from agent.memory import SessionMemory, SourceSummary
from agent.planner import plan
from agent.synthesizer import extract_from_source, check_gaps, synthesize_stream, evaluate_and_refine, evaluate_links
from store.vector import add_to_vector_store
from store.db import (
    create_session,
    log_search,
    log_source,
    log_summary,
    update_session_answer,
)
from tools.search import search
from tools.crawl import fetch_many
from tools.browser import fetch_with_browser
from tools.pdf import extract_pdf_url, is_pdf_url

MAX_REACT_ITERATIONS = int(os.getenv("MAX_REACT_ITERATIONS", "8"))
MAX_SOURCES_PER_ROUND = int(os.getenv("MAX_SOURCES_PER_ROUND", "4"))


def _event(type_: str, **kwargs) -> dict:
    return {"type": type_, **kwargs}


async def run_research(goal: str, session_id: str = None) -> AsyncGenerator[dict, None]:
    """
    Main entry point. Yields progress events as dicts.
    Caller (FastAPI) converts these to SSE JSON lines.
    """
    if not session_id:
        session_id = str(uuid.uuid4())
        await create_session(session_id, goal)
    
    memory = SessionMemory(goal=goal, session_id=session_id)

    yield _event("status", message=f"Starting research session {session_id[:8]}…")

    session_token_budget = int(os.getenv("SESSION_TOKEN_BUDGET", "50000"))
    session_tokens_used = 0

    past = await episodic_memory.search(goal, n=3)
    if past:
        yield _event("status", message=f"Found {len(past)} related past sessions")

    # ── PLAN ──────────────────────────────────────────────────────────────────
    try:
        research_plan = await plan(goal)
        yield _event("status", message=f"Plan ready: {len(research_plan.subtasks)} subtasks")
    except Exception as e:
        yield _event("error", message=f"Planning failed: {e}")
        return

    # ── REACT LOOP ────────────────────────────────────────────────────────────
    pending_queries = [st.query for st in research_plan.subtasks]
    pending_urls = []

    for iteration in range(MAX_REACT_ITERATIONS):
        memory.iteration = iteration

        if not pending_queries and not pending_urls:
            break

        # ── DISPATCH ──────────────────────────────────────────────────────────
        candidate_urls = []
        query = None

        if pending_urls:
            candidate_urls = [pending_urls.pop(0)]
            yield _event("status", message=f"[{iteration+1}] Deep Dive: {candidate_urls[0]}")
        else:
            query = pending_queries.pop(0)
            if memory.is_seen(f"query:{query}"):
                continue
            memory.add_query(query)
            yield _event("status", message=f"[{iteration+1}] Searching: {query}")

        # ── SEARCH ────────────────────────────────────────────────────────────
        if query:
            try:
                results = await search(query)
                await log_search(session_id, query, results)
                yield _event("search", query=query, count=len(results))
                candidate_urls = [
                    r["url"] for r in results
                    if not memory.is_seen(r["url"])
                ][:MAX_SOURCES_PER_ROUND]
            except Exception as e:
                yield _event("error", message=f"Search failed: {e}")
                continue

        for url in candidate_urls:
            memory.add_url(url)

        if not candidate_urls:
            if query:
                yield _event("status", message="No new URLs found for this query")
            continue

        # ── FETCH ─────────────────────────────────────────────────────────────
        # PDF URLs go to the PDF extractor; others go to Crawl4AI
        pdf_urls = [u for u in candidate_urls if is_pdf_url(u)]
        web_urls = [u for u in candidate_urls if not is_pdf_url(u)]

        fetched = []

        if web_urls:
            web_results = await fetch_many(web_urls)
            fetched.extend(web_results)

            # Retry failed URLs with Playwright
            failed_urls = [u for u in web_urls if u not in {r["url"] for r in web_results}]
            for fu in failed_urls[:2]:  # max 2 browser retries per round
                br = await fetch_with_browser(fu)
                if br["ok"]:
                    fetched.append(br)

        for pu in pdf_urls:
            pr = await extract_pdf_url(pu)
            if pr["ok"]:
                fetched.append(pr)

        # ── PARALLEL EXTRACTION ───────────────────────────────────────────────
        extraction_tasks = []
        valid_results = []
        
        for fr in fetched:
            ok = fr.get("ok", False)
            yield _event("fetch", url=fr["url"], ok=ok)

            if ok and fr.get("markdown"):
                session_tokens_used += fr.get("token_count", 0)
                if session_tokens_used > session_token_budget:
                    yield _event("error", message=f"Token budget exceeded ({session_tokens_used} > {session_token_budget}). Stopping fetch.")
                    pending_queries.clear()
                    pending_urls.clear()
                    break

                # Log source to DB
                source_id = await log_source(
                    session_id,
                    fr["url"],
                    fr.get("title", ""),
                    fr.get("domain", ""),
                    fr.get("token_count", 0),
                    True,
                )
                valid_results.append((fr, source_id))
                extraction_tasks.append(extract_from_source(goal, fr["markdown"], fr["url"]))
                # Index in vector store for RAG
                await add_to_vector_store(session_id, fr["url"], fr["markdown"], {
                    "title": fr.get("title", ""),
                    "domain": fr.get("domain", "")
                })

                # ── LINK EXTRACTION (Recursive) ───────────────────────────────
                if iteration < MAX_REACT_ITERATIONS - 3:
                    found_links = re.findall(r'\[.*?\]\((https?://.*?)\)', fr["markdown"])
                    if found_links:
                        gaps = await check_gaps(goal, memory)
                        if gaps:
                            promising = await evaluate_links(goal, gaps, found_links)
                            for p_url in promising:
                                if not memory.is_seen(p_url):
                                    pending_urls.append(p_url)

        if extraction_tasks:
            summaries = await asyncio.gather(*extraction_tasks, return_exceptions=True)
            
            for (fr, source_id), summary_text in zip(valid_results, summaries):
                if isinstance(summary_text, Exception):
                    yield _event("status", message=f"Extraction error for {fr['url']}: {summary_text}")
                    continue

                await log_summary(session_id, source_id, summary_text)

                s = SourceSummary(
                    url=fr["url"],
                    title=fr.get("title", ""),
                    domain=fr.get("domain", ""),
                    summary=summary_text,
                    token_count=fr.get("token_count", 0),
                )
                memory.add_summary(s)
                yield _event("summary", url=fr["url"], text=summary_text[:200] + "…")

        # ── GAP CHECK ─────────────────────────────────────────────────────────
        if memory.has_enough_sources() and iteration < MAX_REACT_ITERATIONS - 2:
            gaps = await check_gaps(goal, memory)
            memory.set_gaps(gaps)

            if gaps:
                yield _event("gap", gaps=gaps)
                # Convert gaps into new search queries for the next iterations
                for gap in gaps[:2]:
                    pending_queries.insert(0, gap)
            else:
                # No gaps — exit the loop early
                yield _event("status", message="No gaps detected. Moving to synthesis.")
                break

        # Small delay between iterations to avoid hammering services
        await asyncio.sleep(0.5)

    # ── SYNTHESIZE ────────────────────────────────────────────────────────────
    if not memory.summaries:
        yield _event("error", message="No sources collected. Cannot synthesize answer.")
        return

    yield _event("status", message=f"Synthesizing answer from {len(memory.summaries)} sources…")

    full_answer = []
    async for chunk in synthesize_stream(memory):
        yield _event("answer_chunk", text=chunk)
        full_answer.append(chunk)

    final_answer = "".join(full_answer)
    
    # ── REFLECT ───────────────────────────────────────────────────────────────
    yield _event("status", message="Refining answer for accuracy…")
    refined_answer = await evaluate_and_refine(goal, final_answer, memory)
    
    await update_session_answer(session_id, refined_answer)

    # ── STORE MEMORY ──────────────────────────────────────────────────────────
    await episodic_memory.store_session(session_id, goal, refined_answer)
    await evaluate_and_store_user_facts(session_id, goal, refined_answer)

    yield _event(
        "done",
        citations=memory.citation_list(),
        iterations=memory.iteration + 1,
        session_id=session_id,
    )
