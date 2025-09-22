# goals/handlers/research.py
# Concrete handler for research/googling/reading/synthesis tasks; uses ctx hooks for web+llm

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..model import Goal, Step, Status
from .base import BaseGoalHandler, HandlerContext

UTCNOW = lambda: datetime.now(timezone.utc)


# ---------- tiny utilities ----------

def _new_step(goal_id: str, name: str, action: Dict[str, Any], *, max_attempts: int = 3, deps: Optional[List[str]] = None) -> Step:
    return Step(
        id=f"s_{uuid.uuid4().hex[:10]}",
        goal_id=goal_id,
        name=name,
        action=action,
        max_attempts=max_attempts,
        deps=list(deps or []),
        status=Status.READY,
    )


def _artifacts_dir(ctx: HandlerContext, goal: Goal) -> Path:
    base = Path(ctx.get("artifacts_dir") or "data/goals/artifacts").resolve()
    d = base / goal.id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_text(dirpath: Path, name: str, text: str) -> str:
    p = dirpath / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def _write_json(dirpath: Path, name: str, obj: Any) -> str:
    p = dirpath / name
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(p)


def _ensure_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        return [x]
    return list(x)


def _unique_preserve(seq: List[Any]) -> List[Any]:
    seen = set()
    out: List[Any] = []
    for s in seq:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _load_latest_json(art_dir: Path, startswith: str) -> Optional[Any]:
    files = sorted([p for p in art_dir.glob(f"{startswith}*.json")], key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return None
    try:
        return json.loads(files[0].read_text(encoding="utf-8"))
    except Exception:
        return None


# ---------- handler ----------

class ResearchHandler(BaseGoalHandler):
    """
    Research pipeline with pluggable hooks:

    Context hooks (expected in `ctx` dict if you want the corresponding step to work):
      - web_search(query: str, k: int) -> list[{"title","url","snippet"}]
      - web_fetch(url: str, timeout: int|None=None) -> str                        # returns plaintext/HTML/markdown
      - llm(prompt: str, **kw) -> str                                            # generic completion
      - rerank(query: str, docs: list[dict]) -> list[int]                        # optional ranker, returns index order

    goal.spec keys (all optional unless noted):
      - queries: list[str] | str       # if missing, we will draft queries with llm or from title
      - urls: list[str]                # seed sources to fetch (skips search if provided)
      - per_query_k: int               # top-k results per query (default 5)
      - fetch_limit: int               # total docs to fetch across all queries/urls (default 10)
      - synth_kind: str                # "memo" (default), "bullets", or "report"
      - output_name: str               # default "research_memo.md"
      - include_citations: bool        # default True
      - style: str                     # optional style hint for writing
    """
    kind: str = "research"

    # ---------- Planning ----------

    def plan(self, goal: Goal, ctx: HandlerContext) -> List[Step]:
        spec = goal.spec or {}
        steps: List[Step] = []

        have_queries = bool(spec.get("queries"))
        have_urls = bool(spec.get("urls"))

        prev: Optional[str] = None

        if not have_queries and not have_urls:
            s = _new_step(goal.id, "draft queries", {"op": "draft_queries"})
            steps.append(s); prev = s.id

        if not have_urls:
            s = _new_step(goal.id, "search", {"op": "search"})
            if prev: s.deps.append(prev)
            steps.append(s); prev = s.id

        s = _new_step(goal.id, "fetch sources", {"op": "fetch"})
        if prev: s.deps.append(prev)
        steps.append(s); prev = s.id

        s = _new_step(goal.id, "synthesize findings", {"op": "synthesize"})
        if prev: s.deps.append(prev)
        steps.append(s)

        return steps

    def is_blocked(self, goal: Goal, ctx: HandlerContext) -> Tuple[bool, Optional[str]]:
        # Research is mostly I/O and LLM bound; no global locks by default.
        return False, None

    # ---------- Execution ----------

    def tick(self, goal: Goal, step: Step, ctx: HandlerContext) -> Optional[Step]:
        op = (step.action or {}).get("op", "")
        art_dir = _artifacts_dir(ctx, goal)
        started_now = False

        if step.started_at is None:
            step.started_at = UTCNOW()
            step.status = Status.RUNNING
            started_now = True

        try:
            if op == "draft_queries":
                queries = self._draft_queries(goal, ctx)
                _write_json(art_dir, f"{step.id}_queries.json", {"queries": queries, "ts": UTCNOW().isoformat()})
                return self._finish_ok(step, art_dir, f"queries={len(queries)}")

            if op == "search":
                queries = _ensure_list((goal.spec or {}).get("queries"))
                if not queries:
                    qpack = _load_latest_json(art_dir, startswith="")
                    if isinstance(qpack, dict) and "queries" in qpack:
                        queries = _ensure_list(qpack["queries"])
                if not queries:
                    # naive fallback from title
                    base = (goal.title or "").strip()
                    if base:
                        queries = [base, f"{base} overview", f"{base} recent developments"]
                    else:
                        raise ValueError("no queries available for search step")

                per_k = int((goal.spec or {}).get("per_query_k", 5))
                search = ctx.get("web_search")
                if not callable(search):
                    raise RuntimeError("ctx.web_search hook not provided")

                results: List[Dict[str, Any]] = []
                for q in queries:
                    rows = _ensure_list(search(q, per_k))
                    for r in rows:
                        r = dict(r or {})
                        r["query"] = q
                        results.append(r)

                # Dedup by URL while preserving order
                urls = _unique_preserve([r.get("url") for r in results if r.get("url")])
                payload = {"queries": queries, "urls": urls, "results": results, "ts": UTCNOW().isoformat()}
                _write_json(art_dir, f"{step.id}_search.json", payload)
                return self._finish_ok(step, art_dir, f"urls={len(urls)} results={len(results)}")

            if op == "fetch":
                # Assemble candidate URLs from spec or prior search artifact
                urls = _ensure_list((goal.spec or {}).get("urls"))
                if not urls:
                    sres = _load_latest_json(art_dir, startswith="")
                    if isinstance(sres, dict) and "urls" in sres:
                        urls = _ensure_list(sres["urls"])
                if not urls:
                    raise ValueError("no URLs to fetch")

                fetch_limit = int((goal.spec or {}).get("fetch_limit", 10))
                web_fetch = ctx.get("web_fetch")
                if not callable(web_fetch):
                    raise RuntimeError("ctx.web_fetch hook not provided")

                fetched: List[Dict[str, Any]] = []
                for i, url in enumerate(urls[:fetch_limit], start=1):
                    try:
                        text = web_fetch(url)
                        doc_name = f"doc_{i:02d}.txt"
                        path = _write_text(art_dir, doc_name, text or "")
                        fetched.append({"url": url, "path": path, "chars": len(text or "")})
                    except Exception as e:
                        fetched.append({"url": url, "error": f"{type(e).__name__}: {e}"})

                _write_json(art_dir, f"{step.id}_docs.json", {"docs": fetched, "ts": UTCNOW().isoformat()})
                return self._finish_ok(step, art_dir, f"fetched={sum(1 for d in fetched if 'path' in d)}")

            if op == "synthesize":
                llm = ctx.get("llm")
                include_citations = bool((goal.spec or {}).get("include_citations", True))
                synth_kind = (goal.spec or {}).get("synth_kind", "memo")
                output_name = (goal.spec or {}).get("output_name", "research_memo.md")
                style = (goal.spec or {}).get("style", "")

                # Load docs manifest
                docs_pack = _load_latest_json(art_dir, startswith="")
                docs: List[Dict[str, Any]] = []
                if isinstance(docs_pack, dict) and "docs" in docs_pack:
                    docs = _ensure_list(docs_pack["docs"])
                if not docs:
                    # Try any doc_*.txt in the artifacts dir
                    for p in sorted(art_dir.glob("doc_*.txt")):
                        docs.append({"url": None, "path": str(p), "chars": p.stat().st_size})

                if not docs:
                    raise ValueError("no documents found for synthesis")

                # Build short notes (first N chars) to fit in prompt budgets; let caller control summarization normally
                snippets: List[Tuple[str, str]] = []
                for d in docs:
                    try:
                        p = Path(d["path"])
                        txt = p.read_text(encoding="utf-8", errors="ignore")
                        snippets.append((d.get("url") or p.name, txt[:6000]))  # keep modest to avoid huge prompts
                    except Exception:
                        continue

                if callable(llm):
                    prompt = _make_synthesis_prompt(goal, synth_kind, include_citations, style, snippets)
                    memo = llm(prompt) or ""
                else:
                    # Minimal offline fallback: stitch together extracts
                    memo = _offline_fallback_memo(goal, synth_kind, include_citations, snippets)

                out_path = _write_text(art_dir, output_name, memo)
                meta = {
                    "goal": asdict(goal),
                    "output": out_path,
                    "docs_used": [{"url": u, "path": p} for (u, _txt), p in zip(snippets, [d.get("path") for d in docs if d.get("path")])],
                    "ts": UTCNOW().isoformat(),
                }
                _write_json(art_dir, f"{step.id}_summary_meta.json", meta)
                # Also emit a tiny txt note
                return self._finish_ok(step, art_dir, f"wrote {Path(out_path).name}")

            # Unknown op
            raise ValueError(f"unknown op: {op!r}")

        except Exception as e:
            step.last_error = f"{type(e).__name__}: {e}"
            step.attempts += 1
            if started_now:
                step.started_at = None
                step.status = Status.READY
            if step.attempts >= step.max_attempts:
                step.status = Status.FAILED
                step.finished_at = UTCNOW()
            return step

    # ---------- helpers ----------

    def _finish_ok(self, step: Step, art_dir: Path, note: str) -> Step:
        step.status = Status.DONE
        step.finished_at = UTCNOW()
        _write_text(art_dir, f"{step.id}_ok.txt", note)
        step.last_error = None
        return step

    def _draft_queries(self, goal: Goal, ctx: HandlerContext) -> List[str]:
        base = (goal.title or "").strip()
        spec_q = _ensure_list((goal.spec or {}).get("queries"))
        if spec_q:
            return _unique_preserve(spec_q)
        llm = ctx.get("llm")
        if callable(llm) and base:
            prompt = (
                "You are a research query generator. Produce 6 diverse, concise web search queries that together cover the topic.\n"
                f"Topic: {base}\n"
                "Return ONLY a JSON array of strings."
            )
            try:
                raw = llm(prompt) or "[]"
                parsed = json.loads(raw) if raw.strip().startswith("[") else json.loads(raw.splitlines()[-1])
                return _unique_preserve([str(x).strip() for x in parsed if str(x).strip()])
            except Exception:
                pass
        # Fallback naive queries
        if base:
            return _unique_preserve([
                base,
                f"{base} overview",
                f"{base} pros and cons",
                f"{base} latest research",
                f"{base} best practices",
                f"{base} pitfalls",
            ])
        return ["emerging research overview", "recent breakthroughs summary"]

def _make_synthesis_prompt(
    goal: Goal,
    kind: str,
    include_citations: bool,
    style: str,
    snippets: List[Tuple[str, str]],
) -> str:
    title = goal.title or "Research Task"
    instructions = {
        "memo": "Write a clear 1–2 page memo with sections: TL;DR, Why it matters, Key findings, Nuances/edge cases, Recommendations, Open questions.",
        "bullets": "Write concise bullet points grouped by theme. Start with a 5-bullet TL;DR.",
        "report": "Write a structured report with headings, an abstract, background, methods (high-level), findings, discussion, and conclusion.",
    }.get(kind, "Write a concise memo.")
    cite = "Include inline citations like [1], [2] that map to a Source list at the end." if include_citations else "Do not include citations."
    style_hint = f"Style hint: {style}\n" if style else ""

    # Build a compact sources block
    src_lines = []
    for i, (src, txt) in enumerate(snippets, start=1):
        src_lines.append(f"[{i}] SOURCE: {src}\n-----\n{txt}\n")

    prompt = (
        f"Task: {title}\n\n"
        f"{instructions}\n{cite}\n{style_hint}\n"
        "You are given excerpts from sources. Read and synthesize them faithfully. If sources disagree, note disagreements.\n"
        "Avoid speculation; highlight confidence and uncertainties.\n\n"
        "SOURCES:\n" + "\n\n".join(src_lines) + "\n"
        "Output strictly in Markdown."
    )
    return prompt


def _offline_fallback_memo(
    goal: Goal,
    kind: str,
    include_citations: bool,
    snippets: List[Tuple[str, str]],
) -> str:
    title = goal.title or "Research Memo"
    lines = [f"# {title}", ""]
    lines.append("*(Offline synthesis fallback: stitched key excerpts. Provide your own LLM for better results.)*")
    lines.append("")
    lines.append("## Key excerpts")
    for i, (src, txt) in enumerate(snippets, start=1):
        head = f"- **[{i}] {src}**"
        lines.append(head)
        lines.append("")
        lines.append("```")
        lines.append(txt[:1200])
        lines.append("```")
        lines.append("")
    if include_citations:
        lines.append("## Sources")
        for i, (src, _txt) in enumerate(snippets, start=1):
            lines.append(f"[{i}] {src}")
    return "\n".join(lines)


__all__ = ["ResearchHandler"]
