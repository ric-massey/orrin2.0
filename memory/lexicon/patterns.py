from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Iterable, Tuple
import re

__all__ = ["DefinitionHit", "DefHit", "extract_definitions", "extract_definitions_rich", "by_term"]

# -------------------------------
# Small helpers
# -------------------------------
_QUOTE_RE = re.compile(r'^[\s\'"`“”‘’]+|[\s\'"`“”‘’]+$')
_WS_RE = re.compile(r"\s+")
_ARTICLE_RE = re.compile(r"^(?:an?|the)\s+", flags=re.I)

def _strip_quotes(s: str) -> str:
    return _QUOTE_RE.sub("", (s or "").strip())

def _clean_spaces(s: str) -> str:
    return _WS_RE.sub(" ", (s or "").strip())

def _norm_term(s: str) -> str:
    return _clean_spaces(_strip_quotes(s))

def _clean_defn(s: str) -> str:
    # Keep leading article (“a/an/the …”) if present; trim quotes & trailing punctuation.
    s = _strip_quotes(s)
    s = s.strip().rstrip(".,;: ")
    return _clean_spaces(s)

def _norm_def_for_key(s: Optional[str]) -> str:
    """Normalized definition for dedup keys: lowercase, no leading article, tight spaces."""
    s = _clean_defn(s or "").lower()
    s = _ARTICLE_RE.sub("", s)  # drop leading a/an/the for keying
    return s

def _split_alias_blob(blob: str) -> List[str]:
    """
    Turn a raw alias clause into individual aliases.
    Handles commas, slashes, 'and', 'or', and phrases like 'and sometimes'.
    Strips quotes/backticks and dedups. Also trims trailing punctuation like '.' or ';'.
    """
    if not blob:
        return []
    # Normalize conjunction-y phrases into commas to simplify splitting
    tmp = re.sub(r"\b(?:and|or)\b", ",", blob, flags=re.I)
    tmp = re.sub(r"\b(?:and\s+)?sometimes\b", ",", tmp, flags=re.I)

    # First split by commas, then split each token by '/'
    parts: List[str] = []
    for chunk in (p for p in (c.strip() for c in tmp.split(",")) if p):
        for piece in (pp for pp in (p.strip() for p in chunk.split("/")) if pp):
            # Remove trailing punctuation that may appear *outside* quotes/backticks
            piece = piece.rstrip(".,;: ")
            # Strip quotes/backticks and normalize spaces
            piece = _norm_term(piece)
            if piece:
                parts.append(piece)

    # Dedup, keep order (case-insensitive)
    seen = set()
    out: List[str] = []
    for a in parts:
        key = a.lower()
        if key and key not in seen:
            seen.add(key)
            out.append(a)
    return out

# -------------------------------
# Result type
# -------------------------------
@dataclass
class DefinitionHit:
    term: str
    definition: Optional[str]
    aliases: List[str]
    pattern: str
    # extras for QA/UX
    confidence: float = 0.75
    span: Tuple[int, int] = (0, 0)

    # Let callers optionally unpack as (term, definition)
    def __iter__(self):
        yield self.term
        yield self.definition

    def to_tuple(self) -> Tuple[str, Optional[str], List[str]]:
        return (self.term, self.definition, list(self.aliases))

# Back-compat alias some callers may use
DefHit = DefinitionHit

# -------------------------------
# Core extractors
# -------------------------------

# Parenthetical definition, but do not cross sentence boundaries for term.
_PAREN_DEF = re.compile(
    r"""
    (?P<term>["'`“”‘’]?[A-Za-z0-9][A-Za-z0-9 \-_/]*?["'`“”‘’]?)      # head term (no dot)
    \s*\(\s*(?!(?:aka|also\s+called)\b)(?P<def>[^)]+?)\s*\)           # (definition) unless aka/also called
    """,
    re.X | re.I | re.M,
)

# "The definition of X is Y"  — allow commas in Y, stop at period/semicolon or EOL.
_DEFINITION_OF = re.compile(
    r"""
    \bdefinition\s+of\s+
    (?P<term>["'`“”‘’]?[A-Za-z0-9][A-Za-z0-9 .\-_\/]*?["'`“”‘’]?)
    \s+is\s+
    (?P<def>[^.;\n]+)
    (?=[.;]|$)
    """,
    re.X | re.I | re.M,
)

# X is/means/stands for Y  — allow commas in Y, stop at period/semicolon or EOL.
_IS_DEF = re.compile(
    r"""
    (?P<term>["'`“”‘’]?[A-Za-z0-9][A-Za-z0-9 .\-_/]*?["'`“”‘’]?)     # term (quoted or not)
    \s+(?:is|means|stands\s+for)\s+                                  # verb
    (?P<def>[^.;\n]+)                                                # definition to sentence/semicolon; commas allowed
    (?=[.;]|$)
    """,
    re.X | re.I | re.M,
)

# Appositive: Foo, a bar, is ...
_APPOSITIVE = re.compile(
    r"""
    (?P<term>[A-Za-z0-9][A-Za-z0-9 .\-_/]*?)\s*,\s+
    (?P<def>(?:an?|the)\s+[^,]{1,})\s*,\s+
    (?:is|are)\b
    """,
    re.X | re.I | re.M,
)

# Colon / dash style: "ECU: engine control unit", "Torque — rotational force", "Drag - resistance"
_COLON_OR_DASH = re.compile(
    r"""
    (?P<term>[A-Za-z0-9][A-Za-z0-9 .\-_\/]*?)\s*
    (?::|—|-)\s+
    (?P<def>[^.\n;]+)
    """,
    re.X | re.M,
)

# AKA in parenthesis: "HTTP (aka ...)"
_AKA_PAREN = re.compile(
    r"""
    (?P<term>[A-Za-z0-9][A-Za-z0-9 .\-_\/]*?)\s*
    \(\s*aka\s*(?P<aliases>[^)]+?)\s*\)
    """,
    re.X | re.I | re.M,
)

# Inline AKA: "JSON aka JavaScript Object Notation"
# Disallow dot in term so we don't span prior sentence fragments like "is common. JSON aka ..."
_AKA_INLINE = re.compile(
    r"""
    (?<!\w)
    (?P<term>[A-Za-z0-9][A-Za-z0-9 \-_\/]*?)\s+
    (?:aka|also\s+called)\s+
    (?P<aliases>[^;\n]+)
    """,
    re.X | re.I | re.M,
)

# Ignore short/stopwordy terms
_PRONOUNS = {"it", "this", "that", "he", "she", "they", "we"}
_MAX_TERM_LEN = 64

def _term_ok(t: str) -> bool:
    if not t:
        return False
    tl = t.lower()
    if tl in _PRONOUNS:
        return False
    if len(t) > _MAX_TERM_LEN:
        return False
    return True

def _mk_hit(term: str, defn: Optional[str], aliases: List[str], pattern: str, m: re.Match, conf: float) -> Optional[DefinitionHit]:
    term = _norm_term(term)
    if not _term_ok(term):
        return None
    if defn is not None:
        defn = _clean_defn(defn)
        if len(defn) < 1:
            return None
        # Filter tautologies: "Foo is Foo" and "Foo is the Foo"
        if _norm_def_for_key(defn) == term.lower():
            return None
    # For AKA patterns, remove alias identical to term; drop hit if none left
    if pattern == "aka":
        aliases = [a for a in aliases if a.lower() != term.lower()]
        if not aliases:
            return None
    # appositive minimum len guard (after cleaning)
    if pattern == "appositive" and defn is not None and len(defn) < 3:
        return None
    return DefinitionHit(term=term, definition=defn, aliases=aliases, pattern=pattern, confidence=conf, span=m.span())

def _dedup_hits(hits: List[DefinitionHit]) -> List[DefinitionHit]:
    """
    Deduplicate hits per term. For definition-bearing patterns, ignore the pattern
    and dedup purely by normalized definition (so 'the engine control unit' collapses
    with 'engine control unit'). For AKA, dedup by sorted alias set.
    """
    out: List[DefinitionHit] = []
    seen_keys = set()
    for h in hits:
        if h.definition is not None:
            key = (h.term.lower(), _norm_def_for_key(h.definition))
        else:
            key = (h.term.lower(), tuple(sorted(a.lower() for a in h.aliases)))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        out.append(h)
    return out

# -------------------------------
# Public API
# -------------------------------
def extract_definitions_rich(text: str) -> List[DefinitionHit]:
    text = text or ""
    hits: List[DefinitionHit] = []

    # 0) Quick filter to ignore code fences (reduce false positives)
    fenced = []
    fence = re.compile(r"^\s*(`{3,}|~{3,}).*?$.*?^\s*\1\s*$", re.M | re.S)  # supports indentation and ~~~
    for m in fence.finditer(text):
        fenced.append((m.start(), m.end()))
    def _inside_fence(pos: int) -> bool:
        return any(a <= pos < b for a, b in fenced)

    # 1) Parenthetical definition
    for m in _PAREN_DEF.finditer(text):
        if _inside_fence(m.start()):
            continue
        hit = _mk_hit(m.group("term"), m.group("def"), [], "paren_def", m, 0.7)
        if hit:
            hits.append(hit)

    # 1b) "The definition of X is Y"
    for m in _DEFINITION_OF.finditer(text):
        if _inside_fence(m.start()):
            continue
        hit = _mk_hit(m.group("term"), m.group("def"), [], "definition_of", m, 0.85)
        if hit:
            hits.append(hit)

    # 2) “X is/means/stands for Y”
    for m in _IS_DEF.finditer(text):
        if _inside_fence(m.start()):
            continue
        hit = _mk_hit(m.group("term"), m.group("def"), [], "is_means_equals", m, 0.8)
        if hit:
            hits.append(hit)

    # 3) Appositive
    for m in _APPOSITIVE.finditer(text):
        if _inside_fence(m.start()):
            continue
        hit = _mk_hit(m.group("term"), m.group("def"), [], "appositive", m, 0.65)
        if hit:
            hits.append(hit)

    # 4) Colon/dash
    for m in _COLON_OR_DASH.finditer(text):
        if _inside_fence(m.start()):
            continue
        hit = _mk_hit(m.group("term"), m.group("def"), [], "colon_dash", m, 0.75)
        if hit:
            hits.append(hit)

    # 5) AKA (parenthetical)
    for m in _AKA_PAREN.finditer(text):
        if _inside_fence(m.start()):
            continue
        term = m.group("term")
        aliases = _split_alias_blob(m.group("aliases"))
        hit = _mk_hit(term, None, aliases, "aka", m, 0.7)
        if hit and hit.aliases:
            hits.append(hit)

    # 6) AKA (inline)
    for m in _AKA_INLINE.finditer(text):
        if _inside_fence(m.start()):
            continue
        term = m.group("term")
        aliases = _split_alias_blob(m.group("aliases"))
        hit = _mk_hit(term, None, aliases, "aka", m, 0.7)
        if hit and hit.aliases:
            hits.append(hit)

    # Dedup & return
    return _dedup_hits(hits)

def extract_definitions(text: str) -> List[Tuple[str, str, List[str]]]:
    """
    Lightweight helper used by the daemon: triples (term, definition, aliases).
    Alias-only hits are omitted here.
    """
    out: List[Tuple[str, str, List[str]]] = []
    for h in extract_definitions_rich(text):
        if h.definition:
            out.append((h.term, h.definition, list(h.aliases)))
    return out

def by_term(hits: Iterable[DefinitionHit], term: str) -> List[DefinitionHit]:
    t = (_norm_term(term)).lower()
    return [h for h in hits if _norm_term(h.term).lower() == t]
