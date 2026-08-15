import hashlib, re, io, asyncio, contextlib, os, json, math, sqlite3, time, threading, logging
from datetime import datetime
from typing import List, Optional, Any
from urllib.parse import quote, quote_plus
import httpx
from bs4 import BeautifulSoup
from langchain_core.messages import filter_messages
from langchain_core.tools import tool
from open_deep_research.configuration import Configuration, SearchAPI

try:
    import arxiv
except ImportError:
    arxiv = None

TOKEN_RE = "[a-z0-9_]{4,}"
NL = chr(10)
MAX_TOOL_TEXT = 6000

def _current_loop():
    try: return asyncio.get_running_loop()
    except RuntimeError:
        try: return asyncio.get_event_loop()
        except RuntimeError: return asyncio.new_event_loop()

class LoopSafeSemaphore:
    def __init__(self, limit):
        self.limit = limit
        self._semaphores = {}
    async def __aenter__(self):
        loop = _current_loop()
        key = id(loop)
        if key not in self._semaphores: self._semaphores[key] = asyncio.Semaphore(self.limit)
        await self._semaphores[key].__aenter__()
        return self
    async def __aexit__(self, exc_type, exc, tb):
        loop = _current_loop()
        sem = self._semaphores.get(id(loop))
        if sem is not None: await sem.__aexit__(exc_type, exc, tb)

class GroqShield:
    def __init__(self, keys):
        self.keys = keys if keys else [""]
        self.cooldowns = {k: 0.0 for k in self.keys}
        self.lock = threading.Lock()
        self.idx = 0
        self.last = None
    def get_key(self, last_failed=None):
        with self.lock:
            now = time.time()
            if last_failed and last_failed in self.cooldowns: self.cooldowns[last_failed] = now + 60.0
            avail = [k for k in self.keys if self.cooldowns.get(k, 0.0) <= now]
            if not avail: return min(self.keys, key=lambda k: self.cooldowns.get(k, 0.0))
            self.idx = (self.idx + 1) % len(avail)
            return avail[self.idx]

_raw_groq = os.environ.get("GROQ_API_KEYS", os.environ.get("GROQ_API_KEY", ""))
_groq_keys = [k.strip() for k in _raw_groq.split(",") if k.strip()]
_shield = GroqShield(_groq_keys)

def get_api_key_for_model(model_name, config, last_failed=None):
    m = str(model_name or "").lower()
    if m.startswith("groq:"): return _shield.get_key(last_failed)

    return None

def get_today_str(): return datetime.now().strftime("%a %b %d, %Y")

def get_config_value(value):
    if value is None: return None
    return value.value if hasattr(value, "value") else value

def _domain_from_url(url):
    try: return str(url).split("//")[-1].split("/")[0].replace("www.", "")
    except Exception: return ""

def _safe_url(url):
    try:
        from urllib.parse import urlparse
        import ipaddress
        p = urlparse(str(url))
        if p.scheme not in ("http", "https"): return False
        h = (p.hostname or "").lower()
        if h == "localhost" or h.endswith(".local") or "metadata.google.internal" in h: return False
        try:
            ip = ipaddress.ip_address(h)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved: return False
        except Exception:
            pass
        return True
    except Exception:
        return False

def _tokens(text): return set(re.findall(TOKEN_RE, str(text or "").lower()))

@tool(description="Search GitHub repositories.")
def github_sniper(query: str) -> str:
    try:
        h = {"Accept": "application/vnd.github.v3+json", "User-Agent": "Omega/2.0"}
        r = httpx.get("https://api.github.com/search/repositories", headers=h, params={"q": query, "sort": "stars", "per_page": 5}, timeout=15)
        r.raise_for_status()
        items = r.json().get("items", [])
        if not items: return "No GitHub repos found."
        return (NL + NL).join(["[" + str(i+1) + "] " + str(x.get("full_name")) + " | Stars: " + str(x.get("stargazers_count")) + NL + "URL: " + str(x.get("html_url")) for i, x in enumerate(items)])
    except Exception as e: return f"GitHub failed: {e}"

@tool(description="Search HuggingFace models.")
def huggingface_sniper(query: str) -> str:
    try:
        r = httpx.get("https://huggingface.co/api/models", params={"search": query, "sort": "downloads", "direction": "-1", "limit": 5}, timeout=15)
        r.raise_for_status()
        models = r.json()
        if not models: return "No HF models found."
        return (NL + NL).join(["[" + str(i+1) + "] " + str(m.get("modelId")) + " | DL: " + str(m.get("downloads", 0)) + NL + "URL: https://huggingface.co/" + str(m.get("modelId")) for i, m in enumerate(models)])
    except Exception as e: return f"HF failed: {e}"

@tool(description="Search ArXiv papers.")
def arxiv_search(query: str) -> str:
    if not arxiv: return "ArXiv not installed."
    try:
        s = arxiv.Search(query=query, max_results=3, sort_by=arxiv.SortCriterion.Relevance)
        out = []
        for i, p in enumerate(s.results()):
            out.append("[" + str(i+1) + "] " + str(p.title) + " (" + str(p.entry_id) + ")" + NL + str(p.summary)[:800])
        return (NL + NL).join(out) if out else "No papers found."
    except Exception: return "ArXiv failed."

@tool(
    description=(
        "Execute deterministic Python computation only. "
        "Network, filesystem, process execution, imports "
        "and reflection are forbidden."
    )
)
def python_repl(code: str) -> str:
    import ast
    import concurrent.futures

    code = str(code or "")

    if len(code) > 12000:
        return (
            "[FALLBACK] Code exceeds sandbox size limit."
        )

    try:
        tree = ast.parse(code)

        if len(list(ast.walk(tree))) > 500:
            return (
                "[FALLBACK] Code complexity limit exceeded."
            )

        forbidden_names = {
            "__import__",
            "__builtins__",
            "exec",
            "eval",
            "open",
            "compile",
            "globals",
            "locals",
            "vars",
            "getattr",
            "setattr",
            "delattr",
            "breakpoint",
            "input",
            "help",
            "dir",
        }

        forbidden_attributes = {
            "__class__",
            "__bases__",
            "__subclasses__",
            "__globals__",
            "__builtins__",
            "__import__",
            "__code__",
            "__closure__",
        }

        for node in ast.walk(tree):

            if isinstance(
                node,
                (
                    ast.Import,
                    ast.ImportFrom,
                ),
            ):
                return (
                    "[FALLBACK] Imports are forbidden."
                )

            if (
                isinstance(node, ast.Name)
                and node.id in forbidden_names
            ):
                return (
                    "[FALLBACK] Forbidden symbol."
                )

            if (
                isinstance(node, ast.Attribute)
                and node.attr
                in forbidden_attributes
            ):
                return (
                    "[FALLBACK] Forbidden attribute."
                )

        def _run():
            out = io.StringIO()

            safe_builtins = {
                "print": print,
                "len": len,
                "range": range,
                "list": list,
                "dict": dict,
                "set": set,
                "tuple": tuple,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "sorted": sorted,
                "enumerate": enumerate,
                "zip": zip,
                "isinstance": isinstance,
                "type": type,
            }

            namespace = {
                "__builtins__": safe_builtins,
                "json": json,
                "re": re,
                "math": math,
                "datetime": datetime,
                "BeautifulSoup": BeautifulSoup,
            }

            with contextlib.redirect_stdout(
                out
            ):
                exec(
                    code,
                    namespace,
                    {},
                )

            return (
                out.getvalue()
                or "OK"
            )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=1
        ) as executor:

            future = executor.submit(
                _run
            )

            try:
                return future.result(
                    timeout=30.0
                )
            except concurrent.futures.TimeoutError:
                return (
                    "[FALLBACK] Sandbox timeout."
                )

    except Exception as exc:
        return (
            "[FALLBACK] "
            + str(exc)
        )



@tool(description="Audit URL pricing.")
def audit_pricing(url: str) -> str:
    if not _safe_url(url): return "AUDIT_FAILED"
    try:
        r = httpx.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10, follow_redirects=True)
        t = BeautifulSoup(r.text, "html.parser").get_text(" ", strip=True).lower()
        red = sum(1 for f in ["contact sales", "book a demo", "enterprise pricing"] if f in t)
        grn = sum(1 for f in ["open source", "free tier", "100% free", "mit license"] if f in t)
        if grn > red and grn > 0: return "VERIFIED_FREE: " + url
        if red > 0 and grn == 0: return "PAID_ENTERPRISE: " + url
        return "UNKNOWN: " + url
    except Exception: return "AUDIT_FAILED"

def jina_scraper(url):
    if not _safe_url(url): return "[JINA FALLBACK]"
    key = os.environ.get("JINA_API_KEY", "")
    h = {"Accept": "application/json", "X-Return-Format": "markdown"}
    if key: h["Authorization"] = "Bearer " + key
    try:
        r = httpx.get("https://r.jina.ai/" + quote(str(url), safe=""), headers=h, timeout=30.0)
        r.raise_for_status()
        data = r.json().get("data", {})
        return str(data.get("content", ""))[:MAX_TOOL_TEXT]
    except Exception: return "[JINA FALLBACK]"

async def validate_urls(urls):
    out = {}
    uniq = list(set([str(u) for u in (urls or []) if str(u).startswith("http")]))[:30]
    if not uniq: return out
    uniq = [u for u in uniq if _safe_url(u)]
    async with httpx.AsyncClient(timeout=5.0, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"}) as client:
        for u in uniq:
            try:
                resp = await client.get(u)
                out[u] = resp.status_code < 400
            except Exception: out[u] = False
    return out

@tool(description="Search web via Jina AI.")
async def jina_search(query: str) -> str:
    key = os.environ.get("JINA_API_KEY", "")
    h = {"Accept": "application/json", "X-Return-Format": "markdown"}
    if key: h["Authorization"] = "Bearer " + key
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get("https://s.jina.ai/" + quote_plus(str(query)), headers=h)
            r.raise_for_status()
            data = r.json().get("data", [])
            if not data: return "No results."
            out = []
            for i, d in enumerate(data[:5]):
                out.append("--- " + str(i+1) + ": " + str(d.get("title")) + " ---" + NL + "URL: " + str(d.get("url")) + NL + str(d.get("content", ""))[:1200])
            return (NL + NL).join(out)
    except Exception as e: return "Jina failed: " + str(e)

@tool(description="Search web via Searxng.")
async def searxng_search(query: str) -> str:
    base = os.environ.get("SEARXNG_BASE_URL", "http://localhost:8080").rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=20.0, headers={"User-Agent": "Mozilla/5.0"}) as client:
            r = await client.get(base + "/search", params={"q": query, "format": "json"})
            r.raise_for_status()
            results = r.json().get("results", [])
            if not results: return "No Searxng results."
            out = []
            for i, item in enumerate(results[:5]):
                out.append("--- " + str(i+1) + ": " + str(item.get("title")) + " ---" + NL + "URL: " + str(item.get("url")) + NL + str(item.get("content", ""))[:1200])
            return (NL + NL).join(out)
    except Exception:
        try: return await jina_search.ainvoke(query)
        except Exception: return "Searxng failed."

@tool(description="Search Wikipedia.")
def wikipedia_rest_search(query: str) -> str:
    try:
        r = httpx.get("https://en.wikipedia.org/w/api.php", params={"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 3}, timeout=10).json()
        hits = r.get("query", {}).get("search", [])
        if not hits: return "No Wikipedia results."
        out = []
        for h in hits:
            title = str(h.get("title", "")).replace(" ", "_")
            out.append("TITLE: " + str(h.get("title", "")) + NL + "URL: https://en.wikipedia.org/wiki/" + title + NL + str(h.get("snippet", "")))
        return (NL + NL).join(out)
    except Exception: return "Wikipedia failed."

@tool(description="Strategic reflection.")
def think_tool(reflection: str) -> str: return "Reflection: " + str(reflection)

@tool("ResearchComplete", description="Signal that the research plan is complete.")
def research_complete_tool() -> str: return "Research complete."

_pdf_semaphore = LoopSafeSemaphore(1)

@tool(description="Ingest PDF via local or Jina fallback.")
async def omega_pdf_ingestor(url: str) -> str:
    if not str(url).lower().endswith(".pdf"): return "Not a PDF."
    if not _safe_url(url): return "[PDF] Unsafe URL."
    async with _pdf_semaphore:
        try:
            async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                r = await client.get(url)
                r.raise_for_status()
                pdf = r.content
        except Exception as e: return "[PDF] Download failed: " + str(e)
        try:
            import pymupdf4llm, tempfile
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as t:
                t.write(pdf)
                tp = t.name
            md = pymupdf4llm.to_markdown(tp)
            os.unlink(tp)
            if len(md.strip()) > 200: return "[TIER1 LOCAL]" + NL + md[:15000]
        except Exception: pass
        try:
            txt = jina_scraper(url)
            if txt and not txt.startswith("[JINA"): return "[TIER2 JINA]" + NL + txt
        except Exception: pass
        return "[PDF] Unable to extract text."

import tempfile as _tmp
OMEGA_MEM = os.path.join(_tmp.gettempdir(), "omega_memory.json")

class OmegaMemory:
    def __init__(self):
        self.data = {"domains": {}}
        try:
            if os.path.exists(OMEGA_MEM):
                with open(OMEGA_MEM, "r", encoding="utf-8") as f: self.data = json.load(f)
        except Exception: pass
    def save(self):
        try:
            t = OMEGA_MEM + ".tmp"
            with open(t, "w", encoding="utf-8") as f: json.dump(self.data, f)
            os.replace(t, OMEGA_MEM)
        except Exception: pass
    def update_domain(self, d, ok):
        if not d: return
        x = self.data.get("domains", {}).get(d, {"trust": 0.5, "hits": 0})
        x["hits"] += 1
        x["trust"] = max(0.0, min(1.0, x.get("trust", 0.5) * 0.9 + (0.1 if ok else -0.05)))
        self.data.setdefault("domains", {})[d] = x
        self.save()
    def get_context_prompt(self):
        top = sorted(self.data.get("domains", {}).items(), key=lambda x: x[1].get("trust", 0.0), reverse=True)[:5]
        if not top: return ""
        return "Trusted: " + ", ".join([str(d) + "(" + str(round(v.get("trust", 0.0), 1)) + ")" for d, v in top])

try: omega_memory = OmegaMemory()
except Exception:
    class _NoOp:
        def get_context_prompt(self): return ""
        def update_domain(self, d, s): pass
    omega_memory = _NoOp()

DB_PATH = os.path.join(_tmp.gettempdir(), "omega.db")

class LocalSQLiteMemory:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.lock = threading.Lock()
        self.c = self.conn.cursor()
        self.c.execute("CREATE TABLE IF NOT EXISTS ev (id INTEGER PRIMARY KEY, claim TEXT, url TEXT)")
        self.conn.commit()
    def store(self, claim, url):
        if not claim or not url: return
        try:
            with self.lock:
                self.c.execute("INSERT INTO ev (claim, url) VALUES (?,?)", (str(claim), str(url)))
                self.conn.commit()
        except Exception: pass
    def recall(self, q, limit=5):
        if not q: return []
        try:
            with self.lock:
                self.c.execute("SELECT claim, url FROM ev WHERE claim LIKE ? ORDER BY id DESC LIMIT ?", ("%" + str(q) + "%", int(limit)))
                return [{"claim": r[0], "url": r[1]} for r in self.c.fetchall()]
        except Exception: return []

try: omega_local_memory = LocalSQLiteMemory()
except Exception:
    class _NoOpDB:
        def store(self, c, u): pass
        def recall(self, q, l=5): return []
    omega_local_memory = _NoOpDB()

_citation_semaphore = LoopSafeSemaphore(5)

def _sentence_chunks(text):
    clean = re.sub(
        r"\s+",
        " ",
        str(text or "")
    ).strip()

    if not clean:
        return []

    return [
        part.strip()
        for part in re.split(
            r"(?<=[.!?])\s+",
            clean,
        )
        if part.strip()
    ]


def _verification_numbers(text):
    values = []

    for raw in re.findall(
        r"[0-9]+(?:\.[0-9]+)?",
        str(text or ""),
    ):
        try:
            values.append(float(raw))
        except Exception:
            pass

    return values


def _numbers_supported(
    claim,
    evidence,
):
    wanted = _verification_numbers(
        claim
    )

    if not wanted:
        return True

    found = _verification_numbers(
        evidence
    )

    for value in wanted:
        if not any(
            abs(value - other)
            <= max(
                0.01,
                abs(value) * 0.01,
            )
            for other in found
        ):
            return False

    return True


def _source_kind(url):
    domain = _domain_from_url(
        url
    ).lower()

    if any(
        x in domain
        for x in [".gov", ".mil"]
    ):
        return "government"

    if any(
        x in domain
        for x in [".edu", "arxiv.org"]
    ):
        return "academic"

    if any(
        x in domain
        for x in [
            "github.com",
            "huggingface.co",
        ]
    ):
        return "technical"

    if any(
        x in domain
        for x in [
            "reuters.com",
            "apnews.com",
            "bbc.com",
            "nature.com",
        ]
    ):
        return "major_media"

    return "general"


async def verify_citations_programmatically(
    nodes
):
    """
    Deterministic source/claim verification.

    VERIFIED:
        Strong local support with consistent numbers.

    PARTIAL:
        Some meaningful support but insufficient strength.

    FAILED:
        Source does not sufficiently support the claim.
    """

    strong = []
    weak = []

    if not nodes:
        return {
            "strong": strong,
            "weak": weak,
        }

    async with httpx.AsyncClient(
        timeout=15.0,
        follow_redirects=True,
        headers={
            "User-Agent": "Mozilla/5.0"
        },
    ) as client:

        for node in list(nodes)[:40]:

            url = str(
                getattr(
                    node,
                    "url",
                    "",
                ) or ""
            ).strip()

            claim = re.sub(
                r"\s+",
                " ",
                str(
                    getattr(
                        node,
                        "claim",
                        "",
                    ) or ""
                ),
            ).strip()

            if (
                not url
                or not claim
                or not _safe_url(url)
            ):
                continue

            try:
                response = await client.get(
                    url
                )

                if response.status_code >= 400:
                    node.verification_status = (
                        "FAILED"
                    )

                    omega_memory.update_domain(
                        _domain_from_url(url),
                        False,
                    )

                    weak.append(node)
                    continue

                page = BeautifulSoup(
                    response.text,
                    "html.parser",
                ).get_text(
                    " ",
                    strip=True,
                )

                claim_tokens = _tokens(
                    claim
                )

                best_span = ""
                best_score = 0.0

                for sentence in _sentence_chunks(
                    page
                )[:400]:

                    sentence_tokens = _tokens(
                        sentence
                    )

                    if (
                        not claim_tokens
                        or not sentence_tokens
                    ):
                        continue

                    overlap = (
                        len(
                            claim_tokens.intersection(
                                sentence_tokens
                            )
                        )
                        / max(
                            1,
                            len(claim_tokens),
                        )
                    )

                    if overlap > best_score:
                        best_score = overlap
                        best_span = sentence[
                            :1800
                        ]

                if claim.lower() in page.lower():
                    best_score = 1.0
                    best_span = claim

                number_ok = _numbers_supported(
                    claim,
                    best_span,
                )

                node.source_kind = _source_kind(
                    url
                )

                node.entailment_score = round(
                    min(
                        1.0,
                        best_score,
                    ),
                    3,
                )

                if (
                    number_ok
                    and best_score >= 0.72
                ):
                    node.evidence_span = (
                        best_span
                    )

                    node.verification_status = (
                        "VERIFIED"
                    )

                    node.provenance_id = (
                        hashlib.sha256(
                            (
                                url
                                + "|"
                                + claim
                                + "|"
                                + best_span
                            ).encode(
                                "utf-8"
                            )
                        ).hexdigest()[:16]
                    )

                    omega_memory.update_domain(
                        _domain_from_url(url),
                        True,
                    )

                    strong.append(node)

                elif (
                    number_ok
                    and best_score >= 0.45
                ):
                    node.evidence_span = (
                        best_span
                    )

                    node.verification_status = (
                        "PARTIAL"
                    )

                    node.provenance_id = (
                        hashlib.sha256(
                            (
                                url
                                + "|"
                                + claim
                                + "|"
                                + best_span
                            ).encode(
                                "utf-8"
                            )
                        ).hexdigest()[:16]
                    )

                    omega_memory.update_domain(
                        _domain_from_url(url),
                        False,
                    )

                    weak.append(node)

                else:
                    node.verification_status = (
                        "FAILED"
                    )

                    omega_memory.update_domain(
                        _domain_from_url(url),
                        False,
                    )

                    weak.append(node)

            except Exception:
                node.verification_status = (
                    "FAILED"
                )
                weak.append(node)

    return {
        "strong": strong,
        "weak": weak,
    }


def calculate_epistemic_saturation(eg, rp):
    evidence = eg or []
    plan = rp or []
    contradictions = sum(1 for n in evidence if getattr(n, "contradicts", []))
    if not plan: return min(1.0, len(evidence) / 20.0)
    plan_words = set()
    for n in plan:
        topic = n.get("topic", "") if isinstance(n, dict) else getattr(n, "topic", "")
        plan_words.update(_tokens(topic))
    if not plan_words: return min(1.0, len(evidence) / 20.0)
    claim_words = set()
    for n in evidence: claim_words.update(_tokens(getattr(n, "claim", "")))
    coverage = len(plan_words.intersection(claim_words)) / len(plan_words)
    evidence_bonus = min(0.25, len(evidence) / 25.0)
    penalty = min(0.2, contradictions * 0.03)
    return max(0.0, min(1.0, coverage * 0.75 + evidence_bonus - penalty))

def programmatic_epistemic_verification(eg, ti):
    if not eg: return {"consensus_report": "No evidence.", "confidence_score": 0.0, "red_team_findings": "N/A", "devils_advocate_critique": "N/A"}
    unique = {}
    for n in eg:
        fp = " ".join(sorted(_tokens(getattr(n, "claim", ""))))
        if fp and fp not in unique: unique[fp] = n
    deduped = list(unique.values())
    year = datetime.now().year
    valid = []
    decay = 0.0
    for n in deduped:
        score = 1.0
        ds = getattr(n, "date_published", None)
        if ds and str(ti).lower() == "current":
            try:
                age = year - int(str(ds)[:4])
                if age > 0:
                    score *= math.exp(-0.2 * age)
                    decay += (1.0 - score)
            except Exception: pass
        if score > 0.35: valid.append((n, score))
    domains = set()
    trust_values = []
    for n, _ in valid:
        d = _domain_from_url(getattr(n, "url", ""))
        if d:
            domains.add(d)
            trust_values.append(float(getattr(omega_memory, "data", {}).get("domains", {}).get(d, {}).get("trust", 0.5)))
    contradictions = sum(1 for n, _ in valid if getattr(n, "contradicts", []))
    _tvals = []
    for _d in domains:
        _d = str(_d or "").lower()
        if any(x in _d for x in ["gov", "edu", "arxiv", "wikipedia", "github", "huggingface"]): _tvals.append(1.0)
        elif any(x in _d for x in ["reuters", "bloomberg", "apnews", "bbc", "nature", "science", "ieee", "acm"]): _tvals.append(0.9)
        elif any(x in _d for x in ["medium", "blogspot", "wordpress", "quora", "reddit"]): _tvals.append(0.35)
        else: _tvals.append(0.6)
    domain_score = min(1.0, (sum(_tvals) / max(1, len(_tvals))) * min(1.0, len(domains) / 4.0)) if domains else 0.0
    fact_score = min(1.0, len(valid) / 10.0)
    trust_score = sum(trust_values) / len(trust_values) if trust_values else 0.5
    penalty = contradictions * 0.05 + min(0.25, decay * 0.01)
    confidence = max(0.05, min(0.99, domain_score * 0.35 + fact_score * 0.35 + trust_score * 0.20 - penalty))
    return {"consensus_report": "Analyzed " + str(len(valid)) + " facts from " + str(len(domains)) + " domains. Confidence: " + str(round(confidence, 2)), "confidence_score": confidence, "red_team_findings": "Removed " + str(len(eg) - len(deduped)) + " duplicates. Contradictions: " + str(contradictions) + ".", "devils_advocate_critique": "Temporal decay penalty: " + str(round(decay, 2)) + "."}

async def get_search_tool(sa):
    if sa == SearchAPI.SEARXNG: return [searxng_search]
    if sa == SearchAPI.NONE: return [jina_search]
    return [jina_search]

async def get_all_tools(config):
    cfg = Configuration.from_runnable_config(config)
    tools = [research_complete_tool, think_tool, omega_pdf_ingestor, github_sniper, huggingface_sniper, wikipedia_rest_search, arxiv_search, audit_pricing]
    if getattr(cfg, "enable_python_repl", True): tools.append(python_repl)
    tools.extend(await get_search_tool(SearchAPI(get_config_value(cfg.search_api))))
    return tools

def get_notes_from_tool_calls(msgs):
    try:
        return [str(m.content) for m in filter_messages(msgs, include_types=["tool"])]
    except Exception:
        return [str(getattr(m, "content", "")) for m in msgs if getattr(m, "type", "") == "tool"]

def is_token_limit_exceeded(e, model=None):
    s = str(e).lower()
    return "context_length" in s or "too long" in s or "resource_exhausted" in s or "413" in s or "maximum context" in s

def get_model_token_limit(m):
    m = str(m or "").lower()
    if "groq:llama-3.3-70b" in m: return 128000
    if "groq:llama-3.1-8b" in m: return 128000
    return None

def check_information_satiation(new, existing, threshold=0.75):
    if not new: return True
    new_text = " ".join([str(x) for x in new])
    old_text = " ".join([str(x) for x in (existing or [])])
    nt = _tokens(new_text)
    if not nt: return False
    ot = _tokens(old_text)
    if not ot: return False
    return len(nt.intersection(ot)) / len(nt) >= float(threshold)

def filter_and_verify_evidence(eg, temporal_intent="Current"):
    """Normalize, validate, deduplicate and cap evidence deterministically."""
    out = []
    seen = set()

    for node in eg or []:
        claim = str(getattr(node, "claim", "")).strip()
        url = str(getattr(node, "url", "")).strip()

        if len(claim) < 20 or not _safe_url(url):
            continue

        key = (
            re.sub(r"\s+", " ", claim.lower()),
            url.rstrip("/").lower()
        )

        if key in seen:
            continue

        seen.add(key)
        out.append(node)

    return out[:80]
