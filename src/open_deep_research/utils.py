"""Omega Supremacy Arsenal: Tools, Scrapers, Memory, and Epistemic Math."""
import re, io, asyncio, contextlib, os, json, math, sqlite3, time, threading
from datetime import datetime, timedelta
from typing import Annotated, Any, List, Optional
from collections import Counter
from urllib.parse import quote, quote_plus
import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, MessageLikeRepresentation, filter_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from open_deep_research.configuration import Configuration, SearchAPI
from open_deep_research.prompts import summarize_webpage_prompt
from open_deep_research.state import ResearchComplete, Summary

try:
    import arxiv
except ImportError:
    arxiv = None

try:
    import aiohttp
except ImportError:
    aiohttp = None


class GroqShield:
    def __init__(self, keys):
        self.keys = keys if keys else [""]
        self.cooldowns = {k: 0 for k in self.keys}
        self.lock = threading.Lock()
        self.idx = 0

    def get_key(self, last_failed=None):
        with self.lock:
            now = time.time()
            if last_failed and last_failed in self.cooldowns:
                self.cooldowns[last_failed] = now + 60.0
            avail = [k for k in self.keys if self.cooldowns.get(k, 0) <= now]
            if not avail:
                return min(self.keys, key=lambda k: self.cooldowns.get(k, 0))
            self.idx = (self.idx + 1) % len(avail)
            return avail[self.idx]


_raw_groq = os.environ.get("GROQ_API_KEYS", os.environ.get("GROQ_API_KEY", ""))
_groq_keys = [k.strip() for k in _raw_groq.split(",") if k.strip()]
_shield = GroqShield(_groq_keys)
groq_burst_semaphore = asyncio.Semaphore(3)


def get_api_key_for_model(model_name, config, last_failed=None):
    if model_name.lower().startswith("groq:"):
        return _shield.get_key(last_failed)
    return None


def get_today_str():
    return datetime.now().strftime("%a %b %d, %Y")


def get_config_value(value):
    if value is None:
        return None
    return value.value if hasattr(value, 'value') else value


@tool(description="Search GitHub repositories by query.")
def github_sniper(query: str) -> str:
    try:
        h = {"Accept": "application/vnd.github.v3+json", "User-Agent": "Omega/2.0"}
        r = httpx.get("https://api.github.com/search/repositories", headers=h, params={"q": query, "sort": "stars", "per_page": 5}, timeout=15)
        r.raise_for_status()
        items = r.json().get("items", [])
        if not items:
            return "No GitHub repos found."
        return "\n\n".join([f"[{i+1}] {x.get('full_name')} | Stars: {x.get('stargazers_count')}\nURL: {x.get('html_url')}\nDesc: {str(x.get('description',''))[:150]}" for i, x in enumerate(items)])
    except Exception as e:
        return f"GitHub failed: {e}"


@tool(description="Search HuggingFace for AI models and datasets.")
def huggingface_sniper(query: str) -> str:
    try:
        r = httpx.get("https://huggingface.co/api/models", params={"search": query, "sort": "downloads", "direction": "-1", "limit": 5}, timeout=15)
        r.raise_for_status()
        models = r.json()
        if not models:
            return "No HuggingFace models found."
        return "\n\n".join([f"[{i+1}] {m.get('modelId')} | Downloads: {m.get('downloads',0)}\nURL: https://huggingface.co/{m.get('modelId')}" for i, m in enumerate(models)])
    except Exception as e:
        return f"HuggingFace failed: {e}"


@tool(description="Search Hacker News for trending stories.")
def hackernews_sniper(query: str) -> str:
    try:
        r = httpx.get("https://hn.algolia.com/api/v1/search", params={"query": query, "tags": "story", "hitsPerPage": 5}, timeout=15)
        r.raise_for_status()
        hits = r.json().get("hits", [])
        if not hits:
            return "No Hacker News stories found."
        return "\n\n".join([f"[{i+1}] {h.get('title')} | Points: {h.get('points',0)} | Comments: {h.get('num_comments',0)}\nURL: https://news.ycombinator.com/item?id={h.get('objectID')}" for i, h in enumerate(hits)])
    except Exception as e:
        return f"HackerNews failed: {e}"


@tool(description="Search ArXiv for academic papers.")
def arxiv_search(query: str) -> str:
    if not arxiv:
        return "ArXiv library not installed."
    try:
        s = arxiv.Search(query=query, max_results=3, sort_by=arxiv.SortCriterion.Relevance)
        out = [f"[{i+1}] {p.title} ({p.entry_id})\nAbstract: {p.summary[:800]}" for i, p in enumerate(s.results())]
        return "\n\n".join(out) if out else "No ArXiv papers found."
    except Exception:
        return "ArXiv search failed."


@tool(description="Search PubMed for biomedical literature.")
def pubmed_search(query: str) -> str:
    try:
        r = httpx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params={"db": "pubmed", "term": query, "retmode": "json", "retmax": 3}, timeout=10).json()
        ids = r.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return "No PubMed articles found."
        f = httpx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi", params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"}, timeout=10).json()
        return "\n\n".join([f"TITLE: {f.get('result',{}).get(uid,{}).get('title','Unknown')}\nPMID: {uid}\nURL: https://pubmed.ncbi.nlm.nih.gov/{uid}/" for uid in ids])
    except Exception:
        return "PubMed search failed."


@tool(description="Search Semantic Scholar for academic papers with citation counts.")
def semantic_scholar_search(query: str) -> str:
    try:
        r = httpx.get("https://api.semanticscholar.org/graph/v1/paper/search", params={"query": query, "limit": 5, "fields": "title,abstract,url,year,citationCount"}, timeout=10).json()
        papers = r.get("data", [])
        if not papers:
            return "No Semantic Scholar results."
        return "\n\n".join([f"[{i+1}] {p.get('title')} ({p.get('year')}) | Citations: {p.get('citationCount',0)}\nURL: {p.get('url')}\nAbstract: {str(p.get('abstract',''))[:500]}" for i, p in enumerate(papers)])
    except Exception:
        return "Semantic Scholar failed."


@tool(description="Execute Python code in a secure sandbox for data processing.")
def python_repl(code: str) -> str:
    import ast
    import concurrent.futures
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name in ['os', 'sys', 'subprocess', 'socket', 'shutil', 'pathlib', 'requests']:
                        return f"[FALLBACK] Import of {a.name} is forbidden."
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split('.')[0] in ['os', 'sys', 'subprocess', 'socket']:
                    return f"[FALLBACK] Import from {node.module} is forbidden."
            elif isinstance(node, ast.Call) and getattr(node.func, 'id', '') in ['eval', 'exec', 'exit', 'quit']:
                return f"[FALLBACK] Use of {node.func.id}() is forbidden."
    except SyntaxError as e:
        return f"[FALLBACK] Syntax error: {e}"

    def _run():
        out = io.StringIO()
        try:
            import pandas as pd
            import numpy as np
            import yfinance as yf
        except ImportError:
            pd, np, yf = None, None, None
        safe_globals = {
            "__builtins__": {"print": print, "len": len, "range": range, "sorted": sorted, "list": list, "dict": dict, "set": set, "str": str, "int": int, "float": float, "min": min, "max": max, "sum": sum, "enumerate": enumerate, "zip": zip, "map": map, "filter": filter, "isinstance": isinstance, "type": type, "abs": abs, "round": round, "bool": bool, "tuple": tuple, "any": any, "all": all, "next": next, "iter": iter},
            "json": json, "re": re, "math": math, "datetime": datetime, "BeautifulSoup": BeautifulSoup, "httpx": httpx, "pd": pd, "np": np, "yf": yf
        }
        with contextlib.redirect_stdout(out):
            exec(code, safe_globals, {})
        return out.getvalue()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(_run).result(timeout=30.0) or "Code executed (no output)."
    except concurrent.futures.TimeoutError:
        return "[FALLBACK] Execution timeout (30s)."
    except Exception as e:
        return f"[FALLBACK] Execution error: {e}"


@tool(description="Audit a URL to verify if the service is 100% free or paid enterprise.")
def audit_pricing(url: str) -> str:
    try:
        r = httpx.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10, follow_redirects=True)
        t = BeautifulSoup(r.text, 'html.parser').get_text(' ', strip=True).lower()
        red = sum(1 for f in ["contact sales", "book a demo", "enterprise pricing", "credit card required"] if f in t)
        grn = sum(1 for f in ["open source", "free tier", "free forever", "100% free", "mit license", "apache license"] if f in t)
        if grn > red and grn > 0:
            return f"VERIFIED_FREE: {url}"
        if red > 0 and grn == 0:
            return f"PAID_ENTERPRISE: {url} (REJECT if free constraint active)"
        return f"UNKNOWN: {url}"
    except Exception:
        return "AUDIT_FAILED"
def jina_scraper(url):
    """Cloudflare Piercer: Uses Jina AI to bypass anti-bot protections."""
    key = os.environ.get("JINA_API_KEY", "")
    h = {"Accept": "application/json", "X-Return-Format": "markdown"}
    if key:
        h["Authorization"] = f"Bearer {key}"
    try:
        safe_url = quote(str(url), safe='')
        r = httpx.get(f"https://r.jina.ai/{safe_url}", headers=h, timeout=30.0)
        r.raise_for_status()
        return r.json().get("data", {}).get("content", "")[:15000]
    except Exception:
        return "[JINA FALLBACK]"


async def validate_urls(urls):
    """Async URL health check."""
    out = {}
    if aiohttp is None:
        return {u: True for u in urls}
    async with aiohttp.ClientSession() as s:
        for u in set(urls):
            if not u or not str(u).startswith("http"):
                out[u] = False
                continue
            try:
                async with s.get(u, allow_redirects=True, timeout=aiohttp.ClientTimeout(total=5)) as r:
                    out[u] = r.status < 400 or r.status in [403, 405]
            except Exception:
                out[u] = False
    return out


async def _crawl_urls(urls):
    """Fetch and clean multiple URLs concurrently."""
    out = {}
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0 ProjectOmega/2.0"}) as c:
        tasks = [c.get(u) for u in urls]
        resps = await asyncio.gather(*tasks, return_exceptions=True)
        for u, r in zip(urls, resps):
            if isinstance(r, Exception) or getattr(r, 'status_code', 400) >= 400:
                jt = jina_scraper(u)
                if not jt.startswith("[JINA"):
                    out[u] = jt
                continue
            try:
                soup = BeautifulSoup(r.text, 'html.parser')
                for s in soup(["script", "style", "nav", "footer", "header"]):
                    s.extract()
                txt = "\n".join([l.strip() for l in soup.get_text(' ', strip=True).splitlines() if len(l.strip()) > 40])
                if len(txt) < 200:
                    jt = jina_scraper(u)
                    if not jt.startswith("[JINA"):
                        txt = jt
                out[u] = txt[:15000]
            except Exception:
                pass
    return out


@tool(description="Search the web using Jina AI Search API. Returns clean Markdown.")
async def jina_search(query: str) -> str:
    key = os.environ.get("JINA_API_KEY", "")
    h = {"Accept": "application/json", "X-Return-Format": "markdown"}
    if key:
        h["Authorization"] = f"Bearer {key}"
    try:
        async with httpx.AsyncClient(timeout=20.0) as c:
            r = await c.get(f"https://s.jina.ai/{quote_plus(str(query))}", headers=h)
            r.raise_for_status()
            data = r.json().get("data", [])
            if not data:
                return "No search results found."
            return "\n\n".join([f"--- SOURCE {i+1}: {d.get('title','Untitled')} ---\nURL: {d.get('url','')}\nCONTENT:\n{d.get('content','')[:1200]}" for i, d in enumerate(data[:5])])
    except Exception as e:
        return f"Jina Search failed: {e}"
@tool(description="Search Wikipedia using the official REST API.")
def wikipedia_rest_search(query: str) -> str:
    try:
        response = httpx.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": 3,
            },
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        hits = payload.get("query", {}).get("search", [])
        if not hits:
            return "No Wikipedia results found."

        results = []
        for h in hits:
            title = h.get("title", "")
            snippet = h.get("snippet", "")
            snippet = snippet.replace('<span class="searchmatch">', "").replace("</span>", "")
            url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            results.append(f"TITLE: {title}\nURL: {url}\nSUMMARY: {snippet}")

        return "\n\n".join(results)
    except Exception as e:
        return f"Wikipedia search failed: {e}"

async def _ddg_fallback(query):
    try:
        with DDGS() as d:
            return [{"title": r.get("title", ""), "url": r.get("href", ""), "content": r.get("body", "")} for r in d.text(query, max_results=5)]
    except Exception:
        return []


@tool(description="Search via SearXNG with DuckDuckGo fallback.")
async def searxng_search(queries: List[str], config: RunnableConfig = None) -> str:
    cfg = Configuration.from_runnable_config(config)
    all_r = {}
    for q in queries:
        results = []
        try:
            async with httpx.AsyncClient(timeout=15.0) as c:
                r = await c.get(f"{cfg.searxng_base_url}/search", params={"q": q, "format": "json"})
                results = r.json().get("results", [])
        except Exception:
            results = await _ddg_fallback(q)
        for x in results:
            u = x.get("url")
            if u and u not in all_r:
                all_r[u] = {"title": x.get("title", ""), "snippet": x.get("content", "")}
    if not all_r:
        return "No search results found."
    crawled = await _crawl_urls(list(all_r.keys())[:10])
    out = "Search results:\n\n"
    for i, (u, d) in enumerate(list(all_r.items())[:8]):
        out += f"--- SOURCE {i+1}: {d['title']} ---\nURL: {u}\nCONTENT:\n{crawled.get(u, d['snippet'])[:1200]}\n\n"
    return out


async def summarize_webpage(model, content):
    try:
        p = summarize_webpage_prompt.format(webpage_content=content[:10000], date=get_today_str())
        s = await asyncio.wait_for(model.ainvoke([HumanMessage(content=p)]), timeout=20.0)
        return f"<summary>\n{s.summary}\n</summary>\n<key_excerpts>\n{s.key_excerpts}\n</key_excerpts>"
    except Exception:
        return content[:2000] + "\n[FALLBACK: Summarization skipped]"


@tool(description="Strategic reflection and planning tool.")
def think_tool(reflection: str) -> str:
    return "Reflection recorded: " + reflection

@tool("ResearchComplete", description="Signal that the research plan is complete.")
def ResearchComplete() -> str:
    return "Research complete."


_pdf_semaphore = asyncio.Semaphore(1)


@tool(description="Ingest PDF via 3-Tier Engine: Local PyMuPDF, Gemini Cloud OCR, or Raw Extraction.")
async def omega_pdf_ingestor(url: str) -> str:
    if not url.lower().endswith('.pdf'):
        return "Not a PDF URL."
    async with _pdf_semaphore:
        try:
            async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as c:
                r = await c.get(url)
                r.raise_for_status()
                pdf = r.content
        except Exception as e:
            return f"[PDF] Download failed: {e}"

        # TIER 1: Local Digital Fast-Path
        try:
            import pymupdf4llm
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as t:
                t.write(pdf)
                tp = t.name
            md = pymupdf4llm.to_markdown(tp)
            os.unlink(tp)
            if len(md.strip()) > 200:
                return f"[TIER 1 LOCAL]\n{md[:15000]}"
        except Exception:
            pass

        # TIER 2: Gemini Cloud OCR (3-Key Rotation)
        gkeys = [k.strip() for k in os.environ.get("GEMINI_API_KEYS", os.environ.get("GEMINI_API_KEY", "")).split(",") if k.strip()]
        for gk in gkeys:
            try:
                import google.generativeai as genai
                import tempfile
                genai.configure(api_key=gk)
                model = genai.GenerativeModel('gemini-2.0-flash')
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as t:
                    t.write(pdf)
                    tp = t.name
                pf = genai.upload_file(tp, mime_type="application/pdf")
                os.unlink(tp)
                resp = model.generate_content(["Extract all text, tables, and layout into clean structured Markdown. Preserve table grids and headers.", pf])
                try:
                    genai.delete_file(pf.name)
                except Exception:
                    pass
                if resp.text:
                    return f"[TIER 2 GEMINI CLOUD]\n{resp.text[:15000]}"
            except Exception:
                continue

        # TIER 3: Emergency Raw Text
        try:
            import fitz
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as t:
                t.write(pdf)
                tp = t.name
            doc = fitz.open(tp)
            txt = "\n".join([p.get_text() for p in doc])
            os.unlink(tp)
            return f"[TIER 3 EMERGENCY]\n{txt[:15000]}"
        except Exception as e:
            return f"[PDF] All tiers failed: {e}"


OMEGA_MEM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "omega_memory.json")


class OmegaMemory:
    def __init__(self):
        self.data = {"domains": {}}
        try:
            if os.path.exists(OMEGA_MEM_PATH):
                with open(OMEGA_MEM_PATH) as f:
                    self.data = json.load(f)
        except Exception:
            pass

    def save(self):
        try:
            tmp = OMEGA_MEM_PATH + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.data, f)
            os.replace(tmp, OMEGA_MEM_PATH)
        except Exception:
            pass

    def update_domain(self, domain, success):
        if not domain:
            return
        d = self.data["domains"].get(domain, {"trust": 0.5, "hits": 0})
        d["hits"] += 1
        d["trust"] = max(0.0, min(1.0, d["trust"] * 0.9 + (0.1 if success else -0.05)))
        self.data["domains"][domain] = d
        self.save()

    def get_context_prompt(self):
        top = sorted(self.data.get("domains", {}).items(), key=lambda x: x[1]["trust"], reverse=True)[:5]
        if not top:
            return ""
        return "<MEMORY_REGISTRY>\nTrusted Domains: " + ", ".join([f"{d}({v['trust']:.2f})" for d, v in top]) + "\n</MEMORY_REGISTRY>"


try:
    omega_memory = OmegaMemory()
except Exception:
    class _NoOp:
        def get_context_prompt(self): return ""
        def update_domain(self, d, s): pass
    omega_memory = _NoOp()


DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "omega_memory.db")


class LocalSQLiteMemory:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.c = self.conn.cursor()
        self.c.execute("CREATE TABLE IF NOT EXISTS evidence (id INTEGER PRIMARY KEY AUTOINCREMENT, claim TEXT, url TEXT, keywords TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
        self.conn.commit()

    def store(self, claim, url):
        if not claim or not url:
            return
        words = re.findall(r'\b[a-zA-Z]{4,}\b', claim.lower())
        kw = json.dumps(list(set(words))[:10])
        try:
            self.c.execute("INSERT INTO evidence (claim, url, keywords) VALUES (?, ?, ?)", (claim, url, kw))
            self.conn.commit()
        except Exception:
            pass

    def recall(self, query, limit=5):
        q_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', query.lower()))
        if not q_words:
            return []
        self.c.execute("SELECT claim, url, keywords FROM evidence ORDER BY timestamp DESC LIMIT 200")
        rows = self.c.fetchall()
        scored = []
        for claim, url, kw_str in rows:
            try:
                db_kw = set(json.loads(kw_str))
                inter = len(q_words.intersection(db_kw))
                union = len(q_words.union(db_kw))
                score = inter / union if union > 0 else 0
                if score > 0:
                    scored.append((score, claim, url))
            except Exception:
                pass
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"claim": c, "url": u} for s, c, u in scored[:limit]]


try:
    omega_local_memory = LocalSQLiteMemory()
except Exception:
    class _NoOpDB:
        def store(self, c, u): pass
        def recall(self, q, l=5): return []
    omega_local_memory = _NoOpDB()
_citation_semaphore = asyncio.Semaphore(5)


async def verify_citations_programmatically(nodes):
    """Zero-Token Citation Executioner: Verifies claims exist at source URLs."""
    verified = []
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0 ProjectOmega/2.0"}) as c:
        for n in nodes:
            async with _citation_semaphore:
                u = getattr(n, "url", "")
                cl = getattr(n, "claim", "")
                if not u or not cl or not u.startswith("http"):
                    continue
                try:
                    r = await c.get(u)
                    if r.status_code >= 400:
                        continue
                    text = BeautifulSoup(r.text, 'html.parser').get_text(' ', strip=True).lower()
                    claim_lower = cl.lower()
                    if claim_lower in text:
                        verified.append(n)
                        continue
                    words = claim_lower.split()
                    if len(words) > 3:
                        text_words = set(text.split())
                        match_count = sum(1 for w in words if w in text_words)
                        if match_count / len(words) >= 0.70:
                            verified.append(n)
                except Exception:
                    pass
    return verified


def calculate_epistemic_saturation(evidence_graph, research_plan):
    """Calculate how much of the research plan is covered by gathered evidence."""
    if not research_plan or not evidence_graph:
        return 0.0
    plan_words = set()
    for node in research_plan:
        topic = node.get("topic", "") if isinstance(node, dict) else getattr(node, "topic", "")
        plan_words.update(re.findall(r'\b\w{4,}\b', topic.lower()))
    if not plan_words:
        return 0.0
    covered_words = set()
    for node in evidence_graph:
        covered_words.update(re.findall(r'\b\w{4,}\b', getattr(node, 'claim', '').lower()))
    overlap = len(plan_words.intersection(covered_words))
    saturation = overlap / len(plan_words)
    volume_bonus = min(len(evidence_graph) / 30.0, 0.2)
    return min(saturation + volume_bonus, 1.0)


def programmatic_epistemic_verification(evidence_graph, temporal_intent):
    """Zero-Token Programmatic Verification: Dedup, decay, and score evidence."""
    if not evidence_graph:
        return {"consensus_report": "No evidence gathered.", "confidence_score": 0.0, "red_team_findings": "N/A", "devils_advocate_critique": "N/A"}

    unique_claims = {}
    for node in evidence_graph:
        fp = " ".join(sorted(set(re.findall(r'\b\w{4,}\b', getattr(node, 'claim', '').lower()))))
        if fp and fp not in unique_claims:
            unique_claims[fp] = node
    deduped = list(unique_claims.values())

    current_year = datetime.now().year
    verified_nodes = []
    decay_penalty = 0.0
    for node in deduped:
        score = 1.0
        ds = getattr(node, 'date_published', None)
        if ds and temporal_intent == "Current":
            try:
                age = current_year - int(str(ds)[:4])
                if age > 0:
                    score *= math.exp(-0.2 * age)
                    decay_penalty += (1.0 - score)
            except Exception:
                pass
        if score > 0.4:
            verified_nodes.append((node, score))

    domains = set()
    for node, _ in verified_nodes:
        u = getattr(node, 'url', '')
        if u:
            try:
                domains.add(u.split("//")[-1].split("/")[0].replace("www.", ""))
            except Exception:
                pass

    source_diversity = min(len(domains) / 5.0, 1.0)
    volume_score = min(len(verified_nodes) / 10.0, 1.0)
    contradictions = sum(1 for n, _ in verified_nodes if getattr(n, 'contradicts', []))
    contradiction_penalty = contradictions * 0.05
    final_confidence = max(0.1, min((source_diversity * 0.4 + volume_score * 0.6) - contradiction_penalty - (decay_penalty * 0.01), 0.99))

    return {
        "consensus_report": f"Analyzed {len(verified_nodes)} unique facts from {len(domains)} domains. Confidence: {final_confidence:.2f}",
        "confidence_score": final_confidence,
        "red_team_findings": f"Programmatic dedup removed {len(evidence_graph) - len(deduped)} duplicate claims.",
        "devils_advocate_critique": f"Temporal decay penalty: {decay_penalty:.2f}. Contradiction penalty: {contradiction_penalty:.2f}."
    }


def filter_and_verify_evidence(evidence_graph, temporal_intent="Current"):
    """Deduplicate evidence and prefer newer/older sources based on temporal intent."""
    if not evidence_graph:
        return []
    unique = {}
    for n in evidence_graph:
        k = "".join(sorted(re.findall(r'\b\w{4,}\b', getattr(n, 'claim', '').lower())))
        if not k:
            continue
        if k not in unique:
            unique[k] = n
        else:
            ed = getattr(unique[k], 'date_published', None)
            nd = getattr(n, 'date_published', None)
            if ed and nd:
                if temporal_intent == "Historical" and str(nd) < str(ed):
                    unique[k] = n
                elif temporal_intent != "Historical" and str(nd) > str(ed):
                    unique[k] = n
    return list(unique.values())


def check_information_satiation(new_claims, existing_claims, threshold=0.75):
    """Detect if new research is yielding diminishing returns."""
    if not existing_claims or not new_claims:
        return False

    def get_words(text):
        return set(re.findall(r'\b\w{4,}\b', text.lower()))

    pool = set()
    for c in existing_claims:
        pool.update(get_words(c))
    if not pool:
        return False
    redundant = sum(1 for c in new_claims if len(get_words(c).intersection(pool)) / max(len(get_words(c)), 1) >= 0.60)
    return (redundant / len(new_claims)) >= threshold


async def get_search_tool(search_api):
    if search_api == SearchAPI.NONE:
        return []
    if search_api == SearchAPI.SEARXNG:
        return [searxng_search, jina_search]
    return [jina_search]


async def get_all_tools(config):
    tools = [
        ResearchComplete, think_tool, omega_pdf_ingestor,
        github_sniper, huggingface_sniper, hackernews_sniper,
        wikipedia_rest_search, arxiv_search, pubmed_search,
        semantic_scholar_search, python_repl, audit_pricing, jina_search
    ]
    cfg = Configuration.from_runnable_config(config)
    tools.extend(await get_search_tool(SearchAPI(get_config_value(cfg.search_api))))
    return tools


def get_notes_from_tool_calls(msgs):
    return [m.content for m in filter_messages(msgs, include_types="tool")]


def is_token_limit_exceeded(e, model=None):
    s = str(e).lower()
    return 'context_length' in s or 'too long' in s or 'resource_exhausted' in s


def get_model_token_limit(m):
    limits = {"groq:llama-3.3-70b": 128000, "groq:llama-3.1-8b": 128000, "gemini-2.0-flash": 1048576}
    for k, v in limits.items():
        if k in m:
            return v
    return None


def remove_up_to_last_ai_message(msgs):
    for i in range(len(msgs) - 1, -1, -1):
        if isinstance(msgs[i], AIMessage):
            return msgs[:i]
    return msgs


def compile_search_results(name, raw):
    return raw