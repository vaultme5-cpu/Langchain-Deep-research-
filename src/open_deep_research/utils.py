import re, io, itertools, asyncio, contextlib, logging, warnings, os
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Dict, List, Literal, Optional
import aiohttp, httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, MessageLikeRepresentation, filter_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, InjectedToolArg, StructuredTool, ToolException, tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.config import get_store
from mcp import McpError
from tavily import AsyncTavilyClient
from open_deep_research.configuration import Configuration, SearchAPI
from open_deep_research.prompts import summarize_webpage_prompt
from open_deep_research.state import ResearchComplete, Summary

try: import wikipedia
except ImportError: wikipedia = None
try: import arxiv
except ImportError: arxiv = None

# Groq Multi-Key Pool
_raw_groq = os.environ.get("GROQ_API_KEYS", os.environ.get("GROQ_API_KEY", ""))
_groq_keys = [k.strip() for k in _raw_groq.split(",") if k.strip()]
_groq_pool = itertools.cycle(_groq_keys) if _groq_keys else itertools.cycle([""])

def get_api_key_for_model(model_name: str, config: RunnableConfig):
    model_name = model_name.lower()
    if model_name.startswith("groq:"): return next(_groq_pool)
    elif model_name.startswith("openai:"): return os.getenv("OPENAI_API_KEY")
    elif model_name.startswith("anthropic:"): return os.getenv("ANTHROPIC_API_KEY")
    return None

def get_tavily_api_key(config: RunnableConfig): return os.getenv("TAVILY_API_KEY")
def get_today_str() -> str:
    now = datetime.now()
    return now.strftime("%a %b ") + str(now.day) + ", " + str(now.year)
def get_config_value(value):
    if value is None: return None
    return value.value if hasattr(value, 'value') else value

@tool(description="Search GitHub for open-source repositories.")
def github_sniper(query: str) -> str:
    try:
        headers = {"Accept": "application/vnd.github.v3+json", "User-Agent": "ProjectOmega/1.0"}
        params = {"q": query, "sort": "stars", "order": "desc", "per_page": 8}
        response = httpx.get("https://api.github.com/search/repositories", headers=headers, params=params, timeout=15)
        response.raise_for_status()
        items = response.json().get("items", [])
        if not items: return "No GitHub repositories found."
        results = []
        for i, repo in enumerate(items):
            results.append("[" + str(i+1) + "] " + str(repo.get("full_name")) + " | Stars: " + str(repo.get("stargazers_count")) + " | Lang: " + str(repo.get("language")) + "\nURL: " + str(repo.get("html_url")) + "\nDesc: " + str(repo.get("description", "")[:150]))
        return "\n\n".join(results)
    except Exception as e: return "GitHub search failed: " + str(e)

@tool(description="Search HuggingFace for AI models and datasets.")
def huggingface_sniper(query: str) -> str:
    try:
        params = {"search": query, "sort": "downloads", "direction": "-1", "limit": 8}
        response = httpx.get("https://huggingface.co/api/models", params=params, timeout=15)
        response.raise_for_status()
        models = response.json()
        if not models: return "No HuggingFace models found."
        results = []
        for i, model in enumerate(models):
            model_id = str(model.get("modelId"))
            results.append("[" + str(i+1) + "] " + model_id + " | Downloads: " + str(model.get("downloads", 0)) + " | Type: " + str(model.get("pipeline_tag", "Unknown")) + "\nURL: https://huggingface.co/" + model_id)
        return "\n\n".join(results)
    except Exception as e: return "HuggingFace search failed: " + str(e)

@tool(description="Search Hacker News for top stories.")
def hackernews_sniper(query: str) -> str:
    try:
        params = {"query": query, "tags": "story", "hitsPerPage": 8}
        response = httpx.get("https://hn.algolia.com/api/v1/search", params=params, timeout=15)
        response.raise_for_status()
        hits = response.json().get("hits", [])
        if not hits: return "No Hacker News stories found."
        results = []
        for i, hit in enumerate(hits):
            hn_url = "https://news.ycombinator.com/item?id=" + str(hit.get("objectID", ""))
            results.append("[" + str(i+1) + "] " + str(hit.get("title")) + " | Points: " + str(hit.get("points", 0)) + " | Comments: " + str(hit.get("num_comments", 0)) + "\nHN Link: " + hn_url)
        return "\n\n".join(results)
    except Exception as e: return "Hacker News search failed: " + str(e)

@tool(description="Search Wikipedia.")
def wikipedia_search(query: str) -> str:
    if not wikipedia: return "Wikipedia library not installed."
    try:
        page = wikipedia.page(query, auto_suggest=True)
        return "SOURCE: Wikipedia (" + str(page.url) + ")\nTITLE: " + str(page.title) + "\nCONTENT: " + str(page.content[:4000])
    except Exception: return "No Wikipedia page found."

@tool(description="Search ArXiv.")
def arxiv_search(query: str) -> str:
    if not arxiv: return "ArXiv library not installed."
    try:
        search = arxiv.Search(query=query, max_results=3, sort_by=arxiv.SortCriterion.Relevance)
        results = []
        for i, paper in enumerate(search.results()):
            results.append("[" + str(i+1) + "] " + str(paper.title) + " (" + str(paper.entry_id) + ")\nAbstract: " + str(paper.summary[:1500]))
        return "\n\n".join(results) if results else "No ArXiv papers found."
    except Exception: return "ArXiv search failed."

@tool(description="Search PubMed.")
def pubmed_search(query: str) -> str:
    try:
        resp = httpx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params={"db": "pubmed", "term": query, "retmode": "json", "retmax": 3}, timeout=10).json()
        ids = resp.get("esearchresult", {}).get("idlist", [])
        if not ids: return "No PubMed articles found."
        fetch_resp = httpx.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi", params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"}, timeout=10).json()
        results = []
        for uid in ids:
            data = fetch_resp.get("result", {}).get(uid, {})
            results.append("TITLE: " + str(data.get("title", "Unknown")) + "\nPMID: " + str(uid) + "\nLINK: https://pubmed.ncbi.nlm.nih.gov/" + str(uid) + "/")
        return "\n\n".join(results)
    except Exception: return "PubMed search failed."

@tool(description="Execute Python code securely.")
def python_repl(code: str) -> str:
    dangerous_patterns = ["import os", "import sys", "import subprocess", "import socket", "while True", "eval(", "exec(", "__import__"]
    for pattern in dangerous_patterns:
        if pattern in code: return "SECURITY VIOLATION: Code contains forbidden pattern: " + pattern
    import concurrent.futures
    def _execute():
        stdout = io.StringIO()
        safe_builtins = {"print": print, "len": len, "range": range, "sorted": sorted, "list": list, "dict": dict, "set": set, "str": str, "int": int, "float": float, "min": min, "max": max, "sum": sum, "enumerate": enumerate, "zip": zip, "map": map, "filter": filter, "isinstance": isinstance, "type": type, "abs": abs, "round": round}
        import json, re
        safe_globals = {"__builtins__": safe_builtins, "json": json, "re": re}
        with contextlib.redirect_stdout(stdout): exec(code, safe_globals, {})
        return stdout.getvalue()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_execute)
            result = future.result(timeout=5.0)
            return result or "Code executed successfully (no output)."
    except concurrent.futures.TimeoutError: return "EXECUTION TIMEOUT: Code took longer than 5 seconds."
    except Exception as e: return "EXECUTION ERROR: " + str(e)

@tool(description="Audit a URL to verify if it is 100% free.")
def audit_pricing(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0 ProjectOmega/1.0"}
        response = httpx.get(url, headers=headers, timeout=10, follow_redirects=True)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator=' ', strip=True).lower()
        red_flags = ["contact sales", "book a demo", "enterprise pricing", "credit card required", "upgrade to pro"]
        green_flags = ["open source", "free tier", "free forever", "playground", "100% free", "mit license", "apache license", "huggingface space", "github.com"]
        red_count = sum(1 for flag in red_flags if flag in text)
        green_count = sum(1 for flag in green_flags if flag in text)
        if green_count > red_count and green_count > 0: return "VERIFIED_FREE: " + url
        elif red_count > 0 and green_count == 0: return "PAID_ENTERPRISE: " + url + " (DISCARD if user requested 100% free)"
        else: return "UNKNOWN: " + url
    except Exception as e: return "AUDIT_FAILED: " + str(e)

async def validate_urls(urls: list) -> dict:
    results = {}
    async with aiohttp.ClientSession() as session:
        for url in set(urls):
            if not url or not str(url).startswith("http"): results[url] = False; continue
            try:
                async with session.head(url, allow_redirects=True, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    results[url] = resp.status < 400
            except Exception: results[url] = False
    return results

SEARXNG_SEARCH_DESCRIPTION = "A free meta-search engine."
_searxng_semaphore = asyncio.Semaphore(4)
async def _fetch_searxng(query: str, base_url: str) -> list:
    async with _searxng_semaphore:
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.get(f"{base_url}/search", params={"q": query, "format": "json"})
                response.raise_for_status()
                return response.json().get("results", [])
        except Exception: return []

async def _fetch_ddg_fallback(query: str) -> list:
    try:
        with DDGS() as ddgs:
            return [{"title": r.get("title", ""), "url": r.get("href", ""), "content": r.get("body", "")} for r in ddgs.text(query, max_results=5)]
    except Exception: return []

async def _crawl_urls(urls: list[str]) -> dict[str, str]:
    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
        results = {}
        browser_config = BrowserConfig(headless=True, verbose=False)
        run_config = CrawlerRunConfig(word_count_threshold=10, bypass_cache=True)
        async with AsyncWebCrawler(config=browser_config) as crawler:
            for url in urls:
                try:
                    result = await crawler.arun(url=url, config=run_config)
                    if result.success and result.markdown: results[url] = result.markdown
                except Exception: pass
        return results
    except Exception: return {}

@tool(description=SEARXNG_SEARCH_DESCRIPTION)
async def searxng_search(queries: List[str], max_results: Annotated[int, InjectedToolArg] = 5, config: RunnableConfig = None) -> str:
    configurable = Configuration.from_runnable_config(config)
    base_url = configurable.searxng_base_url
    all_results = {}
    for q in queries:
        results = await _fetch_searxng(q, base_url)
        if not results: results = await _fetch_ddg_fallback(q)
        for res in results:
            url = res.get("url")
            if url and url not in all_results: all_results[url] = {"title": res.get("title", "No Title"), "snippet": res.get("content", ""), "query": q}
    if not all_results: return "No valid search results found."
    urls_to_crawl = list(all_results.keys())[:max_results * len(queries)]
    crawled_content = await _crawl_urls(urls_to_crawl)
    max_char = configurable.max_content_length
    model_api_key = get_api_key_for_model(configurable.summarization_model, config)
    summarization_model = init_chat_model(model=configurable.summarization_model, max_tokens=configurable.summarization_model_max_tokens, api_key=model_api_key, tags=["langsmith:nostream"]).with_structured_output(Summary).with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    async def noop(): return None
    tasks = [noop() if not crawled_content.get(url, data.get("snippet")) else summarize_webpage(summarization_model, crawled_content.get(url, data.get("snippet"))[:max_char]) for url, data in all_results.items()]
    summaries = await asyncio.gather(*tasks)
    formatted = "Search results:\n\n"
    for i, ((url, data), summary) in enumerate(zip(all_results.items(), summaries)):
        formatted += "\n--- SOURCE " + str(i+1) + ": " + str(data['title']) + " ---\nURL: " + str(url) + "\nSUMMARY:\n" + str(summary if summary else data.get("snippet")) + "\n" + "-"*80 + "\n"
    return formatted

TAVILY_SEARCH_DESCRIPTION = "A search engine optimized for comprehensive, accurate, and trusted results."
@tool(description=TAVILY_SEARCH_DESCRIPTION)
async def tavily_search(queries: List[str], max_results: Annotated[int, InjectedToolArg] = 5, topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general", config: RunnableConfig = None) -> str:
    tavily_client = AsyncTavilyClient(api_key=get_tavily_api_key(config))
    search_results = await asyncio.gather(*[tavily_client.search(q, max_results=max_results, include_raw_content=True, topic=topic) for q in queries])
    unique_results = {}
    for response in search_results:
        for result in response['results']:
            if result['url'] not in unique_results: unique_results[result['url']] = {**result, "query": response['query']}
    configurable = Configuration.from_runnable_config(config)
    model_api_key = get_api_key_for_model(configurable.summarization_model, config)
    summarization_model = init_chat_model(model=configurable.summarization_model, max_tokens=configurable.summarization_model_max_tokens, api_key=model_api_key, tags=["langsmith:nostream"]).with_structured_output(Summary).with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    async def noop(): return None
    tasks = [noop() if not r.get("raw_content") else summarize_webpage(summarization_model, r['raw_content'][:configurable.max_content_length]) for r in unique_results.values()]
    summaries = await asyncio.gather(*tasks)
    formatted = "Search results:\n\n"
    for i, (url, result) in enumerate(unique_results.items()):
        formatted += "\n--- SOURCE " + str(i+1) + ": " + str(result['title']) + " ---\nURL: " + str(url) + "\nSUMMARY:\n" + str(result['content'] if summaries[i] is None else summaries[i]) + "\n" + "-"*80 + "\n"
    return formatted

async def summarize_webpage(model: BaseChatModel, webpage_content: str) -> str:
    try:
        prompt = summarize_webpage_prompt.format(webpage_content=webpage_content, date=get_today_str())
        summary = await asyncio.wait_for(model.ainvoke([HumanMessage(content=prompt)]), timeout=60.0)
        return "<summary>\n" + summary.summary + "\n</summary>\n\n<key_excerpts>\n" + summary.key_excerpts + "\n</key_excerpts>"
    except Exception: return webpage_content

@tool(description="Strategic reflection tool")
def think_tool(reflection: str) -> str: return "Reflection recorded: " + reflection

@tool(description="Search Semantic Scholar.")
def semantic_scholar_search(query: str) -> str:
    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {"query": query, "limit": 5, "fields": "title,abstract,url,year,citationCount"}
        resp = httpx.get(url, params=params, timeout=10).json()
        papers = resp.get("data", [])
        if not papers: return "No Semantic Scholar papers found."
        results = []
        for i, p in enumerate(papers):
            results.append("[" + str(i+1) + "] " + str(p.get("title", "Unknown")) + " (" + str(p.get("year", "Unknown")) + ") | Citations: " + str(p.get("citationCount", 0)) + "\nURL: " + str(p.get("url", "")) + "\nAbstract: " + str(p.get("abstract", "No abstract"))[:1000])
        return "\n\n".join(results)
    except Exception: return "Semantic Scholar search failed."

@tool(description="Find free alternatives to paid software.")
def saas_alternative_sniper(query: str) -> str:
    try:
        alt_query = "site:alternativeto.net OR site:opensource.com " + query + " free alternative open source"
        with DDGS() as ddgs:
            results_list = [{"title": r.get("title", ""), "url": r.get("href", ""), "content": r.get("body", "")} for r in ddgs.text(alt_query, max_results=8)]
        if not results_list: return "No free alternatives found."
        results = []
        for i, res in enumerate(results_list[:8]):
            results.append("[" + str(i+1) + "] " + str(res.get("title", "Unknown")) + "\nURL: " + str(res.get("url", "")) + "\nDetails: " + str(res.get("content", ""))[:250])
        return "\n\n".join(results)
    except Exception as e: return "Alternative search failed: " + str(e)

async def get_search_tool(search_api: SearchAPI):
    if search_api == SearchAPI.TAVILY: return [tavily_search]
    elif search_api == SearchAPI.SEARXNG: return [searxng_search]
    return []

async def get_all_tools(config: RunnableConfig):
    tools = [tool(ResearchComplete), think_tool, github_sniper, huggingface_sniper, hackernews_sniper, wikipedia_search, arxiv_search, pubmed_search, python_repl, audit_pricing, semantic_scholar_search, saas_alternative_sniper]
    configurable = Configuration.from_runnable_config(config)
    tools.extend(await get_search_tool(SearchAPI(get_config_value(configurable.search_api))))
    return tools

def intelligent_memory_reducer(current: Any, new: Any) -> list:
    if current is None: current = []
    new_items = new if isinstance(new, list) else [new]
    combined = current + new_items
    unique, seen = [], set()
    for m in combined:
        if not isinstance(m, str): continue
        fp = "".join(sorted(set(re.findall(r'\b\w{5,}\b', m.lower()))))
        if fp and fp not in seen: unique.append(m); seen.add(fp)
    return unique[-15:]

def advanced_evidence_graph_reducer(current: Any, new: Any) -> list:
    if current is None: current = []
    new_nodes = new if isinstance(new, list) else [new]
    claim_map = {}
    for n in current:
        fp = "".join(sorted(re.findall(r'\b\w{4,}\b', getattr(n, 'claim', '').lower())))
        if fp: claim_map[fp] = n
    next_cite = max([getattr(n, 'citation_index', 0) for n in current] + [0]) + 1
    for node in new_nodes:
        fp = "".join(sorted(re.findall(r'\b\w{4,}\b', getattr(node, 'claim', '').lower())))
        if not fp: continue
        if fp in claim_map:
            ex_date = getattr(claim_map[fp], 'date_published', None)
            new_date = getattr(node, 'date_published', None)
            if ex_date and new_date and str(new_date) > str(ex_date): claim_map[fp] = node
        else:
            node.citation_index = next_cite
            claim_map[fp] = node
            next_cite += 1
    support_counts = {n.citation_index: 0 for n in claim_map.values()}
    for n in claim_map.values():
        for s_idx in getattr(n, 'supports', []):
            if s_idx in support_counts: support_counts[s_idx] += 1
    final_nodes = []
    for n in claim_map.values():
        is_contradicted = False
        for c_idx in getattr(n, 'contradicts', []):
            if support_counts.get(c_idx, 0) > support_counts.get(n.citation_index, 0):
                is_contradicted = True; break
        if not is_contradicted: final_nodes.append(n)
    return final_nodes[-100:] if len(final_nodes) > 100 else final_nodes

@tool(description="Calculate real-time momentum of a topic.")
def trend_velocity_auditor(query: str) -> str:
    try:
        import time
        hn_url = "https://hn.algolia.com/api/v1/search_by_date"
        cutoff = str(int(time.time()) - 604800)
        hn_resp = httpx.get(hn_url, params={"query": query, "tags": "story", "numericFilters": "created_at_i>" + cutoff}, timeout=10).json()
        hn_hits = hn_resp.get("hits", [])
        hn_points = sum(hit.get("points", 0) for hit in hn_hits)
        gh_url = "https://api.github.com/search/repositories"
        gh_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        gh_resp = httpx.get(gh_url, headers={"Accept": "application/vnd.github.v3+json"}, params={"q": query + " pushed:>" + gh_date, "sort": "updated"}, timeout=10).json()
        gh_count = len(gh_resp.get("items", []))
        velocity = hn_points + (gh_count * 10)
        if velocity > 100: status = "TRENDING UP (High Velocity)"
        elif velocity > 20: status = "STABLE (Moderate Activity)"
        else: status = "STAGNANT/DEAD (Low Activity)"
        return "VELOCITY REPORT for '" + query + "':\nStatus: " + status + "\nHN Points (7d): " + str(hn_points) + "\nGH Active Repos (7d): " + str(gh_count) + "\nVelocity Score: " + str(velocity)
    except Exception as e: return "Velocity audit failed: " + str(e)

def get_notes_from_tool_calls(messages: list[MessageLikeRepresentation]):
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]
def anthropic_websearch_called(response):
    try: return response.response_metadata.get("usage", {}).get("server_tool_use", {}).get("web_search_requests", 0) > 0
    except Exception: return False
def openai_websearch_called(response):
    try: return any(t.get("type") == "web_search_call" for t in response.additional_kwargs.get("tool_outputs", []))
    except Exception: return False
def is_token_limit_exceeded(exception: Exception, model_name: str = None) -> bool:
    error_str = str(exception).lower()
    return 'context_length_exceeded' in error_str or 'prompt is too long' in error_str or 'resourceexhausted' in error_str or 'resource_exhausted' in error_str
MODEL_TOKEN_LIMITS = {"groq:llama-3.3-70b-versatile": 128000, "groq:llama-3.1-8b-instant": 128000, "google_genai:gemini-2.0-flash": 1048576, "openai:gpt-4o": 128000, "anthropic:claude-3-5-sonnet": 200000}
def get_model_token_limit(model_string):
    for k, v in MODEL_TOKEN_LIMITS.items():
        if k in model_string: return v
    return None
def remove_up_to_last_ai_message(messages: list[MessageLikeRepresentation]) -> list[MessageLikeRepresentation]:
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage): return messages[:i]
    return messages
def check_information_satiation(new_claims: list[str], existing_claims: list[str], threshold: float = 0.75) -> bool:
    if not existing_claims or not new_claims: return False
    def get_core_words(text: str) -> set: return set(re.findall(r'\b\w{4,}\b', text.lower()))
    existing_word_pool = set()
    for claim in existing_claims: existing_word_pool.update(get_core_words(claim))
    if not existing_word_pool: return False
    redundant_claims = 0
    for new_claim in new_claims:
        new_words = get_core_words(new_claim)
        if not new_words: continue
        if len(new_words.intersection(existing_word_pool)) / len(new_words) >= 0.60: redundant_claims += 1
    return (redundant_claims / len(new_claims)) >= threshold

def filter_and_verify_evidence(evidence_graph: list, temporal_intent: str = "Current") -> list:
    if not evidence_graph: return []
    unique_claims = {}
    for node in evidence_graph:
        claim_key = "".join(sorted(re.findall(r'\b\w{4,}\b', getattr(node, 'claim', '').lower())))
        if not claim_key: continue
        if claim_key not in unique_claims: unique_claims[claim_key] = node
        else:
            existing_date = getattr(unique_claims[claim_key], 'date_published', None)
            new_date = getattr(node, 'date_published', None)
            if existing_date and new_date:
                if temporal_intent == "Historical":
                    if str(new_date) < str(existing_date): unique_claims[claim_key] = node
                elif temporal_intent == "Timeless": pass
                else:
                    if str(new_date) > str(existing_date): unique_claims[claim_key] = node
    return list(unique_claims.values())

async def execute_tool_safely(tool, args, config):
    max_retries = 3
    for attempt in range(max_retries):
        try: return await tool.ainvoke(args, config)
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "429" in error_str or "timeout" in error_str or "resource_exhausted" in error_str:
                await asyncio.sleep(2 ** attempt); continue
            if attempt == max_retries - 1:
                tool_name = getattr(tool, "name", "unknown_tool")
                return f"[TOOL FALLBACK]: {tool_name} failed. Data unavailable."
    return "[TOOL FALLBACK]: Max retries exceeded."

def compile_search_results(tool_name: str, raw_output: str) -> str:
    if not isinstance(raw_output, str) or "failed" in raw_output.lower() or "no " in raw_output.lower()[:20]: return raw_output
    if tool_name == "github_sniper":
        rows = []
        for block in raw_output.split("\n\n"):
            header = re.search(r'\[\d+\]\s*(.*?)\s*\|\s*Stars:\s*(.*?)\s*\|\s*Lang:\s*(.*)', block)
            url = re.search(r'URL:\s*(.*)', block)
            if header and url: rows.append(f"| {header.group(1)} | {header.group(2)} | {header.group(3)} | {url.group(1)} |")
        if rows: return "### GitHub Search Results (Compiled)\n| Repository | Stars | Language | URL |\n|---|---|---|---|\n" + "\n".join(rows)
    elif tool_name == "huggingface_sniper":
        rows = []
        for block in raw_output.split("\n\n"):
            header = re.search(r'\[\d+\]\s*(.*?)\s*\|\s*Downloads:\s*(.*?)\s*\|\s*Type:\s*(.*)', block)
            url = re.search(r'URL:\s*(.*)', block)
            if header and url: rows.append(f"| {header.group(1)} | {header.group(2)} | {header.group(3)} | {url.group(1)} |")
        if rows: return "### HuggingFace Search Results (Compiled)\n| Model | Downloads | Type | URL |\n|---|---|---|---|\n" + "\n".join(rows)
    elif tool_name == "hackernews_sniper":
        rows = []
        for block in raw_output.split("\n\n"):
            header = re.search(r'\[\d+\]\s*(.*?)\s*\|\s*Points:\s*(.*?)\s*\|\s*Comments:\s*(.*)', block)
            url = re.search(r'HN Link:\s*(.*)', block)
            if header and url: rows.append(f"| {header.group(1)} | {header.group(2)} | {header.group(3)} | {url.group(1)} |")
        if rows: return "### HackerNews Search Results (Compiled)\n| Story | Points | Comments | HN Link |\n|---|---|---|---|\n" + "\n".join(rows)
    elif tool_name == "semantic_scholar_search":
        rows = []
        for block in raw_output.split("\n\n"):
            header = re.search(r'\[\d+\]\s*(.*?)\s*\((.*?)\)\s*\|\s*Citations:\s*(.*)', block)
            url = re.search(r'URL:\s*(.*)', block)
            if header and url: rows.append(f"| {header.group(1)} | {header.group(2)} | {header.group(3)} | {url.group(1)} |")
        if rows: return "### Semantic Scholar Results (Compiled)\n| Paper | Year | Citations | URL |\n|---|---|---|---|\n" + "\n".join(rows)
    return raw_output
