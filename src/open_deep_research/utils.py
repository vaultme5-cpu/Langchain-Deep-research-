import re
import asyncio
import logging
import os
import warnings
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Dict, List, Literal, Optional
import aiohttp
import httpx
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

SEARXNG_SEARCH_DESCRIPTION = "A free meta-search engine."
_searxng_semaphore = asyncio.Semaphore(4)

async def _fetch_searxng(query: str, base_url: str) -> list:
    async with _searxng_semaphore:
        params = {"q": query, "format": "json"}
        async with httpx.AsyncClient(timeout=20.0) as client:
            try:
                response = await client.get(f"{base_url}/search", params=params)
                response.raise_for_status()
                return response.json().get("results", [])
            except Exception as e: return []

async def _fetch_ddg_fallback(query: str) -> list:
    try:
        with DDGS() as ddgs:
            return [{"title": r.get("title", ""), "url": r.get("href", ""), "content": r.get("body", "")} for r in ddgs.text(query, max_results=5)]
    except Exception: return []

# INJECTED: The missing Crawl4AI function
async def _crawl_urls(urls: list[str]) -> dict[str, str]:
    """Sector 2: Deep Extraction. Gracefully handles missing crawl4ai."""
    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
    except ImportError:
        logging.warning("crawl4ai not installed in this cloud environment. Falling back to SearXNG snippets.")
        return {}
    results = {}
    try:
        browser_config = BrowserConfig(headless=True, verbose=False)
        run_config = CrawlerRunConfig(word_count_threshold=10, bypass_cache=True)
        async with AsyncWebCrawler(config=browser_config) as crawler:
            for url in urls:
                try:
                    result = await crawler.arun(url=url, config=run_config)
                    if result.success and result.markdown:
                        results[url] = result.markdown
                except Exception:
                    pass
    except Exception as e:
        logging.warning(f"Crawl4AI execution failed: {e}. Falling back to snippets.")
    return results

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
    max_char_to_include = configurable.max_content_length
    model_api_key = get_api_key_for_model(configurable.summarization_model, config)
    summarization_model = init_chat_model(model=configurable.summarization_model, max_tokens=configurable.summarization_model_max_tokens, api_key=model_api_key, tags=["langsmith:nostream"]).with_structured_output(Summary).with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    async def noop(): return None
    summarization_tasks = []
    for url, data in all_results.items():
        content = crawled_content.get(url, data.get("snippet", ""))
        if not content: summarization_tasks.append(noop())
        else: summarization_tasks.append(summarize_webpage(summarization_model, content[:max_char_to_include]))
    summaries = await asyncio.gather(*summarization_tasks)
    
    # BULLETPROOF STRING CONCATENATION
    formatted_output = "Search results: \n\n"
    for i, ((url, data), summary) in enumerate(zip(all_results.items(), summaries)):
        formatted_output += "\n--- SOURCE " + str(i+1) + ": " + str(data['title']) + " ---\n"
        formatted_output += "URL: " + str(url) + "\n"
        final_content = summary if summary else data.get("snippet", "No content available.")
        formatted_output += "SUMMARY:\n" + str(final_content) + "\n"
        formatted_output += "\n" + "-" * 80 + "\n"
    return formatted_output

TAVILY_SEARCH_DESCRIPTION = "A search engine optimized for comprehensive, accurate, and trusted results."
@tool(description=TAVILY_SEARCH_DESCRIPTION)
async def tavily_search(queries: List[str], max_results: Annotated[int, InjectedToolArg] = 5, topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general", config: RunnableConfig = None) -> str:
    search_results = await tavily_search_async(queries, max_results=max_results, topic=topic, include_raw_content=True, config=config)
    unique_results = {}
    for response in search_results:
        for result in response['results']:
            url = result['url']
            if url not in unique_results: unique_results[url] = {**result, "query": response['query']}
    configurable = Configuration.from_runnable_config(config)
    max_char_to_include = configurable.max_content_length
    model_api_key = get_api_key_for_model(configurable.summarization_model, config)
    summarization_model = init_chat_model(model=configurable.summarization_model, max_tokens=configurable.summarization_model_max_tokens, api_key=model_api_key, tags=["langsmith:nostream"]).with_structured_output(Summary).with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    async def noop(): return None
    summarization_tasks = [noop() if not result.get("raw_content") else summarize_webpage(summarization_model, result['raw_content'][:max_char_to_include]) for result in unique_results.values()]
    summaries = await asyncio.gather(*summarization_tasks)
    summarized_results = {url: {'title': result['title'], 'content': result['content'] if summary is None else summary} for url, result, summary in zip(unique_results.keys(), unique_results.values(), summaries)}
    if not summarized_results: return "No valid search results found."
    formatted_output = "Search results: \n\n"
    for i, (url, result) in enumerate(summarized_results.items()):
        formatted_output += "\n\n--- SOURCE " + str(i+1) + ": " + str(result['title']) + " ---\n"
        formatted_output += "URL: " + str(url) + "\n\n"
        formatted_output += "SUMMARY:\n" + str(result['content']) + "\n\n"
        formatted_output += "\n\n" + "-" * 80 + "\n"
    return formatted_output

async def tavily_search_async(search_queries, max_results: int = 5, topic: Literal["general", "news", "finance"] = "general", include_raw_content: bool = True, config: RunnableConfig = None):
    tavily_client = AsyncTavilyClient(api_key=get_tavily_api_key(config))
    search_tasks = [tavily_client.search(query, max_results=max_results, include_raw_content=include_raw_content, topic=topic) for query in search_queries]
    return await asyncio.gather(*search_tasks)

async def summarize_webpage(model: BaseChatModel, webpage_content: str) -> str:
    try:
        prompt_content = summarize_webpage_prompt.format(webpage_content=webpage_content, date=get_today_str())
        summary = await asyncio.wait_for(model.ainvoke([HumanMessage(content=prompt_content)]), timeout=60.0)
        return "<summary>\n" + summary.summary + "\n</summary>\n\n<key_excerpts>\n" + summary.key_excerpts + "\n</key_excerpts>"
    except Exception: return webpage_content

@tool(description="Strategic reflection tool for research planning")
def think_tool(reflection: str) -> str:
    return "Reflection recorded: " + reflection

async def get_mcp_access_token(supabase_token: str, base_mcp_url: str) -> Optional[Dict[str, Any]]:
    try:
        form_data = {"client_id": "mcp_default", "subject_token": supabase_token, "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange", "resource": base_mcp_url.rstrip("/") + "/mcp", "subject_token_type": "urn:ietf:params:oauth:token-type:access_token"}
        async with aiohttp.ClientSession() as session:
            async with session.post(base_mcp_url.rstrip("/") + "/oauth/token", headers={"Content-Type": "application/x-www-form-urlencoded"}, data=form_data) as response:
                if response.status == 200: return await response.json()
    except Exception: pass
    return None

async def get_tokens(config: RunnableConfig):
    store = get_store()
    thread_id = config.get("configurable", {}).get("thread_id")
    user_id = config.get("metadata", {}).get("owner")
    if not thread_id or not user_id: return None
    tokens = await store.aget((user_id, "tokens"), "data")
    if not tokens: return None
    if datetime.now(timezone.utc) > tokens.created_at + timedelta(seconds=tokens.value.get("expires_in", 0)):
        await store.adelete((user_id, "tokens"), "data")
        return None
    return tokens.value

async def set_tokens(config: RunnableConfig, tokens: dict[str, Any]):
    store = get_store()
    thread_id = config.get("configurable", {}).get("thread_id")
    user_id = config.get("metadata", {}).get("owner")
    if thread_id and user_id: await store.aput((user_id, "tokens"), "data", tokens)

async def fetch_tokens(config: RunnableConfig) -> dict[str, Any]:
    current_tokens = await get_tokens(config)
    if current_tokens: return current_tokens
    supabase_token = config.get("configurable", {}).get("x-supabase-access-token")
    mcp_config = config.get("configurable", {}).get("mcp_config")
    if not supabase_token or not mcp_config or not mcp_config.get("url"): return None
    mcp_tokens = await get_mcp_access_token(supabase_token, mcp_config.get("url"))
    if not mcp_tokens: return None
    await set_tokens(config, mcp_tokens)
    return mcp_tokens

def wrap_mcp_authenticate_tool(tool: StructuredTool) -> StructuredTool:
    original_coroutine = tool.coroutine
    async def authentication_wrapper(**kwargs):
        try: return await original_coroutine(**kwargs)
        except BaseException as original_error:
            mcp_error = None
            if isinstance(original_error, McpError): mcp_error = original_error
            elif hasattr(original_error, 'exceptions'):
                for sub_exc in original_error.exceptions:
                    if isinstance(sub_exc, McpError): mcp_error = sub_exc; break
            if not mcp_error: raise original_error
            error_details = mcp_error.error
            if getattr(error_details, "code", None) == -32003:
                msg = getattr(getattr(error_details, "data", None), "message", {}).get("text", "Required interaction")
                raise ToolException(msg) from original_error
            raise original_error
    tool.coroutine = authentication_wrapper
    return tool

async def load_mcp_tools(config: RunnableConfig, existing_tool_names: set[str]) -> list[BaseTool]:
    configurable = Configuration.from_runnable_config(config)
    mcp_tokens = await fetch_tokens(config) if configurable.mcp_config and configurable.mcp_config.auth_required else None
    if not (configurable.mcp_config and configurable.mcp_config.url and configurable.mcp_config.tools and (mcp_tokens or not configurable.mcp_config.auth_required)): return []
    try:
        client = MultiServerMCPClient({"server_1": {"url": configurable.mcp_config.url.rstrip("/") + "/mcp", "headers": {"Authorization": "Bearer " + mcp_tokens['access_token']} if mcp_tokens else None, "transport": "streamable_http"}})
        available_mcp_tools = await client.get_tools()
    except Exception: return []
    configured_tools = []
    for mcp_tool in available_mcp_tools:
        if mcp_tool.name in existing_tool_names or mcp_tool.name not in set(configurable.mcp_config.tools): continue
        configured_tools.append(wrap_mcp_authenticate_tool(mcp_tool))
    return configured_tools

async def get_search_tool(search_api: SearchAPI):
    if search_api == SearchAPI.ANTHROPIC: return [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}]
    elif search_api == SearchAPI.OPENAI: return [{"type": "web_search_preview"}]
    elif search_api == SearchAPI.TAVILY:
        tavily_search.metadata = {**(tavily_search.metadata or {}), "type": "search", "name": "web_search"}
        return [tavily_search]
    elif search_api == SearchAPI.SEARXNG:
        searxng_search.metadata = {**(searxng_search.metadata or {}), "type": "search", "name": "web_search"}
        return [searxng_search]
    return []

async def get_all_tools(config: RunnableConfig):
    tools = [tool(ResearchComplete), think_tool]
    configurable = Configuration.from_runnable_config(config)
    search_api = SearchAPI(get_config_value(configurable.search_api))
    tools.extend(await get_search_tool(search_api))
    existing_tool_names = {tool.name if hasattr(tool, "name") else tool.get("name", "web_search") for tool in tools}
    tools.extend(await load_mcp_tools(config, existing_tool_names))
    return tools



##########################
# OMEGA PRIME SNIPER TOOLS (Platform-Specific APIs, Zero Keys Required)
##########################

@tool(description="Search GitHub for open-source repositories. Returns exact repo names, star counts, primary language, and descriptions. Use this when the user asks about GitHub repos, open-source projects, code libraries, or developer tools.")
def github_sniper(query: str) -> str:
    """Direct API access to GitHub Search. Bypasses web scraping for pure structured data."""
    try:
        headers = {"Accept": "application/vnd.github.v3+json", "User-Agent": "ProjectOmega-DeepResearch/1.0"}
        params = {"q": query, "sort": "stars", "order": "desc", "per_page": 10}
        response = httpx.get("https://api.github.com/search/repositories", headers=headers, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        items = data.get("items", [])
        if not items:
            return "No GitHub repositories found for this query."
        
        results = []
        for i, repo in enumerate(items):
            name = str(repo.get("full_name", "Unknown"))
            stars = str(repo.get("stargazers_count", 0))
            language = str(repo.get("language", "Unknown"))
            description = str(repo.get("description", "No description") or "No description")[:200]
            url = str(repo.get("html_url", ""))
            updated = str(repo.get("updated_at", "Unknown"))[:10]
            results.append("[" + str(i+1) + "] " + name + " | Stars: " + stars + " | Language: " + language + " | Updated: " + updated + "\nURL: " + url + "\nDescription: " + description)
        
        return "\n\n".join(results)
    except Exception as e:
        return "GitHub search failed: " + str(e)

@tool(description="Search HuggingFace for AI models, datasets, and spaces. Returns model names, download counts, likes, and pipeline tags. Use this when the user asks about AI models, LLMs, diffusion models, ML libraries, or HuggingFace resources.")
def huggingface_sniper(query: str) -> str:
    """Direct API access to HuggingFace. Returns trending models with download metrics."""
    try:
        params = {"search": query, "sort": "downloads", "direction": "-1", "limit": 10}
        response = httpx.get("https://huggingface.co/api/models", params=params, timeout=15)
        response.raise_for_status()
        models = response.json()
        if not models:
            return "No HuggingFace models found for this query."
        
        results = []
        for i, model in enumerate(models):
            model_id = str(model.get("modelId", "Unknown"))
            downloads = str(model.get("downloads", 0))
            likes = str(model.get("likes", 0))
            pipeline_tag = str(model.get("pipeline_tag", "Unknown"))
            url = "https://huggingface.co/" + model_id
            results.append("[" + str(i+1) + "] " + model_id + " | Downloads: " + downloads + " | Likes: " + likes + " | Type: " + pipeline_tag + "\nURL: " + url)
        
        return "\n\n".join(results)
    except Exception as e:
        return "HuggingFace search failed: " + str(e)

@tool(description="Search Hacker News for top stories and discussions. Returns story titles, points, comment counts, and URLs. Use this when the user asks about trending tech topics, startup news, developer discussions, or what the engineering community is currently talking about.")
def hackernews_sniper(query: str) -> str:
    """Direct API access to Hacker News via Algolia. Returns top-voted discussions."""
    try:
        params = {"query": query, "tags": "story", "hitsPerPage": 10}
        response = httpx.get("https://hn.algolia.com/api/v1/search", params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        hits = data.get("hits", [])
        if not hits:
            return "No Hacker News stories found for this query."
        
        results = []
        for i, hit in enumerate(hits):
            title = str(hit.get("title", "Untitled"))
            points = str(hit.get("points", 0))
            comments = str(hit.get("num_comments", 0))
            url = str(hit.get("url") or ("https://news.ycombinator.com/item?id=" + str(hit.get("objectID", ""))))
            hn_url = "https://news.ycombinator.com/item?id=" + str(hit.get("objectID", ""))
            created = str(hit.get("created_at", "Unknown"))[:10]
            results.append("[" + str(i+1) + "] " + title + " | Points: " + points + " | Comments: " + comments + " | Date: " + created + "\nArticle: " + url + "\nHN Discussion: " + hn_url)
        
        return "\n\n".join(results)
    except Exception as e:
        return "Hacker News search failed: " + str(e)

@tool(description="Find free and open-source alternatives to paid software. Returns verified alternative names, pricing status, and descriptions. Use this when the user asks for 'free alternatives to X', 'open source alternatives', 'cheaper alternatives', or wants to replace paid SaaS tools.")
def saas_alternative_sniper(query: str) -> str:
    """Targets AlternativeTo.net and open-source directories to find verified free alternatives."""
    try:
        # Strategy: Query SearXNG with site-specific targeting for alternative directories
        alt_query = "site:alternativeto.net OR site:opensource.com OR site:alternativeto.net/software " + query + " free alternative open source"
        headers = {"User-Agent": "ProjectOmega-DeepResearch/1.0"}
        
        # Try to use configured SearXNG first, fall back to DuckDuckGo
        base_url = "http://localhost:8080"
        try:
            from open_deep_research.configuration import Configuration
            # We cannot access config directly here, so we use a sensible default
            # The researcher will have already loaded config in its flow
        except Exception:
            pass
        
        params = {"q": alt_query, "format": "json"}
        results_list = []
        try:
            response = httpx.get(base_url + "/search", params=params, headers=headers, timeout=20)
            response.raise_for_status()
            results_list = response.json().get("results", [])
        except Exception:
            # Fallback to DuckDuckGo
            try:
                with DDGS() as ddgs:
                    results_list = [{"title": r.get("title", ""), "url": r.get("href", ""), "content": r.get("body", "")} for r in ddgs.text(alt_query, max_results=8)]
            except Exception:
                pass
        
        if not results_list:
            return "No free alternatives found for this software."
        
        results = []
        for i, res in enumerate(results_list[:8]):
            title = str(res.get("title", "Unknown"))
            url = str(res.get("url", ""))
            snippet = str(res.get("content", res.get("snippet", "No description")))[:250]
            results.append("[" + str(i+1) + "] " + title + "\nURL: " + url + "\nDetails: " + snippet)
        
        return "\n\n".join(results)
    except Exception as e:
        return "Alternative search failed: " + str(e)


##########################
# OMEGA PRIME UNIVERSAL DOMAIN INGESTORS (Zero API Keys Required)
##########################
try:
    import wikipedia
except ImportError:
    wikipedia = None

try:
    import arxiv
except ImportError:
    arxiv = None

@tool(description="Search Wikipedia for encyclopedic, historical, and general factual knowledge. Use this for foundational context, history, and broad scientific concepts.")
def wikipedia_search(query: str) -> str:
    if not wikipedia: return "Wikipedia library not installed."
    try:
        page = wikipedia.page(query, auto_suggest=True)
        return "SOURCE: Wikipedia (" + str(page.url) + ")\nTITLE: " + str(page.title) + "\nCONTENT: " + str(page.content[:4000])
    except Exception:
        return "No Wikipedia page found for this query."

@tool(description="Search ArXiv for physics, mathematics, and deep computer science pre-prints. Use this for cutting-edge theoretical and academic research.")
def arxiv_search(query: str) -> str:
    if not arxiv: return "ArXiv library not installed."
    try:
        search = arxiv.Search(query=query, max_results=3, sort_by=arxiv.SortCriterion.Relevance)
        results = []
        for i, paper in enumerate(search.results()):
            results.append("[" + str(i+1) + "] " + str(paper.title) + " (" + str(paper.entry_id) + ")\nAbstract: " + str(paper.summary[:1500]))
        return "\n\n".join(results) if results else "No ArXiv papers found."
    except Exception:
        return "ArXiv search failed."

@tool(description="Search PubMed for medical, biological, and clinical trials. Use this for healthcare, biology, pharmacology, and medical research.")
def pubmed_search(query: str) -> str:
    try:
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {"db": "pubmed", "term": query, "retmode": "json", "retmax": 3}
        resp = httpx.get(search_url, params=params, timeout=10).json()
        ids = resp.get("esearchresult", {}).get("idlist", [])
        if not ids: return "No PubMed articles found."
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        fetch_params = {"db": "pubmed", "id": ",".join(ids), "retmode": "json"}
        fetch_resp = httpx.get(fetch_url, params=fetch_params, timeout=10).json()
        results = []
        for uid in ids:
            data = fetch_resp.get("result", {}).get(uid, {})
            title = data.get("title", "Unknown")
            results.append("TITLE: " + str(title) + "\nPMID: " + str(uid) + "\nLINK: https://pubmed.ncbi.nlm.nih.gov/" + str(uid) + "/")
        return "\n\n".join(results)
    except Exception:
        return "PubMed search failed."

@tool(description="Search Semantic Scholar for universal academic papers, citation counts, and abstracts. Use this for cross-disciplinary academic consensus and literature reviews.")
def semantic_scholar_search(query: str) -> str:
    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {"query": query, "limit": 5, "fields": "title,abstract,url,year,citationCount"}
        resp = httpx.get(url, params=params, timeout=10).json()
        papers = resp.get("data", [])
        if not papers: return "No Semantic Scholar papers found."
        results = []
        for i, p in enumerate(papers):
            title = str(p.get("title", "Unknown"))
            year = str(p.get("year", "Unknown"))
            citations = str(p.get("citationCount", 0))
            abstract = str(p.get("abstract", "No abstract"))[:1000]
            paper_url = str(p.get("url", ""))
            results.append("[" + str(i+1) + "] " + title + " (" + year + ") | Citations: " + citations + "\nURL: " + paper_url + "\nAbstract: " + abstract)
        return "\n\n".join(results)
    except Exception:
        return "Semantic Scholar search failed."


##########################
# OMEGA PRIME PROTOCOL 4: Hacker's Sandbox & Post-Hoc Validation
##########################
import io
import contextlib

@tool(description="Execute Python code to filter data, calculate statistics, parse JSON, or sort lists. Use this when you need to programmatically process search results, perform math, or rank items. Returns standard output.")
def python_repl(code: str) -> str:
    stdout = io.StringIO()
    # Restricted builtins for security (prevents malicious OS calls)
    safe_builtins = {"print": print, "len": len, "range": range, "sorted": sorted, "list": list, "dict": dict, "set": set, "str": str, "int": int, "float": float, "min": min, "max": max, "sum": sum, "enumerate": enumerate, "zip": zip, "map": map, "filter": filter, "isinstance": isinstance, "type": type}
    import json, re
    safe_globals = {"__builtins__": safe_builtins, "json": json, "re": re}
    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, safe_globals, {})
        return stdout.getvalue() or "Code executed successfully (no output)."
    except Exception as e:
        return "Execution Error: " + str(e)

async def validate_urls(urls: list) -> dict:
    """Asynchronously pings URLs to ensure 100% citation integrity before report generation."""
    results = {}
    async with aiohttp.ClientSession() as session:
        for url in set(urls):
            if not url or not str(url).startswith("http"):
                results[url] = False
                continue
            try:
                # Try HEAD request first (fast)
                async with session.head(url, allow_redirects=True, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    results[url] = resp.status < 400
            except Exception:
                # Fallback to GET if HEAD is blocked by the server
                try:
                    async with session.get(url, allow_redirects=True, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                        results[url] = resp.status < 400
                except Exception:
                    results[url] = False
    return results

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
    provider = None
    if model_name:
        if model_name.lower().startswith('openai:'): provider = 'openai'
        elif model_name.lower().startswith('anthropic:'): provider = 'anthropic'
        elif 'gemini' in model_name.lower() or 'google' in model_name.lower(): provider = 'gemini'
    if provider == 'openai': return _check_openai_token_limit(exception, error_str)
    elif provider == 'anthropic': return _check_anthropic_token_limit(exception, error_str)
    elif provider == 'gemini': return _check_gemini_token_limit(exception, error_str)
    return _check_openai_token_limit(exception, error_str) or _check_anthropic_token_limit(exception, error_str) or _check_gemini_token_limit(exception, error_str)

def _check_openai_token_limit(exception: Exception, error_str: str) -> bool:
    try: return 'context_length_exceeded' in str(getattr(exception, 'code', '')) or 'prompt is too long' in error_str
    except Exception: return False

def _check_anthropic_token_limit(exception: Exception, error_str: str) -> bool:
    return 'prompt is too long' in error_str

def _check_gemini_token_limit(exception: Exception, error_str: str) -> bool:
    return 'resourceexhausted' in error_str or 'resource_exhausted' in error_str

MODEL_TOKEN_LIMITS = {
    "google_genai:gemini-2.5-pro": 2097152,
    "google_genai:gemini-2.0-flash": 1048576,
    "google_genai:gemini-1.5-pro": 2097152,
    "google_genai:gemini-1.5-flash": 1048576,
    "google:gemini-2.5-pro": 2097152,
    "google:gemini-1.5-pro": 2097152,"openai:gpt-4.1": 1047576, "openai:gpt-4o": 128000, "anthropic:claude-3-5-sonnet": 200000, "google:gemini-1.5-pro": 2097152, "google:gemini-1.5-flash": 1048576, "google:gemini-2.5-pro": 1048576}
def get_model_token_limit(model_string):
    for k, v in MODEL_TOKEN_LIMITS.items():
        if k in model_string: return v
    return None

def remove_up_to_last_ai_message(messages: list[MessageLikeRepresentation]) -> list[MessageLikeRepresentation]:
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage): return messages[:i]
    return messages

def get_today_str() -> str:
    now = datetime.now()
    return now.strftime("%a %b ") + str(now.day) + ", " + str(now.year)

def get_config_value(value):
    if value is None: return None
    return value.value if hasattr(value, 'value') else value

def get_api_key_for_model(model_name: str, config: RunnableConfig):
    model_name = model_name.lower()
    if os.getenv("GET_API_KEYS_FROM_CONFIG", "false").lower() == "true":
        api_keys = config.get("configurable", {}).get("apiKeys", {})
        if model_name.startswith("openai:"): return api_keys.get("OPENAI_API_KEY")
        elif model_name.startswith("anthropic:"): return api_keys.get("ANTHROPIC_API_KEY")
        elif "google" in model_name or "gemini" in model_name: return api_keys.get("GOOGLE_API_KEY")
    else:
        if model_name.startswith("openai:"): return os.getenv("OPENAI_API_KEY")
        elif model_name.startswith("anthropic:"): return os.getenv("ANTHROPIC_API_KEY")
        elif "google" in model_name or "gemini" in model_name: return os.getenv("GOOGLE_API_KEY")
    return None

def get_tavily_api_key(config: RunnableConfig):
    if os.getenv("GET_API_KEYS_FROM_CONFIG", "false").lower() == "true":
        return config.get("configurable", {}).get("apiKeys", {}).get("TAVILY_API_KEY")
    return os.getenv("TAVILY_API_KEY")

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

def filter_and_verify_evidence(evidence_graph: list) -> list:
    if not evidence_graph: return []
    unique_claims = {}
    for node in evidence_graph:
        claim_key = "".join(sorted(re.findall(r'\b\w{4,}\b', getattr(node, 'claim', '').lower())))
        if not claim_key: continue
        if claim_key not in unique_claims: unique_claims[claim_key] = node
        else:
            existing_date = getattr(unique_claims[claim_key], 'date_published', None)
            new_date = getattr(node, 'date_published', None)
            if existing_date and new_date and str(new_date) > str(existing_date): unique_claims[claim_key] = node
    return list(unique_claims.values())
##########################
# OMEGA PRIME UNIVERSAL TOOLS (Domain Ingestors & Sandbox)
##########################
import wikipedia
import arxiv
import io
import contextlib

@tool(description="Search Wikipedia for encyclopedic, historical, and general factual knowledge.")
def wikipedia_search(query: str) -> str:
    try:
        page = wikipedia.page(query, auto_suggest=True)
        return "SOURCE: Wikipedia (" + str(page.url) + ")\nTITLE: " + str(page.title) + "\nCONTENT: " + str(page.content[:4000])
    except Exception:
        return "No Wikipedia page found."

@tool(description="Search ArXiv for scientific, mathematical, and technical academic papers.")
def arxiv_search(query: str) -> str:
    try:
        search = arxiv.Search(query=query, max_results=3, sort_by=arxiv.SortCriterion.Relevance)
        results = []
        for i, paper in enumerate(search.results()):
            results.append("[" + str(i+1) + "] " + str(paper.title) + " (" + str(paper.entry_id) + ")\nAbstract: " + str(paper.summary[:1500]))
        return "\n\n".join(results) if results else "No ArXiv papers found."
    except Exception:
        return "ArXiv search failed."

@tool(description="Search PubMed for medical, biological, and clinical research.")
def pubmed_search(query: str) -> str:
    try:
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {"db": "pubmed", "term": query, "retmode": "json", "retmax": 3}
        resp = httpx.get(search_url, params=params, timeout=10).json()
        ids = resp.get("esearchresult", {}).get("idlist", [])
        if not ids: return "No PubMed articles found."
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        fetch_params = {"db": "pubmed", "id": ",".join(ids), "retmode": "json"}
        fetch_resp = httpx.get(fetch_url, params=fetch_params, timeout=10).json()
        results = []
        for uid in ids:
            data = fetch_resp.get("result", {}).get(uid, {})
            title = data.get("title", "Unknown")
            results.append("TITLE: " + str(title) + "\nPMID: " + str(uid) + "\nLINK: https://pubmed.ncbi.nlm.nih.gov/" + str(uid) + "/")
        return "\n\n".join(results)
    except Exception:
        return "PubMed search failed."

@tool(description="Execute Python code to filter data, calculate statistics, or parse JSON/CSV. Use for programmatic data manipulation.")
def python_repl(code: str) -> str:
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, {"__builtins__": __builtins__}, {})
        return stdout.getvalue() or "Code executed successfully."
    except Exception as e:
        return "Execution Error: " + str(e)
