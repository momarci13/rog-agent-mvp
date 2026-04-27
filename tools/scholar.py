"""Scholar integration: arXiv search and paper ingestion."""
from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import requests


def _retry_request(url: str, params: dict, max_retries: int = 3, backoff: float = 1.0) -> requests.Response | None:
    """Retry HTTP request with exponential backoff."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                print(f"[SCHOLAR] Request failed after {max_retries} attempts: {e}")
                return None
            wait_time = backoff * (2 ** attempt)
            print(f"[SCHOLAR] Request failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time:.1f}s: {e}")
            time.sleep(wait_time)
    return None

DEFAULT_SEARCH_CATEGORIES = ["q-fin", "stat.ML", "math.PR", "math.ST", "stat.AP"]
MATH_SEARCH_CATEGORIES = ["math.PR", "math.ST", "stat.AP", "stat.ML", "q-fin"]
FINANCE_SEARCH_CATEGORIES = ["q-fin", "stat.ML", "math.ST"]


@dataclass
class ArxivPaper:
    """Minimal arXiv paper metadata."""
    arxiv_id: str
    title: str
    authors: list[str]
    published: str
    summary: str
    url: str

    def to_markdown(self) -> str:
        """Format as markdown for KB ingestion."""
        author_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            author_str += ", et al."
        return f"""# {self.title}

**Authors:** {author_str}  
**Published:** {self.published}  
**arXiv:** {self.arxiv_id}  
**URL:** {self.url}

## Abstract

{self.summary}

---

*Source: arXiv. Retrieved dynamically for task context.*
"""

    def to_bibtex(self) -> str:
        """Generate BibTeX entry."""
        authors = " and ".join(self.authors)
        year = self.published.split("-")[0]
        key = re.sub(r"[^a-z0-9]", "", self.title.lower()[:20])
        return f"""@article{{{key}{year},
  author = {{{authors}}},
  title = {{{self.title}}},
  journal = {{arXiv preprint}},
  year = {{{year}}},
  eprint = {{{self.arxiv_id}}},
  url = {{{self.url}}},
  abstract = {{{self.summary[:200]}}}
}}
"""


def search_arxiv(
    query: str,
    n: int = 5,
    category: str | list[str] = "q-fin",
    sort_by: str = "relevance",
) -> list[ArxivPaper]:
    """Search arXiv for papers.

    Args:
        query: Search query (title, abstract, authors)
        n: Number of papers to return
        category: arXiv category or categories to search
        sort_by: Sort order ("relevance" or "submittedDate")

    Returns:
        List of ArxivPaper objects
    """
    if isinstance(category, str):
        categories = [category]
    else:
        categories = list(category)

    category_query = " OR ".join(f"cat:{c}" for c in categories)
    base_url = "http://export.arxiv.org/api/query?"
    search_query = f"({query}) AND ({category_query})"
    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": n,
        "sortBy": sort_by,
        "sortOrder": "descending",
    }

    try:
        response = _retry_request(base_url, params)
        if response is None:
            return []
    except Exception as e:
        print(f"[SCHOLAR] arXiv search failed: {e}")
        return []

    papers = []

    try:
        root = ET.fromstring(response.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        for entry in root.findall("atom:entry", ns):
            arxiv_id_elem = entry.find("atom:id", ns)
            title_elem = entry.find("atom:title", ns)
            authors_elems = entry.findall("atom:author", ns)
            published_elem = entry.find("atom:published", ns)
            summary_elem = entry.find("atom:summary", ns)

            if not all([arxiv_id_elem, title_elem, summary_elem]):
                continue

            arxiv_id = arxiv_id_elem.text.split("/abs/")[-1]
            title = title_elem.text.strip()
            authors = [
                a.find("atom:name", ns).text
                for a in authors_elems
                if a.find("atom:name", ns) is not None
            ]
            published = published_elem.text.split("T")[0] if published_elem else "unknown"
            summary = summary_elem.text.strip()
            url = f"https://arxiv.org/abs/{arxiv_id}"

            papers.append(
                ArxivPaper(
                    arxiv_id=arxiv_id,
                    title=title,
                    authors=authors,
                    published=published,
                    summary=summary,
                    url=url,
                )
            )

        return papers[:n]
    except Exception as e:
        print(f"[SCHOLAR] Failed to parse arXiv response: {e}")
        return []


def scholar_augment_task(
    task_description: str,
    n_papers: int = 5,
    category: str = "q-fin",
) -> tuple[list[ArxivPaper], str]:
    """Search for papers relevant to a task.

    Args:
        task_description: User's task description
        n_papers: Number of papers to fetch
        category: arXiv category ("q-fin", "cs.LG", "stat.AP", etc.)

    Returns:
        (list of papers, markdown context string for LLM)
    """
    try:
        print(f"[SCHOLAR] Analyzing task: '{task_description[:50]}...'")
        keywords = extract_keywords(task_description)

        if not keywords:
            print("[SCHOLAR] No suitable keywords found in task description")
            return [], ""

        query = " ".join(keywords[:5])
        keyset = set(keywords)
        math_terms = {
            "probability", "probabilistic", "bayesian", "bayes", "markov",
            "stochastic", "martingale", "ergodic", "random", "distribution",
            "likelihood", "posterior", "prior", "infer", "inference",
        }
        finance_terms = {
            "finance", "trading", "strategy", "risk", "portfolio", "returns",
            "backtest", "volatility", "leverage", "alpha", "beta", "market",
        }

        if category != "q-fin":
            search_categories = [category] if isinstance(category, str) else list(category)
        elif keyset & math_terms:
            search_categories = MATH_SEARCH_CATEGORIES
        elif keyset & finance_terms:
            search_categories = FINANCE_SEARCH_CATEGORIES
        else:
            search_categories = DEFAULT_SEARCH_CATEGORIES

        print(f"[SCHOLAR] Searching arXiv for: '{query}' in categories {search_categories}")

        papers = search_arxiv(
            query=query,
            n=n_papers,
            category=search_categories,
        )

        if not papers:
            print(f"[SCHOLAR] No papers found for query '{query}' in category '{category}'")
            return [], ""

        print(f"[SCHOLAR] Successfully retrieved {len(papers)} papers")
        for paper in papers[:3]:  # Show first 3 papers
            print(f"[SCHOLAR]   - {paper.title[:60]}... ({paper.arxiv_id})")

        context = f"""## Recently Retrieved Academic Papers (arXiv)

The following {len(papers)} papers are relevant to your task (searched for: {query}):

"""
        for i, paper in enumerate(papers, 1):
            authors_str = ", ".join(paper.authors[:2])
            if len(paper.authors) > 2:
                authors_str += " et al."
            context += f"""{i}. **{paper.title}**  
   {authors_str}  
   arXiv:{paper.arxiv_id} ({paper.published})  

"""

        return papers, context

    except Exception as e:
        print(f"[SCHOLAR] Unexpected error during scholar augmentation: {e}")
        return [], ""


def extract_keywords(text: str, max_keywords: int = 5) -> list[str]:
    """Extract keywords from task description."""
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "be", "have", "do", "run",
        "using", "use", "based", "compute", "calculate", "report", "write",
        "you", "your", "python", "code", "task", "please", "help",
    }

    words = text.lower().split()
    keywords = [
        w.strip(".,;:!?") for w in words
        if w.strip(".,;:!?") not in stop_words and len(w.strip(".,;:!?")) > 3
    ]
    return list(set(keywords))[:max_keywords]
