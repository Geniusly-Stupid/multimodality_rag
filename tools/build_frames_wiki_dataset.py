"""Build a Wikipedia corpus for the Frames Benchmark dataset."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set
from urllib.parse import unquote, urlparse

from datasets import load_dataset

try:
    import wikipediaapi
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'wikipediaapi'. Install it with `pip install wikipedia-api`."
    ) from exc

LINK_PREFIX = "wikipedia_link_"
OUTPUT_DIR = Path("frames_wiki_dataset")
PAGES_DIR = OUTPUT_DIR / "pages"
DISALLOWED_NAMESPACES = {
    "file",
    "category",
    "wikipedia",
    "template",
    "help",
    "portal",
    "special",
    "talk",
    "mediawiki",
    "module",
}
MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 2.0


@dataclass
class PageResult:
    """Container for scraped Wikipedia page data."""

    page_id: str
    title: str
    url: str
    text: str
    images: List[str]


def normalize_link(link: Any) -> Optional[str]:
    """Normalize a raw link string."""
    if not link:
        return None
    link_str = str(link).strip()
    if not link_str:
        return None
    link_str = link_str.rstrip("/")
    if not link_str.lower().startswith("http"):
        return None
    return link_str


def parse_wiki_links_field(value: Any) -> List[str]:
    """Parse the wiki_links column into a Python list."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, (list, tuple)):
                return [str(item) for item in parsed]
        except Exception:
            pass
        cleaned = stripped.strip("[]")
        return [item.strip().strip("'\"") for item in cleaned.split(",") if item.strip()]
    return []


def collect_links_from_example(example: Dict[str, Any]) -> List[str]:
    """Collect and deduplicate all Wikipedia links from a dataset row."""
    raw_links: List[str] = []
    for key, value in example.items():
        if key.startswith(LINK_PREFIX) and value:
            raw_links.append(str(value))
    raw_links.extend(parse_wiki_links_field(example.get("wiki_links")))

    seen: Set[str] = set()
    deduped: List[str] = []
    for raw in raw_links:
        normalized = normalize_link(raw)
        if not normalized:
            continue
        if normalized.endswith("#"):
            normalized = normalized[:-1]
        normalized = normalized.split("#")[0]
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def link_to_title(link: str) -> Optional[str]:
    """Extract a Wikipedia page title from a URL."""
    try:
        parsed = urlparse(link)
    except ValueError:
        return None
    hostname = parsed.netloc.lower()
    if "wikipedia.org" not in hostname:
        return None
    path = parsed.path
    if not path:
        return None
    if not path.startswith("/wiki/"):
        return None
    title = path[len("/wiki/") :]
    if not title:
        return None
    title = unquote(title)
    if ":" in title:
        namespace = title.split(":", 1)[0].lower()
        if namespace in DISALLOWED_NAMESPACES:
            return None
    return title


def slugify_title(title: str) -> str:
    """Create a filesystem-safe identifier from a title."""
    slug = title.strip().replace(" ", "_").replace("/", "_")
    slug = slug.lower()
    slug = re.sub(r"[^0-9a-zA-Z_.-]", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    if not slug:
        slug = "page"
    digest = hashlib.sha1(title.encode("utf-8")).hexdigest()[:8]
    return f"{slug}_{digest}"


def ensure_directory(path: Path) -> None:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def serialize_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write a JSON payload with UTF-8 encoding."""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_lines(path: Path, lines: Sequence[str]) -> None:
    """Write newline-separated strings to disk."""
    path.write_text("\n".join(lines), encoding="utf-8")


class WikipediaCrawler:
    """Wrapper around wikipediaapi with caching and persistence helpers."""

    def __init__(self, language: str = "en") -> None:
        self.api = wikipediaapi.Wikipedia(
            language=language,
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent="frames-wiki-crawler/1.0 (https://github.com/)",
        )
        self.link_cache: Dict[str, Optional[PageResult]] = {}
        self.title_cache: Dict[str, PageResult] = {}

    def crawl(self, link: str) -> Optional[PageResult]:
        """Fetch a Wikipedia page for a link."""
        normalized = normalize_link(link)
        if not normalized:
            print(f"[warn] Skipping invalid link: {link}")
            return None
        if normalized in self.link_cache:
            return self.link_cache[normalized]

        title = link_to_title(normalized)
        if not title:
            print(f"[warn] Could not derive Wikipedia title from link: {link}")
            self.link_cache[normalized] = None
            return None

        cached_by_title = self.title_cache.get(title.lower())
        if cached_by_title:
            self.link_cache[normalized] = cached_by_title
            return cached_by_title

        page = self._fetch_page_with_retries(title)
        if not page:
            self.link_cache[normalized] = None
            return None

        text = (page.text or "").strip()
        image_urls = self._collect_image_urls(page)
        page_id = slugify_title(page.title or title)
        result = PageResult(
            page_id=page_id,
            title=page.title or title,
            url=page.fullurl or normalized,
            text=text,
            images=image_urls,
        )
        self.link_cache[normalized] = result
        self.title_cache[title.lower()] = result
        return result

    def _fetch_page_with_retries(self, title: str) -> Optional[wikipediaapi.WikipediaPage]:
        """Fetch a page with retry/backoff to handle transient errors."""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                page = self.api.page(title)
            except Exception as exc:  # noqa: BLE001 - log and retry
                print(f"[warn] Error fetching '{title}' (attempt {attempt}/{MAX_RETRIES}): {exc}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_WAIT_SECONDS * attempt)
                    continue
                return None
            if not page.exists():
                print(f"[warn] Wikipedia page does not exist: {title}")
                return None
            if page.ns and str(page.ns).lower() in DISALLOWED_NAMESPACES:
                print(f"[warn] Skipping non-article namespace for title '{title}'")
                return None
            return page
        return None

    @staticmethod
    def _collect_image_urls(page: wikipediaapi.WikipediaPage) -> List[str]:
        """Extract absolute image URLs from a wikipediaapi page."""
        image_urls: List[str] = []
        images = getattr(page, "images", None)
        if isinstance(images, dict):
            candidates = images.values()
        elif isinstance(images, Iterable):
            candidates = images
        else:
            candidates = []
        for item in candidates:
            if isinstance(item, str) and item.startswith("http"):
                image_urls.append(item)
        deduped: List[str] = []
        seen: Set[str] = set()
        for url in image_urls:
            if url in seen:
                continue
            seen.add(url)
            deduped.append(url)
        return deduped


def save_page_to_disk(page: PageResult) -> None:
    """Persist a Wikipedia page into the expected folder structure."""
    ensure_directory(PAGES_DIR / page.page_id)
    page_dir = PAGES_DIR / page.page_id
    serialize_json(
        page_dir / "meta.json",
        {"title": page.title, "url": page.url, "num_images": len(page.images)},
    )
    write_lines(page_dir / "text.txt", [page.text])
    write_lines(page_dir / "images.txt", page.images)
    print(
        f"Saved page {page.title} (text: {len(page.text):,} chars, images: {len(page.images)})"
    )


def safe_get(example: Dict[str, Any], key: str) -> Any:
    """Helper to fetch a field from an example."""
    return example.get(key)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PAGES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Frames Benchmark dataset...")
    dataset = load_dataset("google/frames-benchmark")
    test_split = dataset["test"]
    total = len(test_split)
    print(f"Loaded {total} test examples.")

    crawler = WikipediaCrawler(language="en")
    stored_pages: Set[str] = set()

    samples_path = OUTPUT_DIR / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as samples_file:
        for idx, example in enumerate(test_split):
            progress = f"[{idx + 1}/{total}]"
            links = collect_links_from_example(example)
            print(f"{progress} Extracted {len(links)} links.")

            page_ids: List[str] = []
            if links:
                print(f"{progress} Crawling {len(links)} Wikipedia pages...")

            for link in links:
                page = crawler.crawl(link)
                if not page:
                    continue
                if page.page_id not in stored_pages:
                    save_page_to_disk(page)
                    stored_pages.add(page.page_id)
                page_ids.append(page.page_id)

            record = {
                "id": idx,
                "prompt": safe_get(example, "prompt"),
                "answer": safe_get(example, "answer"),
                "reasoning_types": safe_get(example, "reasoning_types"),
                "wiki_links": links,
                "page_ids": page_ids,
            }
            samples_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved structured dataset to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted by user.")
