"""
OpenAlex Works Fetcher - Low-level async data retrieval module.

This module provides async functions to fetch OpenAlex works data with proper
rate limiting and pagination handling. Can be used standalone via CLI or 
imported by the Streamlit app.
"""

import asyncio
import json
import sys
from datetime import datetime
from typing import Dict, List, Optional
import aiohttp
import pandas as pd
from tqdm import tqdm


class OpenAlexFetcher:
    """Async fetcher for OpenAlex works with rate limiting."""
    
    def __init__(self, max_concurrent: int = 3, delay: float = 0.1):
        self.max_concurrent = max_concurrent
        self.delay = delay
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def fetch_page(self, session: aiohttp.ClientSession, url: str) -> Optional[Dict]:
        """Fetch a single page from OpenAlex API."""
        async with self.semaphore:
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        content_type = response.headers.get('content-type', '')
                        if 'application/json' in content_type:
                            data = await response.json()
                            await asyncio.sleep(self.delay)
                            return data
                        else:
                            text = await response.text()
                            print(f"Unexpected content type {content_type} for {url}")
                            print(f"Response preview: {text[:200]}")
                            return None
                    else:
                        print(f"HTTP {response.status} for {url}")
                        text = await response.text()
                        print(f"Error response: {text[:200]}")
                        return None
            except Exception as e:
                print(f"Error fetching {url}: {e}")
                return None
    
    async def fetch_all_pages(self, base_url: str, progress_callback=None) -> List[Dict]:
        """
        Fetch all pages following OpenAlex pagination cursors.
        
        Args:
            base_url: Initial OpenAlex works URL with filters
            progress_callback: Optional callback(page_num, total_records) for progress
            
        Returns:
            List of all work records across all pages
        """
        all_works = []
        
        # Convert website URL to API URL if needed
        current_url = base_url.replace("https://openalex.org/", "https://api.openalex.org/")
        page_num = 0
        
        # Ensure per-page is set to maximum (200)
        if "per-page=" not in current_url:
            separator = "&" if "?" in current_url else "?"
            current_url += f"{separator}per-page=200"
        
        headers = {
            'User-Agent': 'OpenAlex-Fetcher/1.0 (mailto:research@university.edu)',
            'Accept': 'application/json'
        }
        
        async with aiohttp.ClientSession(headers=headers) as session:
            while current_url:
                page_num += 1
                data = await self.fetch_page(session, current_url)
                
                if not data or "results" not in data:
                    break
                
                works = data["results"]
                all_works.extend(works)
                
                if progress_callback:
                    progress_callback(page_num, len(all_works))
                
                # Get next page URL from meta
                current_url = data.get("meta", {}).get("next_cursor")
                
                if not current_url or len(works) == 0:
                    break
        
        return all_works


def reconstruct_abstract_from_inverted_index(inverted_index: Dict) -> str:
    """
    Reconstruct readable abstract text from OpenAlex inverted index.
    
    Args:
        inverted_index: Dictionary where keys are words and values are lists of positions
        
    Returns:
        Reconstructed abstract text
    """
    if not inverted_index:
        return ""
    
    # Create a list to hold words at their positions
    word_positions = []
    
    # Extract all word-position pairs
    for word, positions in inverted_index.items():
        for position in positions:
            word_positions.append((position, word))
    
    # Sort by position to get correct order
    word_positions.sort(key=lambda x: x[0])
    
    # Extract just the words in order
    words = [word for position, word in word_positions]
    
    # Join words with spaces
    return " ".join(words)


def flatten_works_to_dataframe(works: List[Dict]) -> pd.DataFrame:
    """
    Convert raw OpenAlex works to a flattened DataFrame with selected fields.
    
    Args:
        works: List of raw work dictionaries from OpenAlex API
        
    Returns:
        pandas DataFrame with flattened, selected columns
    """
    if not works:
        return pd.DataFrame()
    
    flattened = []
    
    for work in works:
        # Extract author names and join with semicolons
        authors = []
        for authorship in work.get("authorships", []):
            author = authorship.get("author", {})
            if author and author.get("display_name"):
                authors.append(author["display_name"])
        authors_str = "; ".join(authors)
        
        # Extract primary location host organization lineage
        primary_location = work.get("primary_location", {})
        source = primary_location.get("source", {}) if primary_location else {}
        host_org_lineage = source.get("host_organization_lineage", []) if source else []
        
        # Extract open access info
        open_access = work.get("open_access", {})
        is_oa = open_access.get("is_oa", False) if open_access else False
        
        # Reconstruct abstract from inverted index
        abstract_text = ""
        if work.get("abstract_inverted_index"):
            abstract_text = reconstruct_abstract_from_inverted_index(work["abstract_inverted_index"])
        
        flattened_work = {
            "title": work.get("title", ""),
            "publication_year": work.get("publication_year"),
            "doi": work.get("doi", ""),
            "open_access.is_oa": is_oa,
            "cited_by_count": work.get("cited_by_count", 0),
            "authorships.author.display_name": authors_str,
            "abstract": abstract_text
        }
        
        flattened.append(flattened_work)
    
    return pd.DataFrame(flattened)


async def fetch_openalex_works(url: str, progress_callback=None) -> pd.DataFrame:
    """
    Main async function to fetch and flatten OpenAlex works.
    
    Args:
        url: OpenAlex works URL with filters
        progress_callback: Optional callback for progress updates
        
    Returns:
        pandas DataFrame with flattened works data
    """
    fetcher = OpenAlexFetcher()
    works = await fetcher.fetch_all_pages(url, progress_callback)
    return flatten_works_to_dataframe(works)


def generate_filename() -> str:
    """Generate timestamped filename for CSV export."""
    now = datetime.now()
    return f"openalex_export_{now.strftime('%Y-%m-%d_%H-%M')}.csv"


async def main_cli():
    """CLI interface for standalone usage."""
    if len(sys.argv) != 2:
        print("Usage: python fetch_openalex.py 'https://openalex.org/works?...'")
        sys.exit(1)
    
    url = sys.argv[1]
    
    if not url.startswith("https://openalex.org/"):
        print("Error: URL must start with 'https://openalex.org/'")
        sys.exit(1)
    
    print(f"Fetching OpenAlex works from: {url}")
    
    # Progress bar for CLI
    pbar = None
    
    def progress_callback(page_num: int, total_records: int):
        nonlocal pbar
        if pbar is None:
            pbar = tqdm(desc="Fetching", unit=" records")
        pbar.set_description(f"Page {page_num}")
        pbar.n = total_records
        pbar.refresh()
    
    try:
        df = await fetch_openalex_works(url, progress_callback)
        
        if pbar is not None:
            pbar.close()
        
        if df.empty:
            print("No results found.")
            return
        
        filename = generate_filename()
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"\nSaved {len(df)} records to {filename}")
        
    except Exception as e:
        if pbar is not None:
            pbar.close()
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main_cli())
