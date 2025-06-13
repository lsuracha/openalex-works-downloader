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
        self.sjr_cache = {}  # Cache SJR data to avoid repeated lookups
    
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
    
    async def fetch_sjr_quartile(self, session: aiohttp.ClientSession, issn_list: List[str], year: int = None) -> Dict[str, str]:
        """
        Fetch journal quartile information from SCImago Journal Rank database.
        
        Args:
            session: aiohttp client session
            issn_list: List of ISSNs for the journal
            year: Publication year (optional, defaults to most recent)
            
        Returns:
            Dictionary with SJR metrics: {quartile, sjr_score, category, h_index}
        """
        if not issn_list:
            return {"quartile": "", "sjr_score": "", "category": "", "h_index": ""}
        
        # Try each ISSN until we find a match
        for issn in issn_list:
            # Clean ISSN format (remove hyphens for API calls)
            clean_issn = issn.replace("-", "")
            cache_key = f"{clean_issn}_{year or 'latest'}"
            
            if cache_key in self.sjr_cache:
                return self.sjr_cache[cache_key]
            
            try:
                # Use SCImago CSV export API
                # This searches for journals by ISSN and returns CSV data
                base_url = "https://www.scimagojr.com/journalrank.php"
                params = {
                    'country': 'all',
                    'category': 'all', 
                    'area': 'all',
                    'year': year or 2023,  # Default to most recent year
                    'order': 'sjr',
                    'min': 0,
                    'min_type': 'cd',
                    'out': 'xls',  # CSV format
                    'search': issn
                }
                
                async with self.semaphore:
                    async with session.get(base_url, params=params) as response:
                        if response.status == 200:
                            # Parse CSV response to extract quartile info
                            text = await response.text()
                            result = self.parse_sjr_response(text, issn)
                            self.sjr_cache[cache_key] = result
                            await asyncio.sleep(self.delay)
                            return result
                        
            except Exception as e:
                print(f"Error fetching SJR data for ISSN {issn}: {e}")
                continue
        
        # No data found for any ISSN
        result = {"quartile": "", "sjr_score": "", "category": "", "h_index": ""}
        for issn in issn_list:
            clean_issn = issn.replace("-", "")
            cache_key = f"{clean_issn}_{year or 'latest'}"
            self.sjr_cache[cache_key] = result
        
        return result
    
    def parse_sjr_response(self, csv_text: str, target_issn: str) -> Dict[str, str]:
        """
        Parse SCImago CSV response to extract journal quartile information.
        
        Args:
            csv_text: Raw CSV text from SCImago
            target_issn: The ISSN we're looking for
            
        Returns:
            Dictionary with extracted SJR metrics
        """
        try:
            lines = csv_text.strip().split('\n')
            if len(lines) < 2:
                return {"quartile": "", "sjr_score": "", "category": "", "h_index": ""}
            
            # Parse CSV header to find column indices
            header = lines[0].split(';')
            
            # Common SCImago column names (may vary)
            quartile_col = None
            sjr_col = None
            category_col = None
            h_index_col = None
            issn_col = None
            
            for i, col in enumerate(header):
                col_lower = col.lower().strip('"')
                if 'quartile' in col_lower:
                    quartile_col = i
                elif 'sjr' in col_lower and 'score' not in col_lower:
                    sjr_col = i
                elif 'category' in col_lower or 'subject' in col_lower:
                    category_col = i
                elif 'h index' in col_lower or 'h_index' in col_lower:
                    h_index_col = i
                elif 'issn' in col_lower:
                    issn_col = i
            
            # Look for matching ISSN in data rows
            target_clean = target_issn.replace("-", "").strip()
            
            for line in lines[1:]:
                if not line.strip():
                    continue
                    
                cols = line.split(';')
                if len(cols) <= max(filter(None, [quartile_col, sjr_col, category_col, h_index_col, issn_col])):
                    continue
                
                # Check if this row matches our target ISSN
                if issn_col is not None and issn_col < len(cols):
                    row_issn = cols[issn_col].strip('"').replace("-", "")
                    if target_clean in row_issn or row_issn in target_clean:
                        # Extract the data
                        quartile = cols[quartile_col].strip('"') if quartile_col is not None and quartile_col < len(cols) else ""
                        sjr_score = cols[sjr_col].strip('"') if sjr_col is not None and sjr_col < len(cols) else ""
                        category = cols[category_col].strip('"') if category_col is not None and category_col < len(cols) else ""
                        h_index = cols[h_index_col].strip('"') if h_index_col is not None and h_index_col < len(cols) else ""
                        
                        return {
                            "quartile": quartile,
                            "sjr_score": sjr_score,
                            "category": category,
                            "h_index": h_index
                        }
            
        except Exception as e:
            print(f"Error parsing SJR response: {e}")
        
        return {"quartile": "", "sjr_score": "", "category": "", "h_index": ""}


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
        
        # Extract journal information
        primary_location = work.get("primary_location", {})
        source = primary_location.get("source", {}) if primary_location else {}
        
        # Extract journal name and ISSN
        journal_name = source.get("display_name", "") if source else ""
        journal_issn = ""
        if source and source.get("issn"):
            journal_issn = "; ".join(source["issn"])
        
        # Extract open access info
        open_access = work.get("open_access", {})
        is_oa = open_access.get("is_oa", False) if open_access else False
        
        # Reconstruct abstract from inverted index
        abstract_text = ""
        if work.get("abstract_inverted_index"):
            abstract_text = reconstruct_abstract_from_inverted_index(work["abstract_inverted_index"])
        
        flattened_work = {
            "title": work.get("title", ""),
            "journal_name": journal_name,
            "journal_issn": journal_issn,
            "publication_year": work.get("publication_year"),
            "doi": work.get("doi", ""),
            "open_access.is_oa": is_oa,
            "cited_by_count": work.get("cited_by_count", 0),
            "authorships.author.display_name": authors_str,
            "abstract": abstract_text
        }
        
        flattened.append(flattened_work)
    
    return pd.DataFrame(flattened)


async def fetch_openalex_works(url: str, progress_callback=None, include_quartiles: bool = True) -> pd.DataFrame:
    """
    Main async function to fetch and flatten OpenAlex works.
    
    Args:
        url: OpenAlex works URL with filters
        progress_callback: Optional callback for progress updates
        include_quartiles: Whether to fetch SJR quartile information (slower but more accurate)
        
    Returns:
        pandas DataFrame with flattened works data
    """
    fetcher = OpenAlexFetcher()
    works = await fetcher.fetch_all_pages(url, progress_callback)
    
    if include_quartiles:
        return await flatten_works_to_dataframe_with_sjr(works, fetcher)
    else:
        return flatten_works_to_dataframe(works)


def generate_filename() -> str:
    """Generate timestamped filename for CSV export."""
    now = datetime.now()
    return f"openalex_export_{now.strftime('%Y-%m-%d_%H-%M')}.csv"


async def main_cli():
    """CLI interface for standalone usage."""
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python fetch_openalex.py 'https://openalex.org/works?...' [--quartiles]")
        print("  --quartiles: Include SJR journal quartile information (slower)")
        sys.exit(1)
    
    url = sys.argv[1]
    include_quartiles = len(sys.argv) == 3 and sys.argv[2] == "--quartiles"
    
    if not url.startswith("https://openalex.org/"):
        print("Error: URL must start with 'https://openalex.org/'")
        sys.exit(1)
    
    print(f"Fetching OpenAlex works from: {url}")
    if include_quartiles:
        print("📊 Journal quartile lookup enabled (this will be slower)")
    
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
        df = await fetch_openalex_works(url, progress_callback, include_quartiles)
        
        if pbar is not None:
            pbar.close()
        
        if df.empty:
            print("No results found.")
            return
        
        filename = generate_filename()
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"\nSaved {len(df)} records to {filename}")
        
        if include_quartiles:
            quartile_count = (df['journal_quartile'] != '').sum()
            print(f"📊 Found quartile information for {quartile_count} journals")
        
    except Exception as e:
        if pbar is not None:
            pbar.close()
        print(f"Error: {e}")
        sys.exit(1)


async def flatten_works_to_dataframe_with_sjr(works: List[Dict], fetcher: OpenAlexFetcher = None) -> pd.DataFrame:
    """
    Convert raw OpenAlex works to a flattened DataFrame with SJR quartile information.
    
    Args:
        works: List of raw work dictionaries from OpenAlex API
        fetcher: OpenAlexFetcher instance for SJR lookups
        
    Returns:
        pandas DataFrame with flattened, selected columns including SJR quartiles
    """
    if not works:
        return pd.DataFrame()
    
    flattened = []
    
    # Set up session for SJR lookups if fetcher is provided
    if fetcher:
        headers = {
            'User-Agent': 'OpenAlex-Fetcher/1.0 (mailto:research@university.edu)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        }
        session = aiohttp.ClientSession(headers=headers)
    else:
        session = None
    
    try:
        for work in works:
            # Extract author names and join with semicolons
            authors = []
            for authorship in work.get("authorships", []):
                author = authorship.get("author", {})
                if author and author.get("display_name"):
                    authors.append(author["display_name"])
            authors_str = "; ".join(authors)
            
            # Extract journal information
            primary_location = work.get("primary_location", {})
            source = primary_location.get("source", {}) if primary_location else {}
            
            # Extract journal name and ISSN
            journal_name = source.get("display_name", "") if source else ""
            journal_issn_list = source.get("issn", []) if source else []
            journal_issn = "; ".join(journal_issn_list) if journal_issn_list else ""
            
            # Fetch SJR quartile information
            sjr_data = {"quartile": "", "sjr_score": "", "category": "", "h_index": ""}
            if fetcher and session and journal_issn_list:
                publication_year = work.get("publication_year")
                sjr_data = await fetcher.fetch_sjr_quartile(session, journal_issn_list, publication_year)
            
            # Extract open access info
            open_access = work.get("open_access", {})
            is_oa = open_access.get("is_oa", False) if open_access else False
            
            # Reconstruct abstract from inverted index
            abstract_text = ""
            if work.get("abstract_inverted_index"):
                abstract_text = reconstruct_abstract_from_inverted_index(work["abstract_inverted_index"])
            
            flattened_work = {
                "title": work.get("title", ""),
                "journal_name": journal_name,
                "journal_issn": journal_issn,
                "journal_quartile": sjr_data["quartile"],
                "sjr_score": sjr_data["sjr_score"],
                "journal_category": sjr_data["category"],
                "journal_h_index": sjr_data["h_index"],
                "publication_year": work.get("publication_year"),
                "doi": work.get("doi", ""),
                "open_access.is_oa": is_oa,
                "cited_by_count": work.get("cited_by_count", 0),
                "authorships.author.display_name": authors_str,
                "abstract": abstract_text
            }
            
            flattened.append(flattened_work)
    
    finally:
        if session:
            await session.close()
    
    return pd.DataFrame(flattened)


if __name__ == "__main__":
    asyncio.run(main_cli())

