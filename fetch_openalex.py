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
    
    async def fetch_journal_quality_fast(self, session: aiohttp.ClientSession, source_id: str) -> Dict[str, str]:
        """
        Fetch journal quality metrics from OpenAlex Sources API (much faster than SCImago).
        
        Args:
            session: aiohttp client session
            source_id: OpenAlex source ID (e.g., "S137773608")
            
        Returns:
            Dictionary with quality metrics: {impact_factor_estimate, h_index, quality_tier, total_works}
        """
        if not source_id:
            return {"impact_factor_estimate": "", "h_index": "", "quality_tier": "", "total_works": ""}
        
        cache_key = f"source_{source_id}"
        if cache_key in self.sjr_cache:
            return self.sjr_cache[cache_key]
        
        try:
            # Fetch journal metadata from OpenAlex Sources API
            url = f"https://api.openalex.org/sources/{source_id}"
            
            async with self.semaphore:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Extract quality metrics
                        summary_stats = data.get("summary_stats", {})
                        impact_estimate = summary_stats.get("2yr_mean_citedness", 0)
                        h_index = summary_stats.get("h_index", 0)
                        total_works = data.get("works_count", 0)
                        
                        # Calculate quality tier based on OpenAlex metrics
                        quality_tier = self.calculate_quality_tier(impact_estimate, h_index, total_works)
                        
                        result = {
                            "impact_factor_estimate": str(round(impact_estimate, 2)) if impact_estimate else "",
                            "h_index": str(h_index) if h_index else "",
                            "quality_tier": quality_tier,
                            "total_works": str(total_works) if total_works else ""
                        }
                        
                        self.sjr_cache[cache_key] = result
                        await asyncio.sleep(self.delay)
                        return result
                        
        except Exception as e:
            print(f"Error fetching journal quality data for source {source_id}: {e}")
        
        # No data found
        result = {"impact_factor_estimate": "", "h_index": "", "quality_tier": "", "total_works": ""}
        self.sjr_cache[cache_key] = result
        return result
    
    def calculate_quality_tier(self, impact_factor: float, h_index: int, total_works: int) -> str:
        """
        Calculate journal quality tier based on OpenAlex metrics.
        
        Args:
            impact_factor: 2-year mean citedness
            h_index: Journal h-index
            total_works: Total number of publications
        
        Returns:
            Quality tier: Q1, Q2, Q3, Q4, or empty string
        """
        # High-quality journals (Q1): High impact and h-index
        if impact_factor >= 10.0 and h_index >= 200:
            return "Q1"
        elif impact_factor >= 5.0 and h_index >= 100:
            return "Q1"
        elif impact_factor >= 3.0 and h_index >= 80:
            return "Q1"
        
        # Good journals (Q2): Moderate impact and h-index
        elif impact_factor >= 2.0 and h_index >= 50:
            return "Q2"
        elif impact_factor >= 1.5 and h_index >= 30:
            return "Q2"
        
        # Average journals (Q3): Some impact
        elif impact_factor >= 1.0 and h_index >= 20:
            return "Q3"
        elif impact_factor >= 0.5 and h_index >= 10:
            return "Q3"
        
        # Lower-tier journals (Q4): Lower metrics but still indexed
        elif h_index >= 5 and total_works >= 100:
            return "Q4"
        
        # Insufficient data or very low quality
        return ""
    
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


async def fetch_openalex_works(url: str, progress_callback=None, include_quality: bool = True) -> pd.DataFrame:
    """
    Main async function to fetch and flatten OpenAlex works.
    
    Args:
        url: OpenAlex works URL with filters
        progress_callback: Optional callback for progress updates
        include_quality: Whether to fetch journal quality information (faster OpenAlex-based metrics)
        
    Returns:
        pandas DataFrame with flattened works data
    """
    fetcher = OpenAlexFetcher()
    works = await fetcher.fetch_all_pages(url, progress_callback)
    
    if include_quality:
        return await flatten_works_to_dataframe_with_quality(works, fetcher)
    else:
        return flatten_works_to_dataframe(works)


def generate_filename() -> str:
    """Generate timestamped filename for CSV export."""
    now = datetime.now()
    return f"openalex_export_{now.strftime('%Y-%m-%d_%H-%M')}.csv"


async def main_cli():
    """CLI interface for standalone usage."""
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python fetch_openalex.py 'https://openalex.org/works?...' [--quality]")
        print("  --quality: Include fast journal quality information (Q1-Q4 tiers)")
        sys.exit(1)
    
    url = sys.argv[1]
    include_quality = len(sys.argv) == 3 and sys.argv[2] == "--quality"
    
    if not url.startswith("https://openalex.org/"):
        print("Error: URL must start with 'https://openalex.org/'")
        sys.exit(1)
    
    print(f"Fetching OpenAlex works from: {url}")
    if include_quality:
        print("📊 Fast journal quality lookup enabled (using OpenAlex metrics)")
    
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
        df = await fetch_openalex_works(url, progress_callback, include_quality)
        
        if pbar is not None:
            pbar.close()
        
        if df.empty:
            print("No results found.")
            return
        
        filename = generate_filename()
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"\nSaved {len(df)} records to {filename}")
        
        if include_quality:
            quality_count = (df.get('journal_quality_tier', pd.Series()) != '').sum()
            print(f"📊 Found quality information for {quality_count} journals")
        
    except Exception as e:
        if pbar is not None:
            pbar.close()
        print(f"Error: {e}")
        sys.exit(1)


async def flatten_works_to_dataframe_with_quality(works: List[Dict], fetcher: OpenAlexFetcher = None) -> pd.DataFrame:
    """
    Convert raw OpenAlex works to a flattened DataFrame with fast journal quality information.
    
    Args:
        works: List of raw work dictionaries from OpenAlex API
        fetcher: OpenAlexFetcher instance for quality lookups
        
    Returns:
        pandas DataFrame with flattened, selected columns including journal quality metrics
    """
    if not works:
        return pd.DataFrame()
    
    flattened = []
    
    # Set up session for quality lookups if fetcher is provided
    if fetcher:
        headers = {
            'User-Agent': 'OpenAlex-Fetcher/1.0 (mailto:research@university.edu)',
            'Accept': 'application/json'
        }
        session = aiohttp.ClientSession(headers=headers)
        
        # STEP 1: Collect all unique journal source IDs to minimize API calls
        unique_sources = {}  # source_id -> source_id mapping (for deduplication)
        for work in works:
            primary_location = work.get("primary_location", {})
            source = primary_location.get("source", {}) if primary_location else {}
            source_id = source.get("id", "") if source else ""
            
            if source_id and source_id.startswith("https://openalex.org/S"):
                # Extract source ID (e.g., "S137773608" from full URL)
                clean_source_id = source_id.split("/")[-1]
                unique_sources[clean_source_id] = clean_source_id
        
        # STEP 2: Batch fetch quality data for all unique sources
        print(f"Fetching quality data for {len(unique_sources)} unique journals...")
        quality_lookup = {}  # source_id -> quality_data mapping
        
        for source_id in unique_sources.keys():
            quality_data = await fetcher.fetch_journal_quality_fast(session, source_id)
            quality_lookup[source_id] = quality_data
            
    else:
        session = None
        quality_lookup = {}
    
    try:
        # STEP 3: Process each work using the pre-fetched quality data
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
            
            # Get quality information from pre-fetched data
            quality_data = {"impact_factor_estimate": "", "h_index": "", "quality_tier": "", "total_works": ""}
            source_id = source.get("id", "") if source else ""
            if source_id and source_id.startswith("https://openalex.org/S"):
                clean_source_id = source_id.split("/")[-1]
                if clean_source_id in quality_lookup:
                    quality_data = quality_lookup[clean_source_id]
            
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
                "journal_quality_tier": quality_data["quality_tier"],
                "journal_impact_estimate": quality_data["impact_factor_estimate"],
                "journal_h_index": quality_data["h_index"],
                "journal_total_works": quality_data["total_works"],
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

