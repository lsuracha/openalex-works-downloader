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
import os


def normalize_issn(issn: str) -> str:
    """
    Normalize ISSN to a consistent format for lookup.
    Removes hyphens, spaces, and converts to uppercase.
    
    Args:
        issn: ISSN in any format (e.g., "1234-5678", "12345678")
    
    Returns:
        Normalized ISSN string (e.g., "12345678")
    """
    if not issn:
        return ""
    return issn.replace("-", "").replace(" ", "").strip().upper()


class LocalJournalLookup:
    """Local journal quartile lookup using journal_info.csv file."""
    
    def __init__(self, csv_path: str = "journal_info.csv"):
        self.csv_path = csv_path
        self.lookup_table = {}
        self.loaded = False
    
    def load_journal_data(self):
        """Load journal information from CSV file into memory."""
        if self.loaded:
            return
        
        if not os.path.exists(self.csv_path):
            print(f"Warning: Journal lookup file {self.csv_path} not found. Quartile lookup disabled.")
            self.loaded = True
            return
        
        try:
            # Read the CSV file
            df = pd.read_csv(self.csv_path)
            
            # Expected columns: Title, Issn, SJR, SJR Best Quartile, H index
            required_cols = ['Issn', 'SJR Best Quartile']
            if not all(col in df.columns for col in required_cols):
                print(f"Warning: Required columns {required_cols} not found in {self.csv_path}")
                self.loaded = True
                return
            
            # Build lookup table
            for _, row in df.iterrows():
                issn_str = str(row['Issn']).strip('"')  # Remove quotes
                if pd.isna(issn_str) or not issn_str:
                    continue
                
                # Split multiple ISSNs (comma or semicolon separated)
                issns = issn_str.replace(';', ',').split(',')
                
                quartile = str(row['SJR Best Quartile']).strip()
                sjr_score = str(row.get('SJR', '')).replace(',', '') if 'SJR' in row else ''
                h_index = str(row.get('H index', '')) if 'H index' in row else ''
                title = str(row.get('Title', '')) if 'Title' in row else ''
                
                # Create entry for each ISSN
                for issn in issns:
                    normalized_issn = normalize_issn(issn)
                    if normalized_issn:
                        self.lookup_table[normalized_issn] = {
                            'quartile': quartile,
                            'sjr_score': sjr_score,
                            'h_index': h_index,
                            'title': title
                        }
            
            print(f"📚 Loaded {len(self.lookup_table)} journal ISSN mappings from {self.csv_path}")
            self.loaded = True
            
        except Exception as e:
            print(f"Error loading journal data from {self.csv_path}: {e}")
            self.loaded = True
    
    def lookup_by_issn(self, issn_list: List[str]) -> Dict[str, str]:
        """
        Look up journal information by ISSN.
        
        Args:
            issn_list: List of ISSNs to search for
            
        Returns:
            Dictionary with quartile and other info, or empty values if not found
        """
        if not self.loaded:
            self.load_journal_data()
        
        if not issn_list or not self.lookup_table:
            return {"quartile": "", "sjr_score": "", "h_index": "", "category": ""}
        
        # Try each ISSN until we find a match
        for issn in issn_list:
            normalized = normalize_issn(issn)
            if normalized in self.lookup_table:
                entry = self.lookup_table[normalized]
                return {
                    "quartile": entry['quartile'],
                    "sjr_score": entry['sjr_score'],
                    "h_index": entry['h_index'],
                    "category": entry['title']  # Using title as category for now
                }
        
        # No match found
        return {"quartile": "", "sjr_score": "", "h_index": "", "category": ""}


class OpenAlexFetcher:
    """Async fetcher for OpenAlex works with rate limiting."""
    
    def __init__(self, max_concurrent: int = 3, delay: float = 0.1):
        self.max_concurrent = max_concurrent
        self.delay = delay
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.journal_lookup = LocalJournalLookup()  # Local journal lookup instead of cache
    
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
    
    async def fetch_all_pages(self, base_url: str, progress_callback=None, max_pages: int = None) -> List[Dict]:
        """
        Fetch all pages using page-based pagination (more reliable than cursor-based).
        
        Args:
            base_url: Initial OpenAlex works URL with filters
            progress_callback: Optional callback(page_num, total_records) for progress
            max_pages: Optional limit on number of pages to fetch (for testing/safety)
            
        Returns:
            List of all work records across all pages
        """
        all_works = []
        
        # Convert website URL to API URL if needed
        base_api_url = base_url.replace("https://openalex.org/", "https://api.openalex.org/")
        
        # Clean up any existing pagination parameters that users might have in their URLs
        import re
        # Remove existing page= and per-page= parameters
        base_api_url = re.sub(r'[&?]page=\d+', '', base_api_url)
        base_api_url = re.sub(r'[&?]per-page=\d+', '', base_api_url)
        
        # Ensure per-page is set to maximum (200)
        separator = "&" if "?" in base_api_url else "?"
        base_api_url += f"{separator}per-page=200"
        
        headers = {
            'User-Agent': 'OpenAlex-Fetcher/1.0 (mailto:research@university.edu)',
            'Accept': 'application/json'
        }
        
        async with aiohttp.ClientSession(headers=headers) as session:
            page_num = 0
            total_count = None
            
            while True:
                page_num += 1
                
                # Add page parameter to URL
                separator = "&" if "?" in base_api_url else "?"
                current_url = f"{base_api_url}{separator}page={page_num}"
                
                data = await self.fetch_page(session, current_url)
                
                if not data or "results" not in data:
                    break
                
                works = data["results"]
                
                # Get total count from first page
                if total_count is None and "meta" in data:
                    total_count = data["meta"].get("count", 0)
                    print(f"📊 Total available records: {total_count:,}")
                
                # Break if no more results
                if len(works) == 0:
                    break
                
                all_works.extend(works)
                
                if progress_callback:
                    progress_callback(page_num, len(all_works))
                
                # Safety check: respect max_pages limit if set
                if max_pages and page_num >= max_pages:
                    print(f"⚠️ Reached maximum page limit ({max_pages}). Use max_pages=None to fetch all pages.")
                    break
                
                # If we have fewer than 200 results, this is the last page
                if len(works) < 200:
                    break
        
        return all_works
    
    def fetch_journal_quartile_local(self, issn_list: List[str]) -> Dict[str, str]:
        """
        Fetch journal quartile information from local CSV file (much faster than external APIs).
        
        Args:
            issn_list: List of ISSNs to look up
            
        Returns:
            Dictionary with quartile metrics: {quartile, sjr_score, h_index, category}
        """
        return self.journal_lookup.lookup_by_issn(issn_list)

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
        # Note: This function is kept for backward compatibility but now uses OpenAlex lookup
        # The local quartile lookup is the preferred method
        
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
                        
                        await asyncio.sleep(self.delay)
                        return result
                        
        except Exception as e:
            print(f"Error fetching journal quality data for source {source_id}: {e}")
        
        # No data found
        result = {"impact_factor_estimate": "", "h_index": "", "quality_tier": "", "total_works": ""}
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


async def fetch_openalex_works(url: str, progress_callback=None, include_quality: bool = False, include_quartiles: bool = False, max_pages: int = None) -> pd.DataFrame:
    """
    Main async function to fetch and flatten OpenAlex works.
    
    Args:
        url: OpenAlex works URL with filters
        progress_callback: Optional callback for progress updates
        include_quality: Whether to fetch journal quality information (OpenAlex-based metrics)
        include_quartiles: Whether to fetch journal quartile information (local CSV lookup)
        max_pages: Optional limit on number of pages to fetch (default: None = all pages)
        
    Returns:
        pandas DataFrame with flattened works data
    """
    fetcher = OpenAlexFetcher()
    works = await fetcher.fetch_all_pages(url, progress_callback, max_pages=max_pages)
    
    if include_quartiles:
        return flatten_works_to_dataframe_with_quartiles(works, fetcher)
    elif include_quality:
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
        print("Usage: python fetch_openalex.py 'https://openalex.org/works?...' [--no-quartiles]")
        print("  --no-quartiles: Disable journal quartile information (quartiles enabled by default)")
        sys.exit(1)
    
    url = sys.argv[1]
    include_quartiles = not (len(sys.argv) == 3 and sys.argv[2] == "--no-quartiles")
    
    if not url.startswith("https://openalex.org/"):
        print("Error: URL must start with 'https://openalex.org/'")
        sys.exit(1)
    
    print(f"Fetching OpenAlex works from: {url}")
    if include_quartiles:
        print("📚 Journal quartile lookup enabled (using journal_info.csv)")
    else:
        print("📊 Basic export mode (no quartile information)")
    
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
        df = await fetch_openalex_works(url, progress_callback, include_quality=False, include_quartiles=include_quartiles)
        
        if pbar is not None:
            pbar.close()
        
        if df.empty:
            print("No results found.")
            return
        
        filename = generate_filename()
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"\nSaved {len(df)} records to {filename}")
        
        if include_quartiles:
            quartile_count = (df.get('journal_quartile', pd.Series()) != '').sum()
            print(f"📚 Found quartile information for {quartile_count} journals")
        
    except Exception as e:
        if pbar is not None:
            pbar.close()
        print(f"Error: {e}")
        sys.exit(1)


def flatten_works_to_dataframe_with_quartiles(works: List[Dict], fetcher: OpenAlexFetcher = None) -> pd.DataFrame:
    """
    Convert raw OpenAlex works to a flattened DataFrame with local journal quartile information.
    
    Args:
        works: List of raw work dictionaries from OpenAlex API
        fetcher: OpenAlexFetcher instance for quartile lookups
        
    Returns:
        pandas DataFrame with flattened, selected columns including journal quartile metrics
    """
    if not works:
        return pd.DataFrame()

    flattened = []
    
    # Load journal lookup data once
    if fetcher:
        fetcher.journal_lookup.load_journal_data()
    
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
        
        # Get quartile information from local lookup
        quartile_data = {"quartile": "", "sjr_score": "", "h_index": "", "category": ""}
        if fetcher and journal_issn_list:
            quartile_data = fetcher.fetch_journal_quartile_local(journal_issn_list)
        
        # Use journal_category from quartile data if available, otherwise use journal_name
        final_journal_name = quartile_data["category"] if quartile_data["category"] else journal_name
        
        # Extract open access info
        open_access = work.get("open_access", {})
        is_oa = open_access.get("is_oa", False) if open_access else False
        
        # Reconstruct abstract from inverted index
        abstract_text = ""
        if work.get("abstract_inverted_index"):
            abstract_text = reconstruct_abstract_from_inverted_index(work["abstract_inverted_index"])
        
        flattened_work = {
            "title": work.get("title", ""),
            "journal_name": final_journal_name,
            "journal_issn": journal_issn,
            "journal_quartile": quartile_data["quartile"],
            "sjr_score": quartile_data["sjr_score"],
            "journal_h_index": quartile_data["h_index"],
            "publication_year": work.get("publication_year"),
            "doi": work.get("doi", ""),
            "open_access.is_oa": is_oa,
            "cited_by_count": work.get("cited_by_count", 0),
            "authorships.author.display_name": authors_str,
            "abstract": abstract_text
        }
        
        flattened.append(flattened_work)
    
    return pd.DataFrame(flattened)


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

