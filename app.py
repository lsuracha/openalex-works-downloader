"""
OpenAlex Works Downloader - Streamlit Web Interface
Version 3.0 - Now with Fast Journal Quality Metrics

A simple web app that allows students to paste OpenAlex works URLs and download
the complete results as CSV files. Designed to handle up to ~50 concurrent users
on a modest server (2 vCPU, 4GB RAM) thanks to async I/O and Streamlit's
built-in session management.

Usage: streamlit run app.py
"""

import asyncio
from datetime import datetime
from io import StringIO
import streamlit as st
import pandas as pd
from fetch_openalex import fetch_openalex_works, generate_filename


def validate_openalex_url(url: str) -> bool:
    """Check if URL is a valid OpenAlex works URL."""
    return url.strip().startswith("https://openalex.org/") and len(url.strip()) > 25


@st.cache_data(show_spinner=False, ttl=3600)
def cached_fetch_openalex_works(url: str, include_quality: bool = False) -> pd.DataFrame:
    """Cached wrapper for async OpenAlex fetching."""
    return asyncio.run(fetch_openalex_works(url, include_quality=include_quality))


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="OpenAlex Works Downloader",
        page_icon="📚",
        layout="centered"
    )
    
    st.title("📚 OpenAlex Works Downloader")
    st.markdown("""
    Paste any OpenAlex works URL (with your filters) and download the complete dataset as CSV.
    Perfect for research projects, bibliometric analysis, and academic studies.
    """)
    
    # URL input
    url = st.text_input(
        "Paste your OpenAlex works URL (with filters):",
        placeholder="https://openalex.org/works?filter=publication_year:2023,type:journal-article",
        help="Example: https://openalex.org/works?filter=institutions.id:I27837315"
    )
    
    # Options section
    with st.expander("⚙️ Advanced Options", expanded=False):
        include_quality = st.checkbox(
            "Include Journal Quality Metrics (Q1-Q4)",
            value=False,
            help="Fetch journal quality metrics from OpenAlex including estimated impact factor, h-index, and quality tier (Q1-Q4). Uses OpenAlex's own fast API!"
        )
        
        if include_quality:
            st.info("⚡ **Fast journal quality lookup enabled**: Uses OpenAlex's own journal metrics including 2-year impact estimates, h-index, and calculated quality tiers. Lightning fast!")
    
    # Validate URL and enable/disable fetch button
    is_valid_url = validate_openalex_url(url)
    
    if url and not is_valid_url:
        st.error("❌ Please enter a valid OpenAlex URL starting with 'https://openalex.org/'")
    
    # Fetch button
    if st.button("🔍 Fetch Data", disabled=not is_valid_url, type="primary"):
        if not is_valid_url:
            st.error("Please enter a valid OpenAlex URL first.")
            return
        
        try:
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Progress callback for real-time updates
            progress_data = {"page": 0, "records": 0}
            
            def update_progress(page_num: int, total_records: int):
                progress_data["page"] = page_num
                progress_data["records"] = total_records
                
                # Update progress bar (estimate based on page number)
                # Since we don't know total pages, use a growing function
                progress_value = min(0.95, (page_num - 1) * 0.1)
                progress_bar.progress(progress_value)
                
                status_text.text(f"Page {page_num} — {total_records:,} records collected")
            
            # Show initial status
            status_text.text("Starting download...")
            
            # Fetch data with progress updates
            # Note: We can't pass the callback directly to cached function,
            # so we'll use a simpler approach for the Streamlit version
            if include_quality:
                status_text.text("Fetching OpenAlex data with journal quality metrics...")
                with st.spinner("Fetching OpenAlex data and journal quality metrics (fast lookup)..."):
                    df = cached_fetch_openalex_works(url, include_quality=True)
            else:
                with st.spinner("Fetching OpenAlex data..."):
                    df = cached_fetch_openalex_works(url, include_quality=False)
            
            # Complete progress
            progress_bar.progress(1.0)
            status_text.text(f"✅ Download complete! {len(df):,} records fetched.")
            
            if df.empty:
                st.warning("⚠️ No results found for this query. Try adjusting your filters.")
                return
            
            # Display summary
            st.success(f"🎉 Successfully fetched **{len(df):,} works** from OpenAlex!")
            
            # Show preview
            with st.expander("📋 Data Preview (first 5 rows)", expanded=True):
                st.dataframe(df.head(), use_container_width=True)
            
            # Data info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", f"{len(df):,}")
            with col2:
                unique_years = df['publication_year'].nunique()
                st.metric("Unique Years", unique_years)
            with col3:
                with_doi = df['doi'].notna().sum()
                st.metric("With DOI", f"{with_doi:,}")
            
            # Download button
            filename = generate_filename()
            
            # Convert DataFrame to CSV string
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False, encoding='utf-8')
            csv_string = csv_buffer.getvalue()
            
            st.download_button(
                label=f"📥 Download CSV ({len(df):,} rows)",
                data=csv_string.encode('utf-8'),
                file_name=filename,
                mime="text/csv",
                type="primary"
            )
            
            tip_message = "💡 **Tip**: The download includes all available data with columns like author names, "
            if include_quality:
                tip_message += "journal quality tiers (Q1-Q4), impact estimates, h-index, publication year, DOI, citation counts, and more!"
            else:
                tip_message += "publication year, DOI, citation counts, and more! Enable 'Journal Quality Metrics' option for Q1-Q4 rankings."
            
            st.info(tip_message)
            
        except Exception as e:
            st.error(f"❌ Error fetching data: {str(e)}")
            st.info("This might be due to network issues or an invalid URL. Please try again.")
    
    # Footer with usage info
    st.markdown("---")
    st.markdown("""
    **How to use:**
    1. Go to [OpenAlex.org](https://openalex.org/works) and apply your filters
    2. Copy the URL from your browser
    3. Paste it above and click "Fetch Data"
    4. Download your results as CSV
    
    **Supported filters:** Institution, author, venue, publication year, topic, and more!
    """)


if __name__ == "__main__":
    main()
