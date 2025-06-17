"""
Research Reference Downloader - Streamlit Web Interface
Version 4.0.1 - Local Journal Quartile Lookup (Force Redeploy)

A simple web app that allows students to paste OpenAlex works URLs and download
the complete results as CSV files with accurate journal quartile information.
Designed to handle up to ~50 concurrent users on a modest server (2 vCPU, 4GB RAM) 
thanks to async I/O and Streamlit's built-in session management.

Last Updated: June 13, 2025 - Deployed with simplified UI and local quartiles
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
def cached_fetch_openalex_works(url: str, max_pages: int = None) -> pd.DataFrame:
    """Cached wrapper for async OpenAlex fetching with quartile lookup."""
    return asyncio.run(fetch_openalex_works(url, include_quality=False, include_quartiles=True, max_pages=max_pages))


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Research Reference Downloader",
        page_icon="📚",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    # Display logo inline with title
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.markdown("<div style='margin-top: 20px;'>", unsafe_allow_html=True)
        try:
            st.image("stylor_academy_words.svg", width=150)
        except:
            pass  # If logo file not found, continue without it
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="display: flex; align-items: center; height: 100px;">
            <h2 style="margin: 0; font-size: 2.2rem; font-weight: 600;">Research Reference Downloader</h2>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("""
    Paste any OpenAlex works URL (with your filters) and download the complete dataset as CSV with **accurate journal quartile information (Q1-Q4)**.
    Perfect for research projects, bibliometric analysis, and academic studies.
    """)
    
    # URL input
    url = st.text_input(
        "Paste your OpenAlex works URL (with filters):",
        placeholder="https://openalex.org/works?filter=publication_year:2023,type:journal-article",
        help="Example: https://openalex.org/works?filter=institutions.id:I27837315"
    )
    
    # Validate URL and enable/disable fetch button
    is_valid_url = validate_openalex_url(url)
    
    if url and not is_valid_url:
        st.error("❌ Please enter a valid OpenAlex URL starting with 'https://openalex.org/'")
    
    # Info about journal quartiles
    st.info("📚 **Automatic journal quartile lookup enabled**: All downloads include accurate Q1-Q4 journal rankings based on SJR database with over 31,000 journals!")
    
    # Download size options
    with st.expander("⚙️ Download Options", expanded=False):
        download_option = st.radio(
            "Choose download size:",
            options=[
                ("Quick sample (1,000 records)", 5),
                ("Medium dataset (10,000 records)", 50), 
                ("Large dataset (50,000 records)", 250),
                ("Complete dataset (all available records)", None)
            ],
            format_func=lambda x: x[0],
            index=1,  # Default to medium
            help="Larger datasets take longer to process. Start with a sample to test your query."
        )
        max_pages = download_option[1]
        
        if max_pages is None:
            st.warning("⚠️ **Complete dataset**: This will download ALL available records, which could be millions! This may take a very long time and use significant bandwidth.")
        else:
            estimated_records = max_pages * 200
            st.info(f"📊 This will download approximately **{estimated_records:,} records** (up to {max_pages} pages of 200 records each).")
    
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
            
            # Fetch data with quartile information
            status_text.text("Fetching OpenAlex data with journal quartiles...")
            with st.spinner("Fetching OpenAlex data and journal quartiles (local lookup)..."):
                df = cached_fetch_openalex_works(url, max_pages=max_pages)
            
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
            tip_message += "journal quartiles (Q1-Q4), SJR scores, h-index, publication year, DOI, citation counts, and more!"
            
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
