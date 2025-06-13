# OpenAlex Works Downloader

A simple web application that allows students and researchers to download complete OpenAlex datasets as CSV files without any coding required.

## 🎯 Features

- **Simple Interface**: Just paste any OpenAlex works URL and download complete results
- **No Coding Required**: Perfect for students and researchers without programming experience  
- **Fast & Scalable**: Handles up to 50+ concurrent users with async processing
- **Smart Caching**: Prevents duplicate API calls when multiple users request the same data
- **Real-time Progress**: Shows download progress with live updates
- **Lightning-Fast Journal Quality Metrics**: Get Q1-Q4 journal rankings using OpenAlex's own fast API
- **User-Friendly Output**: Clean CSV with readable abstracts and author names
- **⚡ Fast Journal Quality Metrics**: Optional Q1-Q4 journal rankings using OpenAlex's native quality metrics (lightning fast!)

## 🚀 Quick Start

### Online (Recommended)
Use the deployed version at: [Your Streamlit App URL]

### Local Installation
```bash
# Clone the repository
git clone [your-repo-url]
cd OpenAlex

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📖 How to Use

1. Go to [OpenAlex.org](https://openalex.org/works) and apply your research filters
2. Copy the URL from your browser address bar
3. Paste it into the app and click "Fetch Data"
4. Download your results as a CSV file

## 🔍 Example URLs

- All 2023 publications: `https://openalex.org/works?filter=publication_year:2023`
- Journal articles with abstracts: `https://openalex.org/works?filter=publication_year:2023,type:journal-article,has_abstract:true`
- Works from specific institution: `https://openalex.org/works?filter=institutions.id:I27837315`

## 📊 Output Columns

The downloaded CSV includes:
- **title**: Publication title
- **journal_name**: Journal name (if available)
- **journal_issn**: Journal ISSN numbers (semicolon-separated)
- **publication_year**: Year of publication
- **doi**: Digital Object Identifier (if available)
- **open_access.is_oa**: Whether the work is open access
- **cited_by_count**: Number of citations
- **authorships.author.display_name**: All authors (semicolon-separated)
- **abstract**: Full abstract text (when available)

### Optional Journal Quality Metrics (when enabled):
- **journal_quality_tier**: Q1, Q2, Q3, or Q4 ranking based on OpenAlex metrics
- **journal_impact_estimate**: 2-year mean citedness (impact factor estimate)
- **journal_h_index**: Journal h-index
- **journal_total_works**: Total number of works published by the journal

## 🛠 Command Line Usage

You can also use the tool from the command line:

```bash
# Basic usage (no journal quality metrics)
python fetch_openalex.py "https://openalex.org/works?filter=publication_year:2023&per-page=10"

# With fast journal quality metrics (Q1-Q4 rankings)
python fetch_openalex.py "https://openalex.org/works?filter=publication_year:2023&per-page=10" --quality
```

## 🏗 Technical Details

- **Built with**: Python, Streamlit, aiohttp, pandas
- **Rate Limiting**: 3 concurrent requests with 0.1s delays
- **Caching**: 1-hour TTL to reduce API load
- **Capacity**: Designed for ~50 concurrent users on modest hardware
- **⚡ Performance**: Lightning-fast journal quality lookups using OpenAlex's native Sources API (no external scraping!)
- **Rate Limiting**: 3 concurrent requests with 0.1s delays
- **Caching**: 1-hour TTL to reduce API load
- **Capacity**: Designed for ~50 concurrent users on modest hardware

## 📝 License

MIT License - feel free to use for academic and research purposes.

## 🤝 Contributing

Issues and pull requests welcome! This tool is designed to help the research community.
