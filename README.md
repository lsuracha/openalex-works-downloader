# OpenAlex Works Downloader

A simple web application that allows students and researchers to download complete OpenAlex datasets as CSV files with accurate journal quartile information, no coding required.

## 🎯 Features

- **Simple Interface**: Just paste any OpenAlex works URL and download complete results
- **No Coding Required**: Perfect for students and researchers without programming experience  
- **Fast & Scalable**: Handles up to 50+ concurrent users with async processing
- **Smart Caching**: Prevents duplicate API calls when multiple users request the same data
- **Real-time Progress**: Shows download progress with live updates
- **📚 Accurate Journal Quartiles**: Built-in Q1-Q4 journal rankings using local SJR database (31,000+ journals)
- **⚡ Lightning Fast**: No external API dependencies for journal rankings - all lookups are local
- **User-Friendly Output**: Clean CSV with readable abstracts and author names

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
- **journal_name**: Journal name (preferring journal_category when available)
- **journal_issn**: Journal ISSN numbers (semicolon-separated)
- **publication_year**: Year of publication
- **doi**: Digital Object Identifier (if available)
- **open_access.is_oa**: Whether the work is open access
- **cited_by_count**: Number of citations
- **authorships.author.display_name**: All authors (semicolon-separated)
- **abstract**: Full abstract text (when available)

### Standard Journal Quartile Information (Always Included):
- **journal_quartile**: Q1, Q2, Q3, or Q4 ranking based on SJR (SCImago Journal Rank)
- **sjr_score**: SCImago Journal Rank score
- **journal_category**: Journal subject category/title from SJR database
- **journal_h_index**: Journal h-index from SJR database

## 🛠 Command Line Usage

You can also use the tool from the command line:

```bash
# Standard usage with journal quartiles (default behavior)
python fetch_openalex.py "https://openalex.org/works?filter=publication_year:2023&per-page=10"

# Without journal quartiles (faster for large datasets)
python fetch_openalex.py "https://openalex.org/works?filter=publication_year:2023&per-page=10" --no-quartiles

# With OpenAlex journal quality metrics (optional, in addition to quartiles)
python fetch_openalex.py "https://openalex.org/works?filter=publication_year:2023&per-page=10" --quality
```

## 🏗 Technical Details

- **Built with**: Python, Streamlit, aiohttp, pandas
- **Rate Limiting**: 3 concurrent requests with 0.1s delays
- **Caching**: 1-hour TTL to reduce API load
- **Capacity**: Designed for ~50 concurrent users on modest hardware
- **📚 Journal Quartiles**: Local CSV lookup with 31,000+ journals for instant SJR rankings
- **🚀 Performance**: No external API dependencies for quartile lookup - all data is local

## 📝 License

MIT License - feel free to use for academic and research purposes.

## 🤝 Contributing

Issues and pull requests welcome! This tool is designed to help the research community.
