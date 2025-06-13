# Journal Database Update Plan

## Overview
The OpenAlex Works Downloader uses a local CSV database (`journal_info.csv`) containing SJR journal quartile information for over 31,000 journals. This document outlines how to keep this database current.

## Current Database Status
- **File**: `journal_info.csv`
- **Last Updated**: Initial setup (June 2025)
- **Journals**: 31,000+ entries
- **Source**: SCImago Journal Rank (SJR) database
- **Format**: CSV with columns: `issn`, `journal_quartile`, `sjr_score`, `journal_category`, `journal_h_index`

## Update Schedule
**Recommended frequency**: Annually (SJR data is typically updated once per year)

## Update Process

### 1. Data Source
- **Primary**: SCImago Journal & Country Rank portal (https://www.scimagojr.com/)
- **Download**: Available as Excel/CSV export from their website
- **Format**: Contains ISSN, quartile rankings, SJR scores, h-index, and subject categories

### 2. Update Steps

#### Step 1: Download Latest SJR Data
1. Visit https://www.scimagojr.com/journalrank.php
2. Set filters to "All subject areas" and current year
3. Download the complete journal list as CSV/Excel
4. Save as `sjr_raw_[YEAR].csv`

#### Step 2: Data Processing
1. Clean and standardize ISSN format (remove hyphens, ensure 8 digits)
2. Map quartile information (Q1, Q2, Q3, Q4)
3. Extract relevant columns: ISSN, quartile, SJR score, category, h-index
4. Handle duplicate ISSNs (keep highest quartile entry)

#### Step 3: Validation
1. Compare journal count with previous version
2. Spot-check known journals (Nature, Science, etc.)
3. Verify ISSN format consistency
4. Test with sample OpenAlex queries

#### Step 4: Deployment
1. Backup current `journal_info.csv` as `journal_info_backup_[DATE].csv`
2. Replace with new file
3. Test the application with known journal queries
4. Update documentation with new journal count and date

### 3. Quality Assurance

#### Automated Checks
```python
# Example validation script
import pandas as pd

def validate_journal_db(filepath):
    df = pd.read_csv(filepath)
    
    # Check required columns
    required_cols = ['issn', 'journal_quartile', 'sjr_score', 'journal_category', 'journal_h_index']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
    
    # Check ISSN format (8 digits)
    invalid_issns = df[~df['issn'].str.match(r'^\d{8}$')]['issn'].head(10)
    if len(invalid_issns) > 0:
        print(f"Invalid ISSN formats: {invalid_issns.tolist()}")
    
    # Check quartile values
    valid_quartiles = {'Q1', 'Q2', 'Q3', 'Q4'}
    invalid_quartiles = df[~df['journal_quartile'].isin(valid_quartiles)]['journal_quartile'].unique()
    if len(invalid_quartiles) > 0:
        print(f"Invalid quartiles: {invalid_quartiles}")
    
    print(f"Total journals: {len(df):,}")
    print(f"Quartile distribution:")
    print(df['journal_quartile'].value_counts().sort_index())

# Run validation
validate_journal_db('journal_info.csv')
```

#### Manual Spot Checks
Test these known journals after each update:
- Nature (ISSN: 00280836) → Should be Q1
- Science (ISSN: 00368075) → Should be Q1
- PLOS ONE (ISSN: 19326203) → Should be Q1 or Q2
- Check 5-10 journals from your research domain

### 4. Version Control
- Commit new database file to git
- Tag release with date: `git tag journal-db-2026.01`
- Update changelog with journal count and major changes

## Emergency Updates
If critical journal information is missing or incorrect:

1. **Immediate Fix**: Manual edit to `journal_info.csv`
2. **Documentation**: Add note to this file about the manual change
3. **Next Scheduled Update**: Ensure manual changes are preserved

## Monitoring
- Monitor application logs for "Journal not found" messages
- Track user feedback about missing journal quartiles
- Keep list of commonly requested journals not in database

## Archive Strategy
- Keep last 3 versions of journal database
- Store in `archive/` folder with date stamps
- Remove older versions annually to save space

## Contact and Responsibility
- **Primary maintainer**: [Your name/team]
- **Update responsibility**: [Assign to specific person]
- **Notification list**: [Users to notify of updates]

---
*Last updated: June 13, 2025*
*Next scheduled update: January 2026*
