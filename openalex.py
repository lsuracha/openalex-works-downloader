# requirements: pandas ≥1.3 (or polars), python ≥3.8
import gzip, json, glob, pandas as pd, pathlib

SRC_DIR  = pathlib.Path("/Users/yor/Downloads/concepts_snapshot")         # wherever you ran `aws s3 sync`
OUT_CSV  = pathlib.Path(__file__).parent / "openalex_concepts.csv"  # save in same folder as script
FIELDS   = ["id", "display_name", "level",
            "wikidata", "description"]              # pick whatever fields you need

rows = []
# glob every .gz in every date folder
for gz in glob.glob(str(SRC_DIR / "**/*.gz"), recursive=True):
    with gzip.open(gz, "rt", encoding="utf-8") as fh:
        for line in fh:
            doc = json.loads(line)
            rows.append({k: doc.get(k) for k in FIELDS})

df = pd.DataFrame(rows)
print(df.head())                       # sanity-check
df.to_csv(OUT_CSV, index=False)        # or: df.to_parquet("concepts.parquet")