import pandas as pd

print("=== Scanning nurseCharting for delirium-related strings ===")
reader = pd.read_csv(
    'data/raw/eicu/2.0/nurseCharting.csv.gz',
    compression='gzip',
    usecols=['nursingchartcelltypecat',
             'nursingchartcelltypevallabel',
             'nursingchartcelltypevalname',
             'nursingchartvalue'],
    chunksize=3_000_000
)

cats = set()
valnames = set()
vallabels = set()

for chunk in reader:
    cat = chunk['nursingchartcelltypecat'].dropna().str.lower()
    vn = chunk['nursingchartcelltypevalname'].dropna().str.lower()
    vl = chunk['nursingchartcelltypevallabel'].dropna().str.lower()

    cats.update(cat.unique().tolist())

    mask = (vn.str.contains('delirium|score|assess|mental|confusion|agitat|icdsc', na=False) |
            vl.str.contains('delirium|score|assess|mental|confusion|agitat|icdsc', na=False))

    if mask.any():
        valnames.update(vn[mask].unique().tolist())
        vallabels.update(vl[mask].unique().tolist())

print("\nAll nursingchartcelltypecat values:")
for c in sorted(cats):
    print(f"  '{c}'")

print("\nRelevant nursingchartcelltypevalname values:")
for v in sorted(valnames):
    print(f"  '{v}'")

print("\nRelevant nursingchartcelltypevallabel values:")
for v in sorted(vallabels):
    print(f"  '{v}'")
