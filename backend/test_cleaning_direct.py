import sys
sys.path.insert(0, 'd:/Hackathon Projects/0 2025 Statathon/ver 6 mvp scratch/backend')

import pandas as pd
from services.cleaning_engine import CleaningEngine

# Load the file
df = pd.read_csv('d:/Hackathon Projects/0 2025 Statathon/ver 6 mvp scratch/backend/weatherAUS.csv')
print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

# Create engine and run cleaning
engine = CleaningEngine(df)
summary = engine.auto_clean()

print("\n" + "="*80)
print("CLEANING SUMMARY")
print("="*80)
print(f"Issues detected: {summary['issues_detected']}")
print(f"Issues fixed: {summary['issues_fixed']}")
print(f"Outliers detected: {summary['outliers_detected']}")

print("\n" + "="*80)
print(f"OUTLIERS DETAILS ({len(summary['outliers_details'])} columns)")
print("="*80)
for col, details in summary['outliers_details'].items():
    print(f"\n{col}:")
    print(f"  Method: {details['method']}")
    print(f"  Count: {details['count']}")
    print(f"  Percentage: {details['percentage']}%")
    print(f"  Treatment: {details['treatment']}")
