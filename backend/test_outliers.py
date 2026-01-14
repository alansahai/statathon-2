import pandas as pd
import numpy as np
from scipy import stats

# Load the data
df = pd.read_csv('weatherAUS.csv')
print(f'Total rows: {len(df)}')
print(f'Total columns: {len(df.columns)}')

# Get numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(f'\nNumeric columns ({len(numeric_cols)}):')
for col in numeric_cols:
    print(f'  - {col}')

# Check for outliers in each numeric column
print('\n' + '='*100)
print('OUTLIER DETECTION SUMMARY')
print('='*100)

outlier_columns = []
for col in numeric_cols:
    non_null = df[col].dropna()
    if len(non_null) < 10:
        print(f'{col:25s} | SKIPPED: Insufficient data ({len(non_null)} values)')
        continue
    
    # Calculate skewness
    skewness = abs(stats.skew(non_null))
    
    if skewness < 1:  # Z-score method
        mean = non_null.mean()
        std = non_null.std()
        if std == 0:
            print(f'{col:25s} | SKIPPED: Zero standard deviation')
            continue
        z_scores = np.abs((df[col] - mean) / std)
        outlier_count = (z_scores > 3.0).sum()
        method = 'Z-score'
    else:  # IQR method
        Q1 = non_null.quantile(0.25)
        Q3 = non_null.quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0:
            print(f'{col:25s} | SKIPPED: Zero IQR')
            continue
        lower_fence = Q1 - 1.5 * IQR
        upper_fence = Q3 + 1.5 * IQR
        outlier_count = ((df[col] < lower_fence) | (df[col] > upper_fence)).sum()
        method = 'IQR'
    
    if outlier_count > 0:
        outlier_columns.append(col)
        pct = (outlier_count / len(non_null)) * 100
        print(f'✓ {col:25s} | {method:8s} | Outliers: {outlier_count:6d} ({pct:6.2f}%) | Skewness: {skewness:.3f}')
    else:
        print(f'  {col:25s} | {method:8s} | No outliers found      | Skewness: {skewness:.3f}')

print('\n' + '='*100)
print(f'TOTAL COLUMNS WITH OUTLIERS: {len(outlier_columns)} out of {len(numeric_cols)} numeric columns')
print('='*100)
if outlier_columns:
    print('\nColumns with outliers:')
    for col in outlier_columns:
        print(f'  • {col}')
