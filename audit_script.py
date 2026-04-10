import pandas as pd
import numpy as np
from pathlib import Path
import random

BASE_PATH = Path(r'E:\大四上课程\毕业设计\数据\output\quant_training\data\raw')

print('=' * 80)
print('DATA QUALITY AUDIT REPORT')
print('=' * 80)

# 1. DAILY PRICES
print('\n1. DAILY PRICES QUALITY CHECK\n')
daily_dir = BASE_PATH / 'daily_prices'
daily_files = sorted(list(daily_dir.glob('*.parquet')))
print(f'Total daily price files: {len(daily_files)}')

sampled_daily = random.sample(daily_files, min(12, len(daily_files)))
print(f'Sampling {len(sampled_daily)} stocks\n')

for fp in sampled_daily:
    code = fp.stem
    df = pd.read_parquet(fp)
    row_count = len(df)
    file_size_kb = fp.stat().st_size / 1024
    
    date_col = 'trade_date' if 'trade_date' in df.columns else df.columns[0]
    date_vals = pd.to_datetime(df[date_col])
    date_range = f"{date_vals.min().date()} to {date_vals.max().date()}"
    dup_dates = df[date_col].duplicated().sum()
    date_sorted = date_vals.sort_values()
    date_diffs = date_sorted.diff().dt.days
    large_gaps = (date_diffs > 10).sum()
    
    nan_pct = (df.isna().sum() / len(df) * 100).to_dict()
    
    close_col = None
    for col in ['close', 'Close', 'CLOSE']:
        if col in df.columns:
            close_col = col
            break
    
    invalid_closes = 0
    if close_col:
        invalid_closes = ((df[close_col] <= 0) | (df[close_col].isna())).sum()
    
    print(f'Stock: {code}')
    print(f'  Size: {file_size_kb:.1f}KB | Rows: {row_count}')
    print(f'  Dates: {date_range}')
    
    top_nan = sorted(nan_pct.items(), key=lambda x: -x[1])[:3]
    nan_str = ', '.join([f'{c}:{p:.0f}%' for c, p in top_nan])
    print(f'  NaN cols: {nan_str}')
    
    if invalid_closes > 0:
        print(f'  WARNING: {invalid_closes} invalid closes')
    if dup_dates > 0:
        print(f'  WARNING: {dup_dates} duplicate dates')
    if large_gaps > 0:
        print(f'  WARNING: {large_gaps} gaps >10d')
    print()

# 2. BALANCE SHEET
print('\n2. BALANCE SHEET QUALITY CHECK\n')
bs_dir = BASE_PATH / 'balance_sheet'
bs_files = sorted(list(bs_dir.glob('*.parquet')))
print(f'Total balance sheet files: {len(bs_files)}')

sampled_bs = random.sample(bs_files, min(10, len(bs_files)))
print(f'Sampling {len(sampled_bs)} stocks\n')

for fp in sampled_bs:
    code = fp.stem
    df = pd.read_parquet(fp)
    row_count = len(df)
    file_size_kb = fp.stat().st_size / 1024
    
    nan_pct_key = {}
    zero_assets = 0
    for col in ['TOTAL_ASSETS', 'TOTAL_EQUITY', 'TOTAL_PARENT_EQUITY']:
        if col in df.columns:
            nan_pct_key[col] = df[col].isna().sum() / len(df) * 100
            if col == 'TOTAL_ASSETS':
                zero_assets = (df[col] == 0).sum()
    
    date_issues = []
    for date_col in ['NOTICE_DATE', 'REPORT_DATE']:
        if date_col not in df.columns:
            date_issues.append(date_col)
        elif df[date_col].isna().sum() > 0:
            date_issues.append(f'{date_col}(NaN)')
    
    print(f'Stock: {code}')
    print(f'  Size: {file_size_kb:.1f}KB | Rows: {row_count} | Cols: {len(df.columns)}')
    
    key_str = ', '.join([f'{c}:{p:.0f}%' for c, p in nan_pct_key.items()])
    print(f'  Key NaN: {key_str}')
    
    if date_issues:
        print(f'  Date issues: {date_issues}')
    if zero_assets > 0:
        print(f'  WARNING: {zero_assets} zero assets')
    print()

# 3. PROFIT SHEET
print('\n3. PROFIT SHEET QUALITY CHECK\n')
ps_dir = BASE_PATH / 'profit_sheet'
ps_files = sorted(list(ps_dir.glob('*.parquet')))
print(f'Total profit sheet files: {len(ps_files)}')

sampled_ps = random.sample(ps_files, min(10, len(ps_files)))
print(f'Sampling {len(sampled_ps)} stocks\n')

for fp in sampled_ps:
    code = fp.stem
    df = pd.read_parquet(fp)
    row_count = len(df)
    file_size_kb = fp.stat().st_size / 1024
    
    nan_pct_key = {}
    negative_rev = 0
    for col in ['OPERATE_INCOME', 'PARENT_NETPROFIT', 'NETPROFIT']:
        if col in df.columns:
            nan_pct_key[col] = df[col].isna().sum() / len(df) * 100
            if col == 'OPERATE_INCOME':
                negative_rev = (df[col] < 0).sum()
    
    print(f'Stock: {code}')
    print(f'  Size: {file_size_kb:.1f}KB | Rows: {row_count} | Cols: {len(df.columns)}')
    
    key_str = ', '.join([f'{c}:{p:.0f}%' for c, p in nan_pct_key.items()])
    print(f'  Key NaN: {key_str}')
    
    if negative_rev > 0:
        print(f'  WARNING: {negative_rev} negative revenue')
    print()

# 4. FINANCIAL INDICATORS
print('\n4. FINANCIAL INDICATORS QUALITY CHECK\n')
fi_dir = BASE_PATH / 'financial'
fi_files = sorted(list(fi_dir.glob('*.parquet')))
print(f'Total financial files: {len(fi_files)}')

sampled_fi = random.sample(fi_files, min(10, len(fi_files)))
print(f'Sampling {len(sampled_fi)} stocks\n')

for fp in sampled_fi:
    code = fp.stem
    df = pd.read_parquet(fp)
    row_count = len(df)
    col_count = len(df.columns)
    file_size_kb = fp.stat().st_size / 1024
    
    total_cells = len(df) * len(df.columns)
    nan_cells = df.isna().sum().sum()
    overall_nan_pct = nan_cells / total_cells * 100 if total_cells > 0 else 0
    
    nan_by_col = (df.isna().sum() / len(df) * 100)
    high_nan_cols = nan_by_col[nan_by_col > 50]
    
    print(f'Stock: {code}')
    print(f'  Size: {file_size_kb:.1f}KB | Rows: {row_count} | Cols: {col_count}')
    print(f'  Overall NaN: {overall_nan_pct:.1f}%')
    
    if len(high_nan_cols) > 0:
        print(f'  High NaN cols: {len(high_nan_cols)} (>50%)')
        for col, pct in list(high_nan_cols.items())[:3]:
            print(f'    {col}: {pct:.1f}%')
    print()

# 5. OVERALL STATISTICS
print('\n' + '=' * 80)
print('5. OVERALL STATISTICS')
print('=' * 80)
print('\nFile Sizes by Directory:\n')

for dir_name in ['daily_prices', 'balance_sheet', 'profit_sheet', 'financial']:
    dir_path = BASE_PATH / dir_name
    if dir_path.exists():
        files = list(dir_path.glob('*.parquet'))
        sizes = [f.stat().st_size / 1024 for f in files]
        if sizes:
            print(f'{dir_name}: {len(sizes)} files')
            print(f'  Mean: {np.mean(sizes):.1f} KB, Median: {np.median(sizes):.1f} KB')
            print(f'  Range: {np.min(sizes):.1f} - {np.max(sizes):.1f} KB')
            small = sum(1 for s in sizes if s < 1)
            if small > 0:
                print(f'  WARNING: {small} files <1KB')
            print()

print('\n' + '=' * 80)
print('AUDIT COMPLETE')
print('=' * 80)
