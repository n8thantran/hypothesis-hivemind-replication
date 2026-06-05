"""
CGM Analytical Toolkit - Implements all functions from Table 7 (Appendix) of the paper.
Three tiers: Data Processing, Daily Clinical Metrics, Long-term Aggregation & Analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Union
import json
import os
import warnings
warnings.filterwarnings('ignore')


# ==================== TIER 1: DATA PROCESSING & CGM WEARTIME ====================

def load_cgm_data(filepath: str) -> pd.DataFrame:
    """Load CGM data from CSV or Excel file, standardize columns."""
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath, parse_dates=['timestamp'])
        df = df.rename(columns={'timestamp': 'Date', 'glucose_value_mg_dl': 'CGM'})
    else:
        # Shanghai T2DM Excel format
        df = pd.read_excel(filepath)
        df = df.rename(columns={'Date': 'Date', 'CGM (mg / dl)': 'CGM'})
        df = df[['Date', 'CGM']].dropna(subset=['CGM'])
        df['Date'] = pd.to_datetime(df['Date'])
    
    df = df.sort_values('Date').reset_index(drop=True)
    return df


def merge_subject_files(filepaths: List[str]) -> pd.DataFrame:
    """Merge multiple CGM files for one subject (Shanghai T2DM has multiple files)."""
    dfs = []
    for fp in filepaths:
        dfs.append(load_cgm_data(fp))
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset='Date').sort_values('Date').reset_index(drop=True)
    return df


def estimate_cgm_sampling_rate(df: pd.DataFrame) -> int:
    """Infer the device sampling rate in minutes (e.g., 5 min vs. 15 min)."""
    if len(df) < 2:
        return 5  # default
    diffs = df['Date'].diff().dropna().dt.total_seconds() / 60
    # Use median to be robust to gaps
    median_diff = diffs.median()
    if median_diff < 10:
        return 5
    else:
        return 15


def filter_cgm_csv(df: pd.DataFrame, dates: Optional[List[str]] = None,
                   start_time: Optional[str] = None, end_time: Optional[str] = None,
                   start_datetime: Optional[str] = None, end_datetime: Optional[str] = None) -> pd.DataFrame:
    """
    Filter raw CGM data based on a user-specified date list or time window.
    
    Args:
        df: CGM dataframe with 'Date' and 'CGM' columns
        dates: List of date strings (e.g., ['2024-01-12', '2024-01-13'])
        start_time: Time-of-day filter start (e.g., '06:00')
        end_time: Time-of-day filter end (e.g., '12:00')
        start_datetime: Datetime range filter start
        end_datetime: Datetime range filter end
    """
    filtered = df.copy()
    
    if dates is not None and len(dates) > 0:
        date_objs = [pd.Timestamp(d).date() for d in dates]
        filtered = filtered[filtered['Date'].dt.date.isin(date_objs)]
    
    if start_datetime is not None and end_datetime is not None:
        start_dt = pd.Timestamp(start_datetime)
        end_dt = pd.Timestamp(end_datetime)
        filtered = filtered[(filtered['Date'] >= start_dt) & (filtered['Date'] <= end_dt)]
    
    if start_time is not None and end_time is not None:
        start_t = pd.Timestamp(start_time).time()
        end_t = pd.Timestamp(end_time).time()
        if start_t <= end_t:
            filtered = filtered[(filtered['Date'].dt.time >= start_t) & 
                              (filtered['Date'].dt.time <= end_t)]
        else:
            # Crosses midnight
            filtered = filtered[(filtered['Date'].dt.time >= start_t) | 
                              (filtered['Date'].dt.time <= end_t)]
    
    return filtered.sort_values('Date').reset_index(drop=True)


def find_adherence(df: pd.DataFrame, date: str, sampling_rate: int,
                   start_time: Optional[str] = None, end_time: Optional[str] = None) -> float:
    """
    Calculate the percentage of active wear time for a full day or specific time window.
    
    Returns: CGM weartime percentage (0-100)
    """
    date_obj = pd.Timestamp(date).date()
    day_data = df[df['Date'].dt.date == date_obj]
    
    if start_time is not None and end_time is not None:
        start_t = pd.Timestamp(start_time).time()
        end_t = pd.Timestamp(end_time).time()
        day_data = day_data[(day_data['Date'].dt.time >= start_t) & 
                           (day_data['Date'].dt.time <= end_t)]
        # Calculate expected readings for this window
        start_minutes = start_t.hour * 60 + start_t.minute
        end_minutes = end_t.hour * 60 + end_t.minute
        if end_minutes > start_minutes:
            window_minutes = end_minutes - start_minutes
        else:
            window_minutes = 1440 - start_minutes + end_minutes
        expected_readings = window_minutes / sampling_rate
    else:
        # Full day: 24 hours
        expected_readings = 1440 / sampling_rate  # 288 for 5-min, 96 for 15-min
    
    if expected_readings == 0:
        return 0.0
    
    actual_readings = len(day_data)
    weartime = min(100.0, (actual_readings / expected_readings) * 100)
    return round(weartime, 2)


# ==================== TIER 2: DAILY CLINICAL METRICS ====================

def find_BG_time_range(day_data: pd.DataFrame, low: float = 70.0, high: float = 180.0) -> Dict:
    """
    Compute TIR, TBR, TAR for a single day's data.
    Returns percentages and durations in minutes.
    """
    if len(day_data) == 0:
        return {'TIR': -1, 'TBR': -1, 'TAR': -1, 
                'TIR_minutes': -1, 'TBR_minutes': -1, 'TAR_minutes': -1}
    
    values = day_data['CGM'].values
    n = len(values)
    
    in_range = np.sum((values >= low) & (values <= high))
    below_range = np.sum(values < low)
    above_range = np.sum(values > high)
    
    sampling_rate = estimate_cgm_sampling_rate(day_data)
    
    tir = round((in_range / n) * 100, 2)
    tbr = round((below_range / n) * 100, 2)
    tar = round((above_range / n) * 100, 2)
    
    return {
        'TIR': tir,
        'TBR': tbr,
        'TAR': tar,
        'TIR_minutes': round(in_range * sampling_rate, 2),
        'TBR_minutes': round(below_range * sampling_rate, 2),
        'TAR_minutes': round(above_range * sampling_rate, 2)
    }


def find_avg_std_gv_BG(day_data: pd.DataFrame) -> Dict:
    """
    Calculate Mean BG, Standard Deviation, CV, estimated A1c, GMI.
    """
    if len(day_data) == 0:
        return {'mean_glucose': -1, 'std_glucose': -1, 'cv': -1, 'eA1c': -1, 'GMI': -1}
    
    values = day_data['CGM'].values
    mean_bg = np.mean(values)
    std_bg = np.std(values, ddof=1) if len(values) > 1 else 0.0
    cv = (std_bg / mean_bg * 100) if mean_bg > 0 else 0.0
    
    # GMI = 3.31 + 0.02392 × mean_glucose (mg/dL)
    gmi = 3.31 + 0.02392 * mean_bg
    
    # eA1c = (46.7 + mean_glucose) / 28.7
    ea1c = (46.7 + mean_bg) / 28.7
    
    return {
        'mean_glucose': round(mean_bg, 2),
        'std_glucose': round(std_bg, 2),
        'cv': round(cv, 2),
        'eA1c': round(ea1c, 2),
        'GMI': round(gmi, 2)
    }


def find_BG_min_max(day_data: pd.DataFrame) -> Dict:
    """Identify min and max glucose values for a day."""
    if len(day_data) == 0:
        return {'min_glucose': -1, 'max_glucose': -1, 
                'min_time': None, 'max_time': None}
    
    min_idx = day_data['CGM'].idxmin()
    max_idx = day_data['CGM'].idxmax()
    
    return {
        'min_glucose': round(day_data.loc[min_idx, 'CGM'], 2),
        'max_glucose': round(day_data.loc[max_idx, 'CGM'], 2),
        'min_time': str(day_data.loc[min_idx, 'Date']),
        'max_time': str(day_data.loc[max_idx, 'Date'])
    }


def find_hypo_hyper_events(day_data: pd.DataFrame, sampling_rate: int,
                           hypo_threshold: float = 70.0, hyper_threshold: float = 180.0,
                           min_duration_minutes: int = 15) -> Dict:
    """
    Count discrete hypo/hyper events.
    Hypo: <70 mg/dL for 15+ min
    Hyper: >180 mg/dL for 15+ min
    """
    if len(day_data) == 0:
        return {'hypo_events': -1, 'hyper_events': -1}
    
    min_readings = max(1, min_duration_minutes // sampling_rate)
    
    def count_events(values, condition_fn):
        events = 0
        consecutive = 0
        for v in values:
            if condition_fn(v):
                consecutive += 1
            else:
                if consecutive >= min_readings:
                    events += 1
                consecutive = 0
        # Check last streak
        if consecutive >= min_readings:
            events += 1
        return events
    
    values = day_data['CGM'].values
    hypo = count_events(values, lambda v: v < hypo_threshold)
    hyper = count_events(values, lambda v: v > hyper_threshold)
    
    return {
        'hypo_events': hypo,
        'hyper_events': hyper
    }


def extract_features_json(df: pd.DataFrame, dates: List[str], sampling_rate: int) -> Dict:
    """
    Pipeline wrapper: execute all daily metric functions and aggregate results.
    Returns a date-keyed dictionary of all features.
    """
    results = {}
    for date_str in dates:
        date_obj = pd.Timestamp(date_str).date()
        day_data = df[df['Date'].dt.date == date_obj].copy()
        
        weartime = find_adherence(df, date_str, sampling_rate)
        time_range = find_BG_time_range(day_data)
        avg_std = find_avg_std_gv_BG(day_data)
        min_max = find_BG_min_max(day_data)
        events = find_hypo_hyper_events(day_data, sampling_rate)
        
        results[date_str] = {
            'cgm_weartime': weartime,
            **time_range,
            **avg_std,
            **min_max,
            **events
        }
    
    return results


# ==================== TIER 3: LONG-TERM AGGREGATION & ANALYSIS ====================

def get_average(features_dict: Dict, feature_name: str, weartime_threshold: float = 70.0) -> Dict:
    """
    Compute average of a feature across multiple days.
    Returns two values: one for all days, one for days with good weartime (≥70%).
    """
    all_values = []
    good_weartime_values = []
    
    for date_key, features in features_dict.items():
        val = features.get(feature_name, -1)
        weartime = features.get('cgm_weartime', 0)
        
        if val != -1:
            all_values.append(val)
            if weartime >= weartime_threshold:
                good_weartime_values.append(val)
    
    result = {
        'days_all': len(all_values),
        f'avg_{feature_name}_all': round(np.mean(all_values), 2) if all_values else -1,
        'days_sufficient_weartime': len(good_weartime_values),
        f'avg_{feature_name}_sufficient_weartime': round(np.mean(good_weartime_values), 2) if good_weartime_values else -1
    }
    
    return result


def count_satisfied_condition(features_dict: Dict, feature_name: str, 
                              operator: str, threshold: float) -> Dict:
    """Count how many days meet a specific criterion."""
    count = 0
    total = 0
    
    for date_key, features in features_dict.items():
        val = features.get(feature_name, -1)
        if val == -1:
            continue
        total += 1
        
        if operator == '>' and val > threshold:
            count += 1
        elif operator == '>=' and val >= threshold:
            count += 1
        elif operator == '<' and val < threshold:
            count += 1
        elif operator == '<=' and val <= threshold:
            count += 1
        elif operator == '==' and val == threshold:
            count += 1
    
    return {
        'count': count,
        'total': total,
        'percentage': round(count / total * 100, 2) if total > 0 else 0
    }


def feature_range(features_dict: Dict, feature_name: str) -> Dict:
    """Find the global min and max of a feature across multiple days."""
    min_val = float('inf')
    max_val = float('-inf')
    min_date = None
    max_date = None
    
    for date_key, features in features_dict.items():
        val = features.get(feature_name, -1)
        if val == -1:
            continue
        if val < min_val:
            min_val = val
            min_date = date_key
        if val > max_val:
            max_val = val
            max_date = date_key
    
    if min_val == float('inf'):
        return {'min_value': -1, 'max_value': -1, 'min_date': None, 'max_date': None}
    
    return {
        'min_value': round(min_val, 2),
        'max_value': round(max_val, 2),
        'min_date': min_date,
        'max_date': max_date
    }


def compute_difference_ratio(features_dict_a: Dict, features_dict_b: Dict,
                             feature_name: str) -> Dict:
    """
    Compare two time periods: compute the difference and ratio for a given feature.
    """
    avg_a = get_average(features_dict_a, feature_name)
    avg_b = get_average(features_dict_b, feature_name)
    
    val_a = avg_a.get(f'avg_{feature_name}_sufficient_weartime', -1)
    val_b = avg_b.get(f'avg_{feature_name}_sufficient_weartime', -1)
    
    if val_a == -1 or val_b == -1:
        # Fall back to all data
        val_a = avg_a.get(f'avg_{feature_name}_all', -1)
        val_b = avg_b.get(f'avg_{feature_name}_all', -1)
    
    if val_a == -1 or val_b == -1:
        return {'difference': -1, 'ratio': -1, 'higher_group': 'unknown'}

    diff = round(val_a - val_b, 2)
    ratio = round(val_a / val_b, 4) if val_b != 0 else float('inf')
    higher = 'A' if val_a > val_b else ('B' if val_b > val_a else 'equal')
    
    return {
        'group_A_avg': round(val_a, 2),
        'group_B_avg': round(val_b, 2),
        'difference': diff,
        'ratio': ratio,
        'higher_group': higher
    }


def calculate_blood_glucose_excursion(df: pd.DataFrame, dates: List[str],
                                       rate_threshold: float = 2.0,
                                       sampling_rate: int = 5) -> Dict:
    """
    Detect rapid glycemic excursions (spikes/drops).
    rate_threshold: mg/dL per minute
    """
    results = {}
    for date_str in dates:
        date_obj = pd.Timestamp(date_str).date()
        day_data = df[df['Date'].dt.date == date_obj].copy()
        
        if len(day_data) < 2:
            results[date_str] = []
            continue
        
        excursions = []
        values = day_data['CGM'].values
        times = day_data['Date'].values
        
        for i in range(1, len(values)):
            diff = values[i] - values[i-1]
            dt_minutes = (pd.Timestamp(times[i]) - pd.Timestamp(times[i-1])).total_seconds() / 60
            if dt_minutes > 0:
                rate = diff / dt_minutes
                if abs(rate) >= rate_threshold:
                    excursions.append({
                        'start_time': str(pd.Timestamp(times[i-1])),
                        'end_time': str(pd.Timestamp(times[i])),
                        'magnitude': round(abs(diff), 2),
                        'speed': round(abs(rate), 2),
                        'direction': 'rise' if diff > 0 else 'fall'
                    })
        
        results[date_str] = excursions
    
    return results


def plot_daily_trends(df: pd.DataFrame, dates: List[str], 
                      output_path: Optional[str] = None) -> Dict:
    """
    Generate 24-hour aggregate plot (Average Daily Profile).
    Returns mean values per time slot.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    all_data = []
    for date_str in dates:
        date_obj = pd.Timestamp(date_str).date()
        day_data = df[df['Date'].dt.date == date_obj].copy()
        if len(day_data) > 0:
            day_data['time_minutes'] = day_data['Date'].dt.hour * 60 + day_data['Date'].dt.minute
            all_data.append(day_data[['time_minutes', 'CGM']])
    
    if not all_data:
        return {'mean_values': {}, 'std_values': {}}
    
    combined = pd.concat(all_data)
    sampling_rate = estimate_cgm_sampling_rate(df)
    bins = list(range(0, 1441, sampling_rate))
    combined['bin'] = pd.cut(combined['time_minutes'], bins=bins, labels=bins[:-1])
    
    grouped = combined.groupby('bin')['CGM']
    means = grouped.mean()
    stds = grouped.std().fillna(0)
    
    if output_path:
        fig, ax = plt.subplots(figsize=(10, 5))
        hours = [b / 60 for b in means.index.astype(float)]
        ax.plot(hours, means.values, 'g-', linewidth=2, label='Mean Glucose')
        ax.fill_between(hours, means.values - stds.values, means.values + stds.values,
                       alpha=0.2, color='green', label='±1 SD')
        ax.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Low (70 mg/dL)')
        ax.axhline(y=180, color='orange', linestyle='--', alpha=0.5, label='High (180 mg/dL)')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Glucose (mg/dL)')
        ax.set_title(f'Average Daily Glucose Profile ({len(dates)} days)')
        ax.legend()
        ax.set_xlim(0, 24)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
    
    return {
        'mean_values': {str(k): round(v, 2) for k, v in means.items()},
        'num_days': len(dates)
    }


# ==================== SUBJECT DATA MANAGEMENT ====================

class SubjectData:
    """Manages CGM data for a single subject."""
    
    def __init__(self, subject_id: str, df: pd.DataFrame, dataset: str):
        self.subject_id = subject_id
        self.df = df
        self.dataset = dataset
        self.sampling_rate = estimate_cgm_sampling_rate(df)
        self.dates = sorted(df['Date'].dt.date.unique())
        self.date_strings = [str(d) for d in self.dates]
        self.start_date = self.dates[0] if self.dates else None
        self.end_date = self.dates[-1] if self.dates else None
        
        # Pre-compute features for all dates
        self._features_cache = None
    
    def get_features(self, dates: Optional[List[str]] = None) -> Dict:
        """Get features for specified dates (or all dates)."""
        if dates is None:
            dates = self.date_strings
        return extract_features_json(self.df, dates, self.sampling_rate)
    
    def get_all_features(self) -> Dict:
        """Get cached features for all dates."""
        if self._features_cache is None:
            self._features_cache = self.get_features()
        return self._features_cache
    
    def get_tir(self) -> float:
        """Get overall TIR for this subject (for correlation analysis)."""
        features = self.get_all_features()
        avg = get_average(features, 'TIR')
        val = avg.get('avg_TIR_sufficient_weartime', -1)
        if val == -1:
            val = avg.get('avg_TIR_all', -1)
        return val
    
    def summary(self) -> Dict:
        """Get subject summary stats."""
        return {
            'subject_id': self.subject_id,
            'dataset': self.dataset,
            'sampling_rate': self.sampling_rate,
            'num_days': len(self.dates),
            'start_date': str(self.start_date),
            'end_date': str(self.end_date),
            'total_readings': len(self.df),
            'TIR': self.get_tir()
        }


def load_all_subjects(azt1d_dir: str, shanghai_dir: str) -> Dict[str, SubjectData]:
    """Load all 19 subjects as per the paper's mapping (Table 10)."""
    subjects = {}
    
    # AZT1D mapping: P1-P11
    azt1d_mapping = {
        'P1': 'Subject 15', 'P2': 'Subject 23', 'P3': 'Subject 21',
        'P4': 'Subject 20', 'P5': 'Subject 7', 'P6': 'Subject 19',
        'P7': 'Subject 5', 'P8': 'Subject 13', 'P9': 'Subject 6',
        'P10': 'Subject 11', 'P11': 'Subject 4'
    }
    
    for pid, orig_id in azt1d_mapping.items():
        filepath = os.path.join(azt1d_dir, f"{orig_id}.csv")
        if os.path.exists(filepath):
            df = load_cgm_data(filepath)
            subjects[pid] = SubjectData(pid, df, 'AZT1D')
        else:
            print(f"Warning: {filepath} not found for {pid}")
    
    # ShanghaiT2DM mapping: P12-P19
    shanghai_mapping = {
        'P12': '2069', 'P13': '2014', 'P14': '2017', 'P15': '2015',
        'P16': '2078', 'P17': '2001', 'P18': '2055', 'P19': '2074'
    }
    
    import glob
    for pid, orig_id in shanghai_mapping.items():
        pattern = os.path.join(shanghai_dir, f"{orig_id}_*.*")
        files = sorted(glob.glob(pattern))
        if files:
            df = merge_subject_files(files)
            subjects[pid] = SubjectData(pid, df, 'ShanghaiT2DM')
        else:
            print(f"Warning: No files found for {pid} (pattern: {pattern})")
    
    return subjects


if __name__ == '__main__':
    # Quick test of the toolkit
    import sys
    
    # Test with a small synthetic dataset
    dates = pd.date_range('2024-01-01', periods=288*3, freq='5min')  # 3 days
    np.random.seed(42)
    glucose = np.random.normal(140, 30, len(dates))
    glucose = np.clip(glucose, 40, 400)
    
    df = pd.DataFrame({'Date': dates, 'CGM': glucose})
    
    print("=== Testing CGM Toolkit ===")
    print(f"Sampling rate: {estimate_cgm_sampling_rate(df)} min")
    
    test_dates = ['2024-01-01', '2024-01-02', '2024-01-03']
    features = extract_features_json(df, test_dates, 5)
    
    for d, f in features.items():
        print(f"\n{d}:")
        for k, v in f.items():
            print(f"  {k}: {v}")
    
    # Test aggregation
    avg_tir = get_average(features, 'TIR')
    print(f"\nAverage TIR: {avg_tir}")
    
    # Test comparison
    f1 = extract_features_json(df, ['2024-01-01'], 5)
    f2 = extract_features_json(df, ['2024-01-02'], 5)
    diff = compute_difference_ratio(f1, f2, 'TIR')
    print(f"\nDifference: {diff}")
    
    # Test excursion
    exc = calculate_blood_glucose_excursion(df, ['2024-01-01'], rate_threshold=2.0)
    print(f"\nExcursions on 2024-01-01: {len(exc.get('2024-01-01', []))} events")
    
    print("\n=== All tests passed! ===")
