"""
Generate synthetic QA pairs for CGM-Agent evaluation.
Produces 2,470 synthetic + 1,710 user-derived questions = 4,180 total.
Ground truth is computed deterministically using the CGM toolkit.
"""

import json
import re
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from load_subjects import main as load_subjects
from cgm_toolkit import (
    filter_cgm_csv, estimate_cgm_sampling_rate, find_adherence,
    find_BG_time_range, find_avg_std_gv_BG, find_BG_min_max,
    find_hypo_hyper_events, extract_features_json,
    get_average, count_satisfied_condition, feature_range,
    compute_difference_ratio, calculate_blood_glucose_excursion,
    plot_daily_trends
)

random.seed(42)
np.random.seed(42)

def get_valid_dates(df, min_readings=10):
    """Get dates with sufficient data."""
    df['date'] = pd.to_datetime(df['Date']).dt.date
    date_counts = df.groupby('date').size()
    valid = date_counts[date_counts >= min_readings].index.tolist()
    return sorted(valid)

def date_to_str(d):
    if isinstance(d, str):
        return d
    return d.strftime('%Y-%m-%d')

def compute_ground_truth(df, dates, features, sampling_rate, query_type, params=None):
    """Compute ground truth for a query using the toolkit."""
    gt = {}
    dates_str = [date_to_str(d) for d in dates]
    
    try:
        if query_type == 'basic_retrieval':
            # Single day features
            for d in dates_str:
                filtered = filter_cgm_csv(df, dates=[d])
                if filtered.empty:
                    continue
                day_features = extract_features_json(filtered, sampling_rate)
                gt[d] = {}
                for feat in features:
                    if feat in day_features.get(d, {}):
                        gt[d][feat] = day_features[d][feat]
                    elif d in day_features:
                        # Try to find feature in nested structure
                        for k, v in day_features[d].items():
                            if feat.lower() in k.lower():
                                gt[d][feat] = v
                                break
                                
        elif query_type == 'multi_day_average':
            # Average across days
            filtered = filter_cgm_csv(df, dates=dates_str)
            if filtered.empty:
                return gt
            all_features = extract_features_json(filtered, sampling_rate)
            for feat in features:
                avg_result = get_average(all_features, feat)
                key = f"({dates_str[0]}, {dates_str[-1]})"
                gt[key] = avg_result
                
        elif query_type == 'conditional_count':
            filtered = filter_cgm_csv(df, dates=dates_str)
            if filtered.empty:
                return gt
            all_features = extract_features_json(filtered, sampling_rate)
            condition = params.get('condition', 'hypo_events == 0')
            # Parse condition string: "feature_name operator threshold"
            cond_match = re.match(r'(\w+)\s*(>=|<=|==|>|<)\s*([\d.]+)', condition)
            if cond_match:
                feat_name, op, thresh = cond_match.groups()
                count_result = count_satisfied_condition(all_features, feat_name, op, float(thresh))
            else:
                return gt
            key = f"({dates_str[0]}, {dates_str[-1]})"
            gt[key] = count_result
            
        elif query_type == 'feature_range':
            filtered = filter_cgm_csv(df, dates=dates_str)
            if filtered.empty:
                return gt
            all_features = extract_features_json(filtered, sampling_rate)
            feat = features[0]
            range_result = feature_range(all_features, feat)
            key = f"({dates_str[0]}, {dates_str[-1]})"
            gt[key] = range_result
            
        elif query_type == 'period_comparison':
            dates_a = params['dates_a']
            dates_b = params['dates_b']
            filtered_a = filter_cgm_csv(df, dates=[date_to_str(d) for d in dates_a])
            filtered_b = filter_cgm_csv(df, dates=[date_to_str(d) for d in dates_b])
            if filtered_a.empty or filtered_b.empty:
                return gt
            features_a = extract_features_json(filtered_a, sampling_rate)
            features_b = extract_features_json(filtered_b, sampling_rate)
            feat = features[0]
            diff_result = compute_difference_ratio(features_a, features_b, feat)
            key = "comparison"
            gt[key] = diff_result
            
        elif query_type == 'time_window':
            time_start = params.get('time_start', '06:00')
            time_end = params.get('time_end', '12:00')
            filtered = filter_cgm_csv(df, dates=dates_str, 
                                       start_time=time_start, end_time=time_end)
            if filtered.empty:
                return gt
            day_features = extract_features_json(filtered, sampling_rate)
            for d in dates_str:
                if d in day_features:
                    gt[d] = {}
                    for feat in features:
                        if feat in day_features[d]:
                            gt[d][feat] = day_features[d][feat]
                            
        elif query_type == 'excursion':
            for d in dates_str:
                filtered = filter_cgm_csv(df, dates=[d])
                if filtered.empty:
                    continue
                exc = calculate_blood_glucose_excursion(filtered)
                gt[d] = exc
                
        elif query_type == 'trend':
            filtered = filter_cgm_csv(df, dates=dates_str)
            if filtered.empty:
                return gt
            # For trends, ground truth is the mean values per hour
            trend_data = {}
            filtered['hour'] = pd.to_datetime(filtered['Date']).dt.hour
            for h in range(24):
                hour_data = filtered[filtered['hour'] == h]['glucose']
                if len(hour_data) > 0:
                    trend_data[f"hour_{h:02d}"] = round(float(hour_data.mean()), 1)
            key = f"({dates_str[0]}, {dates_str[-1]})"
            gt[key] = trend_data
            
    except Exception as e:
        gt['error'] = str(e)
    
    return gt


def generate_synthetic_questions(subjects, target_per_subject=130):
    """Generate ~130 synthetic questions per subject = ~2470 total."""
    all_questions = []
    
    feature_names = {
        'TIR': 'Time in Range',
        'TBR': 'Time Below Range', 
        'TAR': 'Time Above Range',
        'mean_glucose': 'average blood glucose',
        'std_glucose': 'standard deviation of blood glucose',
        'cv': 'glycemic variability (CV)',
        'eA1c': 'estimated A1c',
        'eGMI': 'estimated Glucose Management Indicator',
        'min_glucose': 'minimum blood glucose',
        'max_glucose': 'maximum blood glucose',
        'cgm_weartime': 'CGM weartime',
        'hypo_events': 'hypoglycemia events',
        'hyper_events': 'hyperglycemia events',
    }
    
    time_windows = [
        ('06:00', '12:00', 'morning'),
        ('12:00', '17:00', 'afternoon'),
        ('17:00', '21:00', 'evening'),
        ('21:00', '06:00', 'overnight'),
        ('00:00', '06:00', 'early morning'),
    ]
    
    conditions = [
        ('hypo_events == 0', 'no hypoglycemia events'),
        ('hyper_events == 0', 'no hyperglycemia events'),
        ('TIR > 70', 'TIR above 70%'),
        ('TIR > 80', 'TIR above 80%'),
        ('cgm_weartime > 70', 'good CGM weartime'),
        ('mean_glucose < 140', 'average glucose below 140'),
        ('mean_glucose < 180', 'average glucose below 180'),
    ]
    
    templates = {
        'basic_retrieval': [
            "What was my {feature_name} on {date}?",
            "Show me my {feature_name} for {date}.",
            "What are my {feature_name} and CGM weartime on {date}?",
        ],
        'multi_day_average': [
            "What is my average {feature_name} from {date_start} to {date_end}?",
            "What are my average {feature_name} over {date_start} to {date_end}? Consider two conditions: 1. Days with any CGM records. 2. Days with good weartime (>70%).",
        ],
        'conditional_count': [
            "How many days had {condition_desc} from {date_start} to {date_end}?",
            "Count the days with {condition_desc} between {date_start} and {date_end}.",
        ],
        'feature_range': [
            "What was my highest {feature_name} from {date_start} to {date_end}?",
            "Which day had the lowest {feature_name} between {date_start} and {date_end}?",
        ],
        'period_comparison': [
            "Compare my {feature_name} between {period_a} and {period_b}.",
            "What's the difference in {feature_name} between {period_a} and {period_b}?",
        ],
        'time_window': [
            "What was my {feature_name} during the {time_desc} on {date}?",
            "Show me my {feature_name} between {time_start} and {time_end} on {date}.",
        ],
        'excursion': [
            "Were there any rapid glucose changes on {date}?",
            "Detect glucose excursions on {date}.",
        ],
        'trend': [
            "Plot my typical daily CGM blood glucose trends for {date_start} to {date_end}.",
            "What are my daily glucose patterns from {date_start} to {date_end}?",
        ],
    }
    
    for subj_id, subj_data in subjects.items():
        df = subj_data.df
        sampling_rate = subj_data.sampling_rate
        valid_dates = get_valid_dates(df)
        
        if len(valid_dates) < 7:
            print(f"  Skipping {subj_id}: only {len(valid_dates)} valid dates")
            continue
            
        questions_for_subject = []
        
        # 1. Basic retrieval (~20 per subject)
        for _ in range(20):
            date = random.choice(valid_dates)
            feat_key = random.choice(list(feature_names.keys()))
            feat_name = feature_names[feat_key]
            template = random.choice(templates['basic_retrieval'])
            question = template.format(feature_name=feat_name, date=date_to_str(date))
            
            gt = compute_ground_truth(df, [date], [feat_key], sampling_rate, 'basic_retrieval')
            
            questions_for_subject.append({
                'subject_id': subj_id,
                'question': question,
                'query_type': 'basic_retrieval',
                'category': 'synthetic',
                'ground_truth': gt,
                'features': [feat_key],
                'dates': [date_to_str(date)],
                'function_calls': ['filter_cgm_csv', 'extract_features_json'],
            })
        
        # 2. Multi-day averages (~20 per subject)
        for _ in range(20):
            n_days = random.choice([3, 5, 7, 14])
            start_idx = random.randint(0, max(0, len(valid_dates) - n_days))
            dates = valid_dates[start_idx:start_idx + n_days]
            feat_key = random.choice(['TIR', 'TBR', 'TAR', 'mean_glucose', 'std_glucose', 'cv', 'eA1c'])
            feat_name = feature_names[feat_key]
            template = random.choice(templates['multi_day_average'])
            question = template.format(
                feature_name=feat_name,
                date_start=date_to_str(dates[0]),
                date_end=date_to_str(dates[-1])
            )
            
            gt = compute_ground_truth(df, dates, [feat_key], sampling_rate, 'multi_day_average')
            
            questions_for_subject.append({
                'subject_id': subj_id,
                'question': question,
                'query_type': 'multi_day_average',
                'category': 'synthetic',
                'ground_truth': gt,
                'features': [feat_key],
                'dates': [date_to_str(d) for d in dates],
                'function_calls': ['filter_cgm_csv', 'extract_features_json', 'get_average'],
            })
        
        # 3. Conditional counting (~15 per subject)
        for _ in range(15):
            n_days = random.choice([7, 14, 21])
            start_idx = random.randint(0, max(0, len(valid_dates) - n_days))
            dates = valid_dates[start_idx:start_idx + n_days]
            cond, cond_desc = random.choice(conditions)
            template = random.choice(templates['conditional_count'])
            question = template.format(
                condition_desc=cond_desc,
                date_start=date_to_str(dates[0]),
                date_end=date_to_str(dates[-1])
            )
            
            gt = compute_ground_truth(df, dates, [], sampling_rate, 'conditional_count',
                                     params={'condition': cond})
            
            questions_for_subject.append({
                'subject_id': subj_id,
                'question': question,
                'query_type': 'conditional_count',
                'category': 'synthetic',
                'ground_truth': gt,
                'features': [],
                'dates': [date_to_str(d) for d in dates],
                'function_calls': ['filter_cgm_csv', 'extract_features_json', 'count_satisfied_condition'],
            })
        
        # 4. Feature range (~15 per subject)
        for _ in range(15):
            n_days = random.choice([7, 14, 21])
            start_idx = random.randint(0, max(0, len(valid_dates) - n_days))
            dates = valid_dates[start_idx:start_idx + n_days]
            feat_key = random.choice(['TIR', 'mean_glucose', 'std_glucose', 'min_glucose', 'max_glucose'])
            feat_name = feature_names[feat_key]
            template = random.choice(templates['feature_range'])
            question = template.format(
                feature_name=feat_name,
                date_start=date_to_str(dates[0]),
                date_end=date_to_str(dates[-1])
            )
            
            gt = compute_ground_truth(df, dates, [feat_key], sampling_rate, 'feature_range')
            
            questions_for_subject.append({
                'subject_id': subj_id,
                'question': question,
                'query_type': 'feature_range',
                'category': 'synthetic',
                'ground_truth': gt,
                'features': [feat_key],
                'dates': [date_to_str(d) for d in dates],
                'function_calls': ['filter_cgm_csv', 'extract_features_json', 'feature_range'],
            })
        
        # 5. Period comparison (~15 per subject)
        for _ in range(15):
            n_days = random.choice([3, 5, 7])
            gap = random.choice([0, 1, 7])
            start_a = random.randint(0, max(0, len(valid_dates) - 2*n_days - gap))
            dates_a = valid_dates[start_a:start_a + n_days]
            start_b = start_a + n_days + gap
            dates_b = valid_dates[start_b:min(start_b + n_days, len(valid_dates))]
            if len(dates_b) < 2:
                dates_b = valid_dates[-n_days:]
            
            feat_key = random.choice(['TIR', 'mean_glucose', 'std_glucose', 'cv'])
            feat_name = feature_names[feat_key]
            period_a = f"{date_to_str(dates_a[0])} to {date_to_str(dates_a[-1])}"
            period_b = f"{date_to_str(dates_b[0])} to {date_to_str(dates_b[-1])}"
            template = random.choice(templates['period_comparison'])
            question = template.format(
                feature_name=feat_name,
                period_a=period_a,
                period_b=period_b
            )
            
            gt = compute_ground_truth(df, dates_a + dates_b, [feat_key], sampling_rate, 
                                     'period_comparison',
                                     params={'dates_a': dates_a, 'dates_b': dates_b})
            
            questions_for_subject.append({
                'subject_id': subj_id,
                'question': question,
                'query_type': 'period_comparison',
                'category': 'synthetic',
                'ground_truth': gt,
                'features': [feat_key],
                'dates': [date_to_str(d) for d in dates_a + dates_b],
                'function_calls': ['filter_cgm_csv', 'extract_features_json', 'compute_difference_ratio'],
            })
        
        # 6. Time window analysis (~15 per subject)
        for _ in range(15):
            date = random.choice(valid_dates)
            tw = random.choice(time_windows)
            time_start, time_end, time_desc = tw
            feat_key = random.choice(['TIR', 'mean_glucose', 'std_glucose', 'min_glucose', 'max_glucose'])
            feat_name = feature_names[feat_key]
            template = random.choice(templates['time_window'])
            question = template.format(
                feature_name=feat_name,
                time_desc=time_desc,
                time_start=time_start,
                time_end=time_end,
                date=date_to_str(date)
            )
            
            gt = compute_ground_truth(df, [date], [feat_key], sampling_rate, 'time_window',
                                     params={'time_start': time_start, 'time_end': time_end})
            
            questions_for_subject.append({
                'subject_id': subj_id,
                'question': question,
                'query_type': 'time_window',
                'category': 'synthetic',
                'ground_truth': gt,
                'features': [feat_key],
                'dates': [date_to_str(date)],
                'function_calls': ['filter_cgm_csv', 'extract_features_json'],
            })
        
        # 7. Excursion detection (~15 per subject)
        for _ in range(15):
            date = random.choice(valid_dates)
            template = random.choice(templates['excursion'])
            question = template.format(date=date_to_str(date))
            
            gt = compute_ground_truth(df, [date], [], sampling_rate, 'excursion')
            
            questions_for_subject.append({
                'subject_id': subj_id,
                'question': question,
                'query_type': 'excursion',
                'category': 'synthetic',
                'ground_truth': gt,
                'features': [],
                'dates': [date_to_str(date)],
                'function_calls': ['filter_cgm_csv', 'calculate_blood_glucose_excursion'],
            })
        
        # 8. Trend visualization (~15 per subject)
        for _ in range(15):
            n_days = random.choice([3, 5, 7])
            start_idx = random.randint(0, max(0, len(valid_dates) - n_days))
            dates = valid_dates[start_idx:start_idx + n_days]
            template = random.choice(templates['trend'])
            question = template.format(
                date_start=date_to_str(dates[0]),
                date_end=date_to_str(dates[-1])
            )
            
            gt = compute_ground_truth(df, dates, [], sampling_rate, 'trend')
            
            questions_for_subject.append({
                'subject_id': subj_id,
                'question': question,
                'query_type': 'trend',
                'category': 'synthetic',
                'ground_truth': gt,
                'features': [],
                'dates': [date_to_str(d) for d in dates],
                'function_calls': ['filter_cgm_csv', 'plot_daily_trends'],
            })
        
        all_questions.extend(questions_for_subject)
        print(f"  {subj_id}: {len(questions_for_subject)} synthetic questions")
    
    return all_questions


def generate_user_derived_questions(subjects, target_per_subject=90):
    """Generate ~90 user-derived questions per subject = ~1710 total.
    These simulate real user questions with ambiguity.
    Includes answerable (798), proxy (399), and unanswerable (513).
    """
    all_questions = []
    
    # Answerable user question templates
    answerable_templates = [
        ("How was my blood glucose yesterday?", ['mean_glucose', 'TIR', 'std_glucose', 'cgm_weartime']),
        ("What's my time in range for the past week?", ['TIR', 'cgm_weartime']),
        ("Am I having too many lows?", ['TBR', 'hypo_events']),
        ("How stable is my glucose?", ['cv', 'std_glucose']),
        ("What's my average sugar level this month?", ['mean_glucose']),
        ("How often am I going high?", ['TAR', 'hyper_events']),
        ("What's my estimated A1c?", ['eA1c', 'eGMI']),
        ("Show me my glucose patterns", ['mean_glucose', 'std_glucose']),
        ("What's my worst day for glucose control?", ['TIR']),
        ("How much time am I spending above range?", ['TAR']),
        ("What are my typical 24-hour glucose patterns?", ['mean_glucose']),
        ("How does my morning glucose compare to evening?", ['mean_glucose']),
        ("What's my glucose variability?", ['cv', 'std_glucose']),
        ("Am I meeting my glucose targets?", ['TIR', 'TBR', 'TAR']),
        ("What's my best day for glucose control?", ['TIR']),
        ("How many hypo events did I have this week?", ['hypo_events']),
        ("What's my glucose range today?", ['min_glucose', 'max_glucose']),
        ("Is my glucose getting better over time?", ['TIR', 'mean_glucose']),
        ("What's my standard deviation of blood glucose?", ['std_glucose']),
        ("How's my overnight glucose?", ['mean_glucose', 'TIR']),
        ("What's my glucose like during the afternoon?", ['mean_glucose', 'TIR']),
        ("Do I have dawn phenomenon?", ['mean_glucose']),
    ]
    
    # Proxy question templates (answerable via time-window proxy)
    proxy_templates = [
        ("How does exercise affect my glucose?", "Compare my TIR between {time_a} and {time_b} on {date}.",
         ['TIR', 'mean_glucose']),
        ("What happens to my glucose after meals?", "What are my glucose excursions on {date}?",
         ['mean_glucose']),
        ("Does late-night eating affect my glucose?", "What's my glucose between 21:00 and 06:00 on {date}?",
         ['mean_glucose', 'TIR']),
        ("How does stress affect my blood sugar?", "What's my glucose variability on {date}?",
         ['cv', 'std_glucose']),
        ("What patterns do I see around my menstrual cycle?", 
         "Plot my typical daily CGM blood glucose trends for {date_start} to {date_end}.",
         ['mean_glucose']),
        ("How long after eating do my glucose levels rise?", "When does my glucose rise fast on {date}?",
         ['mean_glucose']),
        ("Does coffee spike my glucose?", "What's my glucose between 06:00 and 09:00 on {date}?",
         ['mean_glucose', 'max_glucose']),
    ]
    
    # Unanswerable question templates
    unanswerable_templates = [
        "What type of foods are generally safe to consume and do not need massive insulin boluses?",
        "Does splitting insulin boluses help me lower my insulin intake?",
        "Given my CGM data, can you find moments that were likely incorrect blood glucose values?",
        "Does the stability of my blood glucose predict more stability in the coming days?",
        "How does Mounjaro impact insulin intake?",
        "What is my insulin sensitivity factor?",
        "How many carbs should I eat to correct a low?",
        "What's my insulin-to-carb ratio?",
        "Should I adjust my basal rate?",
        "What medication would help my glucose control?",
        "Am I developing insulin resistance?",
        "What's causing my dawn phenomenon?",
        "Should I change my pump settings?",
        "How many units of insulin did I take today?",
        "What did I eat that caused that spike?",
        "How much exercise do I need to lower my glucose?",
        "Is my diabetes getting worse?",
        "Should I see my endocrinologist?",
        "What supplements would help my glucose?",
        "Can you predict my glucose for tomorrow?",
        "How does my sleep quality affect my glucose?",
        "What's my total daily insulin dose?",
        "Am I at risk for diabetic complications?",
        "How does alcohol affect my blood sugar?",
        "What's my carb intake for today?",
        "Should I increase my metformin dose?",
        "How does my weight affect my glucose?",
    ]
    
    for subj_id, subj_data in subjects.items():
        df = subj_data.df
        sampling_rate = subj_data.sampling_rate
        valid_dates = get_valid_dates(df)
        
        if len(valid_dates) < 7:
            continue
        
        questions_for_subject = []
        
        # Answerable questions (~42 per subject → 798 total)
        for _ in range(42):
            template_q, feats = random.choice(answerable_templates)
            # Instantiate with random dates
            date = random.choice(valid_dates)
            n_days = random.choice([1, 3, 7])
            start_idx = max(0, valid_dates.index(date) - n_days + 1)
            dates = valid_dates[start_idx:start_idx + n_days]
            
            # Create instantiated question
            ref_date = date_to_str(date)
            question = template_q  # Keep template as-is for natural language
            
            # Compute ground truth
            if n_days == 1:
                gt = compute_ground_truth(df, [date], feats, sampling_rate, 'basic_retrieval')
            else:
                gt = compute_ground_truth(df, dates, feats, sampling_rate, 'multi_day_average')
            
            questions_for_subject.append({
                'subject_id': subj_id,
                'question': question,
                'query_type': 'user_answerable',
                'category': 'user_derived',
                'is_answerable': True,
                'ground_truth': gt,
                'features': feats,
                'dates': [date_to_str(d) for d in dates],
                'reference_date': ref_date,
                'function_calls': ['filter_cgm_csv', 'extract_features_json'],
            })
        
        # Proxy questions (~21 per subject → 399 total)
        for _ in range(21):
            orig_q, proxy_q, feats = random.choice(proxy_templates)
            date = random.choice(valid_dates)
            dates = valid_dates[max(0, valid_dates.index(date)-6):valid_dates.index(date)+1]
            
            proxy_instantiated = proxy_q.format(
                date=date_to_str(date),
                date_start=date_to_str(dates[0]),
                date_end=date_to_str(dates[-1]),
                time_a="15:00-17:00",
                time_b="17:00-19:00"
            )
            
            gt = compute_ground_truth(df, [date], feats, sampling_rate, 'basic_retrieval')
            
            questions_for_subject.append({
                'subject_id': subj_id,
                'question': orig_q,
                'proxy_question': proxy_instantiated,
                'query_type': 'user_proxy',
                'category': 'user_derived',
                'is_answerable': True,
                'ground_truth': gt,
                'features': feats,
                'dates': [date_to_str(d) for d in dates],
                'reference_date': date_to_str(date),
                'function_calls': ['filter_cgm_csv', 'extract_features_json'],
            })
        
        # Unanswerable questions (~27 per subject → 513 total)
        for _ in range(27):
            question = random.choice(unanswerable_templates)
            
            questions_for_subject.append({
                'subject_id': subj_id,
                'question': question,
                'query_type': 'user_unanswerable',
                'category': 'user_derived',
                'is_answerable': False,
                'ground_truth': {},
                'features': [],
                'dates': [],
                'function_calls': [],
            })
        
        all_questions.extend(questions_for_subject)
        print(f"  {subj_id}: {len(questions_for_subject)} user-derived questions")
    
    return all_questions


def main():
    print("Loading subjects...")
    subjects = load_subjects()
    
    print("\nGenerating synthetic questions...")
    synthetic = generate_synthetic_questions(subjects)
    print(f"Total synthetic: {len(synthetic)}")
    
    print("\nGenerating user-derived questions...")
    user_derived = generate_user_derived_questions(subjects)
    print(f"Total user-derived: {len(user_derived)}")
    
    all_questions = synthetic + user_derived
    print(f"\nTotal questions: {len(all_questions)}")
    
    # Count by category
    cats = {}
    for q in all_questions:
        cat = q.get('query_type', 'unknown')
        cats[cat] = cats.get(cat, 0) + 1
    print("By type:", json.dumps(cats, indent=2))
    
    # Save - convert dates to strings for JSON serialization
    def default_serializer(obj):
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)
    
    with open('results/qa_dataset.json', 'w') as f:
        json.dump(all_questions, f, default=default_serializer, indent=1)
    
    print(f"\nSaved {len(all_questions)} questions to results/qa_dataset.json")
    
    # Summary stats matching paper Table 1
    n_synthetic = len([q for q in all_questions if q['category'] == 'synthetic'])
    n_answerable = len([q for q in all_questions if q.get('query_type') == 'user_answerable'])
    n_proxy = len([q for q in all_questions if q.get('query_type') == 'user_proxy'])
    n_unanswerable = len([q for q in all_questions if q.get('query_type') == 'user_unanswerable'])
    
    print(f"\nDataset composition (cf. paper Table 1):")
    print(f"  Synthetic template-generated: {n_synthetic} (paper: 2,470)")
    print(f"  User-derived answerable: {n_answerable} (paper: 798)")
    print(f"  User-derived proxy: {n_proxy} (paper: 399)")
    print(f"  User-derived unanswerable: {n_unanswerable} (paper: 513)")
    print(f"  Total: {len(all_questions)} (paper: 4,180)")
    
    return all_questions


if __name__ == '__main__':
    main()
