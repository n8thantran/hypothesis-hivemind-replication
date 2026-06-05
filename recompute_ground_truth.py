"""
Recompute ground truth for all QA pairs using SubjectData.get_features().
This fixes the bug in the original generation where extract_features_json 
was called with wrong arguments.
"""

import json
import sys
sys.path.insert(0, '/workspace')
from load_subjects import load_all_subjects
from cgm_toolkit import (
    extract_features_json, get_average, count_satisfied_condition,
    feature_range, compute_difference_ratio, filter_cgm_csv,

)


def recompute_gt(q, subjects):
    """Recompute ground truth for a single question."""
    subject_id = q['subject_id']
    subject = subjects.get(subject_id)
    if not subject:
        return {'error': f'Subject {subject_id} not found'}
    
    query_type = q.get('query_type', '')
    dates = q.get('dates', [])
    features = q.get('features', [])
    category = q.get('category', '')
    
    if category == 'user_derived' and not q.get('is_answerable', True):
        return {}  # Unanswerable questions have no ground truth
    
    if not dates:
        return {'error': 'No dates specified'}
    
    # Validate dates
    valid_dates = [d for d in dates if d in subject.date_strings]
    if not valid_dates:
        return {'error': f'No valid dates found. Requested: {dates[:3]}'}
    
    try:
        if query_type == 'basic_retrieval':
            all_features = subject.get_features(valid_dates)
            if features:
                # Filter to requested features
                gt = {}
                for d in valid_dates:
                    if d in all_features:
                        gt[d] = {}
                        for feat in features:
                            feat_map = get_feature_key_map()
                            mapped_feat = feat_map.get(feat, feat)
                            for k, v in all_features[d].items():
                                if mapped_feat.lower() in k.lower() or feat.lower() in k.lower():
                                    gt[d][k] = v
                            # If no match, include all features for the date
                            if not gt[d]:
                                gt[d] = all_features[d]
                return gt
            else:
                return all_features
        
        elif query_type == 'multi_day_average':
            all_features = subject.get_features(valid_dates)
            gt = {}
            if features:
                feat_map = get_feature_key_map()
                for feat in features:
                    mapped_feat = feat_map.get(feat, feat)
                    avg_result = get_average(all_features, mapped_feat)
                    key = f"({valid_dates[0]}, {valid_dates[-1]})"
                    gt[key] = avg_result
            else:
                key = f"({valid_dates[0]}, {valid_dates[-1]})"
                gt[key] = {}
                for feat_key in ['TIR', 'TBR', 'TAR', 'mean_glucose', 'std_glucose', 'CV']:
                    avg_result = get_average(all_features, feat_key)
                    gt[key][feat_key] = avg_result
            return gt
        
        elif query_type == 'conditional_count':
            all_features = subject.get_features(valid_dates)
            condition = q.get('condition', 'hypo_events == 0')  
            # Try to extract from question
            question = q.get('question', '')
            if 'no hypoglycemia' in question.lower() or 'hypo' in question.lower():
                condition = 'hypo_events == 0'
            elif 'no hyperglycemia' in question.lower() or 'hyper' in question.lower():
                condition = 'hyper_events == 0'
            elif 'TIR' in question or 'time in range' in question.lower():
                condition = 'TIR > 70'
            count_result = count_satisfied_condition(all_features, condition)
            key = f"({valid_dates[0]}, {valid_dates[-1]})"
            return {key: count_result}
        
        elif query_type == 'feature_range':
            all_features = subject.get_features(valid_dates)
            feat_map = get_feature_key_map()
            feat = features[0] if features else 'TIR'
            mapped_feat = feat_map.get(feat, feat)
            range_result = feature_range(all_features, mapped_feat)
            key = f"({valid_dates[0]}, {valid_dates[-1]})"
            return {key: range_result}
        
        elif query_type == 'period_comparison':
            # Split dates into two groups
            mid = len(valid_dates) // 2
            dates_a = valid_dates[:mid]
            dates_b = valid_dates[mid:]
            features_a = subject.get_features(dates_a)
            features_b = subject.get_features(dates_b)
            feat_map = get_feature_key_map()
            feat = features[0] if features else 'mean_glucose'
            mapped_feat = feat_map.get(feat, feat)
            diff_result = compute_difference_ratio(features_a, features_b, mapped_feat)
            return {"comparison": diff_result}
        
        elif query_type == 'time_window':
            # Get time window params from question
            question = q.get('question', '')
            time_start = '06:00'
            time_end = '12:00'
            if 'morning' in question.lower():
                time_start, time_end = '06:00', '12:00'
            elif 'afternoon' in question.lower():
                time_start, time_end = '12:00', '18:00'
            elif 'evening' in question.lower() or 'night' in question.lower():
                time_start, time_end = '18:00', '00:00'
            
            filtered = filter_cgm_csv(subject.df, dates=valid_dates,
                                       start_time=time_start, end_time=time_end)
            if filtered.empty:
                return {'error': 'No data for time window'}
            all_features = extract_features_json(filtered, valid_dates, subject.sampling_rate)
            return all_features
        
        elif query_type == 'excursion':
            all_features = subject.get_features(valid_dates)
            # Also get excursion-specific data
            return all_features
        
        elif query_type == 'trend':
            all_features = subject.get_features(valid_dates)
            return all_features
        
        elif query_type in ('user_answerable', 'user_proxy'):
            all_features = subject.get_features(valid_dates)
            return all_features
        
        else:
            all_features = subject.get_features(valid_dates)
            return all_features
            
    except Exception as e:
        return {'error': str(e)}


def get_feature_key_map():
    """Map feature short names to keys used in extract_features_json."""
    return {
        'TIR': 'TIR',
        'TBR': 'TBR', 
        'TAR': 'TAR',
        'mean': 'mean_glucose',
        'std': 'std_glucose',
        'CV': 'CV',
        'min': 'min_glucose',
        'max': 'max_glucose',
        'eA1c': 'estimated_a1c',
        'eGMI': 'eGMI',
        'avg_glucose': 'mean_glucose',
        'mean_glucose': 'mean_glucose',
        'std_glucose': 'std_glucose',
        'hypo_events': 'hypo_events',
        'hyper_events': 'hyper_events',
        'weartime': 'cgm_weartime',
    }


def main():
    print("Loading subjects...")
    subjects = load_all_subjects('/workspace/data/raw/AZT1D/AZT1D-extracted-glucose-files', '/workspace/data/raw/ShanghaiT2DM_clean')
    print(f"Loaded {len(subjects)} subjects")
    
    with open('results/qa_dataset.json') as f:
        qa_data = json.load(f)
    print(f"Total questions: {len(qa_data)}")
    
    success = 0
    errors = 0
    for i, q in enumerate(qa_data):
        gt = recompute_gt(q, subjects)
        q['ground_truth'] = gt
        if 'error' in str(gt):
            errors += 1
        else:
            success += 1
        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(qa_data)}, success: {success}, errors: {errors}")
    
    print(f"Done: {success} success, {errors} errors out of {len(qa_data)}")
    
    with open('results/qa_dataset.json', 'w') as f:
        json.dump(qa_data, f, default=str, indent=1)
    print("Saved updated qa_dataset.json")


if __name__ == '__main__':
    main()
