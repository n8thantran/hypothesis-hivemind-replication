"""
Load and verify all 19 subjects from AZT1D (P1-P11) and ShanghaiT2DM (P12-P19).
Saves processed data summaries.
"""

import os
import sys
import json
from cgm_toolkit import load_all_subjects

def main():
    azt1d_dir = '/workspace/data/raw/AZT1D/AZT1D-extracted-glucose-files'
    shanghai_dir = '/workspace/data/raw/ShanghaiT2DM_clean'
    
    print(f"AZT1D dir: {azt1d_dir}")
    print(f"Shanghai dir: {shanghai_dir}")
    
    # Load all subjects
    subjects = load_all_subjects(azt1d_dir, shanghai_dir)
    
    print(f"\nLoaded {len(subjects)} subjects")
    print(f"Expected: 19 subjects (P1-P11 AZT1D, P12-P19 ShanghaiT2DM)")
    
    # Generate summaries
    summaries = {}
    for pid in sorted(subjects.keys(), key=lambda x: int(x[1:])):
        s = subjects[pid]
        summary = s.summary()
        summaries[pid] = summary
        print(f"  {pid}: {summary['dataset']}, {summary['num_days']} days, "
              f"{summary['total_readings']} readings, "
              f"sampling={summary['sampling_rate']}min, "
              f"TIR={summary['TIR']:.1f}%")
    
    # Save summaries
    os.makedirs('/workspace/results', exist_ok=True)
    with open('/workspace/results/subject_summaries.json', 'w') as f:
        clean = {}
        for k, v in summaries.items():
            clean[k] = {kk: float(vv) if hasattr(vv, '__float__') and not isinstance(vv, str) else vv 
                       for kk, vv in v.items()}
        json.dump(clean, f, indent=2)
    
    print(f"\nSummaries saved to /workspace/results/subject_summaries.json")
    return subjects

if __name__ == '__main__':
    main()
