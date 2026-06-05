# CGM-Agent Replication Progress

## Current Phase: Implementation - Data Processing

## Implementation Plan
- [ ] 1. Process CGM data for all 19 subjects (AZT1D + ShanghaiT2DM)
- [ ] 2. Implement CGM analytical toolkit (all 12+ functions)
- [ ] 3. Implement question generation (synthetic templates + user-derived)
- [ ] 4. Implement 3-layer agent pipeline (Input Processor, Analytical Agent, Response Generator)
- [ ] 5. Implement evaluation metrics (Precision, Recall, F1, Value Accuracy)
- [ ] 6. Run experiments and generate result tables
- [ ] 7. Create reproduce.sh and final report

## Key Decisions
- Paper uses 19 subjects: P1-P11 from AZT1D (T1D, 5-min), P12-P19 from ShanghaiT2DM (T2D, 15-min)
- Subject mapping (Table 10):
  - P1=Subject15, P2=Subject23, P3=Subject21, P4=Subject20, P5=Subject7
  - P6=Subject19, P7=Subject5, P8=Subject13, P9=Subject6, P10=Subject11, P11=Subject4
  - P12=2069, P13=2014, P14=2017, P15=2015, P16=2078, P17=2001, P18=2055, P19=2074
- Toolkit has 3 tiers: Data Processing, Daily Metrics, Long-term Aggregation
- Evaluation uses ±1% tolerance for value accuracy
- CGM weartime threshold: ≥70%
- TIR: 70-180 mg/dL, TBR: <70 mg/dL, TAR: >180 mg/dL
- Hypo events: <70 mg/dL for 15+ min, Hyper events: >180 mg/dL for 15+ min
- GMI = 3.31 + 0.02392 × mean_glucose (mg/dL)
- eA1c = (46.7 + mean_glucose) / 28.7

## Key Tables to Reproduce
- Table 3: Synthetic query results (Layer 2)
- Table 4: Layer 1 feasibility classification  
- Table 5: Layer 2 real-world query results
- Table 6: Readability analysis
- Table 7: Ablation study
- Table 8: TIR correlation

## Data Sources
- AZT1D: Downloaded from Glucose-ML (PhysioNet), subjects in /workspace/data/raw/glucose-ml-1.0.0/
- ShanghaiT2DM: Downloaded from figshare, extracted to /workspace/data/raw/ShanghaiT2DM_full/

## Completed Work
(none yet)

## Failed Approaches
(none yet)
