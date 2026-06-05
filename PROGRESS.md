# CGM-Agent Replication Progress

## Current Phase: Planning

## Implementation Plan
- [ ] 1. Download and process CGM data (AZT1D + ShanghaiT2DM datasets)
- [ ] 2. Implement CGM analytical toolkit (all 12+ functions)
- [ ] 3. Implement question generation (synthetic templates + user-derived)
- [ ] 4. Implement 3-layer agent pipeline (Input Processor, Analytical Agent, Response Generator)
- [ ] 5. Implement evaluation metrics (Precision, Recall, F1, Value Accuracy)
- [ ] 6. Run experiments and generate result tables
- [ ] 7. Create reproduce.sh and final report

## Key Decisions
- Paper uses 19 subjects: P1-P11 from AZT1D (T1D, 5-min), P12-P19 from ShanghaiT2DM (T2D, 15-min)
- Subject mapping in Table 10 (Appendix)
- Toolkit has 3 tiers: Data Processing, Daily Metrics, Long-term Aggregation
- Evaluation uses ±1% tolerance for value accuracy
- CGM weartime threshold: ≥70%
- TIR: 70-180 mg/dL, TBR: <70 mg/dL, TAR: >180 mg/dL
- Hypo events: <70 mg/dL for 15+ min, Hyper events: >180 mg/dL for 15+ min

## Key Tables to Reproduce
- Table 3: Synthetic query results (Layer 2)
- Table 4: Layer 1 feasibility classification  
- Table 5: Layer 2 real-world query results
- Table 6: Readability analysis
- Table 7: Ablation study
- Table 8: TIR correlation

## Completed Work
(none yet)

## Failed Approaches
(none yet)
