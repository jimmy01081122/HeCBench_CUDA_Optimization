## 1. Scope and Rules Followed                                                                                  
                                                                                                                  
  • Read-Only Scan: Executed inspect-only analysis of the repository paths inside  /home/a/PP . No compiling (    
  make / nvcc ), executing ( ./main ), submitting ( sbatch ), or modifying/overwriting of source, metadata, or    
  summary files was performed.                                                                                    
  • Rules Followed: No scientific speedups were inferred. Stale files were identified but not modified. Mode C was
  detected as  NOT_FOUND  /  NOT_STARTED  and left uninterpreted.                                                 
                                                                                                                  
  ## 2. Hostname Conflict Check                                                                                   
                                                                                                                  
  • Investigation: Inspected all final confirmation files inside :                                  
      • results.csv                                                                                            
      • result.out                                                                                            
      • result.err                                                                                            
      • environment.txt                                                                                            
      • round_summary.md                                                                                            
  • Authoritative Hostname:  gn1221.twcc.ai  for rounds 1 & 2, and  gn1228.twcc.ai  for the final confirmation run.
  • Conflict Status: There is no hostname conflict in the final files on disk. Every single file in the  final/   
  folder (including  environment.txt  line 2,  result.out  line 2,  results.csv  hostname column for all 30 rows, 
  and  round_summary.md  lines 5 & 27) consistently records the hostname as  gn1228.twcc.ai .                     
  • Authoritative Source: The raw Slurm environment output environment.txt is the authoritative source.            
  • Note: The hostname  gn1288  was a typo introduced only in the parent agent's compaction summary text from the 
  previous turn and does not exist in any workspace files.                                                        
                                                                                                                  
  ## 3. Untracked File Preview                                                                                    
                                                                                                                  
  ### A. state_report.md                                                                                           
                                                                                                                  
  • Status: Final draft of a qualitative analysis report.                                                         
  • Unique Info: Yes. Contains deep qualitative reflections on why human-in-the-loop review was critical to turn  
  the partial success/failure of Round 1 into a shape-aware dispatch policy in Round 2, preventing pseudo-speedup 
  claims.                                                                                                         
  • Evidential Value: High. Should be tracked and committed to serve as core qualitative evidence for report/paper
  drafting.                                                                                                       
                                                                                                                  
  ### B. temp_commition.md                                                                                          
                                                                                                                  
  • Status: Progress draft outlining timeline and decision logs for the robust baseline, Round 1, Round 2, and    
  Final confirmation phases.                                                                                      
  • Unique Info: Yes. Outlines why Round 1 was labeled  NEEDS_REVISION , why  impl2  was rejected as a full       
  replacement, and details the exact guidelines and questions for topk-cuda Mode B Round 1.                       
  • Evidential Value: High. Serves as a perfect record of the interactive human-AI collaborative process.         
                                                                                                                  
  ## 4. Stale Summary Inspection                                                                                  
                                                                                                                  
  1. agent_summary.md:                                                                                              
      • MTime: 2026-06-11 16:54:52                                                                                
      • Concise Summary: States that Mode B was stopped before Round 1 pending human review.                      
      • Staleness: Highly Stale. It does not reflect the completed Round 1, Round 2, and Final confirmation.      
      • Incorrect Claims: It claims optimization has not started.                                                 
      • Action: Should be superseded by round_summary.md (or updated with the version written by the Slurm script  
      on the server).                                                                                             
  2. results.csv:                                                                                             
      • MTime: 2026-06-11 16:54:52                                                                                
      • Concise Summary: Contains only 5 baseline rows in 27-column legacy schema.                                
      • Staleness: Highly Stale. Missing all trial-level and optimization round data.                             
      • Incorrect Claims: Outdated baseline representation only.                                                  
      • Action: Should be superseded by results.csv.                                                          
  3. mode_B_report.md:                                                                                             
      • MTime: 2026-06-11 16:49:07                                                                                
      • Concise Summary: Summarizes robust baseline execution for all three benchmarks.                           
      • Staleness: Stale. Claims Mode B guided optimization has not begun.                                        
      • Incorrect Claims: Outdated status of softmax Mode B.                                                      
      • Action: Should be updated to incorporate the final SUCCESS status of  softmax-cuda  Mode B.               
                                                                                                                  
                                                                                                                  
  ## 5. softmax-cuda Mode B Final Evidence                                                                        
                                                                                                                  
  • Checked Files: results.csv, round_summary.md, auditor_report.csv, contradiction_check.csv.                           
  • results.csv check: Row count: 31 (1 header + 30 trial rows). Column count: 33.                                
  • Raw Outputs: Verified that all referenced  .stdout / .stderr  files in  final/raw/  exist, and                
  profiler_status  is  NOT_RUN  for all rows.                                                                     
                                                                                                                  
  ### final/results.csv Trial-Level Per-Slice Summary:                                                            
                                                                                                                  
   Slice S… │ Dispatc… │ Baselin… │ Candida… │ Aggrega… │ Correct… │ Measure… │ Speedup … │ Result T… │ Candidat…
  ──────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┼───────────┼───────────┼───────────
     128    │    1     │ 0.134869 │ 0.134574 │ 1.002197 │   PASS   │  VALID   │   false   │ MEASUREME │   0.04%
            │          │          │          │          │          │          │           │ NT_EQUIVA │           
            │          │          │          │          │          │          │           │ LENT      │
     256    │    1     │ 0.321793 │ 0.321505 │ 1.000895 │   PASS   │  VALID   │   false   │ MEASUREME │   0.11%
            │          │          │          │          │          │          │           │ NT_EQUIVA │           
            │          │          │          │          │          │          │           │ LENT      │
     784    │    2     │ 1.442716 │ 1.036402 │ 1.392043 │   PASS   │  VALID   │   true    │ PARAM_TUN │   1.85%
            │          │          │          │          │          │          │           │ E         │
     1024   │    2     │ 2.104045 │ 1.238443 │ 1.698944 │   PASS   │  VALID   │   true    │ PARAM_TUN │   0.00%
            │          │          │          │          │          │          │           │ E         │
     2048   │    2     │ 2.237452 │ 1.672904 │ 1.337466 │   PASS   │  VALID   │   true    │ PARAM_TUN │   0.02%
            │          │          │          │          │          │          │           │ E         │
                                                                                                                  
  ## 6. softmax-cuda Round 1 and Round 2 Evidence                                                                 
                                                                                                                  
  • Round 1 Summary: round_summary.md. Candidate  impl2  (compound block cached-exp) was rejected because          
  slice=256  had a correctness check failure (trial 1 PASS 2/3, FAIL 1/3) and  slice=128  regressed (0.243x).     
  • Round 1 Patch Summary: patch_summary.md. Explains  impl2  was a compound candidate changing thread layout to   
  block-per-slice and caching exponentials in shared memory.                                                      
  • Round 2 Summary: round_summary.md. Candidate  impl3  (shape-aware dispatch) passed all correctness checks.     
  • Round 2 Patch Summary: patch_summary.md. Explains dispatch logic: slice size 784, 1024, 2048 invokes block-    
  cached-exp  softMax3  ( impl=2 ), whereas slice 128 and 256 invokes optimized  softMax2  ( impl=1 ).            
                                                                                                                  
  ## 7. Mode A and Robust Baseline Evidence                                                                       
                                                                                                                  
  • mode_A_report.md: Documents that Mode A is an agent-only baseline. Exposes high-CV measurement risk (e.g. topk-
  cuda baseline CV ~54.6% caused a fake speedup of 1.4x), showing that robust baselines and paired baselines are  
  required to avoid fake speedups.                                                                                
  • results.csv: Baseline comparison of  impl=0  (naive) vs  impl=1  (existing optimized). Shows  impl=1  is  
  ~37x faster than  impl=0  on large slices, but this is an existing baseline, not a Phase 3 optimization speedup.
  • results.csv: Confirms the robust baseline for Mode B was established using existing optimized  impl=1     
  before any optimization rounds were run, verifying that baseline measurements were stable (CV < 2.3%).          
                                                                                                                  
  ## 8. topk-cuda Optional Supporting Evidence                                                                    
                                                                                                                  
  • Status: Mode B status =  BASELINE_ONLY  (robust baseline completed, optimization rounds not started).         
  • High-CV Issue Documented: Yes, in  mode_A_report.md  (identifies baseline CV of up to 54.6%) and in the robust
  baseline notes (explaining that 7 trials were run up front to suppress CV).                                     
  • Optimization Round: No optimization round exists.                                                             
                                                                                                                  
  ## 9. shmembench-cuda Optional Supporting Evidence                                                              
                                                                                                                  
  • Status: Mode B status =  BASELINE_ONLY  (robust baseline completed, optimization rounds not started).         
  • Official validated comparison:  block_size=256  and  variant=original  is the official validated comparison   
  (which passed correctness checks).                                                                              
  • Diagnostic fail: Block sizes 128, 512, and 1024 are diagnostic failures ( DIAGNOSTIC_FAIL ), failing          
  correctness sweeps or exceeding shared memory compilation limits.                                               
  • Optimization Round: No optimization round exists.                                                             
                                                                                                                  
  ## 10. Authoritative Files                                                                                      
                                                                                                                  
  For writing final reports/papers, only the following files should be cited as authoritative evidence:           
                                                                                                                  
  • Softmax Mode B Final CSV: results.csv                                                                     
  • Softmax Mode B Final Summary: round_summary.md                                                                 
  • Softmax Mode B Final Source: main.cu                                                                  
  • Softmax Mode B Qualitative Analysis: state_report.md                                                          
  • Softmax Mode B Timeline Log: temp_commition.md                                                                  
  • Self-Consistency Audit: auditor_report.csv and contradiction_check.csv                                                  
  • Overall Mode A Report: mode_A_report.md                                                                        
                                                                                                                  
  ## 11. Stale / Superseded / Do-Not-Cite Files                                                                   
                                                                                                                  
  The following files are stale, incorrect, or redundant, and must not be cited in the report:                    
                                                                                                                  
  • Stale Softmax Summaries: agent_summary.md and results.csv at the parent level on local.                    
  • Stale Mode B Reports: mode_B_report.md and its duplicate MODE_B_REPORT.md.                                      
  • Redundant Directory: Top-level directory  phase3/mode_A_agent_only  (redundant duplicate of  topk-            
  cuda/mode_A_agent_only ).                                                                                       
                                                                                                                  
  ## 12. Git Status Items Requiring Human Decision
  
  • deleted  rounds/round_2/advice : Decision needed on whether this deleted advice block is still required for   
  logging or can be safely discarded.
  • untracked  final/state_report.md : Should be added to tracking and committed to preserve core qualitative     
  evidence.
  • untracked  final/temp_commition.md : Should be added to tracking and committed to preserve the timeline log.  
  
  ## 13. Do-Not-Claim List
  
  • Do not claim that Mode C has started.
  • Do not claim that Mode C is effective.
  • Do not claim that  topk-cuda  Mode B optimization has begun or succeeded.
  • Do not claim that  shmembench-cuda  Mode B optimization has begun or succeeded.
  • Do not claim that  impl=3  (shape-aware dispatch) is a new universal kernel optimization.
  • Do not claim that  impl=2  is a universal replacement for all softmax shape sizes.
  • Do not claim any optimization speedup for  slice=128  or  slice=256  under  impl=3 .
  • Do not claim profiler-supported bottleneck conclusions for Mode B.
  • Do not claim that shared-memory cached exponentials alone caused the large-slice speedup.
  
  ## 14. Readiness for Chinese Report Drafting
  
  • Status: The repository is fully ready for Chinese report drafting concerning the  softmax-cuda  Mode B SUCCESS
  results, as all final confirmation metrics are verified and complete.
  • Note: Topk and shmembench Mode B results must be described strictly as frozen at the robust baseline stage,   
  with no optimization speedup claims.