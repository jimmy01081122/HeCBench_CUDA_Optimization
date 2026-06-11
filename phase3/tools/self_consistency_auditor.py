#!/usr/bin/env python3
import sys
import os
import csv

def run_auditor(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return False
        
    rules_checked = {
        "Rule_1_correctness_pass": {"status": "PASS", "notes": []},
        "Rule_2_cv_noisy": {"status": "PASS", "notes": []},
        "Rule_3_noisy_speedup_invalid": {"status": "PASS", "notes": []},
        "Rule_4_cv_extreme_remeasure": {"status": "PASS", "notes": []},
        "Rule_5_mode_a_no_kernel_opt": {"status": "PASS", "notes": []},
        "Rule_6_softmax_baseline_comparison": {"status": "PASS", "notes": []},
        "Rule_7_shmembench_diagnostic_fail": {"status": "PASS", "notes": []},
        "Rule_8_variant_replace_baseline": {"status": "PASS", "notes": []},
        "Rule_9_low_speedup_equivalent": {"status": "PASS", "notes": []},
        "Rule_10_regression_status": {"status": "PASS", "notes": []}
    }
    
    rows = []
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            
    for idx, row in enumerate(rows):
        benchmark = row.get("benchmark", "")
        mode = row.get("mode", "")
        round_val = row.get("round", "")
        case = row.get("case", "")
        variant = row.get("variant", "")
        correctness = row.get("correctness", "")
        status = row.get("status", "")
        result_type = row.get("result_type", "")
        speedup_str = row.get("speedup", "")
        cv_str = row.get("cv", "")
        correctness_status = row.get("correctness_status", "")
        measurement_validity = row.get("measurement_validity", "")
        speedup_claim_valid = row.get("speedup_claim_valid", "")
        require_remeasurement = row.get("require_remeasurement", "")
        notes = row.get("notes", "")
        
        # Parse float values safely
        try:
            speedup = float(speedup_str) if speedup_str and speedup_str.lower() != 'n/a' else None
        except ValueError:
            speedup = None
            
        try:
            # CV could be a float or a percentage (e.g. 54.6% or 0.546)
            if cv_str and cv_str.lower() != 'n/a':
                if cv_str.endswith('%'):
                    cv = float(cv_str.replace('%', '')) / 100.0
                else:
                    cv = float(cv_str)
            else:
                cv = None
        except ValueError:
            cv = None
            
        # 1. correctness_status != PASS -> speedup=n/a, speedup_claim_valid=false
        if correctness_status and correctness_status != 'PASS':
            if speedup_str and speedup_str.lower() != 'n/a':
                rules_checked["Rule_1_correctness_pass"]["status"] = "FAIL"
                rules_checked["Rule_1_correctness_pass"]["notes"].append(f"Row {idx+2}: correctness_status is '{correctness_status}' but speedup is '{speedup_str}'")
            if speedup_claim_valid and speedup_claim_valid.lower() == 'true':
                rules_checked["Rule_1_correctness_pass"]["status"] = "FAIL"
                rules_checked["Rule_1_correctness_pass"]["notes"].append(f"Row {idx+2}: correctness_status is '{correctness_status}' but speedup_claim_valid is 'true'")

        # 2. CV > 15% -> measurement_validity=NOISY
        if cv is not None and cv > 0.15:
            if measurement_validity != 'NOISY':
                rules_checked["Rule_2_cv_noisy"]["status"] = "FAIL"
                rules_checked["Rule_2_cv_noisy"]["notes"].append(f"Row {idx+2}: CV is {cv*100:.2f}% (> 15%) but measurement_validity is '{measurement_validity}'")

        # 3. CV > 15% and speedup > 1.05 -> speedup_claim_valid=false
        if cv is not None and cv > 0.15 and speedup is not None and speedup > 1.05:
            if speedup_claim_valid and speedup_claim_valid.lower() == 'true':
                rules_checked["Rule_3_noisy_speedup_invalid"]["status"] = "FAIL"
                rules_checked["Rule_3_noisy_speedup_invalid"]["notes"].append(f"Row {idx+2}: CV is {cv*100:.2f}% and speedup is {speedup} (>1.05) but speedup_claim_valid is 'true'")

        # 4. CV > 30% -> require_remeasurement=true
        if cv is not None and cv > 0.30:
            if require_remeasurement and require_remeasurement.lower() != 'true':
                rules_checked["Rule_4_cv_extreme_remeasure"]["status"] = "FAIL"
                rules_checked["Rule_4_cv_extreme_remeasure"]["notes"].append(f"Row {idx+2}: CV is {cv*100:.2f}% (> 30%) but require_remeasurement is '{require_remeasurement}'")

        # 5. Mode_A 且無 source change -> 不得標記 KERNEL_OPT
        if mode == 'Mode_A' and result_type == 'KERNEL_OPT':
            rules_checked["Rule_5_mode_a_no_kernel_opt"]["status"] = "FAIL"
            rules_checked["Rule_5_mode_a_no_kernel_opt"]["notes"].append(f"Row {idx+2}: Mode A without source code optimization cannot be labeled KERNEL_OPT")

        # 6. softmax-cuda 的 impl0_to_impl1 comparison -> BASELINE_COMPARISON, not AGENT_OPT
        if 'softmax-cuda' in benchmark and 'impl=0' in case and 'impl=1' in case:
            if result_type in ['KERNEL_OPT', 'AGENT_OPT']:
                rules_checked["Rule_6_softmax_baseline_comparison"]["status"] = "FAIL"
                rules_checked["Rule_6_softmax_baseline_comparison"]["notes"].append(f"Row {idx+2}: softmax baseline comparison of impl0 and impl1 must be BASELINE_COMPARISON")

        # 7. shmembench-cuda 中 block_size != 256 且 correctness FAIL -> DIAGNOSTIC_FAIL
        if 'shmembench-cuda' in benchmark and 'block_size=256' not in case:
            if correctness == 'FAIL' and measurement_validity != 'DIAGNOSTIC_FAIL':
                rules_checked["Rule_7_shmembench_diagnostic_fail"]["status"] = "FAIL"
                rules_checked["Rule_7_shmembench_diagnostic_fail"]["notes"].append(f"Row {idx+2}: shmembench non-256 block size FAIL must be labeled DIAGNOSTIC_FAIL")

        # 8. optional variant 取代 original baseline -> INVALID
        if round_val == 'baseline' and variant != 'original':
            rules_checked["Rule_8_variant_replace_baseline"]["status"] = "FAIL"
            rules_checked["Rule_8_variant_replace_baseline"]["notes"].append(f"Row {idx+2}: baseline round must use original variant, not '{variant}'")

        # 9. If speedup < 1.01, set result_type=MEASUREMENT_EQUIVALENT and speedup_claim_valid=false
        if speedup is not None and speedup < 1.01:
            if result_type not in ['MEASUREMENT_EQUIVALENT', 'REGRESSION', 'BASELINE', 'BASELINE_REMEASUREMENT', 'BASELINE_COMPARISON', 'NAIVE_REFERENCE', 'EXISTING_OPTIMIZED_BASELINE']:
                rules_checked["Rule_9_low_speedup_equivalent"]["status"] = "FAIL"
                rules_checked["Rule_9_low_speedup_equivalent"]["notes"].append(f"Row {idx+2}: speedup is {speedup} (< 1.01) but result_type is '{result_type}'")
            if speedup_claim_valid and speedup_claim_valid.lower() == 'true':
                rules_checked["Rule_9_low_speedup_equivalent"]["status"] = "FAIL"
                rules_checked["Rule_9_low_speedup_equivalent"]["notes"].append(f"Row {idx+2}: speedup is {speedup} (< 1.01) but speedup_claim_valid is 'true'")

        # 10. If speedup < 1.0, set result_type=REGRESSION and speedup_claim_valid=false
        if speedup is not None and speedup < 1.0:
            if result_type != 'REGRESSION':
                rules_checked["Rule_10_regression_status"]["status"] = "FAIL"
                rules_checked["Rule_10_regression_status"]["notes"].append(f"Row {idx+2}: speedup is {speedup} (< 1.0) but result_type is '{result_type}'")
            if speedup_claim_valid and speedup_claim_valid.lower() == 'true':
                rules_checked["Rule_10_regression_status"]["status"] = "FAIL"
                rules_checked["Rule_10_regression_status"]["notes"].append(f"Row {idx+2}: speedup is {speedup} (< 1.0) but speedup_claim_valid is 'true'")
                
    # Write contradiction_check.csv
    output_dir = os.path.dirname(csv_path)
    output_path = os.path.join(output_dir, "contradiction_check.csv")
    
    with open(output_path, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["check", "status", "notes"])
        for rule, val in rules_checked.items():
            notes_str = "; ".join(val["notes"]) if val["notes"] else "no issues"
            writer.writerow([rule, val["status"], notes_str])
            
    print(f"Auditor finished. contradiction_check.csv written to {output_path}")
    
    has_fail = any(val["status"] == "FAIL" for val in rules_checked.values())
    return not has_fail

if __name__ == "__main__":
    csv_file = "results.csv"
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    success = run_auditor(csv_file)
    sys.exit(0 if success else 1)
