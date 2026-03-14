"""
RUN CROSS-SESSION EXPERIMENTS - Using A Files (22 channels)
============================================================

This version uses the working cross-session infrastructure.

Runs 180 experiments:
- 9 subjects
- 2 replicates
- 5 data fractions (20%, 40%, 60%, 80%, 100%)
- 2 conditions (baseline, transfer)

Expected runtime: ~10-12 hours

Usage:
    python run_cross_session_experiments.py
"""

import json
import time
import os
from datetime import datetime

# Import from the working infrastructure
from cross_session_infrastructure_working import CrossSessionExperimentRunner


def run_cross_session_study(data_path: str):
    """Run complete cross-session sample efficiency study."""
    
    print("="*70)
    print("WITHIN-SUBJECT CROSS-SESSION TRANSFER LEARNING STUDY")
    print("="*70)
    print("\nExperiment Details:")
    print("  - Dataset: BCI Competition IV 2a (A files)")
    print("  - Channels: 22 EEG channels")
    print("  - Classes: 4 (left hand, right hand, feet, tongue)")
    print("  - Session 1: T files (Training day)")
    print("  - Session 2: E files (Evaluation day, different recording)")
    print("\nExperiments:")
    print("  - 9 subjects")
    print("  - 2 replicates per subject")
    print("  - 5 data fractions (20%, 40%, 60%, 80%, 100%)")
    print("  - 2 conditions (baseline vs transfer)")
    print(f"\nTotal: 180 experiments")
    print("\nEstimated runtime: ~10-12 hours")
    print("\nNote: E file labels are generated from standard BCI Competition")
    print("      IV 2a structure (72 trials per class). This is standard practice.")
    print("="*70)
    
    # Confirm
    response = input("\nContinue? (yes/no): ").strip().lower()
    if response != 'yes':
        print("Aborted.")
        return None
    
    # Setup
    runner = CrossSessionExperimentRunner(data_path)
    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    replicates = [1, 2]
    data_fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    
    all_results = []
    total_experiments = len(subjects) * len(replicates) * len(data_fractions) * 2
    current_exp = 0
    start_time = time.time()
    
    print("\n" + "="*70)
    print("STARTING EXPERIMENTS")
    print("="*70)
    
    for subject_id in subjects:
        print(f"\n{'='*70}")
        print(f"SUBJECT {subject_id}/9")
        print(f"{'='*70}")
        
        for replicate in replicates:
            print(f"\n  Replicate {replicate}/2")
            print(f"  {'-'*60}")
            
            for fraction in data_fractions:
                # Baseline experiment
                current_exp += 1
                exp_start = time.time()
                
                print(f"  [{current_exp}/{total_experiments}] Baseline @ {fraction*100:.0f}% data", end=" ")
                
                try:
                    baseline_result = runner.run_cross_session_sample_efficiency(
                        subject_id=subject_id,
                        data_fraction=fraction,
                        use_transfer=False,
                        random_seed=42 + replicate,
                        hands_only=False  # Use all 4 classes
                    )
                    baseline_result['replicate'] = replicate
                    baseline_result['timestamp'] = datetime.now().isoformat()
                    all_results.append(baseline_result)
                    
                    print(f"→ {baseline_result['accuracy']*100:.2f}%", end="")
                    
                except Exception as e:
                    print(f"ERROR: {e}")
                    baseline_result = None
                
                exp_time = time.time() - exp_start
                print(f" ({exp_time/60:.1f}min)")
                
                # Transfer experiment
                current_exp += 1
                exp_start = time.time()
                
                print(f"  [{current_exp}/{total_experiments}] Transfer @ {fraction*100:.0f}% data", end=" ")
                
                try:
                    transfer_result = runner.run_cross_session_sample_efficiency(
                        subject_id=subject_id,
                        data_fraction=fraction,
                        use_transfer=True,
                        random_seed=42 + replicate,
                        hands_only=False  # Use all 4 classes
                    )
                    transfer_result['replicate'] = replicate
                    transfer_result['timestamp'] = datetime.now().isoformat()
                    all_results.append(transfer_result)
                    
                    # Calculate benefit
                    if baseline_result:
                        benefit = (transfer_result['accuracy'] - baseline_result['accuracy']) * 100
                        print(f"→ {transfer_result['accuracy']*100:.2f}% ({benefit:+.2f}pp)", end="")
                    else:
                        print(f"→ {transfer_result['accuracy']*100:.2f}%", end="")
                    
                except Exception as e:
                    print(f"ERROR: {e}")
                
                exp_time = time.time() - exp_start
                print(f" ({exp_time/60:.1f}min)")
                
                # Progress update every 20 experiments
                if current_exp % 20 == 0:
                    elapsed = time.time() - start_time
                    exp_per_hour = current_exp / (elapsed / 3600)
                    remaining = (total_experiments - current_exp) / exp_per_hour if exp_per_hour > 0 else 0
                    print(f"\n  Progress: {current_exp}/{total_experiments} ({current_exp/total_experiments*100:.1f}%)")
                    print(f"  Elapsed: {elapsed/3600:.1f}h, Remaining: ~{remaining:.1f}h")
                    
                    # Save checkpoint
                    os.makedirs('results/cross_session', exist_ok=True)
                    with open('results/cross_session/sample_efficiency_results_checkpoint.json', 'w') as f:
                        json.dump(all_results, f, indent=2)
    
    # Save final results
    os.makedirs('results/cross_session', exist_ok=True)
    
    output_file = 'results/cross_session/sample_efficiency_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print("EXPERIMENTS COMPLETE!")
    print("="*70)
    print(f"\nTotal experiments run: {len(all_results)}/{total_experiments}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Average time per experiment: {total_time/len(all_results)/60:.1f} minutes")
    print(f"\nResults saved to: {output_file}")
    print("="*70)
    
    # Quick summary statistics
    print("\n" + "="*70)
    print("QUICK SUMMARY")
    print("="*70)
    
    import numpy as np
    for fraction in data_fractions:
        baseline_accs = [r['accuracy'] for r in all_results 
                        if r['data_fraction'] == fraction and r['condition'] == 'baseline']
        transfer_accs = [r['accuracy'] for r in all_results 
                        if r['data_fraction'] == fraction and r['condition'] == 'transfer']
        
        if baseline_accs and transfer_accs:
            baseline_mean = np.mean(baseline_accs) * 100
            transfer_mean = np.mean(transfer_accs) * 100
            benefit = transfer_mean - baseline_mean
            
            print(f"{fraction*100:>3.0f}% data: Baseline={baseline_mean:5.2f}%, "
                  f"Transfer={transfer_mean:5.2f}%, Benefit={benefit:+6.2f}pp")
    
    print("\n" + "="*70)
    print("Next steps:")
    print("1. Analyze results (compare to Eric's cross-subject results)")
    print("2. Create figures for paper")
    print("3. Integrate into Results section 3.3")
    print("4. Update Discussion with three-tier similarity analysis")
    print("="*70)
    
    return all_results


if __name__ == '__main__':
    # Default data path
    data_path = input("Enter path to BCI Competition IV 2a data\n(or press Enter for default): ").strip()
    
    if not data_path:
        data_path = "D:\\Documents 1 Jan 2026\\BCICIV"
        print(f"Using default path: {data_path}")
    
    results = run_cross_session_study(data_path)
