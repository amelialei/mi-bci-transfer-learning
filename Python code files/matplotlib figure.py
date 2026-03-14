import matplotlib.pyplot as plt
import numpy as np

data_fractions = [20, 40, 60, 80, 100]

# Cross-subject (Eric)
cross_subj_benefit = [-3.31, -1.10, -1.68, -1.41, -1.10]
cross_subj_std = [2.5, 2.0, 2.0, 2.0, 2.0]  # Approximate

# Cross-run (yours)
cross_run_benefit = [5.51, 1.68, -0.09, 3.40, -2.51]
cross_run_std = [3.0, 2.5, 2.5, 2.8, 3.0]  # From analysis

plt.figure(figsize=(8, 5))
plt.plot(data_fractions, cross_subj_benefit, 'ro-', label='Cross-Subject', linewidth=2)
plt.plot(data_fractions, cross_run_benefit, 'bs--', label='Cross-Run (Within-Subject)', linewidth=2)
plt.errorbar(data_fractions, cross_subj_benefit, yerr=cross_subj_std, fmt='none', ecolor='red', alpha=0.3)
plt.errorbar(data_fractions, cross_run_benefit, yerr=cross_run_std, fmt='none', ecolor='blue', alpha=0.3)

plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
plt.xlabel('Calibration Data (%)', fontsize=12)
plt.ylabel('Transfer Learning Benefit (pp)', fontsize=12)
plt.title('Within-Subject vs Cross-Subject Transfer Learning', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Figure_CrossSubject_vs_CrossRun.png', dpi=300)
plt.show()