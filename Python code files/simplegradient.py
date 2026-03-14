"""
Create Similarity Gradient Comparison Figure
This is the KEY figure showing the 8.82pp difference between cross-subject and cross-run transfer
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from Phase 2B (cross-subject) and Phase 3 (cross-run)
fractions = ['20%', '40%', '60%', '80%', '100%']
cross_subject_benefit = [-3.31, -1.10, -1.68, -1.41, -1.10]  # Phase 2B
cross_run_benefit = [5.51, 1.68, -0.09, 3.40, -2.51]  # Phase 3

x = np.arange(len(fractions))
width = 0.35

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Create bars
bars1 = ax.bar(x - width/2, cross_subject_benefit, width,
               label='Cross-Subject (Low Similarity)',
               color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width/2, cross_run_benefit, width,
               label='Cross-Run Within-Subject (Medium Similarity)',
               color='#27ae60', alpha=0.85, edgecolor='black', linewidth=1.2)

# Styling
ax.set_xlabel('Calibration Data Amount', fontsize=14, fontweight='bold')
ax.set_ylabel('Transfer Benefit (percentage points)', fontsize=14, fontweight='bold')
ax.set_title('Transfer Learning Benefit by Source-Target Similarity',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(fractions, fontsize=12)
ax.legend(fontsize=12, loc='upper right')
ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.3)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(-6, 8)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:+.1f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=9, fontweight='bold')

# Annotate the critical 8.82pp difference at 20%
ax.annotate('', xy=(-0.175, -3.31), xytext=(-0.175, 5.51),
            arrowprops=dict(arrowstyle='<->', color='black', lw=2.5))
ax.text(-0.55, 1.1, '8.82pp\ndifference\n(p<0.05)',
        fontsize=11, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

# Add significance markers
ax.text(0 - width/2, -4.5, '*', fontsize=20, ha='center', fontweight='bold', color='red')
ax.text(0 + width/2, 6.5, '†', fontsize=20, ha='center', fontweight='bold', color='green')

# Add legend for significance
ax.text(0.98, 0.02, '* p=0.011 (significant)\n† p=0.089 (marginal)',
        transform=ax.transAxes, fontsize=9,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# Save high-resolution figure
plt.savefig('similarity_gradient_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Created: similarity_gradient_comparison.png")
print("   Resolution: 300 DPI")
print("   This is Figure 9 in the paper - the KEY FINDING!")
plt.show()