"""
create_methodology_flowchart.py

Creates a visual flowchart of the complete TDA pipeline for poster.

Author: Oliver Liu
Date: April 2026
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Colors
color_data = '#E3F2FD'
color_manifold = '#FFE0B2'
color_tda = '#C8E6C9'
color_supervised = '#F8BBD0'
color_unsupervised = '#D1C4E9'
color_results = '#FFCCBC'

def add_box(x, y, width, height, text, color, fontsize=10):
    """Add a rounded box with text."""
    box = FancyBboxPatch((x, y), width, height,
                          boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor='black',
                          linewidth=2, alpha=0.9)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center', fontsize=fontsize,
            fontweight='bold', wrap=True)

def add_arrow(x1, y1, x2, y2):
    """Add an arrow between boxes."""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=30,
                           linewidth=2.5, color='black', alpha=0.7)
    ax.add_patch(arrow)

# Title
ax.text(5, 9.5, 'Multi-Manifold Persistent Homology Pipeline', 
        ha='center', fontsize=18, fontweight='bold')

# Row 1: Data Input
add_box(0.5, 8, 2, 0.8, 'UAVIDS-2025\n122,171 Flows\n23 Features', color_data, 11)

# Row 2: Preprocessing
add_arrow(1.5, 8, 1.5, 7.3)
add_box(0.5, 6.5, 2, 0.8, 'Preprocessing\nIP Encoding\nNormalization', color_data, 10)

# Row 3: Manifold Construction
add_arrow(1.5, 6.5, 1, 5.8)
add_arrow(1.5, 6.5, 3, 5.8)
add_arrow(1.5, 6.5, 5, 5.8)

add_box(0.2, 4.5, 1.5, 1.3, 'C2 Manifold\n(5D)\nSrc/Dst\nAddr/Port\n+ Duration', color_manifold, 9)
add_box(2.5, 4.5, 1.5, 1.3, 'Network\nManifold\n(15D)\nPackets, Bytes\nRates, QoS', color_manifold, 9)
add_box(4.8, 4.5, 1.5, 1.3, 'Physical\nManifold\n(2D)\nDelay\nHop Count', color_manifold, 9)

# Row 4: TDA Computation
add_arrow(0.95, 4.5, 0.95, 4)
add_arrow(3.25, 4.5, 3.25, 4)
add_arrow(5.55, 4.5, 5.55, 4)

add_box(0.2, 3, 1.5, 1, 'Vietoris-Rips\nComplex\n(GUDHI)', color_tda, 9)
add_box(2.5, 3, 1.5, 1, 'Vietoris-Rips\nComplex\n(GUDHI)', color_tda, 9)
add_box(4.8, 3, 1.5, 1, 'Vietoris-Rips\nComplex\n(GUDHI)', color_tda, 9)

# Persistence computation
add_arrow(0.95, 3, 0.95, 2.5)
add_arrow(3.25, 3, 3.25, 2.5)
add_arrow(5.55, 3, 5.55, 2.5)

add_box(0.2, 1.5, 1.5, 1, 'Persistence\nDiagrams\nH₀, H₁, H₂', color_tda, 9)
add_box(2.5, 1.5, 1.5, 1, 'Persistence\nDiagrams\nH₀, H₁, H₂', color_tda, 9)
add_box(4.8, 1.5, 1.5, 1, 'Persistence\nDiagrams\nH₀, H₁, H₂', color_tda, 9)

# Split into two approaches
# Supervised path
add_arrow(0.95, 1.5, 1.5, 0.8)
add_arrow(3.25, 1.5, 2.5, 0.8)
add_arrow(5.55, 1.5, 3.5, 0.8)

add_box(1.5, 0.2, 2.5, 0.6, 'SUPERVISED:\nFeature Extraction\n(Statistics from Diagrams)', 
        color_supervised, 10)

add_arrow(2.75, 0.2, 2.75, -0.5)
add_box(1.5, -1.5, 2.5, 1, 'Train Classifiers:\nRandom Forest\nLogistic Regression\nSVM',
        color_supervised, 10)

add_arrow(2.75, -1.5, 2.75, -2.2)
add_box(1.2, -3, 3.2, 0.8, 'Result: 99.91% Accuracy\n(+3.76% improvement)',
        color_results, 11)

# Unsupervised path
add_arrow(0.95, 1.5, 6.5, 0.8)
add_arrow(3.25, 1.5, 7.5, 0.8)
add_arrow(5.55, 1.5, 8.5, 0.8)

add_box(6.5, 0.2, 2.5, 0.6, 'UNSUPERVISED:\nWasserstein Distance\nto Baseline',
        color_unsupervised, 10)

add_arrow(7.75, 0.2, 7.75, -0.5)
add_box(6.5, -1.5, 2.5, 1, 'Z-Score\nNormalization\n3σ Threshold\nDetection',
        color_unsupervised, 10)

add_arrow(7.75, -1.5, 7.75, -2.2)
add_box(6.2, -3, 3.2, 0.8, 'Result: 84.42% AUC\n(No labels needed)',
        color_results, 11)

# Add legend
legend_elements = [
    mpatches.Patch(facecolor=color_data, edgecolor='black', label='Data Processing'),
    mpatches.Patch(facecolor=color_manifold, edgecolor='black', label='Manifold Construction'),
    mpatches.Patch(facecolor=color_tda, edgecolor='black', label='TDA Computation'),
    mpatches.Patch(facecolor=color_supervised, edgecolor='black', label='Supervised Approach'),
    mpatches.Patch(facecolor=color_unsupervised, edgecolor='black', label='Unsupervised Approach'),
    mpatches.Patch(facecolor=color_results, edgecolor='black', label='Results')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.savefig('results/figures/methodology_flowchart.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/figures/methodology_flowchart.png")
plt.close()

print("\n" + "="*80)
print("✓ METHODOLOGY FLOWCHART COMPLETE")
print("="*80)
print("\nCreated complete visual pipeline from data → results")
print("Shows both supervised and unsupervised approaches side-by-side")
print("="*80)
