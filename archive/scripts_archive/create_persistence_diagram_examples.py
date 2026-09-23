"""
create_persistence_diagram_examples.py

Creates visual examples of persistence diagrams for poster.
Shows Normal Traffic vs Attack Traffic to illustrate topological differences.

Author: Oliver Liu
Date: April 2026
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set style
sns.set_style("whitegrid")

print("Loading data...")

# Load labels
labels = pd.read_csv('outputs/labels.csv')['label']

# Load C2 persistence diagrams
with open('outputs/wasserstein/per_flow_diagrams/c2_all_diagrams.pkl', 'rb') as f:
    c2_diagrams = pickle.load(f)

print(f"Loaded {len(c2_diagrams):,} diagrams")

# Find example indices
normal_idx = np.where(labels == 'Normal Traffic')[0][100]  # Pick 100th normal flow
wormhole_idx = np.where(labels == 'Wormhole Attack')[0][100]  # Pick 100th wormhole
blackhole_idx = np.where(labels == 'Blackhole Attack')[0][100]  # Pick 100th blackhole

print(f"Normal example: flow {normal_idx}")
print(f"Wormhole example: flow {wormhole_idx}")
print(f"Blackhole example: flow {blackhole_idx}")

# Convert GUDHI diagrams to plottable format
def diagram_to_points(diagram, max_edge=5.0):
    """Convert GUDHI persistence diagram to (birth, death) points per dimension."""
    h0_points = []
    h1_points = []
    h2_points = []
    
    for dim, (birth, death) in diagram:
        if death == float('inf'):
            death = max_edge
        
        if dim == 0:
            h0_points.append((birth, death))
        elif dim == 1:
            h1_points.append((birth, death))
        elif dim == 2:
            h2_points.append((birth, death))
    
    return np.array(h0_points), np.array(h1_points), np.array(h2_points)

# Extract points
normal_h0, normal_h1, normal_h2 = diagram_to_points(c2_diagrams[normal_idx])
wormhole_h0, wormhole_h1, wormhole_h2 = diagram_to_points(c2_diagrams[wormhole_idx])
blackhole_h0, blackhole_h1, blackhole_h2 = diagram_to_points(c2_diagrams[blackhole_idx])

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

def plot_diagram(ax, h0, h1, h2, title):
    """Plot persistence diagram with H0, H1, H2."""
    
    # Diagonal line
    ax.plot([0, 5], [0, 5], 'k--', alpha=0.3, linewidth=1, label='Diagonal')
    
    # Plot points
    if len(h0) > 0:
        ax.scatter(h0[:, 0], h0[:, 1], c='blue', s=50, alpha=0.6, label=f'H₀ ({len(h0)})', edgecolors='darkblue')
    if len(h1) > 0:
        ax.scatter(h1[:, 0], h1[:, 1], c='red', s=50, alpha=0.6, label=f'H₁ ({len(h1)})', edgecolors='darkred')
    if len(h2) > 0:
        ax.scatter(h2[:, 0], h2[:, 1], c='green', s=50, alpha=0.6, label=f'H₂ ({len(h2)})', edgecolors='darkgreen')
    
    ax.set_xlabel('Birth', fontsize=12)
    ax.set_ylabel('Death', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')

# Plot all three
plot_diagram(axes[0], normal_h0, normal_h1, normal_h2, 'Normal Traffic')
plot_diagram(axes[1], wormhole_h0, wormhole_h1, wormhole_h2, 'Wormhole Attack')
plot_diagram(axes[2], blackhole_h0, blackhole_h1, blackhole_h2, 'Blackhole Attack')

plt.suptitle('Persistence Diagrams: C2 Manifold (5D)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/figures/persistence_diagram_examples.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/figures/persistence_diagram_examples.png")
plt.close()

# Create barcode visualization
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

def plot_barcode(ax, h0, h1, h2, title):
    """Plot persistence barcode."""
    
    y_offset = 0
    
    # H0 barcodes (blue)
    for i, (birth, death) in enumerate(h0):
        ax.plot([birth, death], [y_offset, y_offset], 'b-', linewidth=3, alpha=0.6)
        y_offset += 1
    
    h0_end = y_offset
    
    # H1 barcodes (red)
    for i, (birth, death) in enumerate(h1):
        ax.plot([birth, death], [y_offset, y_offset], 'r-', linewidth=3, alpha=0.6)
        y_offset += 1
    
    h1_end = y_offset
    
    # H2 barcodes (green)
    for i, (birth, death) in enumerate(h2):
        ax.plot([birth, death], [y_offset, y_offset], 'g-', linewidth=3, alpha=0.6)
        y_offset += 1
    
    # Add dimension labels
    if h0_end > 0:
        ax.text(-0.3, h0_end/2, 'H₀', fontsize=11, fontweight='bold', color='blue', va='center')
    if h1_end > h0_end:
        ax.text(-0.3, (h0_end + h1_end)/2, 'H₁', fontsize=11, fontweight='bold', color='red', va='center')
    if y_offset > h1_end:
        ax.text(-0.3, (h1_end + y_offset)/2, 'H₂', fontsize=11, fontweight='bold', color='green', va='center')
    
    ax.set_xlabel('Filtration Value', fontsize=11)
    ax.set_ylabel('Topological Features', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlim(0, 5)
    ax.set_ylim(-1, y_offset + 1)
    ax.grid(alpha=0.3, axis='x')
    ax.set_yticks([])

plot_barcode(axes[0], normal_h0, normal_h1, normal_h2, 'Normal Traffic Barcode')
plot_barcode(axes[1], wormhole_h0, wormhole_h1, wormhole_h2, 'Wormhole Attack Barcode')
plot_barcode(axes[2], blackhole_h0, blackhole_h1, blackhole_h2, 'Blackhole Attack Barcode')

plt.suptitle('Persistence Barcodes: C2 Manifold', fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('results/figures/persistence_barcode_examples.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/figures/persistence_barcode_examples.png")
plt.close()

print("\n" + "="*80)
print("✓ PERSISTENCE VISUALIZATION COMPLETE")
print("="*80)
print("\nCreated:")
print("  - results/figures/persistence_diagram_examples.png")
print("  - results/figures/persistence_barcode_examples.png")
print("\nThese show the actual topological features TDA extracts!")
print("="*80)
