import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

# Ensure headless generation
matplotlib.use('Agg')

# S3DIS Color Palette
S3DIS_COLORS = np.array([
    [0, 255, 0],     # Ceiling
    [0, 0, 255],     # Floor
    [0, 255, 255],   # Wall
    [255, 255, 0],   # Beam
    [255, 0, 255],   # Column
    [100, 100, 255], # Window
    [200, 200, 100], # Door
    [170, 120, 200], # Table
    [255, 0, 0],     # Chair
    [200, 100, 100], # Sofa
    [10, 200, 100],  # Bookcase
    [200, 200, 200], # Board
    [50, 50, 50]     # Clutter
]) / 255.0

CLASS_NAMES = [
    'Ceiling', 'Floor', 'Wall', 'Beam', 'Column', 'Window', 'Door', 
    'Table', 'Chair', 'Sofa', 'Bookcase', 'Board', 'Clutter'
]

def generate_viz(xyz, lbls_dict, gt, name, output_dir, sample_n=60000, point_size=20.0):
    """
    Generate 3-View Visualization using GridSpec (Original Repository Style).
    Fixed: Adjusted margins to prevent legend overlap.
    """
    # Downsample for performance
    if len(xyz) > sample_n:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(xyz), sample_n, replace=False)
        p_xyz, p_gt = xyz[idx], gt[idx]
        p_lbls_dict = {k: v[idx] for k, v in lbls_dict.items()}
    else:
        p_xyz, p_gt = xyz, gt
        p_lbls_dict = lbls_dict

    def get_c(l):
        c = np.zeros((len(l), 3))
        mask = l != -100
        c[mask] = S3DIS_COLORS[l[mask].astype(int)]
        c[~mask] = [0.8, 0.8, 0.8] # Grey for unlabeled
        return c

    # Prepare stages to plot
    stages = list(p_lbls_dict.keys()) + ["Ground Truth"]
    num_rows = len(stages)
    
    # Setup Figure
    fig = plt.figure(figsize=(24, 5 * num_rows))
    
    # [FIX] Increased bottom margin from 0.05 to 0.12 to accommodate the legend
    gs = gridspec.GridSpec(num_rows, 3, figure=fig, 
                           top=0.98, bottom=0.12, 
                           left=0.10, right=0.98, 
                           hspace=0.1, wspace=0.05)

    for r, stage_name in enumerate(stages):
        if stage_name == "Ground Truth":
            data = p_gt
        else:
            data = p_lbls_dict[stage_name]
            
        c = get_c(data)
        for c_idx in range(3): # 3 Views: XY, XZ, YZ
            ax = fig.add_subplot(gs[r, c_idx])
            dims = [(0,1), (0,2), (1,2)][c_idx]
            
            # Plot scatter
            ax.scatter(p_xyz[:,dims[0]], p_xyz[:,dims[1]], c=c, s=point_size, edgecolors='none', alpha=0.8)
            ax.axis('off')
            
            # Add stage name label on the left of the first column
            if c_idx == 0:
                ax.text(-0.10, 0.5, stage_name, transform=ax.transAxes, 
                        va='center', ha='right', fontsize=18, rotation=90, fontweight='bold')

    # Create Legend
    patches = [mpatches.Patch(color=S3DIS_COLORS[i], label=CLASS_NAMES[i]) for i in range(len(CLASS_NAMES))]
    
    # Legend placement: Lower center, in the reserved bottom margin
    fig.legend(handles=patches, loc='lower center', ncol=7, fontsize=16, 
               bbox_to_anchor=(0.5, 0.02), frameon=False)
    
    # Save
    plt.savefig(os.path.join(output_dir, f"{name}_Viz.png"), dpi=200, bbox_inches='tight')
    plt.close()