import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

# Ensure headless generation
matplotlib.use('Agg')

# ==============================================================================
# Color Palettes & Class Names Definitions
# ==============================================================================

# S3DIS (Indoor)
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

S3DIS_CLASSES = [
    'Ceiling', 'Floor', 'Wall', 'Beam', 'Column', 'Window', 'Door', 
    'Table', 'Chair', 'Sofa', 'Bookcase', 'Board', 'Clutter'
]

# SensatUrban (Outdoor)
# Colors adapted for urban parsing (High contrast)
SENSAT_COLORS = np.array([
    [85, 107, 47],    # 0: Ground (DarkOliveGreen)
    [0, 255, 0],      # 1: Vegetation (Lime)
    [255, 165, 0],    # 2: Building (Orange)
    [47, 79, 79],     # 3: Wall (DarkSlateGray)
    [123, 104, 238],  # 4: Bridge (MediumSlateBlue)
    [128, 0, 128],    # 5: Parking (Purple)
    [255, 0, 255],    # 6: Rail (Magenta)
    [128, 128, 128],  # 7: Traffic Road (Gray)
    [255, 255, 0],    # 8: Street Furniture (Yellow)
    [255, 0, 0],      # 9: Car (Red)
    [0, 255, 255],    # 10: Footpath (Cyan)
    [0, 128, 128],    # 11: Bike (Teal)
    [0, 0, 255]       # 12: Water (Blue)
]) / 255.0

SENSAT_CLASSES = [
    "Ground", "Vegetation", "Building", "Wall", "Bridge", "Parking", 
    "Rail", "Traffic Road", "Street Furn.", "Car", "Footpath", "Bike", "Water"
]

def get_meta(dataset_name):
    """Retrieve colors and class names by dataset name."""
    name = dataset_name.lower()
    if 's3dis' in name:
        return S3DIS_COLORS, S3DIS_CLASSES
    elif 'sensat' in name:
        return SENSAT_COLORS, SENSAT_CLASSES
    else:
        # Fallback to S3DIS if unknown
        return S3DIS_COLORS, S3DIS_CLASSES

def generate_viz(xyz, lbls_dict, gt, name, output_dir, dataset_name='s3dis', sample_n=60000, point_size=20.0):
    """
    Generate 3-View Visualization dynamically based on dataset.
    """
    # Load correct palette
    colors_palette, class_names = get_meta(dataset_name)
    
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
        # Handle -100 (ignore label)
        mask = (l >= 0) & (l < len(colors_palette))
        c[mask] = colors_palette[l[mask].astype(int)]
        c[~mask] = [0.8, 0.8, 0.8] # Grey for unlabeled/ignored
        return c

    # Prepare stages to plot
    stages = list(p_lbls_dict.keys()) + ["Ground Truth"]
    num_rows = len(stages)
    
    # Setup Figure
    fig = plt.figure(figsize=(24, 5 * num_rows))
    
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
            
            # Add stage name label
            if c_idx == 0:
                ax.text(-0.10, 0.5, stage_name, transform=ax.transAxes, 
                        va='center', ha='right', fontsize=18, rotation=90, fontweight='bold')

    # Create Legend Dynamically
    patches = [mpatches.Patch(color=colors_palette[i], label=class_names[i]) for i in range(len(class_names))]
    
    # Legend placement
    fig.legend(handles=patches, loc='lower center', ncol=min(7, len(class_names)), fontsize=14, 
               bbox_to_anchor=(0.5, 0.02), frameon=False)
    
    # Save
    plt.savefig(os.path.join(output_dir, f"{name}_Viz.png"), dpi=200, bbox_inches='tight')
    plt.close()