import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

matplotlib.use('Agg')

S3DIS_COLORS = np.array([[0,255,0],[0,0,255],[0,255,255],[255,255,0],[255,0,255],[100,100,255],[200,200,100],[170,120,200],[255,0,0],[200,100,100],[10,200,100],[200,200,200],[50,50,50]])/255.0
CLASS_NAMES = ['Ceiling', 'Floor', 'Wall', 'Beam', 'Column', 'Window', 'Door', 'Table', 'Chair', 'Sofa', 'Bookcase', 'Board', 'Clutter']

def generate_viz(xyz, lbls_dict, gt, name, output_dir, sample_n=60000, point_size=20.0):
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
        c[~mask] = [0.8, 0.8, 0.8]
        return c

    stages = list(p_lbls_dict.keys()) + ["Ground Truth"]
    num_rows = len(stages)
    
    fig = plt.figure(figsize=(24, 5 * num_rows))
    gs = gridspec.GridSpec(num_rows, 3, figure=fig, 
                           top=0.98, bottom=0.05, 
                           left=0.10, right=0.98, 
                           hspace=0.1, wspace=0.05)

    for r, stage_name in enumerate(stages):
        if stage_name == "Ground Truth":
            data = p_gt
        else:
            data = p_lbls_dict[stage_name]
            
        c = get_c(data)
        for c_idx in range(3):
            ax = fig.add_subplot(gs[r, c_idx])
            dims = [(0,1), (0,2), (1,2)][c_idx]
            ax.scatter(p_xyz[:,dims[0]], p_xyz[:,dims[1]], c=c, s=point_size, edgecolors='none', alpha=0.8)
            ax.axis('off')
            if c_idx == 0:
                ax.text(-0.10, 0.5, stage_name, transform=ax.transAxes, va='center', ha='right', fontsize=18, rotation=90, fontweight='bold')

    patches = [mpatches.Patch(color=S3DIS_COLORS[i], label=CLASS_NAMES[i]) for i in range(len(CLASS_NAMES))]
    fig.legend(handles=patches, loc='lower center', ncol=7, fontsize=16, bbox_to_anchor=(0.5, 0.01), frameon=False)
    
    plt.savefig(os.path.join(output_dir, f"{name}_Viz.png"), dpi=200, bbox_inches='tight')
    plt.close()