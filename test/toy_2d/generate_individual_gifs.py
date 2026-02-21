"""
Generate individual GIFs for each seed.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
sys.path.insert(0, os.path.dirname(__file__))

from generate_combined_gif import run_seed_and_capture_frames, plot_frame_for_seed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

seeds = [36, 56, 61, 62, 84, 153]
budget = 32
frame_step = 4
output_base = 'active_learning/images/toy_2d'

GT_FILL = '#87CEEB'
GT_BORDER = '#00CED1'
PRED_FILL = '#FFB366'
PRED_BORDER = '#FF6600'

for seed in seeds:
    print(f'\n=== Seed {seed} ===')
    output_dir = f'{output_base}/seed_{seed}_32iter'
    os.makedirs(output_dir, exist_ok=True)

    # Run and capture frames
    frames_data = run_seed_and_capture_frames(seed, budget, particles=103)

    # Save frames at intervals
    iterations = list(range(0, budget + 1, frame_step))
    frame_paths = []

    for it in iterations:
        fig, ax = plt.subplots(figsize=(5, 5))
        plot_frame_for_seed(frames_data[it], it, ax, seed)

        # Add legend
        legend_elements = [
            mpatches.Patch(facecolor=GT_FILL, edgecolor=GT_BORDER, linewidth=2, alpha=0.5, label='Ground Truth'),
            mpatches.Patch(facecolor=PRED_FILL, edgecolor=PRED_BORDER, linewidth=2, linestyle='--', alpha=0.4, label='Predicted'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.8, edgecolor='gray')
        ax.set_title(f'Seed {seed} - Iteration {it}', fontsize=12, fontweight='bold')

        save_path = f'{output_dir}/iteration_{it:02d}.png'
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        frame_paths.append(save_path)
        print(f'  Saved: {save_path}')

    # Create GIF
    frames = [Image.open(p) for p in frame_paths]
    durations = [500] * len(frames)
    durations[-1] = 2000

    gif_path = f'{output_dir}/animation.gif'
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    print(f'  GIF saved: {gif_path}')

print('\nDone!')
