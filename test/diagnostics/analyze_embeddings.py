"""Eigenspectrum analysis of training embeddings for covariance prior feasibility."""
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from active_learning.src.utils import load_decoder_model
from active_learning.src.config import DEVICE


def analyze_eigenspectrum(embeddings: torch.Tensor):
    """Compute eigenvalues and eigenvectors via SVD on centered embeddings.

    Returns:
        eigenvalues: Variance explained per component (descending)
        cumulative: Cumulative fraction of variance explained
        Vh: Principal directions (rows of Vh)
    """
    centered = embeddings - embeddings.mean(dim=0)
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    eigenvalues = (S ** 2) / (len(embeddings) - 1)  # unbiased covariance eigenvalues
    cumulative = eigenvalues.cumsum(0) / eigenvalues.sum()
    return eigenvalues, cumulative, Vh


def print_summary(eigenvalues, cumulative):
    """Print eigenvalue table and key thresholds."""
    n = len(eigenvalues)
    ev = eigenvalues.cpu().numpy()
    cv = cumulative.cpu().numpy()

    print(f"\n{'='*60}")
    print(f"  Eigenspectrum Summary  ({n} components)")
    print(f"{'='*60}")
    print(f"  {'PC':<5} {'Eigenvalue':>12} {'Var %':>8} {'Cumul %':>9}")
    print(f"  {'-'*5} {'-'*12} {'-'*8} {'-'*9}")
    for i in range(n):
        var_pct = 100.0 * ev[i] / ev.sum()
        cum_pct = 100.0 * cv[i]
        print(f"  {i+1:<5d} {ev[i]:12.6f} {var_pct:7.2f}% {cum_pct:8.2f}%")

    print(f"\n  Key thresholds:")
    for threshold in [0.90, 0.95, 0.99]:
        k = int((cv >= threshold).nonzero()[0][0]) + 1 if (cv >= threshold).any() else n
        print(f"    {threshold*100:.0f}% variance: {k} components")

    cond = ev[0] / ev[-1] if ev[-1] > 0 else float('inf')
    print(f"\n  Condition number (λ_max/λ_min): {cond:.2f}")
    print(f"  Effective dimensionality (90%): "
          f"{int((cv >= 0.90).nonzero()[0][0]) + 1 if (cv >= 0.90).any() else n}")
    print(f"{'='*60}\n")


def plot_eigenspectrum(eigenvalues, cumulative, save_dir):
    """Bar chart of eigenvalues (log scale) with cumulative variance line."""
    ev = eigenvalues.cpu().numpy()
    cv = cumulative.cpu().numpy()
    n = len(ev)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(range(1, n + 1), ev, color='steelblue', alpha=0.8, label='Eigenvalue')
    ax1.set_yscale('log')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Eigenvalue (log scale)')
    ax1.set_xticks(range(1, n + 1))

    ax2 = ax1.twinx()
    ax2.plot(range(1, n + 1), 100 * cv, color='firebrick', marker='o', markersize=4,
             linewidth=2, label='Cumulative %')
    ax2.set_ylabel('Cumulative Variance Explained (%)')
    ax2.set_ylim(0, 105)

    # Threshold lines
    for thresh, ls in [(90, '--'), (95, ':'), (99, '-.')]:
        ax2.axhline(thresh, color='gray', linestyle=ls, alpha=0.5, linewidth=0.8)
        ax2.text(n + 0.3, thresh, f'{thresh}%', fontsize=8, va='center', color='gray')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    ax1.set_title('Eigenspectrum of Training Embeddings')
    fig.tight_layout()
    fig.savefig(save_dir / 'eigenspectrum.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {save_dir / 'eigenspectrum.png'}")


def plot_pca_2d(embeddings, Vh, save_dir):
    """PC1 vs PC2 scatter of training embeddings."""
    centered = embeddings - embeddings.mean(dim=0)
    proj = (centered @ Vh[:2].T).cpu().numpy()

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=np.arange(len(proj)),
                    cmap='viridis', s=20, alpha=0.8, edgecolors='none')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Training Embeddings — PC1 vs PC2')
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Sample Index')
    ax.set_aspect('equal', adjustable='datalim')
    fig.tight_layout()
    fig.savefig(save_dir / 'pca_2d_scatter.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {save_dir / 'pca_2d_scatter.png'}")


def plot_pca_pairwise(embeddings, Vh, save_dir, n_components=4):
    """Pairwise scatter of top-k PCs (n_components x n_components grid)."""
    centered = embeddings - embeddings.mean(dim=0)
    proj = (centered @ Vh[:n_components].T).cpu().numpy()
    idx = np.arange(len(proj))

    fig, axes = plt.subplots(n_components, n_components, figsize=(12, 12))
    for i in range(n_components):
        for j in range(n_components):
            ax = axes[i, j]
            if i == j:
                ax.hist(proj[:, i], bins=30, color='steelblue', alpha=0.7)
                ax.set_ylabel('Count' if j == 0 else '')
            else:
                ax.scatter(proj[:, j], proj[:, i], c=idx, cmap='viridis',
                           s=8, alpha=0.6, edgecolors='none')
            if i == n_components - 1:
                ax.set_xlabel(f'PC{j+1}')
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(f'PC{i+1}')
            else:
                ax.set_yticklabels([])

    fig.suptitle('Pairwise PCA Scatter (Top 4 Components)', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_dir / 'pca_pairwise.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {save_dir / 'pca_pairwise.png'}")


def main():
    parser = argparse.ArgumentParser(
        description='Eigenspectrum analysis of training embeddings')
    parser.add_argument('--model', type=str, default='models/best_model.pt',
                        help='Path to model checkpoint')
    args = parser.parse_args()

    decoder, embeddings, _ = load_decoder_model(args.model, DEVICE)
    embeddings = embeddings.detach().cpu().float()

    eigenvalues, cumulative, Vh = analyze_eigenspectrum(embeddings)
    print_summary(eigenvalues, cumulative)

    save_dir = Path('active_learning/images/diagnostics/latent')
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Generating plots...")
    plot_eigenspectrum(eigenvalues, cumulative, save_dir)
    plot_pca_2d(embeddings, Vh, save_dir)
    plot_pca_pairwise(embeddings, Vh, save_dir)
    print("Done.")


if __name__ == '__main__':
    main()
