"""
Analyze embeddings and generate all figures for the hypothesis hivemind experiment.

Figures:
  1A: Inter-model heatmap for Task 1 (convergence baseline)
  1B: Inter-model heatmap for Task 2 (diversity desired)
  2A: Intra-model similarity for Task 1
  2B: Intra-model similarity for Task 2
  3A: KDE same-paper vs different-paper for Task 1
  3B: KDE same-paper vs different-paper for Task 2
"""
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import gaussian_kde

OUTPUT_DIR = Path("/workspace/data/outputs")
RESULTS_DIR = Path("/workspace/results")
RESULTS_DIR.mkdir(exist_ok=True)

MODELS = ['Haiku 4.5', 'Sonnet 4.5', 'Sonnet 4.6', 'GPT-5 Nano', 'GPT-5 Mini', 'GPT-5']
MODEL_DISPLAY = {
    'Haiku 4.5': 'Claude Haiku 4.5',
    'Sonnet 4.5': 'Claude Sonnet 4.5',
    'Sonnet 4.6': 'Claude Sonnet 4.6',
    'GPT-5 Nano': 'GPT-5 Nano',
    'GPT-5 Mini': 'GPT-5 Mini',
    'GPT-5': 'GPT-5',
}


def load_embeddings(task_name):
    """Load embeddings and metadata for a task."""
    data = np.load(OUTPUT_DIR / f"{task_name}_embeddings.npz", allow_pickle=True)
    return {
        'embeddings': data['embeddings'],
        'paper_ids': data['paper_ids'],
        'model_names': data['model_names'],
        'sample_idxs': data['sample_idxs'],
    }


def compute_inter_model_similarity(data):
    """
    Compute average cosine similarity between all pairs of models.
    For each pair (model_i, model_j), for each paper, compute average cosine similarity
    between all 10 samples from model_i and all 10 samples from model_j.
    Then average across papers.
    """
    embeddings = data['embeddings']
    paper_ids = data['paper_ids']
    model_names = data['model_names']
    
    unique_papers = sorted(set(paper_ids))
    n_models = len(MODELS)
    
    sim_matrix = np.zeros((n_models, n_models))
    
    for pi, paper in enumerate(unique_papers):
        paper_mask = paper_ids == paper
        
        for i, model_i in enumerate(MODELS):
            mask_i = paper_mask & (model_names == model_i)
            emb_i = embeddings[mask_i]
            
            for j, model_j in enumerate(MODELS):
                mask_j = paper_mask & (model_names == model_j)
                emb_j = embeddings[mask_j]
                
                if len(emb_i) > 0 and len(emb_j) > 0:
                    # Compute pairwise cosine similarities
                    cos_sim = cosine_similarity(emb_i, emb_j)
                    
                    if i == j:
                        # For same model, exclude diagonal (self-similarity)
                        n = cos_sim.shape[0]
                        if n > 1:
                            mask = ~np.eye(n, dtype=bool)
                            sim_matrix[i, j] += cos_sim[mask].mean()
                        else:
                            sim_matrix[i, j] += 1.0
                    else:
                        sim_matrix[i, j] += cos_sim.mean()
    
    sim_matrix /= len(unique_papers)
    return sim_matrix


def compute_intra_model_similarity(data):
    """
    Compute intra-model similarity: for each model, for each paper,
    compute average pairwise cosine similarity among the 10 samples.
    Return per-model distributions across papers.
    """
    embeddings = data['embeddings']
    paper_ids = data['paper_ids']
    model_names = data['model_names']
    
    unique_papers = sorted(set(paper_ids))
    
    results = {}
    for model in MODELS:
        paper_sims = []
        for paper in unique_papers:
            mask = (paper_ids == paper) & (model_names == model)
            emb = embeddings[mask]
            if len(emb) > 1:
                cos_sim = cosine_similarity(emb)
                n = cos_sim.shape[0]
                # Upper triangle (exclude diagonal)
                triu_idx = np.triu_indices(n, k=1)
                paper_sims.append(cos_sim[triu_idx].mean())
        results[model] = paper_sims
    
    return results


def compute_same_diff_paper_similarities(data):
    """
    Compute cosine similarities for:
    - Same paper: all pairs of outputs (any model) for the same paper
    - Different paper: all pairs of outputs (any model) for different papers
    
    To keep computation tractable for different-paper, we sample.
    """
    embeddings = data['embeddings']
    paper_ids = data['paper_ids']
    
    unique_papers = sorted(set(paper_ids))
    
    # Same paper similarities
    same_paper_sims = []
    for paper in unique_papers:
        mask = paper_ids == paper
        emb = embeddings[mask]
        if len(emb) > 1:
            cos_sim = cosine_similarity(emb)
            n = cos_sim.shape[0]
            triu_idx = np.triu_indices(n, k=1)
            same_paper_sims.extend(cos_sim[triu_idx].tolist())
    
    # Different paper similarities - sample for tractability
    rng = np.random.RandomState(42)
    diff_paper_sims = []
    n_samples = min(len(same_paper_sims) * 2, 500000)
    
    # Group embeddings by paper
    paper_embs = {}
    for paper in unique_papers:
        mask = paper_ids == paper
        paper_embs[paper] = embeddings[mask]
    
    paper_list = list(paper_embs.keys())
    for _ in range(n_samples):
        p1, p2 = rng.choice(len(paper_list), 2, replace=False)
        emb1 = paper_embs[paper_list[p1]]
        emb2 = paper_embs[paper_list[p2]]
        i1 = rng.randint(len(emb1))
        i2 = rng.randint(len(emb2))
        sim = cosine_similarity(emb1[i1:i1+1], emb2[i2:i2+1])[0, 0]
        diff_paper_sims.append(sim)
    
    return same_paper_sims, diff_paper_sims


def plot_heatmap(sim_matrix, title, filename):
    """Plot inter-model similarity heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6.5))
    
    display_names = [MODEL_DISPLAY[m] for m in MODELS]
    
    im = ax.imshow(sim_matrix, cmap='YlOrRd', vmin=0.4, vmax=1.0, aspect='equal')
    
    ax.set_xticks(range(len(MODELS)))
    ax.set_yticks(range(len(MODELS)))
    ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(display_names, fontsize=10)
    
    # Add text annotations
    for i in range(len(MODELS)):
        for j in range(len(MODELS)):
            text = f'{sim_matrix[i, j]:.2f}'
            color = 'white' if sim_matrix[i, j] > 0.75 else 'black'
            ax.text(j, i, text, ha='center', va='center', fontsize=11, color=color, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Average Cosine Similarity', fontsize=11)
    
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    
    # Add provider grouping lines
    ax.axhline(y=2.5, color='black', linewidth=2)
    ax.axvline(x=2.5, color='black', linewidth=2)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")


def plot_intra_model(results_task1, results_task2):
    """Plot intra-model similarity box/violin plots for both tasks."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    display_names = [MODEL_DISPLAY[m] for m in MODELS]
    
    for ax, results, title in zip(axes, [results_task1, results_task2],
                                   ['(a) Recover underlying hypothesis', '(b) Generate novel hypothesis']):
        data_list = [results[m] for m in MODELS]
        
        bp = ax.boxplot(data_list, tick_labels=display_names, patch_artist=True,
                       medianprops=dict(color='black', linewidth=2))
        
        # Color by provider
        colors = ['#3498db', '#3498db', '#3498db', '#e74c3c', '#e74c3c', '#e74c3c']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_ylabel('Cosine Similarity', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0.3, 1.0)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'figure2_intra_model_similarity.pdf', dpi=150, bbox_inches='tight')
    plt.savefig(RESULTS_DIR / 'figure2_intra_model_similarity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved figure2_intra_model_similarity")


def plot_kde(same_sims_t1, diff_sims_t1, same_sims_t2, diff_sims_t2):
    """Plot KDE of same-paper vs different-paper cosine similarities."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, same_sims, diff_sims, title in zip(
        axes,
        [same_sims_t1, same_sims_t2],
        [diff_sims_t1, diff_sims_t2],
        ['(a) Recover underlying hypothesis', '(b) Generate novel hypothesis']
    ):
        x = np.linspace(-0.2, 1.0, 500)
        
        kde_same = gaussian_kde(same_sims)
        kde_diff = gaussian_kde(diff_sims)
        
        ax.fill_between(x, kde_same(x), alpha=0.4, color='#e74c3c', label='Same paper')
        ax.fill_between(x, kde_diff(x), alpha=0.4, color='#3498db', label='Different papers')
        ax.plot(x, kde_same(x), color='#e74c3c', linewidth=2)
        ax.plot(x, kde_diff(x), color='#3498db', linewidth=2)
        
        ax.set_xlabel('Cosine Similarity', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'figure3_kde_distributions.pdf', dpi=150, bbox_inches='tight')
    plt.savefig(RESULTS_DIR / 'figure3_kde_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved figure3_kde_distributions")


def save_metrics(sim_t1, sim_t2, intra_t1, intra_t2, same_t1, diff_t1, same_t2, diff_t2):
    """Save numerical metrics to a JSON file."""
    metrics = {
        'task1_inter_model_similarity': {
            'matrix': sim_t1.tolist(),
            'models': MODELS,
            'mean': float(sim_t1.mean()),
            'mean_off_diagonal': float(sim_t1[~np.eye(6, dtype=bool)].mean()),
        },
        'task2_inter_model_similarity': {
            'matrix': sim_t2.tolist(),
            'models': MODELS,
            'mean': float(sim_t2.mean()),
            'mean_off_diagonal': float(sim_t2[~np.eye(6, dtype=bool)].mean()),
        },
        'task1_intra_model': {m: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for m, v in intra_t1.items()},
        'task2_intra_model': {m: {'mean': float(np.mean(v)), 'std': float(np.std(v))} for m, v in intra_t2.items()},
        'task1_same_paper_mean': float(np.mean(same_t1)),
        'task1_diff_paper_mean': float(np.mean(diff_t1)),
        'task2_same_paper_mean': float(np.mean(same_t2)),
        'task2_diff_paper_mean': float(np.mean(diff_t2)),
    }
    
    with open(RESULTS_DIR / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics.json")
    return metrics


def main():
    print("Loading embeddings...")
    data_t1 = load_embeddings('task1')
    data_t2 = load_embeddings('task2')
    
    print(f"Task 1: {len(data_t1['embeddings'])} embeddings")
    print(f"Task 2: {len(data_t2['embeddings'])} embeddings")
    
    # Figure 1: Inter-model heatmaps
    print("\nComputing inter-model similarities...")
    sim_t1 = compute_inter_model_similarity(data_t1)
    sim_t2 = compute_inter_model_similarity(data_t2)
    
    print("\nTask 1 inter-model similarity matrix:")
    print(np.array2string(sim_t1, precision=3))
    print(f"Mean off-diagonal: {sim_t1[~np.eye(6, dtype=bool)].mean():.3f}")
    
    print("\nTask 2 inter-model similarity matrix:")
    print(np.array2string(sim_t2, precision=3))
    print(f"Mean off-diagonal: {sim_t2[~np.eye(6, dtype=bool)].mean():.3f}")
    
    plot_heatmap(sim_t1, '(A) Convergence desired: Recover underlying hypothesis', 'figure1a_heatmap_task1.pdf')
    plot_heatmap(sim_t1, '(A) Convergence desired: Recover underlying hypothesis', 'figure1a_heatmap_task1.png')
    plot_heatmap(sim_t2, '(B) Diversity desired: Generate novel hypothesis', 'figure1b_heatmap_task2.pdf')
    plot_heatmap(sim_t2, '(B) Diversity desired: Generate novel hypothesis', 'figure1b_heatmap_task2.png')
    
    # Figure 2: Intra-model similarities
    print("\nComputing intra-model similarities...")
    intra_t1 = compute_intra_model_similarity(data_t1)
    intra_t2 = compute_intra_model_similarity(data_t2)
    
    for task_name, intra in [('Task 1', intra_t1), ('Task 2', intra_t2)]:
        print(f"\n{task_name} intra-model similarities:")
        for m in MODELS:
            vals = intra[m]
            print(f"  {m}: mean={np.mean(vals):.3f}, std={np.std(vals):.3f}")
    
    plot_intra_model(intra_t1, intra_t2)
    
    # Figure 3: KDE distributions
    print("\nComputing same/different paper similarities...")
    same_t1, diff_t1 = compute_same_diff_paper_similarities(data_t1)
    same_t2, diff_t2 = compute_same_diff_paper_similarities(data_t2)
    
    print(f"Task 1: same-paper mean={np.mean(same_t1):.3f}, diff-paper mean={np.mean(diff_t1):.3f}")
    print(f"Task 2: same-paper mean={np.mean(same_t2):.3f}, diff-paper mean={np.mean(diff_t2):.3f}")
    
    plot_kde(same_t1, diff_t1, same_t2, diff_t2)
    
    # Save metrics
    metrics = save_metrics(sim_t1, sim_t2, intra_t1, intra_t2, same_t1, diff_t1, same_t2, diff_t2)
    
    print("\n=== SUMMARY ===")
    print(f"Task 1 (convergence baseline) inter-model mean: {metrics['task1_inter_model_similarity']['mean_off_diagonal']:.3f}")
    print(f"Task 2 (diversity desired) inter-model mean: {metrics['task2_inter_model_similarity']['mean_off_diagonal']:.3f}")
    print(f"Key finding: Inter-model similarity remains {'high' if metrics['task2_inter_model_similarity']['mean_off_diagonal'] > 0.5 else 'low'} even for diversity task")


if __name__ == "__main__":
    main()
