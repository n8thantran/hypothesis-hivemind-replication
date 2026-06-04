"""Analysis and visualization for the Hypothesis Hivemind experiment."""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from itertools import combinations
from scipy.stats import gaussian_kde
from config import MODEL_ORDER, PAPER_URLS

PAPER_IDS = [url.split("id=")[1] for url in PAPER_URLS]

# Output directories
os.makedirs("results", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)

NUM_SAMPLES = 10


def load_embeddings(filepath):
    """Load embeddings from JSON file and organize by model and paper."""
    with open(filepath) as f:
        data = json.load(f)
    
    # Organize: embeddings[model_name][paper_id] = list of 10 embedding vectors
    embeddings = {}
    for key, emb in data.items():
        parts = key.split("|")
        model_name = parts[0]
        paper_id = parts[1]
        sample_idx = int(parts[2])
        
        if model_name not in embeddings:
            embeddings[model_name] = {}
        if paper_id not in embeddings[model_name]:
            embeddings[model_name][paper_id] = [None] * NUM_SAMPLES
        embeddings[model_name][paper_id][sample_idx] = np.array(emb)
    
    return embeddings


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def compute_inter_model_similarity(embeddings):
    """Compute average inter-model cosine similarity matrix.
    
    For each pair of models (i, j), for each paper:
    - Compute mean cosine similarity across all 10×10 pairs
    - Average across all papers
    Returns 6×6 matrix.
    """
    n_models = len(MODEL_ORDER)
    sim_matrix = np.zeros((n_models, n_models))
    
    for i, model_i in enumerate(MODEL_ORDER):
        for j, model_j in enumerate(MODEL_ORDER):
            if i == j:
                # Self-similarity: compute intra-model
                paper_sims = []
                for paper_id in PAPER_IDS:
                    embs = embeddings[model_i][paper_id]
                    sims = []
                    for a_idx in range(NUM_SAMPLES):
                        for b_idx in range(a_idx + 1, NUM_SAMPLES):
                            if embs[a_idx] is not None and embs[b_idx] is not None:
                                sims.append(cosine_similarity(embs[a_idx], embs[b_idx]))
                    if sims:
                        paper_sims.append(np.mean(sims))
                sim_matrix[i, j] = np.mean(paper_sims) if paper_sims else 0
            else:
                # Cross-model similarity
                paper_sims = []
                for paper_id in PAPER_IDS:
                    embs_i = embeddings[model_i][paper_id]
                    embs_j = embeddings[model_j][paper_id]
                    sims = []
                    for a_idx in range(NUM_SAMPLES):
                        for b_idx in range(NUM_SAMPLES):
                            if embs_i[a_idx] is not None and embs_j[b_idx] is not None:
                                sims.append(cosine_similarity(embs_i[a_idx], embs_j[b_idx]))
                    if sims:
                        paper_sims.append(np.mean(sims))
                sim_matrix[i, j] = np.mean(paper_sims) if paper_sims else 0
    
    return sim_matrix


def compute_intra_model_similarity(embeddings):
    """Compute average intra-model cosine similarity per model.
    
    For each model, for each paper:
    - Compute mean cosine similarity of all C(10,2)=45 pairs
    - Average across papers
    Returns dict: model_name -> mean similarity
    """
    results = {}
    for model in MODEL_ORDER:
        paper_sims = []
        for paper_id in PAPER_IDS:
            embs = embeddings[model][paper_id]
            sims = []
            for a_idx in range(NUM_SAMPLES):
                for b_idx in range(a_idx + 1, NUM_SAMPLES):
                    if embs[a_idx] is not None and embs[b_idx] is not None:
                        sims.append(cosine_similarity(embs[a_idx], embs[b_idx]))
            if sims:
                paper_sims.append(np.mean(sims))
        results[model] = np.mean(paper_sims) if paper_sims else 0
    return results


def compute_same_diff_paper_sims(embeddings):
    """Compute cosine similarities for same-paper vs different-paper pairs.
    
    Pools across all models:
    - Same-paper: cosine similarities between embeddings from any two different models for the SAME paper
    - Different-paper: cosine similarities between embeddings from any two different models for DIFFERENT papers
    """
    same_paper_sims = []
    diff_paper_sims = []
    
    model_pairs = list(combinations(range(len(MODEL_ORDER)), 2))
    
    for mi, mj in model_pairs:
        model_i = MODEL_ORDER[mi]
        model_j = MODEL_ORDER[mj]
        
        for pi, paper_i in enumerate(PAPER_IDS):
            # Same paper: model_i, paper_i vs model_j, paper_i
            embs_i = embeddings[model_i][paper_i]
            embs_j = embeddings[model_j][paper_i]
            # Sample a few pairs to keep manageable
            for a_idx in range(NUM_SAMPLES):
                for b_idx in range(NUM_SAMPLES):
                    if embs_i[a_idx] is not None and embs_j[b_idx] is not None:
                        same_paper_sims.append(cosine_similarity(embs_i[a_idx], embs_j[b_idx]))
            
            # Different paper: sample a few different papers
            # To keep it manageable, compare with 3 random other papers
            rng = np.random.RandomState(42 + pi)
            other_papers = [p for p in range(len(PAPER_IDS)) if p != pi]
            selected = rng.choice(other_papers, size=min(3, len(other_papers)), replace=False)
            
            for pj in selected:
                paper_j = PAPER_IDS[pj]
                embs_j_diff = embeddings[model_j][paper_j]
                # Sample fewer pairs for different papers
                for a_idx in range(0, NUM_SAMPLES, 2):  # every other sample
                    for b_idx in range(0, NUM_SAMPLES, 2):
                        if embs_i[a_idx] is not None and embs_j_diff[b_idx] is not None:
                            diff_paper_sims.append(cosine_similarity(embs_i[a_idx], embs_j_diff[b_idx]))
    
    return np.array(same_paper_sims), np.array(diff_paper_sims)


def plot_heatmap(sim_matrix, title, filename, vmin=None, vmax=None):
    """Plot a heatmap of inter-model cosine similarities."""
    fig, ax = plt.subplots(figsize=(8, 6.5))
    
    # Use a colormap similar to the paper (looks like a warm colormap)
    cmap = plt.cm.YlOrRd
    
    if vmin is None:
        vmin = sim_matrix.min() - 0.02
    if vmax is None:
        vmax = sim_matrix.max() + 0.02
    
    im = ax.imshow(sim_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Average Cosine Similarity', fontsize=12)
    
    # Set ticks
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels(MODEL_ORDER, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(MODEL_ORDER, fontsize=10)
    
    # Add text annotations
    for i in range(len(MODEL_ORDER)):
        for j in range(len(MODEL_ORDER)):
            text_color = 'white' if sim_matrix[i, j] > (vmin + vmax) / 2 else 'black'
            ax.text(j, i, f'{sim_matrix[i, j]:.3f}', ha='center', va='center',
                    color=text_color, fontsize=9, fontweight='bold')
    
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_intra_model_bars(intra_sims, title, filename):
    """Plot bar chart of intra-model similarities."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    models = MODEL_ORDER
    values = [intra_sims[m] for m in models]
    
    # Color by provider
    colors = []
    for m in models:
        if 'Claude' in m:
            colors.append('#E07B39')  # Orange for Anthropic
        else:
            colors.append('#4A90D9')  # Blue for OpenAI
    
    bars = ax.bar(range(len(models)), values, color=colors, width=0.6, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Average Cosine Similarity', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.0)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#E07B39', edgecolor='black', label='Anthropic'),
                       Patch(facecolor='#4A90D9', edgecolor='black', label='OpenAI')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def plot_kde(same_sims, diff_sims, title, filename):
    """Plot KDE of same-paper vs different-paper cosine similarity distributions."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # KDE for same-paper
    kde_same = gaussian_kde(same_sims, bw_method=0.1)
    x_range = np.linspace(0, 1, 500)
    ax.fill_between(x_range, kde_same(x_range), alpha=0.4, color='#E07B39', label='Same paper')
    ax.plot(x_range, kde_same(x_range), color='#E07B39', linewidth=2)
    
    # KDE for different-paper
    kde_diff = gaussian_kde(diff_sims, bw_method=0.1)
    ax.fill_between(x_range, kde_diff(x_range), alpha=0.4, color='#4A90D9', label='Different paper')
    ax.plot(x_range, kde_diff(x_range), color='#4A90D9', linewidth=2)
    
    ax.set_xlabel('Cosine Similarity', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def main():
    print("Loading embeddings...")
    task1_embs = load_embeddings("cache/task1_embeddings.json")
    task2_embs = load_embeddings("cache/task2_embeddings.json")
    
    # Verify data
    for task_name, embs in [("Task 1", task1_embs), ("Task 2", task2_embs)]:
        total = sum(len(papers) for papers in embs.values())
        print(f"  {task_name}: {len(embs)} models, {total} model×paper entries")
    
    # ====== Inter-model similarity heatmaps (Figure 1) ======
    print("\nComputing inter-model similarities...")
    sim_matrix_t1 = compute_inter_model_similarity(task1_embs)
    sim_matrix_t2 = compute_inter_model_similarity(task2_embs)
    
    print("\nTask 1 (convergence desired) similarity matrix:")
    print(np.array2string(sim_matrix_t1, precision=3))
    print(f"  Mean off-diagonal: {np.mean(sim_matrix_t1[np.triu_indices(6, k=1)]):.4f}")
    
    print("\nTask 2 (diversity desired) similarity matrix:")
    print(np.array2string(sim_matrix_t2, precision=3))
    print(f"  Mean off-diagonal: {np.mean(sim_matrix_t2[np.triu_indices(6, k=1)]):.4f}")
    
    # Determine shared vmin/vmax for consistent color scale
    all_vals = np.concatenate([sim_matrix_t1.flatten(), sim_matrix_t2.flatten()])
    vmin = max(0, all_vals.min() - 0.03)
    vmax = min(1, all_vals.max() + 0.03)
    
    plot_heatmap(sim_matrix_t1, 
                 "Task 1: Recover Underlying Hypothesis\n(Convergence Desired)",
                 "results/plots/heatmap_A.pdf", vmin=vmin, vmax=vmax)
    plot_heatmap(sim_matrix_t2,
                 "Task 2: Generate Novel Hypothesis\n(Diversity Desired)",
                 "results/plots/heatmap_B.pdf", vmin=vmin, vmax=vmax)
    
    # ====== Intra-model similarity bar charts (Figure 2) ======
    print("\nComputing intra-model similarities...")
    intra_t1 = compute_intra_model_similarity(task1_embs)
    intra_t2 = compute_intra_model_similarity(task2_embs)
    
    print("\nIntra-model similarities (Task 1):")
    for m in MODEL_ORDER:
        print(f"  {m}: {intra_t1[m]:.4f}")
    
    print("\nIntra-model similarities (Task 2):")
    for m in MODEL_ORDER:
        print(f"  {m}: {intra_t2[m]:.4f}")
    
    plot_intra_model_bars(intra_t1,
                          "Intra-Model Similarity: Recover Underlying Hypothesis",
                          "results/plots/intra_model_repetition_underlying_hypotheses.pdf")
    plot_intra_model_bars(intra_t2,
                          "Intra-Model Similarity: Generate Novel Hypothesis",
                          "results/plots/intra_model_repetition_new_hypotheses.pdf")
    
    # ====== KDE plots (Figure 3) ======
    print("\nComputing same-paper vs different-paper similarities...")
    same_t1, diff_t1 = compute_same_diff_paper_sims(task1_embs)
    same_t2, diff_t2 = compute_same_diff_paper_sims(task2_embs)
    
    print(f"  Task 1: {len(same_t1)} same-paper, {len(diff_t1)} diff-paper similarities")
    print(f"    Same-paper mean: {same_t1.mean():.4f}, Different-paper mean: {diff_t1.mean():.4f}")
    print(f"  Task 2: {len(same_t2)} same-paper, {len(diff_t2)} diff-paper similarities")
    print(f"    Same-paper mean: {same_t2.mean():.4f}, Different-paper mean: {diff_t2.mean():.4f}")
    
    plot_kde(same_t1, diff_t1,
             "Task 1: Same vs Different Paper Similarity",
             "results/plots/intra_inter_kde_pooled_A.pdf")
    plot_kde(same_t2, diff_t2,
             "Task 2: Same vs Different Paper Similarity",
             "results/plots/intra_inter_kde_pooled_B.pdf")
    
    # ====== Save numerical results ======
    results = {
        "inter_model_similarity_task1": sim_matrix_t1.tolist(),
        "inter_model_similarity_task2": sim_matrix_t2.tolist(),
        "inter_model_mean_offdiag_task1": float(np.mean(sim_matrix_t1[np.triu_indices(6, k=1)])),
        "inter_model_mean_offdiag_task2": float(np.mean(sim_matrix_t2[np.triu_indices(6, k=1)])),
        "intra_model_similarity_task1": {m: float(v) for m, v in intra_t1.items()},
        "intra_model_similarity_task2": {m: float(v) for m, v in intra_t2.items()},
        "kde_stats": {
            "task1_same_paper_mean": float(same_t1.mean()),
            "task1_diff_paper_mean": float(diff_t1.mean()),
            "task2_same_paper_mean": float(same_t2.mean()),
            "task2_diff_paper_mean": float(diff_t2.mean()),
        },
        "model_order": MODEL_ORDER,
    }
    
    with open("results/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved metrics to results/metrics.json")
    
    print("\n===== SUMMARY =====")
    print(f"Inter-model similarity (Task 1, convergence desired): {results['inter_model_mean_offdiag_task1']:.4f}")
    print(f"Inter-model similarity (Task 2, diversity desired):   {results['inter_model_mean_offdiag_task2']:.4f}")
    print(f"Key finding: Inter-model similarity remains {'HIGH' if results['inter_model_mean_offdiag_task2'] > 0.6 else 'moderate'} even for Task 2")
    print("All plots saved to results/plots/")


if __name__ == "__main__":
    main()
