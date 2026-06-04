"""Main experiment runner for the Hypothesis Hivemind replication.

This script:
1. For each paper, generates experiment summaries (Task 1 Stage A) using one model
2. For each paper × model, generates 10 hypothesis samples for Task 1 (from summary)
3. For each paper × model, generates 10 novel hypothesis samples for Task 2 (from full text)
4. Embeds all outputs
5. Saves everything to cache for analysis
"""
import os
import json
import time
import argparse
from tqdm import tqdm
from config import (
    MODELS, MODEL_ORDER, NUM_SAMPLES, PAPERS_DIR, CACHE_DIR, DATA_DIR,
    TASK1_SUMMARY_SYSTEM, TASK1_SUMMARY_USER,
    TASK1_HYPOTHESIS_SYSTEM, TASK1_HYPOTHESIS_USER,
    TASK2_SYSTEM, TASK2_USER,
)
from api_utils import call_llm, get_embeddings

# Maximum chars of paper text to send (to fit in context windows)
MAX_PAPER_TEXT = 80000


def load_paper_texts():
    """Load all paper texts from disk."""
    with open(os.path.join(DATA_DIR, "paper_index.json")) as f:
        index = json.load(f)
    
    papers = {}
    for paper_id, info in index.items():
        txt_path = info["txt_path"]
        if os.path.exists(txt_path):
            with open(txt_path) as f:
                text = f.read()
            # Truncate if too long
            if len(text) > MAX_PAPER_TEXT:
                text = text[:MAX_PAPER_TEXT]
            papers[paper_id] = text
    
    return papers


def load_cache(cache_file):
    """Load cache from disk."""
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return json.load(f)
    return {}


def save_cache(cache, cache_file):
    """Save cache to disk."""
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=2)


def generate_experiment_summaries(papers, cache_file):
    """Generate experiment summaries for all papers (Task 1 Stage A).
    
    Uses Claude Sonnet 4.5 as the summarizer (arbitrary choice - the paper
    doesn't specify which model generates summaries, just that summaries
    are given to all models for Task 1).
    """
    cache = load_cache(cache_file)
    summarizer_model = MODELS["Claude Sonnet 4.5"]
    
    for paper_id, text in tqdm(papers.items(), desc="Generating summaries"):
        if paper_id in cache:
            continue
        
        user_msg = TASK1_SUMMARY_USER + text
        response = call_llm(
            summarizer_model,
            TASK1_SUMMARY_SYSTEM,
            user_msg,
            max_tokens=2000,
            temperature=0.3,  # Low temp for factual summary
        )
        
        if response:
            # Try to extract the experiments_summary from JSON
            try:
                data = json.loads(response)
                summary = data.get("experiments_summary", response)
            except json.JSONDecodeError:
                summary = response
            
            cache[paper_id] = summary
            save_cache(cache, cache_file)
            print(f"  Summary for {paper_id}: {len(summary)} chars")
        else:
            print(f"  FAILED to generate summary for {paper_id}")
        
        time.sleep(0.5)
    
    return cache


def generate_hypotheses(papers, summaries, task, cache_file, num_samples=NUM_SAMPLES):
    """Generate hypotheses for all papers × models.
    
    task: "task1" or "task2"
    """
    cache = load_cache(cache_file)
    
    paper_ids = sorted(papers.keys())
    
    for model_name in MODEL_ORDER:
        model_id = MODELS[model_name]
        print(f"\n=== {model_name} ({model_id}) ===")
        
        for paper_id in tqdm(paper_ids, desc=f"{model_name}"):
            cache_key = f"{model_name}|{paper_id}"
            
            # Check if we already have enough samples
            existing = cache.get(cache_key, [])
            if len(existing) >= num_samples:
                continue
            
            needed = num_samples - len(existing)
            
            for sample_idx in range(needed):
                if task == "task1":
                    summary = summaries.get(paper_id, "")
                    if not summary:
                        print(f"  No summary for {paper_id}, skipping")
                        break
                    user_msg = summary + "\n\n" + TASK1_HYPOTHESIS_USER
                    system_msg = TASK1_HYPOTHESIS_SYSTEM
                else:  # task2
                    paper_text = papers[paper_id]
                    user_msg = paper_text + "\n\n" + TASK2_USER
                    system_msg = TASK2_SYSTEM
                
                response = call_llm(
                    model_id,
                    system_msg,
                    user_msg,
                    max_tokens=2000,
                    temperature=1.0,
                )
                
                if response:
                    existing.append(response)
                else:
                    print(f"  FAILED sample {sample_idx} for {model_name}/{paper_id}")
                
                time.sleep(0.3)
            
            cache[cache_key] = existing
            save_cache(cache, cache_file)
    
    return cache


def generate_embeddings(hypotheses_cache, embeddings_cache_file):
    """Generate embeddings for all hypotheses."""
    cache = load_cache(embeddings_cache_file)
    
    # Collect all unique texts that need embedding
    texts_to_embed = []
    text_keys = []
    
    for cache_key, hypotheses in hypotheses_cache.items():
        for i, hyp in enumerate(hypotheses):
            emb_key = f"{cache_key}|{i}"
            if emb_key not in cache:
                texts_to_embed.append(hyp)
                text_keys.append(emb_key)
    
    print(f"Need to embed {len(texts_to_embed)} texts ({len(cache)} already cached)")
    
    # Batch embedding (up to 100 at a time)
    batch_size = 50
    for start in tqdm(range(0, len(texts_to_embed), batch_size), desc="Embedding"):
        batch_texts = texts_to_embed[start:start + batch_size]
        batch_keys = text_keys[start:start + batch_size]
        
        embeddings = get_embeddings(batch_texts)
        
        if embeddings:
            for key, emb in zip(batch_keys, embeddings):
                cache[key] = emb
            save_cache(cache, embeddings_cache_file)
        else:
            print(f"  FAILED to embed batch starting at {start}")
            # Try one by one
            for text, key in zip(batch_texts, batch_keys):
                emb = get_embeddings([text])
                if emb:
                    cache[key] = emb[0]
                else:
                    print(f"  FAILED to embed {key}")
                time.sleep(1)
            save_cache(cache, embeddings_cache_file)
        
        time.sleep(0.3)
    
    return cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["summaries", "task1", "task2", "embed_task1", "embed_task2", "all"],
                       default="all", help="Which step to run")
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    args = parser.parse_args()
    
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Load papers
    papers = load_paper_texts()
    print(f"Loaded {len(papers)} papers")
    
    summary_cache = os.path.join(CACHE_DIR, "summaries.json")
    task1_cache = os.path.join(CACHE_DIR, "task1_hypotheses.json")
    task2_cache = os.path.join(CACHE_DIR, "task2_hypotheses.json")
    task1_emb_cache = os.path.join(CACHE_DIR, "task1_embeddings.json")
    task2_emb_cache = os.path.join(CACHE_DIR, "task2_embeddings.json")
    
    if args.step in ("summaries", "all"):
        print("\n=== Step 1: Generate experiment summaries ===")
        summaries = generate_experiment_summaries(papers, summary_cache)
        print(f"Generated {len(summaries)} summaries")
    
    if args.step in ("task1", "all"):
        print("\n=== Step 2: Generate Task 1 hypotheses ===")
        summaries = load_cache(summary_cache)
        task1_hyps = generate_hypotheses(papers, summaries, "task1", task1_cache, args.num_samples)
        print(f"Generated Task 1 hypotheses for {len(task1_hyps)} model-paper pairs")
    
    if args.step in ("task2", "all"):
        print("\n=== Step 3: Generate Task 2 hypotheses ===")
        task2_hyps = generate_hypotheses(papers, {}, "task2", task2_cache, args.num_samples)
        print(f"Generated Task 2 hypotheses for {len(task2_hyps)} model-paper pairs")
    
    if args.step in ("embed_task1", "all"):
        print("\n=== Step 4: Embed Task 1 hypotheses ===")
        task1_hyps = load_cache(task1_cache)
        task1_embs = generate_embeddings(task1_hyps, task1_emb_cache)
        print(f"Generated {len(task1_embs)} Task 1 embeddings")
    
    if args.step in ("embed_task2", "all"):
        print("\n=== Step 5: Embed Task 2 hypotheses ===")
        task2_hyps = load_cache(task2_cache)
        task2_embs = generate_embeddings(task2_hyps, task2_emb_cache)
        print(f"Generated {len(task2_embs)} Task 2 embeddings")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
