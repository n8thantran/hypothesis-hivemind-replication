"""Parallelized experiment runner for faster API calls."""
import os
import json
import time
import argparse
import concurrent.futures
from threading import Lock
from config import (
    MODELS, MODEL_ORDER, NUM_SAMPLES, PAPERS_DIR, CACHE_DIR, DATA_DIR,
    TASK1_SUMMARY_SYSTEM, TASK1_SUMMARY_USER,
    TASK1_HYPOTHESIS_SYSTEM, TASK1_HYPOTHESIS_USER,
    TASK2_SYSTEM, TASK2_USER,
)
from api_utils import call_llm, get_embeddings

MAX_PAPER_TEXT = 80000
MAX_WORKERS = 8  # concurrent API calls


def load_paper_texts():
    with open(os.path.join(DATA_DIR, "paper_index.json")) as f:
        index = json.load(f)
    papers = {}
    for paper_id, info in index.items():
        txt_path = info["txt_path"]
        if os.path.exists(txt_path):
            with open(txt_path) as f:
                text = f.read()
            if len(text) > MAX_PAPER_TEXT:
                text = text[:MAX_PAPER_TEXT]
            papers[paper_id] = text
    return papers


class ThreadSafeCache:
    def __init__(self, filepath):
        self.filepath = filepath
        self.lock = Lock()
        if os.path.exists(filepath):
            with open(filepath) as f:
                self.data = json.load(f)
        else:
            self.data = {}
        self._save_counter = 0
    
    def get(self, key, default=None):
        with self.lock:
            return self.data.get(key, default)
    
    def set(self, key, value):
        with self.lock:
            self.data[key] = value
            self._save_counter += 1
            if self._save_counter % 10 == 0:
                self._save()
    
    def append(self, key, value):
        with self.lock:
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)
            self._save_counter += 1
            if self._save_counter % 10 == 0:
                self._save()
    
    def _save(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        with open(self.filepath, "w") as f:
            json.dump(self.data, f)
    
    def save(self):
        with self.lock:
            self._save()
    
    def __len__(self):
        with self.lock:
            return len(self.data)
    
    def keys(self):
        with self.lock:
            return list(self.data.keys())
    
    def items(self):
        with self.lock:
            return list(self.data.items())


def generate_single_hypothesis(args_tuple):
    """Generate a single hypothesis (worker function for thread pool)."""
    model_name, model_id, paper_id, sample_idx, system_msg, user_msg = args_tuple
    
    response = call_llm(
        model_id, system_msg, user_msg,
        max_tokens=2000, temperature=1.0,
    )
    return (model_name, paper_id, sample_idx, response)


def run_task(papers, summaries, task, cache_file, num_samples=NUM_SAMPLES):
    """Run hypothesis generation for a task with parallel API calls."""
    cache = ThreadSafeCache(cache_file)
    
    paper_ids = sorted(papers.keys())
    
    # Build work items
    work_items = []
    for model_name in MODEL_ORDER:
        model_id = MODELS[model_name]
        for paper_id in paper_ids:
            cache_key = f"{model_name}|{paper_id}"
            existing = cache.get(cache_key, [])
            needed = num_samples - len(existing)
            
            if needed <= 0:
                continue
            
            for sample_idx in range(needed):
                if task == "task1":
                    summary = summaries.get(paper_id, "")
                    if not summary:
                        continue
                    user_msg = summary + "\n\n" + TASK1_HYPOTHESIS_USER
                    system_msg = TASK1_HYPOTHESIS_SYSTEM
                else:
                    user_msg = papers[paper_id] + "\n\n" + TASK2_USER
                    system_msg = TASK2_SYSTEM
                
                work_items.append((model_name, model_id, paper_id, sample_idx, system_msg, user_msg))
    
    print(f"Total work items: {len(work_items)}")
    
    if not work_items:
        print("Nothing to do!")
        return cache
    
    completed = 0
    failed = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(generate_single_hypothesis, item): item for item in work_items}
        
        for future in concurrent.futures.as_completed(futures):
            try:
                model_name, paper_id, sample_idx, response = future.result()
                cache_key = f"{model_name}|{paper_id}"
                
                if response:
                    cache.append(cache_key, response)
                    completed += 1
                else:
                    failed += 1
                
                if (completed + failed) % 50 == 0:
                    print(f"  Progress: {completed} completed, {failed} failed / {len(work_items)} total")
                    cache.save()
                    
            except Exception as e:
                failed += 1
                print(f"  Error: {e}")
    
    cache.save()
    print(f"Finished: {completed} completed, {failed} failed")
    return cache


def run_embeddings(hypotheses_cache_file, embeddings_cache_file):
    """Generate embeddings for all hypotheses."""
    with open(hypotheses_cache_file) as f:
        hyp_data = json.load(f)
    
    cache = ThreadSafeCache(embeddings_cache_file)
    
    # Collect texts needing embedding
    texts_to_embed = []
    text_keys = []
    for cache_key, hypotheses in hyp_data.items():
        for i, hyp in enumerate(hypotheses):
            emb_key = f"{cache_key}|{i}"
            if cache.get(emb_key) is None:
                texts_to_embed.append(hyp)
                text_keys.append(emb_key)
    
    print(f"Need to embed {len(texts_to_embed)} texts ({len(cache)} already cached)")
    
    # Batch embedding
    batch_size = 100
    for start in range(0, len(texts_to_embed), batch_size):
        batch_texts = texts_to_embed[start:start + batch_size]
        batch_keys = text_keys[start:start + batch_size]
        
        embeddings = get_embeddings(batch_texts)
        if embeddings:
            for key, emb in zip(batch_keys, embeddings):
                cache.set(key, emb)
        else:
            # Try smaller batches
            for i in range(0, len(batch_texts), 10):
                sub_texts = batch_texts[i:i+10]
                sub_keys = batch_keys[i:i+10]
                embs = get_embeddings(sub_texts)
                if embs:
                    for key, emb in zip(sub_keys, embs):
                        cache.set(key, emb)
                time.sleep(0.5)
        
        if (start // batch_size) % 5 == 0:
            print(f"  Embedded {start + len(batch_texts)}/{len(texts_to_embed)}")
        
        time.sleep(0.2)
    
    cache.save()
    print(f"Total embeddings: {len(cache)}")
    return cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["summaries", "task1", "task2", "embed_task1", "embed_task2", "all"],
                       default="all")
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()
    
    global MAX_WORKERS
    MAX_WORKERS = args.workers
    
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    papers = load_paper_texts()
    print(f"Loaded {len(papers)} papers")
    
    summary_file = os.path.join(CACHE_DIR, "summaries.json")
    task1_file = os.path.join(CACHE_DIR, "task1_hypotheses.json")
    task2_file = os.path.join(CACHE_DIR, "task2_hypotheses.json")
    task1_emb_file = os.path.join(CACHE_DIR, "task1_embeddings.json")
    task2_emb_file = os.path.join(CACHE_DIR, "task2_embeddings.json")
    
    if args.step in ("summaries", "all"):
        print("\n=== Generating summaries ===")
        from run_experiment import generate_experiment_summaries
        summaries = generate_experiment_summaries(papers, summary_file)
    
    if args.step in ("task1", "all"):
        print("\n=== Task 1: Hypothesis recovery ===")
        with open(summary_file) as f:
            summaries = json.load(f)
        run_task(papers, summaries, "task1", task1_file, args.num_samples)
    
    if args.step in ("task2", "all"):
        print("\n=== Task 2: Novel hypothesis generation ===")
        run_task(papers, {}, "task2", task2_file, args.num_samples)
    
    if args.step in ("embed_task1", "all"):
        print("\n=== Embedding Task 1 ===")
        run_embeddings(task1_file, task1_emb_file)
    
    if args.step in ("embed_task2", "all"):
        print("\n=== Embedding Task 2 ===")
        run_embeddings(task2_file, task2_emb_file)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
