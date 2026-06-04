"""
Compute embeddings for all hypotheses using text-embedding-3-small via OpenRouter.
"""
import json
import os
import asyncio
import aiohttp
import numpy as np
from pathlib import Path

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
EMBED_URL = "https://openrouter.ai/api/v1/embeddings"
EMBED_MODEL = "openai/text-embedding-3-small"
OUTPUT_DIR = Path("/workspace/data/outputs")

MODELS = ['Haiku 4.5', 'Sonnet 4.5', 'Sonnet 4.6', 'GPT-5 Nano', 'GPT-5 Mini', 'GPT-5']


async def get_embeddings_batch(session, texts: list, max_retries=5) -> list:
    """Get embeddings for a batch of texts."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": EMBED_MODEL,
        "input": texts,
    }
    
    for attempt in range(max_retries):
        try:
            timeout = aiohttp.ClientTimeout(total=120)
            async with session.post(EMBED_URL, headers=headers, json=payload, timeout=timeout) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # Sort by index to maintain order
                    embeddings = sorted(data["data"], key=lambda x: x["index"])
                    return [e["embedding"] for e in embeddings]
                elif resp.status == 429:
                    wait = min(2 ** attempt * 3, 60)
                    print(f"  Rate limited, waiting {wait}s...")
                    await asyncio.sleep(wait)
                else:
                    text = await resp.text()
                    print(f"  Error {resp.status}: {text[:200]}")
                    await asyncio.sleep(2 ** attempt)
        except Exception as e:
            print(f"  Exception: {e}")
            await asyncio.sleep(2 ** attempt)
    
    return None


async def embed_task(task_name: str):
    """Compute embeddings for all hypotheses in a task."""
    hyp_file = OUTPUT_DIR / f"{task_name}_hypotheses.json"
    emb_file = OUTPUT_DIR / f"{task_name}_embeddings.npz"
    
    if emb_file.exists():
        print(f"{task_name} embeddings already exist, skipping")
        return
    
    hypotheses = json.loads(hyp_file.read_text())
    
    # Collect all texts with metadata
    all_texts = []
    metadata = []  # (paper_id, model_name, sample_idx)
    
    for paper_id in sorted(hypotheses.keys()):
        for model_name in MODELS:
            samples = hypotheses[paper_id].get(model_name, [])
            for idx, text in enumerate(samples):
                all_texts.append(text)
                metadata.append((paper_id, model_name, idx))
    
    print(f"{task_name}: {len(all_texts)} texts to embed")
    
    # Batch embed (max 100 per batch for the API)
    BATCH_SIZE = 100
    all_embeddings = []
    
    sem = asyncio.Semaphore(3)  # Limit concurrent embedding requests
    
    async with aiohttp.ClientSession() as session:
        for i in range(0, len(all_texts), BATCH_SIZE):
            batch = all_texts[i:i+BATCH_SIZE]
            async with sem:
                result = await get_embeddings_batch(session, batch)
            
            if result is None:
                print(f"  FAILED batch {i//BATCH_SIZE}")
                # Fill with zeros as fallback
                result = [[0.0] * 1536] * len(batch)
            
            all_embeddings.extend(result)
            
            if (i // BATCH_SIZE) % 5 == 0:
                print(f"  Embedded {i+len(batch)}/{len(all_texts)}")
            
            await asyncio.sleep(0.2)
    
    # Save as numpy arrays
    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    
    # Also save metadata
    paper_ids = [m[0] for m in metadata]
    model_names = [m[1] for m in metadata]
    sample_idxs = [m[2] for m in metadata]
    
    np.savez(emb_file,
             embeddings=embeddings_array,
             paper_ids=np.array(paper_ids),
             model_names=np.array(model_names),
             sample_idxs=np.array(sample_idxs))
    
    print(f"Saved {task_name} embeddings: {embeddings_array.shape}")


async def main():
    import sys
    task = sys.argv[1] if len(sys.argv) > 1 else "all"
    
    if task in ("task1", "all"):
        await embed_task("task1")
    if task in ("task2", "all"):
        await embed_task("task2")


if __name__ == "__main__":
    asyncio.run(main())
