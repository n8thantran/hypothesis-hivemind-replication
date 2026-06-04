"""
Generate hypotheses using 6 frontier models via OpenRouter API.

Task 1: Recover underlying hypothesis from experiment summary
Task 2: Propose novel hypotheses from full paper text

For each task: 10 samples × 6 models × 50 papers = 3000 outputs
"""
import json
import os
import time
import sys
import asyncio
import aiohttp
from pathlib import Path
from typing import Optional

# Configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS = [
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-sonnet-4.6",
    "openai/gpt-5-nano",
    "openai/gpt-5-mini",
    "openai/gpt-5",
]

MODEL_SHORT_NAMES = {
    "anthropic/claude-haiku-4.5": "Haiku 4.5",
    "anthropic/claude-sonnet-4.5": "Sonnet 4.5",
    "anthropic/claude-sonnet-4.6": "Sonnet 4.6",
    "openai/gpt-5-nano": "GPT-5 Nano",
    "openai/gpt-5-mini": "GPT-5 Mini",
    "openai/gpt-5": "GPT-5",
}

NUM_SAMPLES = 10
DATA_DIR = Path("/workspace/data")
OUTPUT_DIR = Path("/workspace/data/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Prompts from the paper (Appendix A, Box 1)
SUMMARY_SYSTEM_PROMPT = (
    "You are a helpful assistant for summarizing key details of "
    "experiments and methodologies from scientific papers."
)
SUMMARY_USER_INSTRUCTION = (
    "Summarize the following research paper, focusing ONLY on this question: "
    "Carefully analyze ONLY the experiments performed or methods used. "
    "Do NOT include results, abstract, introduction, or discussion. "
    "Output MUST be valid JSON of the form: "
    '{"title": "<paper title>", "experiments_summary": "<concise summary>"} '
    "Do NOT wrap the JSON in markdown code fences. "
    "Paper text:\n"
)

UNDERLYING_SYSTEM_PROMPT = (
    "You are a scientific reasoning assistant. Given a description of the "
    "experiments and methods from a research paper, infer the underlying "
    "hypothesis being tested - the core scientific claim the experiments were "
    "designed to validate. A hypothesis is a specific, testable, and falsifiable "
    "prediction about the relationship between variables. Output ONLY the "
    "hypothesis as a single declarative sentence. Do not include preamble, "
    "explanation, or any other text."
)
UNDERLYING_USER_INSTRUCTION = (
    "Generate a single testable hypothesis based on the experiment description "
    "above. Express it as one declarative sentence (e.g. 'If X, then Y because Z')."
)

NOVEL_SYSTEM_PROMPT = (
    "You are an expert research scientist. Given the context of a research paper, "
    "your task is to generate a single novel hypothesis that logically extends "
    "beyond the paper's existing findings - not a restatement of them. The "
    "hypothesis must be: (1) grounded in a gap or open question identified in "
    "the paper, (2) specific and testable, (3) falsifiable. Output ONLY the "
    "hypothesis as a single declarative sentence with no preamble or explanation."
)
NOVEL_USER_INSTRUCTION = (
    "Based on the research context above, generate one novel hypothesis that "
    "extends beyond what this paper has already established."
)


def truncate_text(text: str, max_chars: int = 80000) -> str:
    """Truncate text to fit within context limits."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[... truncated ...]"


# Semaphore to limit concurrent requests per model
MODEL_SEMAPHORES = {}

async def call_model(session, model: str, system_prompt: str, user_message: str,
                     temperature: float = 1.0, max_retries: int = 5) -> Optional[str]:
    """Call a model via OpenRouter API with retries."""
    # Use per-model semaphore to avoid overwhelming any single model
    if model not in MODEL_SEMAPHORES:
        MODEL_SEMAPHORES[model] = asyncio.Semaphore(5)
    
    async with MODEL_SEMAPHORES[model]:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature,
            "max_tokens": 16000,
        }
        
        for attempt in range(max_retries):
            try:
                timeout = aiohttp.ClientTimeout(total=180)
                async with session.post(BASE_URL, headers=headers, json=payload, timeout=timeout) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = data.get("choices", [{}])[0].get("message", {}).get("content")
                        if content is not None:
                            return content.strip()
                        else:
                            # Some models return content as None with reasoning
                            print(f"    Null content from {model}, retrying...")
                            await asyncio.sleep(1)
                            continue
                    elif resp.status == 429:
                        wait = min(2 ** attempt * 3, 60)
                        print(f"    Rate limited ({model}), waiting {wait}s...")
                        await asyncio.sleep(wait)
                    else:
                        text = await resp.text()
                        print(f"    Error {resp.status} for {model}: {text[:200]}")
                        await asyncio.sleep(2 ** attempt)
            except asyncio.TimeoutError:
                print(f"    Timeout for {model}, attempt {attempt+1}")
                await asyncio.sleep(5)
            except Exception as e:
                print(f"    Exception for {model}: {e}")
                await asyncio.sleep(2 ** attempt)
        
        return None


async def generate_summaries(paper_texts: dict) -> dict:
    """Generate experiment summaries for all papers (Task 1 prerequisite)."""
    output_file = OUTPUT_DIR / "experiment_summaries.json"
    if output_file.exists():
        print("Loading existing summaries...")
        return json.loads(output_file.read_text())
    
    summaries = {}
    model = "anthropic/claude-sonnet-4.5"
    
    async with aiohttp.ClientSession() as session:
        for i, (paper_id, text) in enumerate(sorted(paper_texts.items())):
            print(f"  [{i+1}/{len(paper_texts)}] Summarizing {paper_id}...")
            user_msg = SUMMARY_USER_INSTRUCTION + truncate_text(text, 60000)
            result = await call_model(session, model, SUMMARY_SYSTEM_PROMPT, user_msg, temperature=0.3)
            if result:
                summaries[paper_id] = result
                print(f"    Got summary ({len(result)} chars)")
            else:
                print(f"    FAILED to get summary")
            await asyncio.sleep(0.5)
    
    output_file.write_text(json.dumps(summaries, indent=2))
    print(f"Saved {len(summaries)} summaries")
    return summaries


async def generate_hypotheses_for_paper_model(
    session, paper_id: str, model: str, model_short: str,
    system_prompt: str, user_message: str, num_samples: int = 10
) -> list:
    """Generate multiple hypothesis samples for one paper-model pair."""
    tasks = []
    for _ in range(num_samples):
        tasks.append(call_model(session, model, system_prompt, user_message, temperature=1.0))
    
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


async def generate_task_hypotheses(input_texts: dict, task_name: str,
                                    system_prompt: str, user_instruction: str,
                                    max_text_chars: int = 60000) -> dict:
    """Generic function to generate hypotheses for a task."""
    output_file = OUTPUT_DIR / f"{task_name}_hypotheses.json"
    
    if output_file.exists():
        results = json.loads(output_file.read_text())
    else:
        results = {}
    
    # Count existing
    total_expected = len(input_texts) * len(MODELS) * NUM_SAMPLES
    total_existing = sum(
        len(samples)
        for paper_data in results.values()
        for samples in paper_data.values()
    )
    
    if total_existing >= total_expected * 0.95:
        print(f"{task_name} already complete ({total_existing}/{total_expected} samples)")
        return results
    
    print(f"Resuming {task_name} ({total_existing}/{total_expected} samples)...")
    
    connector = aiohttp.TCPConnector(limit=30)
    async with aiohttp.ClientSession(connector=connector) as session:
        for i, (paper_id, text) in enumerate(sorted(input_texts.items())):
            if paper_id not in results:
                results[paper_id] = {}
            
            truncated_text = truncate_text(text, max_text_chars)
            user_msg = truncated_text + "\n\n" + user_instruction
            
            # Process all models for this paper concurrently
            model_tasks = []
            models_to_process = []
            
            for model in MODELS:
                model_short = MODEL_SHORT_NAMES[model]
                if model_short in results[paper_id] and len(results[paper_id][model_short]) >= NUM_SAMPLES:
                    continue
                models_to_process.append((model, model_short))
                model_tasks.append(
                    generate_hypotheses_for_paper_model(
                        session, paper_id, model, model_short,
                        system_prompt, user_msg, NUM_SAMPLES
                    )
                )
            
            if not model_tasks:
                continue
            
            print(f"  [{i+1}/{len(input_texts)}] {paper_id} ({len(models_to_process)} models)...")
            
            model_results = await asyncio.gather(*model_tasks)
            
            for (model, model_short), samples in zip(models_to_process, model_results):
                results[paper_id][model_short] = samples
                print(f"    {model_short}: {len(samples)}/{NUM_SAMPLES}")
            
            # Save after each paper
            output_file.write_text(json.dumps(results, indent=2))
            
            # Brief pause between papers
            await asyncio.sleep(0.5)
    
    output_file.write_text(json.dumps(results, indent=2))
    total_final = sum(len(s) for p in results.values() for s in p.values())
    print(f"{task_name} complete: {total_final} total samples")
    return results


async def main():
    paper_texts = json.loads((DATA_DIR / "paper_texts.json").read_text())
    print(f"Loaded {len(paper_texts)} paper texts")
    
    task = sys.argv[1] if len(sys.argv) > 1 else "all"
    
    if task in ("summaries", "all"):
        print("\n=== Generating experiment summaries ===")
        summaries = await generate_summaries(paper_texts)
    
    if task in ("task1", "all"):
        print("\n=== Task 1: Generating underlying hypotheses ===")
        summaries_file = OUTPUT_DIR / "experiment_summaries.json"
        if summaries_file.exists():
            summaries = json.loads(summaries_file.read_text())
        else:
            print("ERROR: Need summaries first. Run with 'summaries' argument.")
            return
        await generate_task_hypotheses(
            summaries, "task1",
            UNDERLYING_SYSTEM_PROMPT, UNDERLYING_USER_INSTRUCTION,
            max_text_chars=60000
        )
    
    if task in ("task2", "all"):
        print("\n=== Task 2: Generating novel hypotheses ===")
        await generate_task_hypotheses(
            paper_texts, "task2",
            NOVEL_SYSTEM_PROMPT, NOVEL_USER_INSTRUCTION,
            max_text_chars=60000
        )


if __name__ == "__main__":
    asyncio.run(main())
