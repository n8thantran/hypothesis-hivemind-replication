"""
Generate experiment summaries for each paper using Claude Sonnet 4.5.
These summaries are used as input for Task 1 (recover underlying hypothesis).
"""
import json
import os
import asyncio
import aiohttp
from pathlib import Path

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "anthropic/claude-sonnet-4.5"

DATA_DIR = Path("/workspace/data")
OUTPUT_DIR = Path("/workspace/data/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = (
    "You are a helpful assistant for summarizing key details of "
    "experiments and methodologies from scientific papers."
)
USER_INSTRUCTION = (
    "Summarize the following research paper, focusing ONLY on this question: "
    "Carefully analyze ONLY the experiments performed or methods used. "
    "Do NOT include results, abstract, introduction, or discussion. "
    "Output MUST be valid JSON of the form: "
    '{"title": "<paper title>", "experiments_summary": "<concise summary>"} '
    "Do NOT wrap the JSON in markdown code fences. "
    "Paper text:\n"
)

def truncate_text(text: str, max_chars: int = 80000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[... truncated ...]"


async def generate_summary(session, paper_id: str, paper_text: str) -> dict:
    """Generate an experiment summary for a single paper."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_INSTRUCTION + truncate_text(paper_text)},
        ],
        "temperature": 0.3,
        "max_tokens": 4000,
    }
    
    for attempt in range(5):
        try:
            timeout = aiohttp.ClientTimeout(total=180)
            async with session.post(BASE_URL, headers=headers, json=payload, timeout=timeout) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content")
                    if content:
                        # Try to parse as JSON
                        try:
                            result = json.loads(content)
                            return {"paper_id": paper_id, "summary": result}
                        except json.JSONDecodeError:
                            # Return raw text
                            return {"paper_id": paper_id, "summary": {"experiments_summary": content}}
                elif resp.status == 429:
                    wait_time = 2 ** attempt * 5
                    print(f"  Rate limited for {paper_id}, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    text = await resp.text()
                    print(f"  Error {resp.status} for {paper_id}: {text[:200]}")
                    await asyncio.sleep(5)
        except Exception as e:
            print(f"  Exception for {paper_id}: {e}")
            await asyncio.sleep(5)
    
    return {"paper_id": paper_id, "summary": None}


async def main():
    # Load paper texts
    texts_file = DATA_DIR / "paper_texts.json"
    if not texts_file.exists():
        print("Error: paper_texts.json not found. Run extract_text.py first.")
        return
    
    with open(texts_file) as f:
        paper_texts = json.load(f)
    
    print(f"Generating summaries for {len(paper_texts)} papers using {MODEL}...")
    
    # Check for existing summaries
    output_file = OUTPUT_DIR / "experiment_summaries.json"
    existing = {}
    if output_file.exists():
        with open(output_file) as f:
            existing_list = json.load(f)
            existing = {item["paper_id"]: item for item in existing_list}
        print(f"  Found {len(existing)} existing summaries")
    
    # Generate missing summaries
    sem = asyncio.Semaphore(5)
    
    async def bounded_generate(session, paper_id, text):
        if paper_id in existing and existing[paper_id].get("summary"):
            return existing[paper_id]
        async with sem:
            result = await generate_summary(session, paper_id, text)
            print(f"  Generated summary for {paper_id}")
            return result
    
    async with aiohttp.ClientSession() as session:
        tasks = [bounded_generate(session, pid, txt) for pid, txt in paper_texts.items()]
        results = await asyncio.gather(*tasks)
    
    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    success_count = sum(1 for r in results if r.get("summary"))
    print(f"Done. {success_count}/{len(results)} summaries generated.")
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
