"""Utilities for calling OpenRouter API for LLM generation and embeddings."""
import os
import time
import json
import requests
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, EMBEDDING_MODEL

REASONING_MODELS = {"openai/gpt-5", "openai/gpt-5-nano"}

def call_llm(model_id, system_prompt, user_message, max_tokens=2000, temperature=1.0, max_retries=5):
    """Call an LLM via OpenRouter API.
    
    Returns the generated text or None on failure.
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    
    # Reasoning models need much higher max_tokens to produce output
    effective_max_tokens = 16000 if model_id in REASONING_MODELS else max_tokens
    
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": effective_max_tokens,
        "temperature": temperature,
    }
    
    # Longer timeout for reasoning models
    timeout = 180 if model_id in REASONING_MODELS else 120
    
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            
            if resp.status_code == 200:
                data = resp.json()
                if "choices" in data and len(data["choices"]) > 0:
                    content = data["choices"][0]["message"]["content"]
                    if content and content.strip():
                        return content.strip()
                    else:
                        print(f"  Empty content from {model_id}, attempt {attempt+1}")
                else:
                    print(f"  No choices in response from {model_id}: {data}")
            elif resp.status_code == 429:
                wait = min(30, 2 ** (attempt + 1))
                print(f"  Rate limited (429), waiting {wait}s...")
                time.sleep(wait)
                continue
            else:
                print(f"  HTTP {resp.status_code} from {model_id}: {resp.text[:200]}")
                
        except Exception as e:
            print(f"  Error calling {model_id}: {e}")
        
        time.sleep(2 * (attempt + 1))
    
    return None


def get_embeddings(texts, max_retries=5):
    """Get embeddings for a list of texts using OpenRouter.
    
    Returns list of embedding vectors or None on failure.
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": EMBEDDING_MODEL,
        "input": texts,
    }
    
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{OPENROUTER_BASE_URL}/embeddings",
                headers=headers,
                json=payload,
                timeout=120,
            )
            
            if resp.status_code == 200:
                data = resp.json()
                if "data" in data:
                    # Sort by index to ensure correct order
                    sorted_data = sorted(data["data"], key=lambda x: x["index"])
                    return [d["embedding"] for d in sorted_data]
                else:
                    print(f"  No data in embedding response: {data}")
            elif resp.status_code == 429:
                wait = min(30, 2 ** (attempt + 1))
                print(f"  Rate limited (429) for embeddings, waiting {wait}s...")
                time.sleep(wait)
                continue
            else:
                print(f"  HTTP {resp.status_code} for embeddings: {resp.text[:200]}")
                
        except Exception as e:
            print(f"  Error getting embeddings: {e}")
        
        time.sleep(2 * (attempt + 1))
    
    return None


def get_embedding_single(text, max_retries=5):
    """Get embedding for a single text."""
    result = get_embeddings([text], max_retries=max_retries)
    if result:
        return result[0]
    return None
