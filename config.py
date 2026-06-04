"""Configuration for the Hypothesis Hivemind replication."""
import os

# API Configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Models
MODELS = {
    "Claude Haiku 4.5": "anthropic/claude-haiku-4.5",
    "Claude Sonnet 4.5": "anthropic/claude-sonnet-4.5",
    "Claude Sonnet 4.6": "anthropic/claude-sonnet-4.6",
    "GPT-5 Nano": "openai/gpt-5-nano",
    "GPT-5 Mini": "openai/gpt-5-mini",
    "GPT-5": "openai/gpt-5",
}

# Model display order (for figures)
MODEL_ORDER = [
    "Claude Haiku 4.5",
    "Claude Sonnet 4.5",
    "Claude Sonnet 4.6",
    "GPT-5 Nano",
    "GPT-5 Mini",
    "GPT-5",
]

# Embedding model
EMBEDDING_MODEL = "openai/text-embedding-3-small"

# Experiment parameters
NUM_SAMPLES = 10  # independent samples per model per task per paper
TEMPERATURE = 1.0  # default temperature for generation

# Paths
DATA_DIR = "data"
PAPERS_DIR = os.path.join(DATA_DIR, "papers")
RESULTS_DIR = "results"
CACHE_DIR = "cache"

# Paper URLs from appendix
PAPER_URLS = [
    "https://openreview.net/pdf?id=0SPoKR8Xrk",
    "https://openreview.net/pdf?id=0uWNuJ1xtz",
    "https://openreview.net/pdf?id=12ZCZVKm7r",
    "https://openreview.net/pdf?id=24lzMGlvnq",
    "https://openreview.net/pdf?id=35aDuh7ndX",
    "https://openreview.net/pdf?id=3WZkuWlzmN",
    "https://openreview.net/pdf?id=3pAVbjWMXW",
    "https://openreview.net/pdf?id=4U2k4uw43B",
    "https://openreview.net/pdf?id=4Xh9oL5rH0",
    "https://openreview.net/pdf?id=57YLCp7n2V",
    "https://openreview.net/pdf?id=5OsnDm1CdX",
    "https://openreview.net/pdf?id=6pjxodugzO",
    "https://openreview.net/pdf?id=7brF4sMQq3",
    "https://openreview.net/pdf?id=7cbwuA5k0T",
    "https://openreview.net/pdf?id=7l75CbxtmC",
    "https://openreview.net/pdf?id=8JFITrNy3K",
    "https://openreview.net/pdf?id=9JSO4qf1RQ",
    "https://openreview.net/pdf?id=A21WF9M1Um",
    "https://openreview.net/pdf?id=AQkGpEMGWA",
    "https://openreview.net/pdf?id=Bg4Hn9Qq3w",
    "https://openreview.net/pdf?id=Cfj7uBu5dy",
    "https://openreview.net/pdf?id=Ciw6DbDa4U",
    "https://openreview.net/pdf?id=Ei3eF8B8XH",
    "https://openreview.net/pdf?id=EuACaJblk4",
    "https://openreview.net/pdf?id=Gzf8k2wPdF",
    "https://openreview.net/pdf?id=InZczCC8X1",
    "https://openreview.net/pdf?id=YKxwBMK8Nl",
    "https://openreview.net/pdf?id=a3LKICpDO2",
    "https://openreview.net/pdf?id=aECXy5Jgm4",
    "https://openreview.net/pdf?id=acfR6umMJt",
    "https://openreview.net/pdf?id=amn6lBDjXm",
    "https://openreview.net/pdf?id=auRe7zr32I",
    "https://openreview.net/pdf?id=bmgU7yWBeC",
    "https://openreview.net/pdf?id=cEgjPFdLvl",
    "https://openreview.net/pdf?id=cFTvHHXvt6",
    "https://openreview.net/pdf?id=ctyy8EJYQj",
    "https://openreview.net/pdf?id=dEtRvi7G5i",
    "https://openreview.net/pdf?id=dmeAH1hVR8",
    "https://openreview.net/pdf?id=e8bcQehZ15",
    "https://openreview.net/pdf?id=eUiZg9uUt4",
    "https://openreview.net/pdf?id=egi8g2U0ZX",
    "https://openreview.net/pdf?id=enQdbinvNd",
    "https://openreview.net/pdf?id=farKrjdsIH",
    "https://openreview.net/pdf?id=g6Sj1OFjAu",
    "https://openreview.net/pdf?id=gifMFKvAl5",
    "https://openreview.net/pdf?id=hFzjgQzoVU",
    "https://openreview.net/pdf?id=hQCdhenqre",
    "https://openreview.net/pdf?id=hk6iX4mg3B",
    "https://openreview.net/pdf?id=iFHaZzs6Kz",
    "https://openreview.net/pdf?id=j3aOU8Ahue",
]

# Prompts from the paper (Box in Appendix)
TASK1_SUMMARY_SYSTEM = (
    "You are a helpful assistant for summarizing key details of experiments "
    "and methodologies from scientific papers."
)

TASK1_SUMMARY_USER = (
    "Summarize the following research paper, focusing ONLY on this question:\n"
    "Carefully analyze ONLY the experiments performed or methods used.\n"
    "Do NOT include results, abstract, introduction, or discussion.\n"
    "Output MUST be valid JSON of the form:\n"
    '{\n'
    '  "title": "<paper title>",\n'
    '  "experiments_summary": "<concise summary>"\n'
    '}\n'
    "Do NOT wrap the JSON in markdown code fences.\n"
    "Paper text:\n"
)

TASK1_HYPOTHESIS_SYSTEM = (
    "You are a scientific reasoning assistant. Given a description of the "
    "experiments and methods from a research paper, infer the underlying "
    "hypothesis being tested - the core scientific claim the experiments were "
    "designed to validate. A hypothesis is a specific, testable, and falsifiable "
    "prediction about the relationship between variables. Output ONLY the "
    "hypothesis as a single declarative sentence. Do not include preamble, "
    "explanation, or any other text."
)

TASK1_HYPOTHESIS_USER = (
    "Generate a single testable hypothesis based on the experiment description "
    "above. Express it as one declarative sentence (e.g. 'If X, then Y because Z')."
)

TASK2_SYSTEM = (
    "You are an expert research scientist. Given the context of a research paper, "
    "your task is to generate a single novel hypothesis that logically extends "
    "beyond the paper's existing findings - not a restatement of them. The "
    "hypothesis must be: (1) grounded in a gap or open question identified in "
    "the paper, (2) specific and testable, (3) falsifiable. Output ONLY the "
    "hypothesis as a single declarative sentence with no preamble or explanation."
)

TASK2_USER = (
    "Based on the research context above, generate one novel hypothesis that "
    "extends beyond what this paper has already established."
)
