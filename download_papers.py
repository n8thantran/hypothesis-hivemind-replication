"""Download the 50 papers from the NeurIPS 2025 AI4Mat track."""
import os
import time
import requests
from pathlib import Path

PAPER_DIR = Path("/workspace/data/papers")
PAPER_DIR.mkdir(parents=True, exist_ok=True)

# All 50 paper URLs from the paper's Appendix
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

def download_paper(url, output_dir):
    """Download a single paper from OpenReview."""
    paper_id = url.split("id=")[1]
    output_path = output_dir / f"{paper_id}.pdf"
    
    if output_path.exists() and output_path.stat().st_size > 1000:
        print(f"  Already exists: {paper_id}")
        return True
    
    try:
        response = requests.get(url, timeout=30, headers={
            'User-Agent': 'Mozilla/5.0 (research paper download)'
        })
        if response.status_code == 200 and len(response.content) > 1000:
            output_path.write_bytes(response.content)
            print(f"  Downloaded: {paper_id} ({len(response.content)} bytes)")
            return True
        else:
            print(f"  Failed: {paper_id} (status={response.status_code}, size={len(response.content)})")
            return False
    except Exception as e:
        print(f"  Error: {paper_id}: {e}")
        return False

def main():
    print(f"Downloading {len(PAPER_URLS)} papers...")
    success = 0
    for i, url in enumerate(PAPER_URLS):
        print(f"[{i+1}/{len(PAPER_URLS)}]", end="")
        if download_paper(url, PAPER_DIR):
            success += 1
        time.sleep(1)  # Be polite
    print(f"\nDownloaded {success}/{len(PAPER_URLS)} papers")

if __name__ == "__main__":
    main()
