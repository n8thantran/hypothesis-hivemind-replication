"""Download papers from OpenReview and extract text."""
import os
import time
import requests
import fitz  # PyMuPDF
import json
from config import PAPER_URLS, PAPERS_DIR, DATA_DIR

def download_pdf(url, filepath, max_retries=3):
    """Download a PDF from a URL."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=60, headers={
                "User-Agent": "Mozilla/5.0 (research replication)"
            })
            if resp.status_code == 200 and len(resp.content) > 1000:
                with open(filepath, "wb") as f:
                    f.write(resp.content)
                return True
            else:
                print(f"  Attempt {attempt+1}: status={resp.status_code}, size={len(resp.content)}")
        except Exception as e:
            print(f"  Attempt {attempt+1}: error={e}")
        time.sleep(2 * (attempt + 1))
    return False

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()
    except Exception as e:
        print(f"  Error extracting text: {e}")
        return ""

def get_paper_id(url):
    """Extract paper ID from OpenReview URL."""
    return url.split("id=")[-1]

def main():
    os.makedirs(PAPERS_DIR, exist_ok=True)
    
    papers = {}
    
    for i, url in enumerate(PAPER_URLS):
        paper_id = get_paper_id(url)
        pdf_path = os.path.join(PAPERS_DIR, f"{paper_id}.pdf")
        txt_path = os.path.join(PAPERS_DIR, f"{paper_id}.txt")
        
        print(f"[{i+1}/{len(PAPER_URLS)}] Paper {paper_id}")
        
        # Download PDF if not already present
        if not os.path.exists(pdf_path):
            print(f"  Downloading...")
            success = download_pdf(url, pdf_path)
            if not success:
                print(f"  FAILED to download {paper_id}")
                continue
            time.sleep(1)  # Be polite
        else:
            print(f"  PDF already exists")
        
        # Extract text if not already done
        if not os.path.exists(txt_path):
            print(f"  Extracting text...")
            text = extract_text_from_pdf(pdf_path)
            if text:
                with open(txt_path, "w") as f:
                    f.write(text)
                print(f"  Extracted {len(text)} chars")
            else:
                print(f"  FAILED to extract text")
                continue
        else:
            with open(txt_path, "r") as f:
                text = f.read()
            print(f"  Text already extracted ({len(text)} chars)")
        
        papers[paper_id] = {
            "url": url,
            "pdf_path": pdf_path,
            "txt_path": txt_path,
            "text_length": len(text),
        }
    
    # Save paper index
    index_path = os.path.join(DATA_DIR, "paper_index.json")
    with open(index_path, "w") as f:
        json.dump(papers, f, indent=2)
    
    print(f"\nDownloaded and extracted {len(papers)}/{len(PAPER_URLS)} papers")
    print(f"Index saved to {index_path}")

if __name__ == "__main__":
    main()
