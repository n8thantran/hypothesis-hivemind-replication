"""Extract text from all downloaded PDFs using PyMuPDF."""
import json
import fitz  # PyMuPDF
from pathlib import Path

PAPER_DIR = Path("/workspace/data/papers")
OUTPUT_FILE = Path("/workspace/data/paper_texts.json")

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF file."""
    try:
        doc = fitz.open(str(pdf_path))
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n".join(text_parts)
    except Exception as e:
        print(f"Error extracting {pdf_path.name}: {e}")
        return ""

def main():
    paper_texts = {}
    pdf_files = sorted(PAPER_DIR.glob("*.pdf"))
    print(f"Extracting text from {len(pdf_files)} PDFs...")
    
    for pdf_path in pdf_files:
        paper_id = pdf_path.stem
        text = extract_text_from_pdf(pdf_path)
        if text:
            paper_texts[paper_id] = text
            print(f"  {paper_id}: {len(text)} chars")
        else:
            print(f"  {paper_id}: FAILED")
    
    OUTPUT_FILE.write_text(json.dumps(paper_texts, indent=2))
    print(f"\nExtracted text from {len(paper_texts)}/{len(pdf_files)} papers")
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
