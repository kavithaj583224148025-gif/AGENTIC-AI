"""
Document Search Tool Implementation (Local File Search)
------------------------------------------------------
Searches and extracts relevant snippets from files in the documents/ folder.
No external vector databases or frameworks required (pure Python).
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

DOCUMENTS_DIR = Path(__file__).parent / "documents"


def get_available_documents() -> List[str]:
    """Returns a list of document filenames in the documents directory."""
    if not DOCUMENTS_DIR.exists():
        return []
    valid_exts = {".txt", ".md", ".json", ".csv", ".log", ".py"}
    return [
        f.name for f in DOCUMENTS_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in valid_exts
    ]


def search_file(file_path: Path, query_terms: List[str]) -> List[Dict[str, Any]]:
    """Searches a single file for occurrences of query terms by paragraph/block."""
    matches = []
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as err:
        return [{"error": f"Could not read {file_path.name}: {err}"}]

    # Split into paragraphs or logical sections
    blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
    
    for block in blocks:
        block_lower = block.lower()
        score = sum(1 for term in query_terms if term in block_lower)
        if score > 0:
            matches.append({
                "snippet": block,
                "score": score
            })

    # Sort blocks by match score descending
    matches.sort(key=lambda m: m["score"], reverse=True)
    return matches[:3]


def execute_document_search(query: str, filename: Optional[str] = None) -> Dict[str, Any]:
    """
    Main entry point for the document_search tool.
    
    Args:
        query: Keywords or question to search for in local documents.
        filename: Optional specific file name to target.
        
    Returns:
        dict: JSON-serializable dictionary with search matches.
    """
    if not DOCUMENTS_DIR.exists():
        DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        return {
            "status": "error",
            "error": f"The documents directory was empty. Created '{DOCUMENTS_DIR}'. Please add files."
        }

    available_files = get_available_documents()
    if not available_files:
        return {
            "status": "error",
            "error": f"No text or markdown files found in '{DOCUMENTS_DIR}'."
        }

    clean_query = query.strip()
    query_terms = [t.lower() for t in clean_query.split() if len(t) > 2]
    if not query_terms:
        query_terms = [clean_query.lower()]

    # If specific filename requested
    target_files = []
    if filename and filename.strip():
        target_name = filename.strip()
        matched_target = [f for f in available_files if f.lower() == target_name.lower()]
        if matched_target:
            target_files = [DOCUMENTS_DIR / matched_target[0]]
        else:
            return {
                "status": "error",
                "error": f"File '{filename}' not found. Available documents: {available_files}"
            }
    else:
        target_files = [DOCUMENTS_DIR / name for name in available_files]

    search_results = []
    for file_path in target_files:
        matches = search_file(file_path, query_terms)
        for match in matches:
            if "snippet" in match:
                search_results.append({
                    "file": file_path.name,
                    "relevance_score": match["score"],
                    "snippet": match["snippet"]
                })

    # Sort overall results by relevance
    search_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

    if not search_results:
        return {
            "status": "success",
            "query": clean_query,
            "count": 0,
            "message": f"No direct matches found for '{clean_query}'. Available files: {available_files}",
            "available_documents": available_files
        }

    return {
        "status": "success",
        "query": clean_query,
        "count": len(search_results),
        "results": search_results[:5],
        "available_documents": available_files
    }


if __name__ == "__main__":
    import json
    res = execute_document_search("leave policy annual vacation")
    print("Test Document Search:")
    print(json.dumps(res, indent=2))
