import os
import json
import csv
import time
from pathlib import Path
from rag.config import load_config
from rag.pipeline import QueryPipeline
from rag.storage.db_manager import DatabaseManager
from rag.ingestion import ingest_path
from rag.generation.llm_client import LLMClient
from rag.models.model_manager import get_model_manager
from rich.console import Console

# Setup protobuf fallback for PaddleOCR on Windows
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

DATA_DIR = Path("tests/evaluation/definitive_test_data")
GT_FILE = DATA_DIR / "ground_truth.json"
REPORT_FILE = Path("evaluation_report.csv")
WORKSPACE_DIR = Path(os.path.expanduser("~/.motif/workspaces/evaluation"))

console = Console()

def evaluate_metrics(llm_client, query, expected, actual, contexts):
    """Uses the local model to grade Faithfulness and Correctness."""
    
    # Correctness
    correctness_prompt = f"""You are an impartial grading assistant.
Evaluate if the 'Generated Answer' correctly answers the 'Query' based on the 'Expected Answer'.
Score from 0.0 (completely wrong) to 1.0 (perfectly correct). 
Only output a single float number between 0.0 and 1.0. Do not output anything else.

Query: {query}
Expected Answer: {expected}
Generated Answer: {actual}
"""
    correctness_prompt_str = f"<|im_start|>user\n{correctness_prompt}<|im_end|>\n<|im_start|>assistant\n"
    try:
        correctness_res = llm_client.generate(correctness_prompt_str, temperature=0.0, max_tokens=10)
        # Parse float
        correctness_score = float(correctness_res.strip().replace("Score:", "").strip())
    except Exception as e:
        console.print(f"[red]Error parsing correctness score: {e}[/red]")
        correctness_score = 0.0

    # Faithfulness
    context_str = "\n".join([c.excerpt for c in contexts])
    faithfulness_prompt = f"""You are an impartial grading assistant.
Evaluate if the 'Generated Answer' is fully supported by the 'Context'. 
Score from 0.0 (hallucinated or ungrounded) to 1.0 (fully supported by context).
Only output a single float number between 0.0 and 1.0. Do not output anything else.

Context:
{context_str}

Generated Answer: {actual}
"""
    faithfulness_prompt_str = f"<|im_start|>user\n{faithfulness_prompt}<|im_end|>\n<|im_start|>assistant\n"
    try:
        if not contexts:
            faithfulness_score = 0.0
        else:
            faithfulness_res = llm_client.generate(faithfulness_prompt_str, temperature=0.0, max_tokens=10)
            faithfulness_score = float(faithfulness_res.strip().replace("Score:", "").strip())
    except Exception as e:
        console.print(f"[red]Error parsing faithfulness score: {e}[/red]")
        faithfulness_score = 0.0

    return correctness_score, faithfulness_score


def main():
    if not GT_FILE.exists():
        console.print("[red]Ground truth file not found! Run prepare_test_data.py first.[/red]")
        return
        
    with open(GT_FILE, "r") as f:
        ground_truth = json.load(f)

    # 1. Setup RAG configuration for Evaluation Workspace
    config = load_config()
    
    # We will use the evaluation workspace
    os.environ["MOTIF_WORKSPACE"] = str(WORKSPACE_DIR)
    
    import shutil
    # shutil.rmtree(WORKSPACE_DIR, ignore_errors=True)
    
    # Database schema is created automatically if missing by the stores.
    
    # 2. Ingest
    console.print(f"[blue]Ingesting {DATA_DIR}...[/blue]")
    ingest_result = ingest_path(DATA_DIR, config, recursive=False, console=console)
    console.print(f"[green]Ingested {ingest_result.files_processed} files, {ingest_result.chunks_added} chunks.[/green]")
    
    # 3. Setup Pipeline & Evaluator LLM
    pipeline = QueryPipeline(config)
    evaluator_llm = get_model_manager().get_llm(config)
    
    results = []
    
    for i, item in enumerate(ground_truth):
        query = item["query"]
        expected = item["expected_answer"]
        fmt = item["format"]
        
        console.print(f"\n[cyan]Evaluating {i+1}/{len(ground_truth)} [{fmt.upper()}][/cyan]")
        console.print(f"Q: {query}")
        
        # Run Pipeline
        try:
            res = pipeline.answer(query, history=[])
            actual = res.text
            contexts = res.citations
            retrieval_latency = res.retrieval_latency_ms
            generation_latency = res.generation_latency_ms
        except Exception as e:
            console.print(f"[red]Pipeline error: {e}[/red]")
            actual = str(e)
            contexts = []
            retrieval_latency = 0
            generation_latency = 0
            
        console.print(f"A: {actual}")
            
        # Grade
        correctness, faithfulness = evaluate_metrics(evaluator_llm, query, expected, actual, contexts)
        
        console.print(f"[yellow]Score -> Correctness: {correctness}, Faithfulness: {faithfulness}[/yellow]")
        
        results.append({
            "format": fmt,
            "query": query,
            "expected_answer": expected,
            "generated_answer": actual,
            "correctness": correctness,
            "faithfulness": faithfulness,
            "retrieval_latency_ms": retrieval_latency,
            "generation_latency_ms": generation_latency,
            "passages_retrieved": len(contexts)
        })

    # Write report
    keys = results[0].keys()
    with open(str(REPORT_FILE), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
        
    console.print(f"\n[green]Evaluation complete. Results saved to {REPORT_FILE}[/green]")

if __name__ == "__main__":
    main()
