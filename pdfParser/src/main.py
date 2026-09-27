"""CLI entrypoint for extracting legal clause trees and Clause[] arrays from PDF and DOCX contracts.

Usage:
    python pdfParser/src/main.py sample_work_contract.pdf
    python pdfParser/src/main.py aaa.pdf --debug
    python pdfParser/src/main.py contract.pdf --output output/contract.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

try:
    from pdfParser.src.pipeline import ContractParserPipeline
except ImportError:
    try:
        from ml.src.pipeline import ContractParserPipeline
    except ImportError:
        from pipeline import ContractParserPipeline


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="General-purpose contract clause boundary detection and hierarchy extraction system."
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to contract file (.pdf or .docx)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output/contract.json",
        help="Target output JSON path (default: output/contract.json)",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Print human-readable debug output with block role annotations.",
    )
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_arguments()
    file_path = args.file_path

    # If relative path doesn't exist directly, check inside pdfParser folder as fallback
    if not os.path.exists(file_path):
        alt_path = os.path.join(_SRC_DIR.parent, file_path)
        if os.path.exists(alt_path):
            file_path = alt_path
        else:
            print(f"Error: Contract file '{file_path}' not found.", file=sys.stderr)
            sys.exit(1)

    print(f"[*] Processing contract document: {file_path}")
    pipeline = ContractParserPipeline()

    try:
        results = pipeline.parse_file(file_path)
    except Exception as e:
        print(f"Error processing contract: {e}", file=sys.stderr)
        sys.exit(1)

    if args.debug:
        print("\n" + "=" * 80)
        print("HUMAN-READABLE CLAUSE BOUNDARY DEBUG LOG")
        print("=" * 80)
        for block in results["debug_blocks"]:
            role = block["role"].ljust(15)
            snippet = block["text"]
            print(f"[{role}] {snippet}")
        print("=" * 80 + "\n")

    output_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    output_data = {
        "document": results["document"],
        "total_blocks": results["total_blocks"],
        "total_nodes": results["total_nodes"],
        "total_clauses": results["total_clauses"],
        "tree": results["tree"],
        "clauses": results["clauses"],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"[✓] Successfully extracted {results['total_clauses']} clauses (from {results['total_nodes']} tree nodes across {results['total_blocks']} blocks).")
    print(f"[✓] Output written to: {output_path}")


if __name__ == "__main__":
    main()
