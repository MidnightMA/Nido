#!/usr/bin/env python3
"""Convert HPLT Persian->English Marian translation model to CTranslate2 INT8 format."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def convert_marian_to_ct2(source_dir: Path, output_dir: Path, quantization: str = "int8") -> bool:
    print(f"Source model:  {source_dir}")
    print(f"Output CT2 dir: {output_dir}")
    print(f"Quantization:  {quantization}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Try ct2-transformers-converter CLI
    converter = shutil.which("ct2-transformers-converter") or shutil.which("ct2-opus-mt-converter")
    if converter:
        cmd = [
            converter,
            "--model_dir",
            str(source_dir),
            "--output_dir",
            str(output_dir),
            "--quantization",
            quantization,
            "--force",
        ]
        print(f"Executing: {' '.join(cmd)}")
        res = subprocess.run(cmd)
        if res.returncode == 0:
            print("Conversion succeeded via CLI converter.")
            _copy_spm_assets(source_dir, output_dir)
            return True

    # 2. Try Python CTranslate2 Converter API
    try:
        import ctranslate2.converters

        print("Using ctranslate2.converters.TransformersConverter...")
        conv = ctranslate2.converters.TransformersConverter(
            model_name_or_path=str(source_dir),
            copy_files=["source.spm", "target.spm", "spiece.model"],
        )
        conv.convert(
            output_dir=str(output_dir),
            quantization=quantization,
            force=True,
        )
        print("Conversion succeeded via Python API.")
        _copy_spm_assets(source_dir, output_dir)
        return True
    except Exception as e:
        print(f"Python API conversion error: {e}", file=sys.stderr)
        return False


def _copy_spm_assets(source_dir: Path, output_dir: Path) -> None:
    """Ensure SentencePiece tokenizer models are present in output directory."""
    for pattern in ("*.spm", "*.model", "*vocab*.json"):
        for src_file in source_dir.glob(pattern):
            dest = output_dir / src_file.name
            if not dest.exists():
                print(f"Copying tokenizer asset: {src_file.name} -> {dest}")
                shutil.copy(src_file, dest)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Marian translation model to CTranslate2 INT8.")
    parser.add_argument(
        "--source",
        type=str,
        default="~/.local/share/nido/models/marian_raw_hplt",
        help="Path to raw downloaded Marian model.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="~/.local/share/nido/models/translation-ct2",
        help="Output directory for CTranslate2 model.",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="int8",
        choices=["int8", "int16", "float16", "float32"],
        help="Quantization type (default: int8 for fast CPU inference).",
    )

    args = parser.parse_args()
    source_p = Path(args.source).expanduser().resolve()
    output_p = Path(args.output).expanduser().resolve()

    if not source_p.exists():
        print(f"Error: Source directory does not exist: {source_p}", file=sys.stderr)
        return 1

    success = convert_marian_to_ct2(source_p, output_p, quantization=args.quantization)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
