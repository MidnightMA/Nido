#!/usr/bin/env python3
"""Convert HPLT Persian->English Marian translation model to CTranslate2 INT8 format."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def convert_marian_to_ct2(source_dir: Path, output_dir: Path, quantization: str = "int8") -> bool:
    print(f"Source model:   {source_dir}")
    print(f"Output CT2 dir: {output_dir}")
    print(f"Quantization:   {quantization}")

    try:
        import numpy as np
        import sentencepiece as spm
        from ctranslate2.converters import utils
        from ctranslate2.converters.marian import (
            MarianConverter,
            _get_model_config,
            _SUPPORTED_ACTIVATIONS,
            set_transformer_spec,
        )
        from ctranslate2.specs import transformer_spec

        npz_candidates = list(source_dir.glob("*.npz"))
        if not npz_candidates:
            print(f"Error: No .npz model file found in {source_dir}", file=sys.stderr)
            return False
        model_path = npz_candidates[0]
        print(f"Using Marian model: {model_path.name}")

        spm_candidates = list(source_dir.glob("*.spm")) + list(source_dir.glob("*.model"))
        if not spm_candidates:
            print(f"Error: No .spm model file found in {source_dir}", file=sys.stderr)
            return False
        spm_path = spm_candidates[0]
        print(f"Using SentencePiece vocabulary: {spm_path.name}")

        sp_proc = spm.SentencePieceProcessor()
        sp_proc.load(str(spm_path))
        vocab_size = sp_proc.get_piece_size()
        vocab_tokens = [sp_proc.id_to_piece(i) for i in range(vocab_size)]
        print(f"Loaded {len(vocab_tokens)} vocabulary tokens.")

        class HPLTMarianConverter(MarianConverter):
            def __init__(self, m_path: str, tokens: list[str]):
                super().__init__(m_path, [])
                self._vocab_tokens = tokens

            def _load(self):
                model = np.load(self._model_path)
                config = _get_model_config(model)
                vocabs = [self._vocab_tokens, self._vocab_tokens]

                activation = config.get("transformer-ffn-activation", "relu")
                pre_norm = "n" in config.get("transformer-preprocess", "")
                postprocess_emb = config.get("transformer-postprocess-emb", "")

                check = utils.ConfigurationChecker()
                check(
                    config.get("type", "transformer") == "transformer",
                    "Option --type must be 'transformer'",
                )
                check(
                    activation in _SUPPORTED_ACTIVATIONS,
                    f"Option --transformer-ffn-activation {activation} is not supported",
                )
                check.validate()

                alignment_layer = config.get("transformer-guided-alignment-layer", -1)
                if alignment_layer == "last":
                    alignment_layer = -1
                elif isinstance(alignment_layer, (int, str)):
                    alignment_layer = int(alignment_layer) - 1 if int(alignment_layer) > 0 else -1

                layernorm_embedding = "n" in postprocess_emb
                enc_depth = int(config.get("enc-depth", 6))
                dec_depth = int(config.get("dec-depth", 6))
                heads = int(config.get("transformer-heads", 8))

                m_spec = transformer_spec.TransformerSpec.from_config(
                    (enc_depth, dec_depth),
                    heads,
                    pre_norm=pre_norm,
                    activation=_SUPPORTED_ACTIVATIONS[activation],
                    alignment_layer=alignment_layer,
                    alignment_heads=1,
                    layernorm_embedding=layernorm_embedding,
                )
                set_transformer_spec(m_spec, model)
                m_spec.register_source_vocabulary(vocabs[0])
                m_spec.register_target_vocabulary(vocabs[-1])
                m_spec.config.add_source_eos = True
                return m_spec

        output_dir.mkdir(parents=True, exist_ok=True)
        converter = HPLTMarianConverter(str(model_path), vocab_tokens)
        converter.convert(output_dir=str(output_dir), quantization=quantization, force=True)

        _copy_spm_assets(source_dir, output_dir)
        print("Conversion and INT8 quantization succeeded.")
        return True
    except Exception as e:
        print(f"Conversion error: {e}", file=sys.stderr)
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
