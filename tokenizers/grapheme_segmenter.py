# tokenizers/grapheme_segmenter.py
# -*- coding: utf-8 -*-
"""
Unicode grapheme segmentation for Indic Tokenization Lab.

We use the third-party `regex` module's support for \\X to split text into
**extended grapheme clusters**, following Unicode Text Segmentation (UAX #29).

This is the GPE-style pretokenization step: we treat grapheme clusters as
the atomic "characters" for downstream BPE training, as proposed in
Grapheme Pair Encoding (Velayuthan & Sarveswaran).
"""

from __future__ import annotations

from typing import Iterator, List

import regex as re  # pip install regex

# Basic Devanagari block for optional filtering
DEVANAGARI_START = 0x0900
DEVANAGARI_END = 0x097F

# Unicode extended grapheme cluster regex. \X matches one grapheme cluster.
_GRAPHEME_RE = re.compile(r"\X", re.UNICODE)

def iter_graphemes(text: str) -> Iterator[str]:
    """
    Yield Unicode extended grapheme clusters from `text`.

    This is generic (all scripts), and is the core primitive used by
    segment_devanagari_graphemes().
    """
    for match in _GRAPHEME_RE.finditer(text):
        yield match.group(0)

def _contains_devanagari(grapheme: str) -> bool:
    """
    Return True if this grapheme contains at least one Devanagari code point.
    """
    for ch in grapheme:
        cp = ord(ch)
        if DEVANAGARI_START <= cp <= DEVANAGARI_END:
            return True
    return False

def segment_devanagari_graphemes(
    text: str,
    keep_non_devanagari: bool = True,
) -> List[str]:
    """
    Segment `text` into extended grapheme clusters, focusing on Devanagari.

    Parameters
    ----------
    text : str
        Input Unicode text (can be mixed-script).
    keep_non_devanagari : bool
        - If True (default), keep all graphemes (Devanagari + others).
        - If False, filter to graphemes that contain at least one
          Devanagari code point.

    Returns
    -------
    List[str]
        A list of grapheme cluster strings, in order.
    """
    clusters: List[str] = []
    for g in iter_graphemes(text):
        if keep_non_devanagari:
            clusters.append(g)
        else:
            if _contains_devanagari(g):
                clusters.append(g)
    return clusters

def debug_print_graphemes(text: str) -> None:
    """
    Utility: print grapheme clusters, their indices, and code points.
    """
    for i, g in enumerate(iter_graphemes(text)):
        codepoints = " ".join(f"U+{ord(ch):04X}" for ch in g)
        print(f"{i:3d}: {repr(g)}  ({codepoints})")

def _main() -> None:
    """
    CLI entry point for debugging.

    Usage:
        python -m tokenizers.grapheme_segmenter "किशोरी"
        echo "यहाँ आपका हिंदी वाक्य जाएगा।" | python -m tokenizers.grapheme_segmenter
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Debug Unicode grapheme segmentation for Devanagari."
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Text to segment. If omitted, read from stdin.",
    )
    parser.add_argument(
        "--dev-only",
        action="store_true",
        help="Only print graphemes that contain Devanagari code points.",
    )

    args = parser.parse_args()

    if args.text is not None:
        text = args.text
    else:
        text = sys.stdin.read()

    if not text:
        print("No input text provided.", file=sys.stderr)
        sys.exit(1)

    if args.dev_only:
        clusters = segment_devanagari_graphemes(text, keep_non_devanagari=False)
    else:
        clusters = list(iter_graphemes(text))

    for i, g in enumerate(clusters):
        codepoints = " ".join(f"U+{ord(ch):04X}" for ch in g)
        print(f"{i:3d}: {repr(g)}  ({codepoints})")

if __name__ == "__main__":
    _main()

