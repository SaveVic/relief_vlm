"""
relief
======
Experiment pipeline for temple relief analysis: ground truth from
segmentation masks, a grid of LLM calls via Replicate, and evaluation that
produces the paper's tables.

Normal workflow (see README):

    python -m relief fetch --zip-url "<Drive link>"
    python -m relief gt
    python -m relief split
    python -m relief plan --plan dev
    python -m relief smoke
    python -m relief run --plan dev
    python -m relief usage
    python -m relief evaluate
"""

__version__ = "0.2.0"
