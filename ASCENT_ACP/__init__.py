"""ASCENT-ACP: Atmospheric Suborbital Classification & Evaluation Network Tool - Aerosol, Cloud, and Precipitation."""

from .ascent_acp import run_ascent_acp_merge
from .config import PipelineConfig

__all__ = ['run_ascent_acp_merge', 'PipelineConfig', 'run_pipeline']


def run_pipeline(*args, **kwargs):
    """Lazy wrapper for ASCENT_ACP.pipeline.run_pipeline (defers heavy imports)."""
    from .pipeline import run_pipeline as _run

    return _run(*args, **kwargs)

__version__ = '0.1.0'


