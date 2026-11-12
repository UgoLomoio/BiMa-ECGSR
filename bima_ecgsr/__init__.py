"""BiMa-ECGSR: 12-lead ECG reconstruction from reduced lead sets with bidirectional Mamba models.

Models: BiMa-SR (regression, one forward pass) and DiBiMa-SR (diffusion, one-step or DDIM-6 sampling).
Paths are configured in :mod:`bima_ecgsr.config` (environment-variable overridable).
"""
__version__ = "0.1.0"
