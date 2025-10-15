"""
CGAMP: Graph-Based Causal Learning and Protein Language Models for Multi-Class Prediction of Antimicrobial Peptides.

This package includes:
- Graph neural network models (in Net/)
- Data preprocessing and protein graph generation (in utils/)
"""

__version__ = "1.0.0"
__author__ = "Kly Lab"

# from CGAMP import protein_init
from CGAMP.utils import protein_init
from CGAMP.Net import model_mol
