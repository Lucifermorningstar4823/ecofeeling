"""
Echo Feeling
============
Multimodal sentiment analysis library for e-commerce review mining.
Integrates text, emoji, and sticker modalities via a six-phase pipeline.

Authors: Swathi Lekshmi SS, Romin Varghese, R Arun, Ivin Issac
Supervisor: Hridhya J, College of Engineering Karunagappally
Paper: AI-Powered Review Mining for Subjective Text Understanding (SAMANWAYA'26)
"""

from .api import analyze, analyze_multimodal
from .engine import EchoFeelingEngine

__version__ = "2.0.0"
__all__ = ["analyze", "analyze_multimodal", "EchoFeelingEngine"]
