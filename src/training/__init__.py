"""Torch-free logic for the restaurant-agent LoRA training workflow.

Everything here is importable without a GPU, torch, transformers, or a network:
config parsing, transcript -> chat-record conversion, tool-schema extraction, and
label masking (tokenizer only). The heavy lifting lives in ``scripts/train_lora.py``.
"""
