from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


Device = Literal["auto", "cpu", "cuda"]
LLMProvider = Literal["groq"]


@dataclass(frozen=True)
class Settings:
    # Corpus source can be:
    # - a single .txt file (recommended for this repo)
    # - a directory containing .txt files
    # - a directory containing .pdf files
    # - a single .pdf file
    pdf_dir: Path = Path("Books/text.txt")
    vector_db_dir: Path = Path("vector_db")
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_provider: LLMProvider = "groq"
    groq_model: str = "llama-3.3-70b-versatile"
    device: Device = "auto"
    top_k: int = 8
    temperature: float = 0.3

    def resolved_device(self) -> Literal["cpu", "cuda"]:
        if self.device in ("cpu", "cuda"):
            return self.device
        try:
            import torch  # type: ignore

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
