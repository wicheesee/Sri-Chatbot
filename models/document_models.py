from pydantic import BaseModel, Field
from typing import Optional

class DocumentSearchArgs(BaseModel):
    """Input untuk tool document search."""
    query: str = Field(description="Query atau pertanyaan untuk mencari dalam dokumen")
    top_k: int = Field(default=5, description="Jumlah hasil pencarian yang dikembalikan (1-10)", ge=1, le=10)
