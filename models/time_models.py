from pydantic import BaseModel, Field

class GetCurrentTimeArgs(BaseModel):
    """Input untuk tool get_current_time."""
    timezone: str = Field(default="Asia/Jakarta", description="Zona waktu untuk mendapatkan waktu saat ini, contoh: 'Asia/Jakarta' atau 'America/New_York'.")
