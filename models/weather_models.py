from pydantic import BaseModel, Field

class GetWeatherArgs(BaseModel):
    city: str = Field(description="Nama kota untuk mendapatkan informasi cuaca")