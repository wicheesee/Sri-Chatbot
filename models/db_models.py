from pydantic import BaseModel, Field
from typing import Optional

class GetUMKMByIdArgs(BaseModel):
    umkm_id: int = Field(description="ID dari UMKM yang ingin diambil informasinya")

class GetProductsByUMKMArgs(BaseModel):
    umkm_id: int = Field(description="ID dari UMKM untuk mengambil semua produknya")

class SearchUMKMByNameArgs(BaseModel):
    name: str = Field(description="Nama UMKM (atau sebagian nama) yang ingin dicari")

class SearchProductByNameArgs(BaseModel):
    product_name: str = Field(description="Nama produk (atau sebagian nama) yang ingin dicari")