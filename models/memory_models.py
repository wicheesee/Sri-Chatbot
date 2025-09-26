from pydantic import BaseModel, Field
from typing import Optional

class EmptyArgs(BaseModel):
    """No arguments needed"""
    pass

class SaveInfoArgs(BaseModel):
    information: str = Field(
        description="Informasi penting tentang user yang ingin disimpan (nama, alamat, preferensi, dll)"
    )

class AnalyzeMessageArgs(BaseModel):
    user_message: str = Field(
        description="Pesan user yang akan dianalisis untuk mengekstrak informasi penting"
    )

class DeleteMemoryArgs(BaseModel):
    memory_identifier: str = Field(
        description="Identifier untuk menghapus memori: bisa berupa kata kunci konten (contoh: 'alamat', 'warna favorit') atau ID memori parsial"
    )

class UpdateMemoryArgs(BaseModel):
    old_info: str = Field(
        description="Informasi lama yang ingin diubah/diupdate"
    )
    new_info: str = Field(
        description="Informasi baru sebagai pengganti informasi lama"
    )