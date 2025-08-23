from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from Langgraph import react_graph 
from fastapi.middleware.cors import CORSMiddleware

# Uploadfile
from fastapi.staticfiles import StaticFiles
from fastapi import UploadFile, File, Form
from fastapi.responses import JSONResponse
import os
from datetime import datetime
import magic  # python-magic untuk deteksi MIME type
from PIL import Image  # Pillow untuk validasi gambar
import io

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Allowed image formats
ALLOWED_IMAGE_TYPES = {
    "image/jpeg": [".jpg", ".jpeg"],
    "image/png": [".png"]
}

# Maximum file size (5MB)
MAX_FILE_SIZE = 5 * 1024 * 1024

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

class ChatRequest(BaseModel):
    message: str

def validate_image_file(file: UploadFile) -> tuple[bool, str]:
    """Validasi file gambar"""
    try:
        # Check file size
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset to beginning
        
        if file_size > MAX_FILE_SIZE:
            return False, f"File terlalu besar. Maksimal {MAX_FILE_SIZE // (1024*1024)}MB"
        
        if file_size == 0:
            return False, "File kosong"
        
        # Read file content
        contents = file.file.read()
        file.file.seek(0)  
        
        # Check MIME type using python-magic
        mime_type = magic.from_buffer(contents, mime=True)
        
        if mime_type not in ALLOWED_IMAGE_TYPES:
            return False, f"Format file tidak didukung. Hanya menerima: PNG, JPG, JPEG"
        
        # Validate file extension
        file_ext = os.path.splitext(file.filename.lower())[1]
        if file_ext not in ALLOWED_IMAGE_TYPES[mime_type]:
            return False, f"Ekstensi file tidak sesuai dengan format gambar"
        
        # Try to open and validate as image using PIL
        try:
            image = Image.open(io.BytesIO(contents))
            image.verify()  # Verify it's a valid image
        except Exception:
            return False, "File bukan gambar yang valid"
        
        return True, "Valid"
        
    except Exception as e:
        return False, f"Error validasi file: {str(e)}"

@app.post("/chat")
def chat(request: ChatRequest):
    messages = [HumanMessage(content=request.message)]
    result = react_graph.invoke({"messages": messages})
    for m in result["messages"]:
        try:
            m.pretty_print()
        except Exception as e:
            print(f"[RAW] {m}")
    responses = [m.content for m in result["messages"] if hasattr(m, "content")]
    return {"reply": responses[-1] if responses else ""}

@app.post("/upload")
async def upload_umkm_file(
    umkm_id: int = Form(...), 
    file: UploadFile = File(...)
):
    try:
        # Validate image file
        is_valid, message = validate_image_file(file)
        if not is_valid:
            raise HTTPException(status_code=400, detail=message)
        
        # Generate safe filename
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        file_ext = os.path.splitext(file.filename.lower())[1]
        safe_filename = f"{umkm_id}_{timestamp}{file_ext}"
        file_path = os.path.join(UPLOAD_DIR, safe_filename)

        # Save file
        contents = await file.read()
        with open(file_path, "wb") as buffer:
            buffer.write(contents)
        
        file_url = f"http://localhost:8000/uploads/{safe_filename}"

        return JSONResponse({
            "status": "success",
            "message": "Gambar berhasil diupload",
            "file_name": safe_filename,
            "file_url": file_url,
            "file_size": len(contents),
            "mime_type": magic.from_buffer(contents, mime=True)
        })  
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error upload file: {str(e)}")

# Endpoint tambahan untuk mendapatkan info file
@app.get("/upload/{filename}")
async def get_file_info(filename: str):
    try:
        file_path = os.path.join(UPLOAD_DIR, filename)
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File tidak ditemukan")
        
        file_size = os.path.getsize(file_path)
        with open(file_path, "rb") as f:
            mime_type = magic.from_buffer(f.read(1024), mime=True)
        
        return {
            "filename": filename,
            "file_size": file_size,
            "mime_type": mime_type,
            "created_at": datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))