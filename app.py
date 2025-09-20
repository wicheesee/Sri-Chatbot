from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from LangGraph import react_graph
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any, List, Union
import logging 
import uvicorn

logging.basicConfig(level=logging.INFO) # Basic logging
logger = logging.getLogger(__name__)

# Uploadfile
from fastapi.staticfiles import StaticFiles
from fastapi import UploadFile, File, Form
from fastapi.responses import JSONResponse
import os
from datetime import datetime
import magic  # python-magic untuk deteksi MIME type
from PIL import Image  # Pillow untuk validasi gambar
import io
import json
import re
import psycopg2
from dotenv import load_dotenv

load_dotenv()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Database connection
DB_HOST = os.environ.get("DB_HOST")
DB_PORT = os.environ.get("DB_PORT")
DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("DB_USER")
DB_PASS = os.environ.get("DB_PASS")

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
    user_id: str

class ChatResponse(BaseModel):
    reply: str
    data: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None
    images: List[str] = []

def get_db_connection():
    """Membuat koneksi ke database PostgreSQL"""
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )

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
        
        contents = file.file.read()
        file.file.seek(0)
        
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
            image.verify() 
        except Exception:
            return False, "File bukan gambar yang valid"
        return True, "Valid"
        
    except Exception as e:
        return False, f"Error validasi file: {str(e)}"

def extract_images_from_data(data: Union[Dict, List]) -> List[str]:
    """Extract image URLs from structured data"""
    images = []
    
    def find_images(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in ['product_image', 'umkm_image'] and value:
                    images.append(value)
                elif isinstance(value, (dict, list)):
                    find_images(value)
        elif isinstance(obj, list):
            for item in obj:
                find_images(item)
    
    find_images(data)
    return images

def extract_structured_data(messages):
    """Extract structured data (UMKM info, products, etc.) from tool responses"""
    structured_data = None
    images = []
    
    try:
        for message in messages:
            if not hasattr(message, 'content'):
                continue
                
            content = message.content
            if not isinstance(content, str):
                continue
                
            content = content.strip()
            
            if not content:
                continue
            
            try:
                # Try to parse JSON content
                if content.startswith('{') or content.startswith('['):
                    parsed_data = json.loads(content)
                    
                    # Handle new response format with status/data structure
                    if isinstance(parsed_data, dict):
                        if 'status' in parsed_data and 'data' in parsed_data:
                            # New format: {"status": "success", "data": [...]}
                            if parsed_data['status'] == 'success':
                                actual_data = parsed_data['data']
                                structured_data = parsed_data  # Keep full structure for frontend
                                images.extend(extract_images_from_data(actual_data))
                        else:
                            # Old format: direct data
                            structured_data = parsed_data
                            images.extend(extract_images_from_data(parsed_data))
                    elif isinstance(parsed_data, list):
                        # List of items (old format)
                        structured_data = {"status": "success", "data": parsed_data, "count": len(parsed_data)}
                        images.extend(extract_images_from_data(parsed_data))
                        
            except json.JSONDecodeError:
                urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
                for url in urls:
                    if any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.png']):
                        images.append(url)
                        
    except Exception as e:
        print(f"Error extracting structured data: {e}")
    
    return structured_data, list(set(images))  # Remove duplicates

@app.post("/upload-umkm")
async def upload_umkm_image(
    umkm_id: int = Form(...), 
    file: UploadFile = File(...)
):
    """Upload gambar profil UMKM dan update database"""
    try:
        # Validate image file
        is_valid, message = validate_image_file(file)
        if not is_valid:
            raise HTTPException(status_code=400, detail=message)
        
        # Check if UMKM exists
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT umkm_name FROM UMKM_Profile WHERE umkm_id = %s", (umkm_id,))
        umkm = cur.fetchone()
        
        if not umkm:
            cur.close()
            conn.close()
            raise HTTPException(status_code=404, detail=f"UMKM dengan ID {umkm_id} tidak ditemukan")
        
        # Generate safe filename
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        file_ext = os.path.splitext(file.filename.lower())[1]
        safe_filename = f"umkm_{umkm_id}_{timestamp}{file_ext}"
        file_path = os.path.join(UPLOAD_DIR, safe_filename)

        # Save file
        contents = await file.read()
        with open(file_path, "wb") as buffer:
            buffer.write(contents)
        
        # Update database dengan filename
        cur.execute("""
            UPDATE UMKM_Profile 
            SET umkm_image = %s 
            WHERE umkm_id = %s
        """, (safe_filename, umkm_id))
        conn.commit()
        cur.close()
        conn.close()
        
        file_url = f"http://localhost:8000/uploads/{safe_filename}"

        return JSONResponse({
            "status": "success",
            "message": f"Gambar profil UMKM '{umkm[0]}' berhasil diupload",
            "umkm_id": umkm_id,
            "umkm_name": umkm[0],
            "file_name": safe_filename,
            "file_url": file_url,
            "file_size": len(contents),
            "mime_type": magic.from_buffer(contents, mime=True)
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error upload file: {str(e)}")

@app.post("/upload-product")
async def upload_product_image(
    product_id: int = Form(...), 
    file: UploadFile = File(...)
):
    """Upload gambar produk dan update database"""
    try:
        # Validate image file
        is_valid, message = validate_image_file(file)
        if not is_valid:
            raise HTTPException(status_code=400, detail=message)
        
        # Check if product exists and get product + UMKM info
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT p.product_name, p.umkm_id, u.umkm_name 
            FROM Product p 
            JOIN UMKM_Profile u ON p.umkm_id = u.umkm_id 
            WHERE p.product_id = %s
        """, (product_id,))
        product = cur.fetchone()
        
        if not product:
            cur.close()
            conn.close()
            raise HTTPException(status_code=404, detail=f"Produk dengan ID {product_id} tidak ditemukan")
        
        product_name, umkm_id, umkm_name = product
        
        # Generate safe filename
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        file_ext = os.path.splitext(file.filename.lower())[1]
        safe_filename = f"product_{product_id}_{timestamp}{file_ext}"
        file_path = os.path.join(UPLOAD_DIR, safe_filename)

        # Save file
        contents = await file.read()
        with open(file_path, "wb") as buffer:
            buffer.write(contents)
        
        # Update database with filename
        cur.execute("""
            UPDATE Product 
            SET product_image = %s 
            WHERE product_id = %s
        """, (safe_filename, product_id))
        conn.commit()
        cur.close()
        conn.close()
        
        file_url = f"http://localhost:8000/uploads/{safe_filename}"

        return JSONResponse({
            "status": "success",
            "message": f"Gambar produk '{product_name}' berhasil diupload",
            "product_id": product_id,
            "product_name": product_name,
            "umkm_id": umkm_id,
            "umkm_name": umkm_name,
            "file_name": safe_filename,
            "file_url": file_url,
            "file_size": len(contents),
            "mime_type": magic.from_buffer(contents, mime=True)
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error upload file: {str(e)}")


@app.get("/umkm/{umkm_id}")
async def get_umkm_info(umkm_id: int):
    """Get UMKM info by ID"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT umkm_id, umkm_name, umkm_about, umkm_notelp, umkm_email, 
                   umkm_alamat, umkm_sosmed, umkm_image
            FROM UMKM_Profile
            WHERE umkm_id = %s
        """, (umkm_id,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        
        if not row:
            raise HTTPException(status_code=404, detail="UMKM tidak ditemukan")
        
        return {
            "umkm_id": row[0],
            "umkm_name": row[1],
            "umkm_about": row[2],
            "umkm_notelp": row[3],
            "umkm_email": row[4],
            "umkm_alamat": row[5],
            "umkm_sosmed": row[6],
            "umkm_image": f"http://localhost:8000/uploads/{row[7]}" if row[7] else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/product/{product_id}")
async def get_product_info(product_id: int):
    """Get product info by ID"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT p.product_id, p.product_name, p.product_desc, p.product_price, 
                   p.product_stock, p.product_image, p.umkm_id, u.umkm_name
            FROM Product p
            JOIN UMKM_Profile u ON p.umkm_id = u.umkm_id
            WHERE p.product_id = %s
        """, (product_id,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        
        if not row:
            raise HTTPException(status_code=404, detail="Produk tidak ditemukan")
        
        return {
            "product_id": row[0],
            "product_name": row[1],
            "product_desc": row[2],
            "product_price": float(row[3]) if row[3] else None,
            "product_stock": row[4],
            "product_image": f"http://localhost:8000/uploads/{row[5]}" if row[5] else None,
            "umkm_id": row[6],
            "umkm_name": row[7]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to see all file that i've uploaded 
async def list_uploaded_files():
    try:
        files = []
        for filename in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, filename)
            if os.path.isfile(file_path):
                files.append({
                    "filename": filename,
                    "url": f"http://localhost:8000/uploads/{filename}",
                    "size": os.path.getsize(file_path),
                    "created_at": datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
                })
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        # Import dari orchestrator
        from LangGraph import react_graph
        
        # Config dengan user_id yang passed via configurable
        config = {
            "configurable": {
                "thread_id": "1234567",
                "user_id": request.user_id  # Untuk tools akses via config
            }
        }
        
        logger.info(f"Processing chat for user_id: {request.user_id}")
        
        # Direct invoke dengan MessagesState (no initial_state needed)
        result = react_graph.invoke(
            {"messages": [HumanMessage(content=request.message)]},  # Built-in MessagesState
            config=config  # Tools ambil user_id dari sini
        )
        
        # Debug output
        for m in result["messages"]:
            try:
                m.pretty_print()
            except Exception as e:
                print(f"[RAW] {m}")
        
        # Extract final response
        responses = [m.content for m in result["messages"] if hasattr(m, "content") and m.content]
        final_reply = responses[-1] if responses else "Maaf, saya tidak bisa memberikan respons."
        
        # Extract structured data and images (dari function yang sudah ada)
        structured_data, images = extract_structured_data(result["messages"])
        
        print(f"DEBUG: Extracted structured_data: {structured_data}")
        print(f"DEBUG: Extracted images: {images}")
        
        response = ChatResponse(
            reply=final_reply,
            data=structured_data,
            images=images
        )

        print(f"DEBUG: Final response: {response}")
        return response
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        import traceback
        traceback.print_exc()
        
        return ChatResponse(
            reply="Maaf, terjadi kesalahan saat memproses permintaan Anda. Silakan coba lagi.",
            data={"status": "error", "message": str(e)},
            images=[]
        )
    
# Endpoint untuk testing memory
@app.get("/memory/{user_id}")
def get_user_memory(user_id: str):
    """Endpoint untuk melihat semua memori user (untuk debugging)"""
    try:
        from tools.memory_tool import redis_store
        namespace = ("memories", user_id)
        memories = redis_store.search(namespace, query="")
        
        return {
            "user_id": user_id,
            "total_memories": len(memories),
            "memories": [d.value["data"] for d in memories]
        }
    except Exception as e:
        return {"error": str(e)}

port=int(os.getenv("PORT"))

if __name__ == '__main__':
    logger.info(f"Starting Uvicorn server on http://0.0.0.0:{port}")
    uvicorn.run("app:app", host="127.0.0.1", port=port, reload=True, log_level="info")
