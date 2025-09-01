from langchain.tools import StructuredTool
from typing import Dict, Any, List, Optional
import psycopg2
import os
from dotenv import load_dotenv
from models.db_models import GetUMKMByIdArgs, GetProductsByUMKMArgs, SearchUMKMByNameArgs, SearchProductByNameArgs

load_dotenv()

DB_HOST = os.environ.get("DB_HOST")
DB_PORT = os.environ.get("DB_PORT")
DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("DB_USER")
DB_PASS = os.environ.get("DB_PASS")

# Base URL untuk file upload
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")

def get_connection():
    """Membuat koneksi ke database PostgreSQL"""
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )

def format_image_url(image_filename: str) -> Optional[str]:
    """Format image filename menjadi full URL"""
    if not image_filename:
        return None
    # Jika sudah full URL, return as is
    if image_filename.startswith('http'):
        return image_filename
    # Jika hanya filename, buat full URL
    return f"{BASE_URL}/uploads/{image_filename}"

def get_umkm_by_id(umkm_id: int) -> Dict[str, Any]:
    """Mengambil informasi UMKM berdasarkan ID dengan image URL"""
    try:
        conn = get_connection()
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
        
        if row:
            return {
                "status": "success",
                "data": {
                    "umkm_id": row[0],
                    "umkm_name": row[1],
                    "umkm_about": row[2],
                    "umkm_notelp": row[3],
                    "umkm_email": row[4],
                    "umkm_alamat": row[5],
                    "umkm_sosmed": row[6],
                    "umkm_image": format_image_url(row[7]) if len(row) > 7 and row[7] else None
                }
            }
        return {"status": "error", "message": "UMKM tidak ditemukan"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_products_by_umkm(umkm_id: int) -> Dict[str, Any]:
    """Mengambil semua produk berdasarkan UMKM ID dengan image URLs"""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT product_id, product_name, product_desc, product_price, 
                   product_stock, product_image
            FROM Product
            WHERE umkm_id = %s
        """, (umkm_id,))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        products = []
        for r in rows:
            products.append({
                "product_id": r[0],
                "product_name": r[1],
                "product_desc": r[2],
                "product_price": float(r[3]) if r[3] else None,
                "product_stock": r[4],
                "product_image": format_image_url(r[5])
            })
        
        return {
            "status": "success",
            "data": products,
            "count": len(products)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def search_umkm_by_name(name: str) -> Dict[str, Any]:
    """Mencari UMKM berdasarkan nama (LIKE search) dengan image URLs"""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT umkm_id, umkm_name, umkm_alamat, umkm_sosmed, umkm_image
            FROM UMKM_Profile
            WHERE umkm_name ILIKE %s
        """, (f"%{name}%",))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        results = []
        for r in rows:
            results.append({
                "umkm_id": r[0],
                "umkm_name": r[1],
                "umkm_alamat": r[2],
                "umkm_sosmed": r[3],
                "umkm_image": format_image_url(r[4]) if len(r) > 4 and r[4] else None
            })
        
        return {
            "status": "success",
            "data": results,
            "count": len(results)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def search_product_by_name(product_name: str) -> Dict[str, Any]:
    """Mencari produk berdasarkan nama dengan image URLs"""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT p.product_id, p.product_name, p.product_desc, p.product_price, 
                   p.product_stock, p.product_image, u.umkm_name, p.umkm_id
            FROM Product p
            JOIN UMKM_Profile u ON p.umkm_id = u.umkm_id
            WHERE p.product_name ILIKE %s
        """, (f"%{product_name}%",))
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        products = []
        for r in rows:
            products.append({
                "product_id": r[0],
                "product_name": r[1],
                "product_desc": r[2],
                "product_price": float(r[3]) if r[3] else None,
                "product_stock": r[4],
                "product_image": format_image_url(r[5]),
                "umkm_name": r[6],
                "umkm_id": r[7]
            })
        
        return {
            "status": "success",
            "data": products,
            "count": len(products)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

db_tools = [
    StructuredTool.from_function(
        func=get_umkm_by_id,
        name="get_umkm_by_id",
        args_schema=GetUMKMByIdArgs,
        description="Mengambil detail UMKM berdasarkan ID, termasuk gambar profil UMKM"
    ),
    StructuredTool.from_function(
        func=get_products_by_umkm,
        name="get_products_by_umkm",
        args_schema=GetProductsByUMKMArgs,
        description="Mengambil daftar produk dari sebuah UMKM berdasarkan umkm_id, termasuk gambar produk"
    ),
    StructuredTool.from_function(
        func=search_umkm_by_name,
        name="search_umkm_by_name",
        args_schema=SearchUMKMByNameArgs,
        description="Mencari UMKM berdasarkan nama (mirip/LIKE), termasuk gambar profil UMKM"
    ),
    StructuredTool.from_function(
        func=search_product_by_name,
        name="search_product_by_name",
        args_schema=SearchProductByNameArgs,
        description="Mencari produk berdasarkan nama (mirip/LIKE), termasuk gambar produk dan info UMKM"
    ),
]