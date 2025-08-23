from langchain.tools import StructuredTool
from typing import Dict, Any, List
import psycopg2
import os
from dotenv import load_dotenv
from models.db_models import GetUMKMByIdArgs, GetProductsByUMKMArgs, SearchUMKMByNameArgs

load_dotenv()

DB_HOST = os.environ.get("DB_HOST")
DB_PORT = os.environ.get("DB_PORT")
DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("DB_USER")
DB_PASS = os.environ.get("DB_PASS")

def get_connection():
    """Membuat koneksi ke database PostgreSQL"""
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )

def get_umkm_by_id(umkm_id: int) -> Dict[str, Any]:
    """Mengambil informasi UMKM berdasarkan ID"""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT umkm_id, umkm_name, umkm_about, umkm_notelp, umkm_email, umkm_alamat, umkm_sosmed
            FROM UMKM_Profile
            WHERE umkm_id = %s
        """, (umkm_id,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row:
            return {
                "umkm_id": row[0],
                "umkm_name": row[1],
                "umkm_about": row[2],
                "umkm_notelp": row[3],
                "umkm_email": row[4],
                "umkm_alamat": row[5],
                "umkm_sosmed": row[6]
            }
        return {"error": "UMKM tidak ditemukan"}
    except Exception as e:
        return {"error": str(e)}

def get_products_by_umkm(umkm_id: int) -> List[Dict[str, Any]]:
    """Mengambil semua produk berdasarkan UMKM ID"""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT product_id, product_name, product_desc, product_price, product_stock, product_image
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
                "product_image": r[5]
            })
        return products
    except Exception as e:
        return {"error": str(e)}

def search_umkm_by_name(name: str) -> List[Dict[str, Any]]:
    """Mencari UMKM berdasarkan nama (LIKE search)"""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT umkm_id, umkm_name, umkm_alamat, umkm_sosmed
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
                "umkm_sosmed": r[3]
            })
        return results
    except Exception as e:
        return {"error": str(e)}

db_tools = [
    StructuredTool.from_function(
        func=get_umkm_by_id,
        name="get_umkm_by_id",
        args_schema=GetUMKMByIdArgs,
        description="Mengambil detail UMKM berdasarkan ID"
    ),
    StructuredTool.from_function(
        func=get_products_by_umkm,
        name="get_products_by_umkm",
        args_schema=GetProductsByUMKMArgs,
        description="Mengambil daftar produk dari sebuah UMKM berdasarkan umkm_id"
    ),
    StructuredTool.from_function(
        func=search_umkm_by_name,
        name="search_umkm_by_name",
        args_schema=SearchUMKMByNameArgs,
        description="Mencari UMKM berdasarkan nama (mirip/LIKE)"
    ),
]