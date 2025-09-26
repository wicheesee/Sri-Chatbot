import os
import uuid
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langgraph.store.redis import RedisStore
from langgraph.store.base import IndexConfig
from langchain.tools import StructuredTool
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel
from models.memory_models import EmptyArgs, SaveInfoArgs, AnalyzeMessageArgs, DeleteMemoryArgs, UpdateMemoryArgs
import re
from typing import List, Optional

load_dotenv()
REDIS_URL = os.getenv("REDIS_URL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Redis setup
index_config: IndexConfig = {
    "dims": 1536,
    "embed": GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001"),
    "ann_index_config": {"vector_type":     "vector"},
    "distance_type": "cosine",
}

# Global Redis store
redis_store = None
with RedisStore.from_conn_string(REDIS_URL, index=index_config) as _redis_store:
    _redis_store.setup()
    redis_store = _redis_store

# LLM untuk memory detection
memory_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
    convert_system_message_to_human=True,
)

def get_user_context(config: RunnableConfig) -> str:
    """
    WAJIB DIPANGGIL PERTAMA di setiap conversation untuk mendapatkan konteks user yang tersimpan. 
    Tool ini akan memberikan context lengkap tentang user untuk membantu memberikan response yang personal.
    SELALU gunakan tool ini terlebih dahulu sebelum merespons user.
    """
    try:
        user_id = config["configurable"]["user_id"]
        namespace = ("memories", user_id)
        
        # Search semua memori untuk konteks umum
        memories = redis_store.search(namespace, query="")
        
        if memories:
            context_info = []
            for i, memory in enumerate(memories, 1):
                # Add memory ID untuk reference saat delete/update
                memory_data = memory.value['data']
                memory_id = memory.key
                context_info.append(f"{i}. {memory_data} [ID: {memory_id[:8]}...]")
            
            result = f"Konteks yang tersimpan tentang user ({len(memories)} items):\n" + "\n".join(context_info)
            
            print(f"\n--- [TOOL: get_user_context] ---")
            print(f"User ID: {user_id}")
            print(f"Found {len(memories)} memories")
            print(f"--------------------------------\n")
            
            return result
        else:
            return "Belum ada informasi yang tersimpan tentang user ini."
            
    except Exception as e:
        return f"Error mengambil konteks user: {str(e)}"

def save_important_info(information: str, config: RunnableConfig) -> str:
    """
    Tool untuk menyimpan informasi penting tentang user yang disebutkan dalam percakapan.
    Gunakan tool ini ketika user memberikan informasi personal seperti nama, lokasi, preferensi, hobi, pekerjaan, dll.
    """
    try:
        user_id = config["configurable"]["user_id"]
        namespace = ("memories", user_id)
        memory_id = str(uuid.uuid4())
        redis_store.put(namespace, memory_id, {"data": information})
        
        print(f"\n--- [TOOL: save_important_info] ---")
        print(f"User ID: {user_id}")
        print(f"Saved: {information}")
        print(f"Memory ID: {memory_id}")
        print(f"------------------------------------\n")
        return f"✓ Informasi berhasil disimpan: {information} [ID: {memory_id[:8]}...]"
        
    except Exception as e:
        return f"Error menyimpan informasi: {str(e)}"

def delete_user_memory(memory_identifier: str, config: RunnableConfig) -> str:
    """
    Tool untuk menghapus memori user tertentu. 
    User bisa minta hapus dengan menyebut konten memori atau ID memori.
    Contoh: "hapus info alamat saya" atau "hapus memory tentang warna favorit"
    """
    try:
        user_id = config["configurable"]["user_id"]
        namespace = ("memories", user_id)
        
        # Get all memories dulu
        memories = redis_store.search(namespace, query="")
        
        if not memories:
            return "Tidak ada memori yang tersimpan untuk dihapus."
        
        deleted_count = 0
        deleted_items = []
        
        # Cari berdasarkan partial match atau memory ID
        for memory in memories:
            memory_data = memory.value['data'].lower()
            memory_id = memory.key
            identifier_lower = memory_identifier.lower()
            
            # Match berdasarkan content atau ID
            if (identifier_lower in memory_data or 
                memory_id.startswith(memory_identifier) or
                identifier_lower in memory_id.lower()):
                
                redis_store.delete(namespace, memory_id)
                deleted_items.append(memory.value['data'])
                deleted_count += 1
        
        print(f"\n--- [TOOL: delete_user_memory] ---")
        print(f"User ID: {user_id}")
        print(f"Deleted {deleted_count} memories")
        print(f"Items: {deleted_items}")
        print(f"-------------------------------------\n")
        
        if deleted_count > 0:
            return f"✓ Berhasil menghapus {deleted_count} memori: {', '.join(deleted_items)}"
        else:
            return f"Tidak ditemukan memori yang cocok dengan '{memory_identifier}'. Coba gunakan kata kunci yang lebih spesifik."
            
    except Exception as e:
        return f"Error menghapus memori: {str(e)}"

def clear_all_user_memory(config: RunnableConfig) -> str:
    """
    Tool untuk menghapus SEMUA memori user (reset lengkap).
    Hanya gunakan jika user secara eksplisit minta hapus semua data.
    """
    try:
        user_id = config["configurable"]["user_id"]
        namespace = ("memories", user_id)
        
        # Get all memories
        memories = redis_store.search(namespace, query="")
        
        if not memories:
            return "Tidak ada memori yang perlu dihapus."
        
        deleted_count = len(memories)
        
        # Delete semua memories
        for memory in memories:
            redis_store.delete(namespace, memory.key)
        
        print(f"\n--- [TOOL: clear_all_user_memory] ---")
        print(f"User ID: {user_id}")
        print(f"Deleted ALL {deleted_count} memories")
        print(f"---------------------------------------\n")
        
        return f"✓ Berhasil menghapus semua {deleted_count} memori user. Data Anda telah direset lengkap."
        
    except Exception as e:
        return f"Error menghapus semua memori: {str(e)}"

def update_user_memory(old_info: str, new_info: str, config: RunnableConfig) -> str:
    """
    Tool untuk mengupdate/mengoreksi informasi user yang sudah tersimpan.
    Contoh: update info "alamat" dari "Jakarta" ke "Bekasi"
    """
    try:
        user_id = config["configurable"]["user_id"]
        namespace = ("memories", user_id)
        
        # Cari memory yang match dengan old_info
        memories = redis_store.search(namespace, query="")
        updated_count = 0
        updated_items = []
        
        for memory in memories:
            memory_data = memory.value['data']
            memory_id = memory.key
            
            # Check if old_info ada di memory
            if old_info.lower() in memory_data.lower():
                # Replace dengan new_info
                updated_data = memory_data.replace(old_info, new_info)
                
                # Update di Redis
                redis_store.put(namespace, memory_id, {"data": updated_data})
                
                updated_items.append(f"'{memory_data}' → '{updated_data}'")
                updated_count += 1
        
        print(f"\n--- [TOOL: update_user_memory] ---")
        print(f"User ID: {user_id}")
        print(f"Updated {updated_count} memories")
        print(f"Changes: {updated_items}")
        print(f"-----------------------------------\n")
        
        if updated_count > 0:
            return f"✓ Berhasil mengupdate {updated_count} memori:\n" + "\n".join(updated_items)
        else:
            return f"Tidak ditemukan memori yang mengandung '{old_info}' untuk diupdate."
            
    except Exception as e:
        return f"Error mengupdate memori: {str(e)}"

def list_user_memories(config: RunnableConfig) -> str:
    """
    Tool untuk menampilkan daftar lengkap memori user dengan ID untuk referensi.
    Berguna untuk debugging atau saat user ingin review data yang tersimpan.
    """
    try:
        user_id = config["configurable"]["user_id"]
        namespace = ("memories", user_id)
        
        memories = redis_store.search(namespace, query="")
        
        if not memories:
            return "Belum ada memori yang tersimpan."
        
        memory_list = []
        for i, memory in enumerate(memories, 1):
            memory_data = memory.value['data']
            memory_id = memory.key
            created_time = memory.created_at if hasattr(memory, 'created_at') else 'N/A'
            
            memory_list.append(
                f"{i}. {memory_data}\n"
                f"   ID: {memory_id}\n"
                f"   Created: {created_time}"
            )
        
        result = f"Daftar Memori User ({len(memories)} total):\n\n" + "\n\n".join(memory_list)
        
        print(f"\n--- [TOOL: list_user_memories] ---")
        print(f"User ID: {user_id}")
        print(f"Listed {len(memories)} memories")
        print(f"-----------------------------------\n")
        
        return result
        
    except Exception as e:
        return f"Error menampilkan daftar memori: {str(e)}"

def analyze_and_save_info(user_message: str, config: RunnableConfig) -> str:
    """
    Tool untuk menganalisis pesan user dan otomatis menyimpan preferensi songket yang disebutkan.
    Menggunakan rule-based detection dulu untuk efisiensi, baru LLM jika diperlukan.
    """
    try:
        user_id = config["configurable"]["user_id"]
        
        # Rule-based detection untuk info preferensi songket & personal (lebih cepat)
        important_patterns = {
            'nama': r'nama\s+saya\s+(\w+)|saya\s+(\w+)|panggil\s+saya\s+(\w+)',
            'alamat': r'alamat\s+saya\s+([\w\s,]+)|saya\s+tinggal\s+di\s+([\w\s,]+)|domisili\s+di\s+([\w\s,]+)|dari\s+([\w\s,]+)',
            'warna_favorit': r'warna\s+favorit\s+saya\s+([\w\s]+)|saya\s+suka\s+warna\s+([\w\s]+)|warna\s+kesukaan\s+([\w\s]+)',
            'motif_favorit': r'motif\s+favorit\s+saya\s+([\w\s]+)|saya\s+suka\s+motif\s+([\w\s]+)|motif\s+kesukaan\s+([\w\s]+)|motif\s+(lepus|bungo|tabur|pucuk\s+rebung|cantik\s+manis|tajuk|pulir|limar)',
            'preferensi_baju': r'saya\s+suka\s+(kebaya|baju\s+kurung|kemeja|dress)\s*([\w\s]*)|lebih\s+suka\s+(pakai|pake)\s+([\w\s]+)|gaya\s+berpakaian\s+([\w\s]+)',
            'jenis_songket': r'saya\s+suka\s+songket\s+([\w\s]+)|songket\s+favorit\s+saya\s+([\w\s]+)|songket\s+(palembang|sumatera|tradisional|modern|klasik)',
            'acara_khusus': r'untuk\s+acara\s+(pernikahan|wisuda|formal|adat|tradisional)|mau\s+dipake\s+(nikah|wedding|wisuda|formal)',
            'budget_range': r'budget\s+saya\s+([\w\s]+)|harga\s+sekitar\s+([\d\.,juta\s]+)|kisaran\s+harga\s+([\d\.,juta\s]+)',
            'ukuran_baju': r'ukuran\s+saya\s+([SMLXLsmlxl\d]+)|size\s+([SMLXLsmlxl\d]+)|badan\s+saya\s+(kecil|sedang|besar)',
        }
        
        extracted_info = []
        message_lower = user_message.lower()
        
        for category, pattern in important_patterns.items():
            match = re.search(pattern, message_lower)
            if match:
                # Ambil group yang tidak None
                value = next((g for g in match.groups() if g), "")
                if value:
                    extracted_info.append(f"{category.title()}: {value.title()}")
        
        # Jika ada info terdeteksi dengan rule-based, simpan langsung
        if extracted_info:
            namespace = ("memories", user_id)
            saved_items = []
            
            for info in extracted_info:
                memory_id = str(uuid.uuid4())
                redis_store.put(namespace, memory_id, {"data": info})
                saved_items.append(info)
            
            print(f"\n--- [TOOL: analyze_and_save_info - RULE-BASED] ---")
            print(f"User ID: {user_id}")
            print(f"Detected: {', '.join(saved_items)}")
            print(f"-----------------------------------------------\n")
            
            return f"✓ Terdeteksi dan disimpan (rule-based): {'; '.join(saved_items)}"
        
        # Fallback ke LLM analysis untuk kasus yang tidak terdeteksi rule-based
        analysis_prompt = f"""
        Analisis pesan user berikut dan tentukan apakah ada informasi personal/preferensi songket yang perlu disimpan:

        Pesan user: "{user_message}"

        Tugas:
        1. Identifikasi informasi seperti: nama, alamat, warna favorit, motif kesukaan, preferensi baju, jenis songket, acara khusus, budget, ukuran, dll.
        2. Fokus pada preferensi yang berkaitan dengan songket Palembang dan seni tradisional
        3. Jika ada informasi penting, ekstrak dalam format yang jelas
        4. Jika tidak ada informasi penting, jawab "NONE"

        Format response:
        - Jika ada info penting: "INFO: [informasi yang diekstrak]"
        - Jika tidak ada: "NONE"

        Contoh:
        User: "Nama saya Widya, saya suka warna emas dan motif lepus"
        Response: "INFO: Nama user adalah Widya, menyukai warna emas, motif favorit adalah lepus"

        User: "Bagaimana cuaca hari ini?"
        Response: "NONE"
        """
        
        analysis_response = memory_llm.invoke([
            SystemMessage(content="Anda adalah AI analyzer yang bertugas mengekstrak informasi penting dari pesan user."),
            HumanMessage(content=analysis_prompt)
        ])
        
        analysis_result = analysis_response.content.strip()
        
        if analysis_result.startswith("INFO:"):
            # Extract informasi dan simpan
            info_to_save = analysis_result.replace("INFO:", "").strip()
            
            namespace = ("memories", user_id)
            memory_id = str(uuid.uuid4())
            redis_store.put(namespace, memory_id, {"data": info_to_save})
            
            print(f"\n--- [TOOL: analyze_and_save_info - LLM] ---")
            print(f"User ID: {user_id}")
            print(f"Analysis: {analysis_result}")
            print(f"Saved: {info_to_save}")
            print(f"-------------------------------------\n")
            
            return f"✓ Terdeteksi dan disimpan (LLM): {info_to_save}"
        else:
            return "Tidak ada informasi penting yang perlu disimpan dari pesan ini."
    except Exception as e:
        return f"Error dalam analisis: {str(e)}"


memory_tools = [
    StructuredTool.from_function(
        func=get_user_context,
        name="get_user_context",
        args_schema=EmptyArgs,
        description="WAJIB DIPANGGIL PERTAMA di setiap conversation untuk mendapatkan konteks user yang tersimpan. Gunakan tool ini sebelum merespons user untuk mendapatkan informasi personal yang sudah diketahui sebelumnya."
    ),
    StructuredTool.from_function(
        func=analyze_and_save_info,
        name="analyze_and_save_info", 
        args_schema=AnalyzeMessageArgs,
        description="Menganalisis pesan user dengan rule-based detection (cepat) dan LLM fallback untuk otomatis mendeteksi dan menyimpan preferensi songket, seni, dan informasi personal yang perlu diingat."
    ),
    StructuredTool.from_function(
        func=save_important_info,
        name="save_important_info",
        args_schema=SaveInfoArgs,
        description="Menyimpan informasi penting tentang user yang disebutkan dalam percakapan (nama, lokasi, preferensi, dll)."
    ),
    StructuredTool.from_function(
        func=delete_user_memory,
        name="delete_user_memory",
        args_schema=DeleteMemoryArgs,
        description="Menghapus memori user tertentu berdasarkan konten atau ID. Gunakan ketika user minta hapus informasi spesifik seperti 'hapus alamat saya' atau 'lupakan preferensi warna'."
    ),
    StructuredTool.from_function(
        func=update_user_memory,
        name="update_user_memory",
        args_schema=UpdateMemoryArgs,
        description="Mengupdate/mengoreksi informasi user yang sudah tersimpan. Gunakan ketika user mau ganti informasi lama dengan yang baru."
    ),
    StructuredTool.from_function(
        func=clear_all_user_memory,
        name="clear_all_user_memory",
        args_schema=EmptyArgs,
        description="Menghapus SEMUA memori user (reset lengkap). HANYA gunakan jika user secara eksplisit minta hapus semua data pribadi mereka."
    ),
    StructuredTool.from_function(
        func=list_user_memories,
        name="list_user_memories",
        args_schema=EmptyArgs,
        description="Menampilkan daftar lengkap semua memori user dengan ID. Berguna untuk debugging atau ketika user ingin review data yang tersimpan."
    )
]


# Tambahkan ini di kode untuk debug
print("RedisStore type:", type(redis_store))
print("Available methods:", dir(redis_store))
print("Search method doc:", redis_store.search.__doc__)

# Test apakah benar-benar menggunakan embedding
test_data = "nama saya John"
memory_id = "test_123"

# Simpan data
redis_store.put(namespace, memory_id, {"data": test_data})

# Search dengan query mirip
result1 = redis_store.search(namespace, query="John")
result2 = redis_store.search(namespace, query="nama John")  
result3 = redis_store.search(namespace, query="completely different query")

print("Results:", len(result1), len(result2), len(result3))