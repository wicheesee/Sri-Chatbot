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
from models.memory_models import EmptyArgs, SaveInfoArgs, AnalyzeMessageArgs

load_dotenv()
REDIS_URL = os.getenv("REDIS_URL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Redis setup
index_config: IndexConfig = {
    "dims": 1536,
    "embed": GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001"),
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
            for memory in memories:
                context_info.append(f"- {memory.value['data']}")
            
            result = f"Konteks yang tersimpan tentang user:\n" + "\n".join(context_info)
            
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
        return f"✓ Informasi berhasil disimpan: {information}"
        
    except Exception as e:
        return f"Error menyimpan informasi: {str(e)}"

def analyze_and_save_info(user_message: str, config: RunnableConfig) -> str:
    """
    Tool untuk menganalisis pesan user dan otomatis menyimpan informasi penting yang disebutkan.
    Tool ini menggunakan LLM untuk mendeteksi informasi personal yang perlu disimpan.
    """
    try:
        user_id = config["configurable"]["user_id"]
        
        # LLM analysis untuk deteksi info penting
        analysis_prompt = f"""
        Analisis pesan user berikut dan tentukan apakah ada informasi personal/penting yang perlu disimpan:

        Pesan user: "{user_message}"

        Tugas:
        1. Identifikasi informasi personal seperti: nama, lokasi, pekerjaan, hobi, preferensi, keluarga, dll.
        2. Jika ada informasi penting, ekstrak dalam format yang jelas
        3. Jika tidak ada informasi penting, jawab "NONE"

        Format response:
        - Jika ada info penting: "INFO: [informasi yang diekstrak]"
        - Jika tidak ada: "NONE"

        Contoh:
        User: "Nama saya Widya dan saya tinggal di Bekasi"
        Response: "INFO: Nama user adalah Widya, User tinggal di Bekasi"

        User: "Apa kabar hari ini?"
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
            
            print(f"\n--- [TOOL: analyze_and_save_info] ---")
            print(f"User ID: {user_id}")
            print(f"Analysis: {analysis_result}")
            print(f"Saved: {info_to_save}")
            print(f"-------------------------------------\n")
            
            return f"✓ Terdeteksi dan disimpan informasi penting: {info_to_save}"
        else:
            return "Tidak ada informasi penting yang perlu disimpan dari pesan ini."
    except Exception as e:
        return f"Error dalam analisis: {str(e)}"

# Memory tools dengan config approach
memory_tools = [
    StructuredTool.from_function(
        func=get_user_context,
        name="get_user_context",
        args_schema=EmptyArgs,
        description="WAJIB DIPANGGIL PERTAMA di setiap conversation untuk mendapatkan konteks user yang tersimpan. Gunakan tool ini sebelum merespons user untuk mendapatkan informasi personal yang sudah diketahui sebelumnya."
    ),
    StructuredTool.from_function(
        func=save_important_info,
        name="save_important_info",
        args_schema=SaveInfoArgs,
        description="Menyimpan informasi penting tentang user yang disebutkan dalam percakapan (nama, lokasi, preferensi, dll)."
    ),
    StructuredTool.from_function(
        func=analyze_and_save_info,
        name="analyze_and_save_info", 
        args_schema=AnalyzeMessageArgs,
        description="Menganalisis pesan user dengan LLM untuk otomatis mendeteksi dan menyimpan informasi penting yang perlu diingat."
    )
]