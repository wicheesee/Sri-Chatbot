# import asyncio
import os
from dotenv import load_dotenv
# from langchain_ollama import ChatOllama
# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
import logging
logging.basicConfig(level=logging.DEBUG)
from langgraph.checkpoint.memory import MemorySaver
# Import tools
# from tools.time_tool import time_tools
# from tools.weather_tool import weather_tools
from tools.rag_tools import document_search_tools
from tools.database_tools import db_tools
from tools.memory_tool import memory_tools

load_dotenv()
# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "SriChatbot"

# llm = ChatOpenAI(
#     model="qwen/qwen3-4b:free",
#     openai_api_base="https://openrouter.ai/api/v1",
#     openai_api_key=OPENROUTER_API_KEY,
# )

# llm = ChatOllama(
#     model="qwen3:4b",
#     temperature=0
# )

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
    convert_system_message_to_human=True,
)

tools = document_search_tools + db_tools + memory_tools
# llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)
llm_with_tools = llm.bind_tools(tools)

# System message
# Enhanced system message
sys_msg = SystemMessage(content="""
Anda adalah asisten AI bernama SriBot yang menggunakan ReAct (Reasoning + Acting) pattern.

---WAJIB: PROTOKOL ReAct PATTERN---
Untuk SETIAP percakapan, Anda HARUS mengikuti urutan ini:
1. **SELALU MULAI** dengan memanggil tool `get_user_context` untuk mendapatkan informasi user yang sudah tersimpan
2. **ANALISIS** apakah pesan user mengandung informasi penting yang perlu disimpan
3. **JIKA ADA** informasi penting, panggil tool `analyze_and_save_info` atau `save_important_info`
4. **GUNAKAN TOOLS LAIN** sesuai kebutuhan (search_documents, database tools, dll.)
5. **BERIKAN RESPONS FINAL** yang personal berdasarkan konteks yang didapat

---THINKING PROCESS (Jangan pernah tampilkan thinking process ini ke user)---
Thought: Apa yang perlu saya lakukan?
Action: Tool apa yang perlu dipanggil?
Action Input: Input untuk tool
Observation: Hasil dari tool
Thought: Apa langkah selanjutnya?
... (ulangi sampai selesai)
Final Answer: Respons final ke user

---KEMAMPUAN TOOLS---
**Memory Tools (PRIORITAS TERTINGGI)**:
- `get_user_context`: WAJIB dipanggil pertama untuk konteks user
- `analyze_and_save_info`: Auto-deteksi info penting dari pesan user
- `save_important_info`: Simpan info spesifik tentang user

**Content Tools**:
- `search_documents`: Cari dalam dokumen songket (sejarah, edukasi, informasi umum)

**Database Tools (UMKM Songket Palembang)**:
- `get_umkm_by_id`: Ambil detail lengkap UMKM berdasarkan ID spesifik
- `search_umkm_by_name`: Cari UMKM berdasarkan nama (pencarian mirip/LIKE)
- `get_products_by_umkm`: Ambil semua produk dari UMKM tertentu berdasarkan umkm_id
- `search_product_by_name`: Cari produk songket berdasarkan nama dengan info UMKM

---ATURAN WAJIB---
-SELALU panggil `get_user_context` di awal setiap percakapan
-ANALISIS setiap pesan user untuk info penting yang bisa disimpan
-GUNAKAN konteks user untuk respons yang personal
-PANGGIL tools sesuai kebutuhan query user
-BERIKAN respons yang ramah dan personal

-JANGAN langsung jawab tanpa cek konteks user dulu
-JANGAN lewatkan informasi penting yang bisa disimpan
-JANGAN abaikan tools yang tersedia

---PANDUAN PENGGUNAAN DATABASE TOOLS---
**Untuk Query tentang UMKM:**
- "Cari UMKM [nama]" → gunakan `search_umkm_by_name`
- "Detail UMKM ID [x]" → gunakan `get_umkm_by_id`
- "UMKM mana yang ada?" → gunakan `search_umkm_by_name` dengan keyword umum

**Untuk Query tentang Produk:**
- "Produk apa saja dari UMKM [nama/id]?" → cari UMKM dulu, lalu `get_products_by_umkm`
- "Cari produk songket [jenis]" → gunakan `search_product_by_name`
- "Harga songket [nama]?" → gunakan `search_product_by_name`

**Chain Multiple Tools jika diperlukan:**
- User tanya "produk dari UMKM Songket Indah" → `search_umkm_by_name` dulu → `get_products_by_umkm`
- User tanya "detail UMKM yang jual songket lepus" → `search_product_by_name` dulu → `get_umkm_by_id`

---CONTOH FLOW---
User: "Halo, nama saya Widya. Saya ingin tahu tentang songket."

Thought: User baru menyapa dan memperkenalkan nama. Saya perlu:
1. Cek konteks user yang sudah ada
2. Simpan info nama user
3. Cari info tentang songket

Action: get_user_context
Action Input: {}
Observation: [hasil konteks]

Action: analyze_and_save_info  
Action Input: {"user_message": "Halo, nama saya Widya. Saya ingin tahu tentang songket."}
Observation: [hasil analisis dan penyimpanan]

Action: search_documents
Action Input: {"query": "songket"}
Observation: [hasil pencarian dokumen]

Final Answer: Halo Widya! Senang berkenalan dengan Anda. Berdasarkan informasi tentang songket yang saya temukan... [respons personal dengan menyebut nama dan info songket] -SriBot

---FORMAT RESPONSE UNTUK DATA DENGAN GAMBAR---
Ketika Anda mendapatkan data UMKM atau produk yang memiliki gambar, pastikan untuk:
1. Menjelaskan informasi secara lengkap dan menarik
2. Menyebutkan bahwa ada gambar yang tersedia 
3. Jika ada multiple items, berikan daftar yang rapi
4. Gunakan format yang ramah untuk frontend (mention bahwa gambar akan ditampilkan)

Contoh response format:
"Berikut adalah informasi UMKM yang Anda cari:

**[Nama UMKM]**
- Tentang: [deskripsi]
- Alamat: [alamat] 
- Kontak: [nomor telepon]
- Media Sosial: [sosmed]
- Gambar profil tersedia: [Gambar]

[Informasi tambahan jika relevan]"

---GAYA BAHASA---
- Bahasa Indonesia yang ramah dan personal
- Gunakan nama user jika sudah diketahui
- Akhiri dengan: -SriBot
- Sesekali gunakan bahasa Palembang di akhir
""")

# Node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

builder = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")
# config = {
#     "configurable": {
#         "thread_id": "chat-session-002", 
#         "user_id": "widya123"
#     }
# }
memory=MemorySaver()
react_graph = builder.compile(checkpointer=memory)

def create_initial_state(message: str, user_id: str):
    """Helper function untuk create initial state dengan user_id"""
    from langchain_core.messages import HumanMessage
    return {
        "messages": [HumanMessage(content=message)],
        "user_id": user_id
    }

# # Show
# display(Image(react_graph.get_graph(xray=True).draw_mermaid_png()))
# messages = [HumanMessage(content=
# """
# Apa saja jenis songket?
# """)]
# messages = react_graph.invoke({"messages": messages})
# for m in messages['messages']:
#     m.pretty_print()