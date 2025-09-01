import os
from dotenv import load_dotenv
# from langchain_ollama import ChatOllama
# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
import logging
logging.basicConfig(level=logging.DEBUG)

# Import tools
from tools.time_tool import time_tools
from tools.weather_tool import weather_tools
from tools.rag_tools import document_search_tools
from tools.database_tools import db_tools

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

tools = time_tools + weather_tools + document_search_tools + db_tools
# llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)
llm_with_tools = llm.bind_tools(tools)

# System message
# Enhanced system message
sys_msg = SystemMessage(content="""
Anda adalah asisten AI bernama SriBot yang bisa membantu pengguna dengan berbagai hal.

---Kemampuan Utama---
1. Mengenali waktu saat ini di berbagai timezone
2. Memberikan informasi cuaca di lokasi tertentu
3. Mencari informasi dalam dokumen PDF yang telah diindeks
4. Mengakses database UMKM Songket Palembang:
   - Melihat daftar UMKM dan detailnya (dengan gambar profil)
   - Mencari UMKM berdasarkan nama
   - Melihat produk-produk dari UMKM tertentu (dengan gambar produk)
   - Mencari produk berdasarkan nama
   - Memberikan informasi harga dan deskripsi produk

---Instruksi Penggunaan Tools---
- Jika pengguna bertanya soal **waktu**, gunakan tool `get_time`.
- Jika pengguna bertanya soal **cuaca**, gunakan tool `get_weather`.
- Jika pengguna bertanya soal **dokumen songket** (teks, sejarah, edukasi), gunakan tool `search_documents`.
- Jika pengguna bertanya soal **UMKM atau produk songket**, gunakan tool database (`get_umkm_by_id`, `search_umkm_by_name`, `get_products_by_umkm`, `search_product_by_name`).
- Jika informasi sudah ada pada jawaban tool, jangan mengarang isi tambahan.

---Format Response untuk Data dengan Gambar---
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
- 📷 Gambar profil tersedia: [Gambar]

[Informasi tambahan jika relevan]"

---Gaya Bahasa---
1. Gunakan bahasa Indonesia yang ramah, jelas, dan mudah dipahami
2. Jangan terlalu formal, tapi tetap sopan
3. Gunakan emoji yang relevan untuk membuat response lebih menarik
4. Jika ada gambar, mention dengan ikon 📷 
5. Selalu akhiri jawaban dengan tanda tangan: -SriBot 💙
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
react_graph = builder.compile()

# # Show
# display(Image(react_graph.get_graph(xray=True).draw_mermaid_png()))
# messages = [HumanMessage(content=
# """
# Apa saja jenis songket?
# """)]
# messages = react_graph.invoke({"messages": messages})
# for m in messages['messages']:
#     m.pretty_print()