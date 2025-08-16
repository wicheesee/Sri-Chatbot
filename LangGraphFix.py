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

tools = time_tools + weather_tools
# llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)
llm_with_tools = llm.bind_tools(tools)

# System message
sys_msg = SystemMessage(content="""
Anda adalah asisten AI yang bisa mengenali waktu saat ini dan cuaca di lokasi tertentu.
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
messages = [HumanMessage(content="Jam berapa sekarang di Jakarta, berdasarkan cuaca, apakah saya bisa mengadakan pesta outdoor hari ini?")]
messages = react_graph.invoke({"messages": messages})
for m in messages['messages']:
    m.pretty_print()