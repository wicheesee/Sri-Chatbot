from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from Langgraph import react_graph 
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(request: ChatRequest):
    messages = [HumanMessage(content=request.message)]
    result = react_graph.invoke({"messages": messages})
    responses = []
    for m in result["messages"]:
        if hasattr(m, "content"):
            responses.append(m.content)
    return {"reply": responses[-1]}  
