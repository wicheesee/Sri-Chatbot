from pydantic import BaseModel

class EmptyArgs(BaseModel):
    pass

class SaveInfoArgs(BaseModel):
    information: str

class AnalyzeMessageArgs(BaseModel):
    user_message: str

    