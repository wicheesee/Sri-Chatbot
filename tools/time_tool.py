from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from datetime import datetime
from models.time_models import  GetCurrentTimeArgs
import pytz

def get_current_time(timezone: str = "Asia/Jakarta") -> str:
    """Mendapatkan waktu saat ini berdasarkan timezone"""
    tz = pytz.timezone(timezone)
    current_time = datetime.now(tz)
    return current_time.strftime("%Y-%m-%d %H:%M:%S %Z")

time_tools = [
    StructuredTool.from_function(
        func=get_current_time,
        name="get_current_time",
        description="Mendapatkan waktu saat ini berdasarkan timezone",
        args_schema=GetCurrentTimeArgs
    )
]