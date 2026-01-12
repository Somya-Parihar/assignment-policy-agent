import operator
from typing import Annotated, TypedDict, Optional
from langchain_core.messages import BaseMessage

class UserInfo(TypedDict):
    age: Optional[int]
    location: Optional[str]
    income: Optional[int]

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    user_info: UserInfo
    dialog_state: str  # e.g., 'support', 'collecting_info', 'completed', 'finished'