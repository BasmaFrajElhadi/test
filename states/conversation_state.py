from typing import Annotated, TypedDict, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages

class ConversationState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
