"""
Fixed for LangChain v1 (langchain 1.4.2 / langchain-core 1.6.5).

Why the old code broke:
  langchain.chains (ConversationChain) and the old memory classes
  (ConversationBufferMemory) were removed in LangChain v1. They simply
  don't exist in this version anymore, so no reinstall fixes it.

This rewrite shows the same "no memory vs memory" idea two ways:
  1. A simple manual-history version (closest to your original code).
  2. The recommended v1 way: LangGraph + a checkpointer for real,
     session-persistent memory.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------------------------------------------------------------
# 1) No memory: every call is independent, nothing is remembered
# ---------------------------------------------------------------
print(llm.invoke("My name is Ashish. Remember that.").content)
print(llm.invoke("Who am I?").content)  # forgets — no shared history

# ---------------------------------------------------------------
# 2) Manual memory: keep a running list of messages yourself
# ---------------------------------------------------------------
history = []

def chat_with_memory(user_input: str) -> str:
    history.append(HumanMessage(content=user_input))
    response = llm.invoke(history)
    history.append(response)
    return response.content

print(chat_with_memory("My name is Ashish."))
print(chat_with_memory("Who am I?"))  # remembers — same history list

# Fresh session: brand-new empty history
fresh_history = []

def chat_fresh(user_input: str) -> str:
    fresh_history.append(HumanMessage(content=user_input))
    response = llm.invoke(fresh_history)
    fresh_history.append(response)
    return response.content

print(chat_fresh("Who am I?"))  # forgets again — new list

# Reusing the earlier `history` list picks the memory back up
print(chat_with_memory("Who am I?"))  # recalls correctly

# ---------------------------------------------------------------
# 3) The v1-recommended way: LangGraph with a checkpointer.
#    This is what production code should actually use — memory
#    keyed by a thread_id, so you can resume any session later.
# ---------------------------------------------------------------
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.memory import InMemorySaver

def call_model(state: MessagesState):
    return {"messages": llm.invoke(state["messages"])}

graph = StateGraph(state_schema=MessagesState)
graph.add_node("model", call_model)
graph.add_edge(START, "model")

app = graph.compile(checkpointer=InMemorySaver())

# Each thread_id is an isolated "session" with its own memory
session_a = {"configurable": {"thread_id": "ashish-session"}}

out = app.invoke({"messages": [HumanMessage(content="My name is Ashish.")]}, session_a)
print(out["messages"][-1].content)

out = app.invoke({"messages": [HumanMessage(content="Who am I?")]}, session_a)
print(out["messages"][-1].content)  # remembers — same thread_id

session_b = {"configurable": {"thread_id": "new-session"}}
out = app.invoke({"messages": [HumanMessage(content="Who am I?")]}, session_b)
print(out["messages"][-1].content)  # forgets — different thread_id
