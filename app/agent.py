from typing import Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langsmith import traceable
from typing_extensions import Annotated, TypedDict

from app.config import get_settings


class AgentState(TypedDict):
    """State for the production RAG agent.
    Uses Annotated with add_messages reducer for message history.

    Attributes:
        messages: Annotated[list[BaseMessage], add_messages]
            List of messages in the conversation.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    error: Optional[str]
    retry_count: int
    model_used: str


class ProductionRAGAgent:
    """Production RAG agent that uses a graph to manage conversation flow.
    - Retry on failure with model fallback
    - Graceful error handling
    - LangSmith tracing
    """

    def __init__(self):
        settings = get_settings()

        self.primary_llm = ChatOpenAI(
            model=settings.primary_model,
            temperature=0.0,
            timeout=30,
            max_retries=0,
        )
        self.fallback_llm = ChatOpenAI(
            model=settings.fallback_model, temperature=0.0, timeout=30, max_retries=0
        )
        self.max_retries = settings.max_retries
        self.graph = self.build_graph()

    def build_graph(self):
        """Builds and returns the LangGraph state graph for the agent."""

        def process_message(state: AgentState) -> dict:
            """Processes the incoming message and returns the updated state."""
            try:
                response = self.primary_llm.invoke(state["messages"])
                return {
                    "messages": [response],
                    "error": None,
                    "retry_count": 0,
                    "model_used": "primary",
                }
            except Exception as e:
                return {
                    "messages": [],
                    "error": str(e),
                    "retry_count": state["retry_count"] + 1,
                    "model_used": "primary",
                }

        def try_fallback(state: AgentState) -> dict:
            """Fallback to secondary LLM if primary fails."""
            try:
                response = self.fallback_llm.invoke(state["messages"])
                return {
                    "messages": [response],
                    "error": None,
                    "retry_count": 0,
                    "model_used": "secondary",
                }
            except Exception as e:
                return {
                    "messages": [],
                    "error": str(e),
                    "retry_count": state["retry_count"] + 1,
                    "model_used": "secondary",
                }

        def handle_error(state: AgentState) -> dict:
            """Handles errors and decides whether to retry or fallback."""
            return {
                "messages": [AIMessage(content=f"Error: {state['error']}")],
                "error": None,
                "retry_count": 0,
                "model_used": "primary",
            }

        def route_after_process(state: AgentState) -> str:
            """Routes after processing based on model attempts."""

            if state.get("error") is None:
                return "done"
            elif state["retry_count"] < self.max_retries:
                return "fallback"
            else:
                return "error"

        def route_after_fallback(state: AgentState) -> str:
            """Routes after fallback based on model attempts."""
            if state.get("error") is None:
                return "done"
            elif state["retry_count"] < self.max_retries:
                return "fallback"
            else:
                return "error"

        graph = StateGraph(AgentState)

        graph.add_node("process", process_message)
        graph.add_node("fallback", try_fallback)
        graph.add_node("error", handle_error)

        graph.add_edge(START, "process")
        graph.add_conditional_edges(
            "process",
            route_after_process,
            {"done": END, "fallback": "fallback", "error": "error"},
        )
        graph.add_conditional_edges(
            "fallback", route_after_fallback, {"done": END, "error": "error"}
        )

        graph.add_edge("error", END)

        return graph.compile()

    @traceable(name="production_agent_invoke")
    def invoke(self, message: str) -> dict:
        """Invoke the production agent with a user message.

        Args:
            message (str): The user message to process.

        Returns:
            dict: The result of the agent invocation.
            {"response": str, "error": str | None, "model_used": str}
        """

        result = self.graph.invoke(
            {
                "messages": [HumanMessage(content=message)],
                "error": None,
                "retry_count": 0,
                "model_used": "",
            }
        )
        return {
            "response": result["messages"][-1].content,
            "error": result.get("error", None),
            "model_used": result.get("model_used", "unknown"),
        }
