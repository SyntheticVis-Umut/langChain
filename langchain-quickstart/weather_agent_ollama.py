"""
Real-World Weather Agent Example - LangChain Quickstart
Adapted for Ollama (local LLM) instead of OpenAI
Demonstrates:
1. Detailed system prompts
2. Tools that integrate with external data
3. Model configuration with Ollama
4. Structured output
5. Conversational memory
6. Full agent creation and execution
7. LangSmith tracing and observability
"""

import os
from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_ollama import ChatOllama
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver

# LangSmith tracing is automatically enabled when environment variables are set:
# - LANGSMITH_TRACING=true
# - LANGSMITH_API_KEY=<your-api-key>
# - LANGSMITH_PROJECT=<project-name> (optional but recommended)
# Get your API key from https://smith.langchain.com


# Define system prompt
SYSTEM_PROMPT = """You are an expert weather forecaster, who speaks in puns.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location."""

# Define context schema
@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str


# Define tools
@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user information based on user ID."""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"


# Configure model - using Ollama (local LLM)
# Ollama configuration from environment variables
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
ollama_model = os.getenv("OLLAMA_MODEL", "granite3.1-moe:3b")

model = ChatOllama(
    model=ollama_model,
    base_url=ollama_base_url,
    temperature=0.5,
    timeout=60,
    num_ctx=4096,
)


# Define response format
@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    # A punny response (always required)
    punny_response: str
    # Any interesting information about the weather if available
    weather_conditions: str | None = None


# Set up memory
checkpointer = InMemorySaver()

# Create agent
# Use ToolStrategy for structured output with Ollama (works with any tool-calling model)
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),  # Use ToolStrategy for Ollama compatibility
    checkpointer=checkpointer
)


# Run agent
if __name__ == "__main__":
    # Check if environment variables are set
    if not ollama_base_url:
        print("Error: OLLAMA_BASE_URL environment variable is not set.")
        print("Please set: export OLLAMA_BASE_URL=<your-ollama-url>")
        exit(1)
    
    print(f"Connecting to Ollama at: {ollama_base_url}")
    print(f"Using model: {ollama_model}")
    
    # LangSmith tracing (optional but recommended)
    if os.getenv("LANGSMITH_TRACING") == "true":
        langsmith_project = os.getenv("LANGSMITH_PROJECT", "weather-agent-ollama")
        print(f"✅ LangSmith tracing enabled (project: {langsmith_project})")
        print(f"   View traces at: https://smith.langchain.com")
    else:
        print("ℹ️  LangSmith tracing disabled. Set LANGSMITH_TRACING=true to enable.")
    
    # `thread_id` is a unique identifier for a given conversation.
    config = {"configurable": {"thread_id": "1"}}

    print("=== First Question ===")
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "what is the weather outside?"}]},
        config=config,
        context=Context(user_id="1")
    )

    # Debug: Check for tool calls in messages
    tool_calls_made = False
    if response.get("messages"):
        for msg in response["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_calls_made = True
                print(f"🔧 Tool calls made: {msg.tool_calls}")
                break
    
    # Handle structured response (may not always be present with Ollama)
    if 'structured_response' in response:
        print("✅ Structured Response:")
        print(response['structured_response'])
    else:
        print("⚠️  No structured response available (falling back to message content)")
        if not tool_calls_made:
            print("⚠️  Warning: No tool calls were made. The model may be too small for reliable tool calling.")
            print("   Consider using a larger model (3b+ parameters) for better tool calling support.")
        # Fallback: print the last message if structured response not available
        if response.get("messages"):
            last_msg = response["messages"][-1]
            if hasattr(last_msg, "content") and last_msg.content:
                print(f"Agent Response: {last_msg.content}")
            else:
                print("Response:", response)
        else:
            print("Response:", response)
    # Expected: ResponseFormat with punny_response and weather_conditions

    print("\n=== Follow-up Question ===")
    # Note that we can continue the conversation using the same `thread_id`.
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "thank you!"}]},
        config=config,
        context=Context(user_id="1")
    )

    # Debug: Check for tool calls in messages
    tool_calls_made = False
    if response.get("messages"):
        for msg in response["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_calls_made = True
                print(f"🔧 Tool calls made: {msg.tool_calls}")
                break
    
    # Handle structured response (may not always be present with Ollama)
    if 'structured_response' in response:
        print("✅ Structured Response:")
        print(response['structured_response'])
    else:
        print("⚠️  No structured response available (falling back to message content)")
        if not tool_calls_made:
            print("⚠️  Warning: No tool calls were made. The model may be too small for reliable tool calling.")
            print("   Consider using a larger model (3b+ parameters) for better tool calling support.")
        # Fallback: print the last message if structured response not available
        if response.get("messages"):
            last_msg = response["messages"][-1]
            if hasattr(last_msg, "content") and last_msg.content:
                print(f"Agent Response: {last_msg.content}")
            else:
                print("Response:", response)
        else:
            print("Response:", response)
    # Expected: ResponseFormat with punny_response (weather_conditions may be None)

