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
8. Error handling and graceful degradation
9. Streaming support for real-time feedback
10. Improved tool descriptions for better tool calling
"""

import os
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_ollama import ChatOllama
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver

# Configure logging - simplified format without timestamps
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


# ANSI color codes for colorful output
class Colors:
    """ANSI color codes for terminal output."""
    # Reset
    RESET = '\033[0m'
    
    # Text colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Background colors
    BG_BLUE = '\033[44m'
    BG_CYAN = '\033[46m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    
    # Styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'


def colorize(text: str, color: str, bold: bool = False) -> str:
    """Apply color and optional bold styling to text."""
    style = Colors.BOLD if bold else ''
    return f"{style}{color}{text}{Colors.RESET}"


def print_header(text: str, emoji: str = "🤖"):
    """Print a colorful header with emoji."""
    print(f"\n{colorize('═' * 60, Colors.BRIGHT_CYAN, bold=True)}")
    print(f"{emoji}  {colorize(text, Colors.BRIGHT_CYAN, bold=True)}")
    print(f"{colorize('═' * 60, Colors.BRIGHT_CYAN, bold=True)}")


def print_section(text: str, emoji: str = "📋"):
    """Print a section header with emoji."""
    print(f"\n{emoji}  {colorize(text, Colors.BRIGHT_BLUE, bold=True)}")
    print(f"{colorize('─' * 60, Colors.CYAN)}")


def print_success(text: str, emoji: str = "✅"):
    """Print a success message with emoji."""
    print(f"{emoji}  {colorize(text, Colors.BRIGHT_GREEN)}")


def print_warning(text: str, emoji: str = "⚠️"):
    """Print a warning message with emoji."""
    print(f"{emoji}  {colorize(text, Colors.BRIGHT_YELLOW)}")


def print_error(text: str, emoji: str = "❌"):
    """Print an error message with emoji."""
    print(f"{emoji}  {colorize(text, Colors.BRIGHT_RED)}")


def print_info(text: str, emoji: str = "ℹ️"):
    """Print an info message with emoji."""
    print(f"{emoji}  {colorize(text, Colors.BRIGHT_BLUE)}")


def print_tool_call(tool_name: str):
    """Print a tool call with colorful formatting."""
    print(f"\n{colorize('🔧', Colors.BRIGHT_MAGENTA)}  {colorize('Calling tool:', Colors.MAGENTA, bold=True)} {colorize(tool_name, Colors.BRIGHT_MAGENTA, bold=True)}")
    print(f"{colorize('─' * 60, Colors.MAGENTA)}")

# LangSmith tracing is automatically enabled when environment variables are set:
# - LANGSMITH_TRACING=true
# - LANGSMITH_API_KEY=<your-api-key>
# - LANGSMITH_PROJECT=<project-name> (optional but recommended)
# Get your API key from https://smith.langchain.com


# Define system prompt with improved clarity
SYSTEM_PROMPT = """You are an expert weather forecaster who provides weather information with a fun, punny personality.

You have access to two tools:

1. get_weather_for_location(city: str): Use this tool to get weather information for a specific city or location.
   - Call this when the user explicitly mentions a city name or location.
   - Example: "What's the weather in New York?" -> use get_weather_for_location("New York")

2. get_user_location(): Use this tool to retrieve the user's current location.
   - Call this when the user asks about weather at their current location or "where I am".
   - Example: "What's the weather outside?" -> first use get_user_location(), then get_weather_for_location()

IMPORTANT: Always determine the location before providing weather information. If the user's question implies their current location, use get_user_location() first to find out where they are, then use get_weather_for_location() with that location.

Always respond with puns and a friendly, engaging tone while providing accurate weather information."""

# Define context schema
@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str


# Define tools with improved error handling and descriptions
@tool
def get_weather_for_location(city: str) -> str:
    """
    Get current weather information for a specific city or location.
    
    This tool retrieves weather data including temperature, conditions, and forecast
    for the specified location. Use this when the user asks about weather in a
    specific place.
    
    Args:
        city: The name of the city or location to get weather for (e.g., "New York", "London", "Tokyo")
    
    Returns:
        A string containing weather information for the specified location.
    
    Example:
        get_weather_for_location("San Francisco") -> "It's always sunny in San Francisco!"
    """
    try:
        if not city or not city.strip():
            return "Error: Please provide a valid city name."
        
        # In a real implementation, this would call an actual weather API
        # For now, this is a mock implementation
        result = f"It's always sunny in {city.strip()}!"
        return result
    except Exception as e:
        logger.error(f"Weather error: {str(e)}")
        return f"Error fetching weather data for {city}. Please try again later."


@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """
    Retrieve the user's current location based on their user ID.
    
    This tool looks up the user's location from their profile or session data.
    Use this when the user asks about weather at their current location or
    "where I am" without specifying a city.
    
    Args:
        runtime: The tool runtime context containing user information
    
    Returns:
        A string containing the user's location (e.g., "Florida", "San Francisco")
    
    Example:
        get_user_location(runtime) -> "Florida"
    """
    try:
        if not runtime or not runtime.context:
            return "Error: Unable to retrieve user location. Context is missing."
        
        user_id = runtime.context.user_id
        if not user_id:
            return "Error: User ID not found in context."
        
        # In a real implementation, this would query a user database
        location = "Florida" if user_id == "1" else "SF"
        return location
    except Exception as e:
        logger.error(f"Location error: {str(e)}")
        return "Error retrieving user location. Please try again later."


# Configure model - using Ollama (local LLM)
# Ollama configuration from environment variables
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# Validate configuration
if not ollama_base_url:
    raise ValueError("OLLAMA_BASE_URL environment variable must be set")

# Model configuration with improved defaults
# Note: num_ctx controls context window size, adjust based on model capabilities
model = ChatOllama(
    model=ollama_model,
    base_url=ollama_base_url,
    temperature=0.5,  # Lower temperature for more consistent tool calling
    timeout=60,  # Increase timeout for slower models
    num_ctx=4096,  # Context window size (adjust based on model)
    streaming=True,  # Enable streaming for incremental responses
    # Additional options for better tool calling:
    # num_predict=512,  # Max tokens to generate
    # top_p=0.9,  # Nucleus sampling
    # repeat_penalty=1.1,  # Reduce repetition
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


def extract_response(response: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """
    Extract and format the agent response, handling both structured and unstructured outputs.
    
    This helper function handles the complexity of extracting responses from the agent,
    including structured responses, tool calls, and fallback to message content.
    
    Args:
        response: The response dictionary from agent.invoke()
        verbose: Whether to print debug information
    
    Returns:
        A dictionary containing:
        - structured_response: The structured response if available
        - message_content: The message content as fallback
        - tool_calls: List of tool calls made
        - has_structured: Boolean indicating if structured response is available
    """
    result = {
        "structured_response": None,
        "message_content": None,
        "tool_calls": [],
        "has_structured": False
    }
    
    # Check for tool calls in messages
    if response.get("messages"):
        for msg in response["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                result["tool_calls"] = msg.tool_calls
                break
    
    # Handle structured response (preferred)
    if 'structured_response' in response and response['structured_response']:
        result["structured_response"] = response['structured_response']
        result["has_structured"] = True
    else:
        # Fallback to message content
        if response.get("messages"):
            last_msg = response["messages"][-1]
            if hasattr(last_msg, "content") and last_msg.content:
                result["message_content"] = last_msg.content
    
    return result


def print_response(extracted: Dict[str, Any], show_tool_calls: bool = True):
    """
    Print the extracted response in a user-friendly format with colors.
    
    Args:
        extracted: The result dictionary from extract_response()
        show_tool_calls: Whether to display tool call information
    """
    if show_tool_calls and extracted["tool_calls"]:
        print_tool_call("Multiple tools")
        for tool_call in extracted["tool_calls"]:
            tool_name = tool_call.get("name", "unknown")
            print(f"  {colorize('•', Colors.BRIGHT_MAGENTA)} {colorize(tool_name, Colors.BRIGHT_MAGENTA)}")
    
    if extracted["has_structured"]:
        print_success("Structured Response:", "✅")
        print(f"{colorize(str(extracted['structured_response']), Colors.BRIGHT_WHITE)}")
    else:
        print_warning("No structured response available (falling back to message content)")
        if not extracted["tool_calls"]:
            print_warning("No tool calls were made. The model may be too small for reliable tool calling.")
            print_info("Consider using a larger model (3b+ parameters) for better tool calling support.")
        
        if extracted["message_content"]:
            print(f"{colorize('Agent Response:', Colors.BRIGHT_GREEN, bold=True)} {colorize(extracted['message_content'], Colors.BRIGHT_WHITE)}")
        else:
            print_warning("No response content available")


def stream_agent_response(agent, messages: list, config: dict, context: Context):
    """
    Stream agent response using stream_mode="messages" for token-level streaming.
    Based on: https://docs.langchain.com/oss/python/langchain/streaming
    
    Args:
        agent: The agent instance
        messages: List of input messages
        config: Agent configuration (thread_id, etc.)
        context: Runtime context
    """
    tool_calls_shown = set()  # Track which tool calls we've shown
    
    for token, metadata in agent.stream(
        {"messages": messages},
        config=config,
        context=context,
        stream_mode="messages"  # Stream LLM tokens as they're generated
    ):
        # Extract node information from metadata
        node = metadata.get("langgraph_node", "unknown")
        
        # Process content_blocks from the token
        if hasattr(token, "content_blocks"):
            content_blocks = token.content_blocks
        elif isinstance(token, dict) and "content_blocks" in token:
            content_blocks = token["content_blocks"]
        else:
            content_blocks = []
        
        # Process each content block
        for block in content_blocks:
            if isinstance(block, dict):
                block_type = block.get("type", "")
                
                # Handle text tokens - stream incrementally with color
                if block_type == "text" and "text" in block:
                    text = block["text"]
                    if text:
                        # Colorize the streaming text
                        print(f"{colorize(text, Colors.BRIGHT_WHITE)}", end="", flush=True)
                
                # Handle tool call chunks
                elif block_type == "tool_call_chunk":
                    tool_name = block.get("name")
                    tool_args = block.get("args", "")
                    tool_id = block.get("id")
                    
                    # Show tool call info when we have the name
                    if tool_name and tool_id and tool_name not in tool_calls_shown:
                        print_tool_call(tool_name)
                        tool_calls_shown.add(tool_name)
                    # Stream tool args as they're generated (optional - usually not needed)
                    elif tool_args and isinstance(tool_args, str) and tool_args.strip():
                        # Only print if it's meaningful content (not just partial JSON)
                        pass
            
            # Handle content_blocks as objects (if they have attributes)
            elif hasattr(block, "type"):
                if block.type == "text" and hasattr(block, "text"):
                    print(f"{colorize(block.text, Colors.BRIGHT_WHITE)}", end="", flush=True)
    
    print()  # Final newline after streaming


# Run agent
if __name__ == "__main__":
    # Check if environment variables are set
    if not ollama_base_url:
        print_error("OLLAMA_BASE_URL environment variable is not set.")
        print_info(f"Please set: {colorize('export OLLAMA_BASE_URL=<your-ollama-url>', Colors.BRIGHT_CYAN)}")
        exit(1)
    
    # Print colorful header
    print_header("LLM Configuration", "🤖")
    print(f"{colorize('Provider:', Colors.CYAN, bold=True)} {colorize('Ollama', Colors.BRIGHT_GREEN, bold=True)}")
    print(f"{colorize('Base URL:', Colors.CYAN, bold=True)} {colorize(ollama_base_url, Colors.BRIGHT_WHITE)}")
    print(f"{colorize('Model:', Colors.CYAN, bold=True)} {colorize(ollama_model, Colors.BRIGHT_WHITE)}")
    print(f"{colorize('LLM Class:', Colors.CYAN, bold=True)} {colorize('ChatOllama', Colors.BRIGHT_WHITE)}")
    print(f"{colorize('Temperature:', Colors.CYAN, bold=True)} {colorize('0.5', Colors.BRIGHT_YELLOW)}")
    print(f"{colorize('Context Window:', Colors.CYAN, bold=True)} {colorize('4096', Colors.BRIGHT_YELLOW)}")
    
    # LangSmith tracing (optional but recommended)
    if os.getenv("LANGSMITH_TRACING") == "true":
        langsmith_project = os.getenv("LANGSMITH_PROJECT", "weather-agent")
        print_success(f"LangSmith tracing enabled (project: {colorize(langsmith_project, Colors.BRIGHT_CYAN, bold=True)})", "✅")
        print_info(f"View traces at: {colorize('https://smith.langchain.com', Colors.BRIGHT_CYAN, bold=True)}", "🔗")
    else:
        print_warning("LangSmith tracing disabled. Set LANGSMITH_TRACING=true to enable.", "ℹ️")
    
    # `thread_id` is a unique identifier for a given conversation.
    config = {"configurable": {"thread_id": "1"}}

    # Example 1: First question with streaming
    print_section("First Question", "💬")
    print(f"{colorize('User:', Colors.BRIGHT_BLUE, bold=True)} {colorize('what is the weather outside?', Colors.WHITE)}")
    print(f"{colorize('Agent:', Colors.BRIGHT_GREEN, bold=True)} ", end="")
    try:
        stream_agent_response(
            agent=agent,
            messages=[{"role": "user", "content": "what is the weather outside?"}],
            config=config,
            context=Context(user_id="1")
        )
        
    except Exception as e:
        logger.error(f"Streaming error: {str(e)}")
        print_error(f"Error: {str(e)}")
        print_warning("Please check your Ollama connection and model availability.")
        # Fallback to regular invoke if streaming fails
        print_warning("Falling back to non-streaming mode...", "🔄")
        try:
            response = agent.invoke(
                {"messages": [{"role": "user", "content": "what is the weather outside?"}]},
                config=config,
                context=Context(user_id="1")
            )
            extracted = extract_response(response, verbose=False)
            print_response(extracted, show_tool_calls=True)
        except Exception as fallback_error:
            logger.error(f"Fallback error: {str(fallback_error)}")
            print_error(f"Fallback error: {str(fallback_error)}")

    # Example 2: Follow-up question with streaming (continuing conversation)
    print_section("Follow-up Question", "💬")
    print(f"{colorize('User:', Colors.BRIGHT_BLUE, bold=True)} {colorize('thank you! see you later!', Colors.WHITE)}")
    print(f"{colorize('Agent:', Colors.BRIGHT_GREEN, bold=True)} ", end="")
    # Note that we can continue the conversation using the same `thread_id`.
    try:
        stream_agent_response(
            agent=agent,
            messages=[{"role": "user", "content": "thank you! see you later!"}],
            config=config,
            context=Context(user_id="1")
        )
        
    except Exception as e:
        logger.error(f"Streaming error: {str(e)}")
        print_error(f"Error: {str(e)}")
        print_warning("Please check your Ollama connection and model availability.")
        # Fallback to regular invoke if streaming fails
        print_warning("Falling back to non-streaming mode...", "🔄")
        try:
            response = agent.invoke(
                {"messages": [{"role": "user", "content": "thank you! see you later!"}]},
                config=config,
                context=Context(user_id="1")
            )
            extracted = extract_response(response, verbose=False)
            print_response(extracted, show_tool_calls=True)
        except Exception as fallback_error:
            logger.error(f"Fallback error: {str(fallback_error)}")
            print_error(f"Fallback error: {str(fallback_error)}")
    
    # Final message
    print()
    print_success("Conversation completed!", "✨")


