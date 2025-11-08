"""
Real-World Weather Agent Example - LangChain Quickstart
Adapted for Ollama (local LLM) instead of OpenAI
Demonstrates:
1. Detailed system prompts
2. Tools that integrate with external data
3. Model configuration with Ollama
4. Conversational memory
5. Full agent creation and execution
6. LangSmith tracing and observability
7. Error handling and graceful degradation
8. Streaming support for real-time feedback
9. Improved tool descriptions for better tool calling
10. Middleware for monitoring and control (ModelCallLimitMiddleware)
"""

import os
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    AgentMiddleware,
    AgentState,
)
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


def sanitize_text(text: str) -> str:
    """
    Sanitize text to handle UTF-8 encoding errors.
    Removes or replaces invalid UTF-8 surrogate characters.
    
    Args:
        text: The text to sanitize
    
    Returns:
        Sanitized text safe for UTF-8 encoding
    """
    if not text:
        return text
    
    # Convert to string if not already
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ""
    
    try:
        # First, try to remove invalid surrogate characters
        # Surrogates are in the range U+D800 to U+DFFF
        text = ''.join(
            char for char in text 
            if not (0xD800 <= ord(char) <= 0xDFFF)
        )
        
        # Try to encode/decode to ensure valid UTF-8
        text.encode('utf-8').decode('utf-8')
        return text
    except (UnicodeEncodeError, UnicodeDecodeError, UnicodeError):
        # Handle invalid UTF-8 characters by replacing them
        try:
            # Replace invalid characters with replacement character
            return text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
        except Exception:
            # Final fallback: remove all problematic characters
            return ''.join(
                char for char in text 
                if ord(char) < 0xD800 or ord(char) > 0xDFFF
            )

# LangSmith tracing is automatically enabled when environment variables are set:
# - LANGSMITH_TRACING=true
# - LANGSMITH_API_KEY=<your-api-key>
# - LANGSMITH_PROJECT=<project-name> (optional but recommended)
# Get your API key from https://smith.langchain.com


# Define system prompt with improved clarity and memory awareness
SYSTEM_PROMPT = """You are an expert weather forecaster who provides weather information with a fun, punny personality.

CRITICAL MEMORY RULES - READ CAREFULLY:
1. You have FULL ACCESS to the conversation history. ALWAYS check previous messages BEFORE calling any tool.
2. If you already know the user's location from a previous get_user_location() call, DO NOT call it again. Use the location you already know.
3. If you already know the weather for a location from a previous get_weather_for_location() call, DO NOT call it again. Use the weather information you already know.
4. Only call each tool ONCE per conversation when you first need that information.

You have access to two tools:

1. get_user_location(): Retrieves the user's current location.
   - BEFORE calling: Check conversation history - have you already called this tool?
   - If YES: Use the location from the previous call (e.g., "Florida", "SF")
   - If NO: Call this tool to get the location
   - Example: If you see "get_user_location() -> Florida" in history, the user is in Florida. DO NOT call the tool again.

2. get_weather_for_location(city: str): Gets weather for a specific city/location.
   - BEFORE calling: Check conversation history - have you already called this tool for this location?
   - If YES: Use the weather information from the previous call
   - If NO: Call this tool with the location
   - Example: If you see "get_weather_for_location('Florida') -> It's always sunny in Florida!" in history, you already know the weather. DO NOT call the tool again.

STEP-BY-STEP WORKFLOW FOR EACH USER QUESTION:
Step 1: Read ALL previous messages in the conversation history
Step 2: Look for any get_user_location() calls - if found, note the location (e.g., "Florida")
Step 3: Look for any get_weather_for_location() calls - if found, note the weather information
Step 4: If you already have both location AND weather from history, answer using that information WITHOUT calling any tools
Step 5: If you're missing location, call get_user_location() ONLY if you haven't called it before
Step 6: If you're missing weather, call get_weather_for_location() ONLY if you haven't called it for that location before

CONCRETE EXAMPLE:
- User asks: "what to wear today"
- You call: get_user_location() -> "Florida"
- You call: get_weather_for_location("Florida") -> "It's always sunny in Florida!"
- User asks: "where to visit in my city under this weather?"
- You see in history: location="Florida", weather="It's always sunny in Florida!"
- You answer using that information WITHOUT calling any tools

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


# Custom middleware for monitoring and logging
class MonitoringMiddleware(AgentMiddleware):
    """Middleware to monitor agent execution with colorful output."""
    
    def __init__(self, debug: bool = False):
        """Initialize middleware with optional debug mode."""
        self.debug = debug
    
    def before_model(self, state: AgentState, runtime) -> dict[str, Any] | None:
        """Called before model invocation."""
        # Debug: Show what messages the model will see
        if self.debug and state.get("messages"):
            messages = state["messages"]
            print(f"\n{colorize('🔍 DEBUG: Messages sent to model:', Colors.BRIGHT_YELLOW, bold=True)}")
            print(f"{colorize('─' * 60, Colors.YELLOW)}")
            for i, msg in enumerate(messages[-5:], 1):  # Show last 5 messages
                if hasattr(msg, "content"):
                    content = str(msg.content)[:100]  # First 100 chars
                    msg_type = type(msg).__name__
                    print(f"{colorize(f'{i}.', Colors.YELLOW)} {colorize(msg_type, Colors.BRIGHT_YELLOW)}: {colorize(content, Colors.WHITE)}")
                elif hasattr(msg, "name"):  # Tool message
                    tool_name = msg.name
                    tool_content = str(msg.content)[:100] if hasattr(msg, "content") else ""
                    print(f"{colorize(f'{i}.', Colors.YELLOW)} {colorize('ToolMessage', Colors.BRIGHT_MAGENTA)} ({tool_name}): {colorize(tool_content, Colors.WHITE)}")
            print(f"{colorize('─' * 60, Colors.YELLOW)}\n")
        
        # Track model calls (could be used for metrics)
        return None
    
    def after_model(self, state: AgentState, runtime) -> dict[str, Any] | None:
        """
        Called after model invocation.
        Intercept redundant tool calls and replace with cached results if available.
        """
        from langchain.messages import ToolMessage
        
        if state.get("messages"):
            last_msg = state["messages"][-1]
            
            # Check if model wants to call tools
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                messages = state["messages"]
                new_tool_calls = []
                tool_results_to_inject = []
                
                # Check each tool call
                for tool_call in last_msg.tool_calls:
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("args", {})
                    
                    # Check if we already have this information in conversation history
                    should_skip = False
                    cached_result = None
                    
                    if tool_name == "get_user_location":
                        # Look for previous get_user_location result in history
                        for msg in messages:
                            if hasattr(msg, "name") and msg.name == "get_user_location":
                                if hasattr(msg, "content"):
                                    # Sanitize the cached result to prevent UTF-8 errors
                                    raw_result = str(msg.content)
                                    cached_result = sanitize_text(raw_result)
                                    should_skip = True
                                    if self.debug:
                                        print(f"{colorize('⚠️  Skipping redundant tool call:', Colors.BRIGHT_YELLOW)} {colorize('get_user_location', Colors.MAGENTA)} - using cached result: {colorize(cached_result, Colors.WHITE)}")
                                    break
                    
                    elif tool_name == "get_weather_for_location":
                        city = tool_args.get("city", "")
                        # Look for previous get_weather_for_location result for this city
                        for msg in messages:
                            if hasattr(msg, "name") and msg.name == "get_weather_for_location":
                                # Check if this is for the same city (or user location)
                                # We need to check the tool call that led to this result
                                # For simplicity, if we have any weather result, use it
                                if hasattr(msg, "content"):
                                    # Sanitize the cached result to prevent UTF-8 errors
                                    raw_result = str(msg.content)
                                    cached_result = sanitize_text(raw_result)
                                    should_skip = True
                                    if self.debug:
                                        print(f"{colorize('⚠️  Skipping redundant tool call:', Colors.BRIGHT_YELLOW)} {colorize('get_weather_for_location', Colors.MAGENTA)} - using cached result")
                                    break
                    
                    if should_skip and cached_result:
                        # Don't add this tool call, but inject a fake tool result
                        # Sanitize the cached result to prevent UTF-8 errors
                        sanitized_result = sanitize_text(str(cached_result))
                        tool_results_to_inject.append({
                            "tool_call_id": tool_call.get("id", ""),
                            "name": tool_name,
                            "content": sanitized_result
                        })
                    else:
                        # Keep the tool call as-is
                        new_tool_calls.append(tool_call)
                
                # If we're skipping any tools, modify the message
                if tool_results_to_inject:
                    # Create a new message without the skipped tool calls
                    if new_tool_calls:
                        # Some tools still need to be called
                        # Create new message with only non-skipped tool calls
                        from langchain.messages import AIMessage
                        msg_content = last_msg.content if hasattr(last_msg, "content") else ""
                        sanitized_content = sanitize_text(str(msg_content)) if msg_content else ""
                        new_msg = AIMessage(
                            content=sanitized_content,
                            tool_calls=new_tool_calls
                        )
                        # Replace the last message
                        new_messages = list(messages[:-1]) + [new_msg]
                        
                        # Inject fake tool results
                        for tool_result in tool_results_to_inject:
                            # Ensure content is sanitized
                            sanitized_content = sanitize_text(str(tool_result["content"]))
                            fake_tool_msg = ToolMessage(
                                content=sanitized_content,
                                tool_call_id=tool_result["tool_call_id"],
                                name=tool_result["name"]
                            )
                            new_messages.append(fake_tool_msg)
                        
                        return {"messages": new_messages}
                    else:
                        # All tools were skipped - inject fake results and remove tool calls
                        from langchain.messages import AIMessage
                        msg_content = last_msg.content if hasattr(last_msg, "content") else ""
                        sanitized_content = sanitize_text(str(msg_content)) if msg_content else ""
                        new_msg = AIMessage(
                            content=sanitized_content
                        )
                        new_messages = list(messages[:-1]) + [new_msg]
                        
                        # Inject fake tool results
                        for tool_result in tool_results_to_inject:
                            # Ensure content is sanitized
                            sanitized_content = sanitize_text(str(tool_result["content"]))
                            fake_tool_msg = ToolMessage(
                                content=sanitized_content,
                                tool_call_id=tool_result["tool_call_id"],
                                name=tool_result["name"]
                            )
                            new_messages.append(fake_tool_msg)
                        
                        return {"messages": new_messages}
        
        return None
    
    def before_tool(self, state: AgentState, runtime) -> dict[str, Any] | None:
        """Called before tool execution."""
        # Tool execution monitoring (already handled in stream_agent_response)
        return None
    
    def after_tool(self, state: AgentState, runtime) -> dict[str, Any] | None:
        """
        Called after tool execution.
        Cache location information in state so the agent remembers it.
        Based on: https://docs.langchain.com/oss/python/langchain/short-term-memory
        """
        # Get the last tool message to extract tool results
        if state.get("messages"):
            last_msg = state["messages"][-1]
            
            # Check if this is a tool message from get_user_location
            if hasattr(last_msg, "name") and hasattr(last_msg, "content"):
                tool_name = last_msg.name
                tool_result = last_msg.content
                
                # Cache get_user_location results in state
                if tool_name == "get_user_location" and tool_result:
                    # Only cache if it's a valid location (not an error message)
                    if not tool_result.startswith("Error:"):
                        # Store in state so agent can access it
                        # The agent will see this in conversation history, but we can also store it explicitly
                        return {"user_location": tool_result}
        
        return None


# Set up memory
checkpointer = InMemorySaver()

# Configure middleware
# ModelCallLimitMiddleware prevents infinite loops and excessive API calls
middleware = [
    ModelCallLimitMiddleware(
        thread_limit=20,  # Max 20 model calls per conversation thread
        run_limit=10,  # Max 10 model calls per single invocation
        exit_behavior="end",  # Gracefully end when limit reached
    ),
    MonitoringMiddleware(debug=False),  # Custom monitoring middleware with debug mode
]

# Create agent with middleware
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    checkpointer=checkpointer,
    middleware=middleware,  # Add middleware for monitoring and control
)


def show_conversation_history(agent, config: dict, checkpointer):
    """
    Show the conversation history from agent state.
    Useful for debugging memory access.
    
    Args:
        agent: The agent instance
        config: Agent configuration with thread_id
        checkpointer: The checkpointer instance
    """
    try:
        # Try to get state from checkpointer
        # The checkpointer.get() returns a Checkpoint object
        checkpoint_tuple = checkpointer.get(config)
        
        if checkpoint_tuple:
            # Checkpoint is returned as a tuple (checkpoint, metadata)
            if isinstance(checkpoint_tuple, tuple):
                checkpoint = checkpoint_tuple[0]
            else:
                checkpoint = checkpoint_tuple
            
            # Access messages from checkpoint
            if hasattr(checkpoint, 'channel_values') and checkpoint.channel_values:
                messages = checkpoint.channel_values.get("messages", [])
            elif isinstance(checkpoint, dict):
                messages = checkpoint.get("channel_values", {}).get("messages", [])
            else:
                messages = []
            
            # Fallback: Try to get messages from a dummy invoke
            if not messages:
                try:
                    # Use agent's internal state access
                    # This is a workaround - invoke with empty message to get state
                    dummy_response = agent.invoke(
                        {"messages": []},
                        config=config
                    )
                    if dummy_response and dummy_response.get("messages"):
                        messages = dummy_response["messages"]
                except Exception:
                    pass
            
            print()
            print_section("Conversation History (Memory)", "💾")
            print(f"{colorize(f'Total messages: {len(messages)}', Colors.CYAN, bold=True)}")
            print()
            
            if not messages:
                print_warning("No messages found in conversation history.")
                print_info("This might mean the conversation hasn't started yet, or there's an issue accessing memory.")
                return
            
            for i, msg in enumerate(messages, 1):
                if hasattr(msg, "content"):
                    content = str(msg.content)
                    msg_type = type(msg).__name__
                    role = "User" if "Human" in msg_type else "Agent" if "AI" in msg_type else msg_type
                    
                    # Truncate long messages
                    if len(content) > 150:
                        content = content[:150] + "..."
                    
                    print(f"{colorize(f'{i}.', Colors.YELLOW)} {colorize(f'[{role}]', Colors.BRIGHT_CYAN, bold=True)}")
                    print(f"   {colorize(content, Colors.WHITE)}")
                    
                    # Show tool calls if present
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            tool_name = tc.get("name", "unknown")
                            tool_args = tc.get("args", {})
                            print(f"   {colorize('🔧 Tool:', Colors.MAGENTA)} {colorize(tool_name, Colors.BRIGHT_MAGENTA)} {colorize(str(tool_args), Colors.DIM)}")
                
                elif hasattr(msg, "name"):  # Tool message
                    tool_name = msg.name
                    tool_content = str(msg.content) if hasattr(msg, "content") else ""
                    if len(tool_content) > 150:
                        tool_content = tool_content[:150] + "..."
                    print(f"{colorize(f'{i}.', Colors.YELLOW)} {colorize('[Tool Result]', Colors.BRIGHT_MAGENTA, bold=True)}")
                    print(f"   {colorize(f'{tool_name}:', Colors.MAGENTA)} {colorize(tool_content, Colors.WHITE)}")
                
                print()
        else:
            print_warning("No conversation history found in state.")
            
    except Exception as e:
        logger.error(f"Error accessing conversation history: {str(e)}")
        print_error(f"Error accessing conversation history: {str(e)}")


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
            # Sanitize message content to handle UTF-8 encoding errors
            message_content = sanitize_text(str(extracted['message_content']))
            print(f"{colorize('Agent Response:', Colors.BRIGHT_GREEN, bold=True)} {colorize(message_content, Colors.BRIGHT_WHITE)}")
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
    
    try:
        for token, metadata in agent.stream(
            {"messages": messages},
            config=config,
            context=context,
            stream_mode="messages"  # Stream LLM tokens as they're generated
        ):
            # Extract node information from metadata
            node = metadata.get("langgraph_node", "unknown")
            
            # Only process tokens from the model node, skip tool node outputs
            if node != "model":
                continue
            
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
                            # Sanitize text to handle UTF-8 encoding errors
                            text = sanitize_text(str(text))
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
                        text = sanitize_text(str(block.text))
                        if text:
                            print(f"{colorize(text, Colors.BRIGHT_WHITE)}", end="", flush=True)
        
        print()  # Final newline after streaming
    
    except UnicodeEncodeError as e:
        # Handle UTF-8 encoding errors gracefully
        logger.error(f"UTF-8 encoding error during streaming: {str(e)}")
        print()
        print_warning("Encoding error occurred. Attempting to recover...")
        # Try to get the response without streaming
        raise


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
    print(f"{colorize('Memory:', Colors.CYAN, bold=True)} {colorize('InMemorySaver (Short-term memory enabled)', Colors.BRIGHT_GREEN)}")
    print(f"{colorize('Middleware:', Colors.CYAN, bold=True)} {colorize('ModelCallLimitMiddleware, MonitoringMiddleware', Colors.BRIGHT_WHITE)}")
    
    # LangSmith tracing (optional but recommended)
    if os.getenv("LANGSMITH_TRACING") == "true":
        langsmith_project = os.getenv("LANGSMITH_PROJECT", "weather-agent")
        print_success(f"LangSmith tracing enabled (project: {colorize(langsmith_project, Colors.BRIGHT_CYAN, bold=True)})", "✅")
        print_info(f"View traces at: {colorize('https://smith.langchain.com', Colors.BRIGHT_CYAN, bold=True)}", "🔗")
    else:
        print_warning("LangSmith tracing disabled. Set LANGSMITH_TRACING=true to enable.", "ℹ️")
    
    # `thread_id` is a unique identifier for a given conversation.
    # Using the same thread_id maintains conversation memory across interactions
    # Based on: https://docs.langchain.com/oss/python/langchain/short-term-memory
    config = {"configurable": {"thread_id": "1"}}
    
    # Interactive loop - wait for user input
    print()
    print_success("Weather Agent is ready! Type your questions below.", "🚀")
    print_info("💾 Short-term memory enabled - I'll remember our conversation!", "💾")
    print_info("Type 'exit', 'quit', or press Ctrl+C to stop.", "ℹ️")
    print_info("Type '/history' to view conversation history, '/debug' to toggle debug mode.", "🔍")
    print()
    
    # Debug mode flag
    debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
    
    try:
        while True:
            # Get user input
            try:
                user_input = input(f"{colorize('You:', Colors.BRIGHT_BLUE, bold=True)} ").strip()
            except (EOFError, KeyboardInterrupt):
                # Handle Ctrl+C or EOF
                print()
                print_warning("Exiting...", "👋")
                break
            
            # Check for exit commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                print()
                print_success("Goodbye! Thanks for using Weather Agent!", "👋")
                break
            
            # Check for debug commands
            if user_input.lower() == '/history':
                show_conversation_history(agent, config, checkpointer)
                continue
            
            if user_input.lower() == '/debug':
                debug_mode = not debug_mode
                if debug_mode:
                    print_success("Debug mode enabled - will show messages sent to model", "🔍")
                    # Update middleware with debug mode
                    for mw in middleware:
                        if isinstance(mw, MonitoringMiddleware):
                            mw.debug = True
                else:
                    print_info("Debug mode disabled", "🔍")
                    # Update middleware with debug mode
                    for mw in middleware:
                        if isinstance(mw, MonitoringMiddleware):
                            mw.debug = False
                continue
            
            # Skip empty input
            if not user_input:
                continue
            
            # Process user input with agent
            print(f"{colorize('Agent:', Colors.BRIGHT_GREEN, bold=True)} ", end="")
            try:
                stream_agent_response(
                    agent=agent,
                    messages=[{"role": "user", "content": user_input}],
                    config=config,
                    context=Context(user_id="1")
                )
                
            except Exception as e:
                error_msg = sanitize_text(str(e))
                logger.error(f"Streaming error: {error_msg}")
                print_error(f"Error: {error_msg}")
                print_warning("Please check your Ollama connection and model availability.")
                # Fallback to regular invoke if streaming fails
                print_warning("Falling back to non-streaming mode...", "🔄")
                try:
                    response = agent.invoke(
                        {"messages": [{"role": "user", "content": user_input}]},
                        config=config,
                        context=Context(user_id="1")
                    )
                    extracted = extract_response(response, verbose=False)
                    print_response(extracted, show_tool_calls=True)
                except Exception as fallback_error:
                    fallback_msg = sanitize_text(str(fallback_error))
                    logger.error(f"Fallback error: {fallback_msg}")
                    print_error(f"Fallback error: {fallback_msg}")
            
            print()  # Add spacing between interactions
            
    except KeyboardInterrupt:
        print()
        print_warning("Interrupted by user. Exiting...", "👋")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print_error(f"Unexpected error: {str(e)}")


