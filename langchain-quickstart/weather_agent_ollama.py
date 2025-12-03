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
7. Streaming support for real-time feedback
8. Improved tool descriptions for better tool calling
"""

import os
from dataclasses import dataclass
from typing import Any, Dict

from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver


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


# Define tools with improved descriptions
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
    if not city or not city.strip():
        return "Error: Please provide a valid city name."
    
    # In a real implementation, this would call an actual weather API
    # For now, this is a mock implementation
    result = f"It's always sunny in {city.strip()}!"
    return result


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
    if not runtime or not runtime.context:
        return "Error: Unable to retrieve user location. Context is missing."
    
    user_id = runtime.context.user_id
    if not user_id:
        return "Error: User ID not found in context."
    
    # In a real implementation, this would query a user database
    location = "Alaska" if user_id == "1" else "Norway"
    return location


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
    temperature=0.7,  # Lower temperature for more consistent tool calling
    timeout=60,  # Increase timeout for slower models
    num_ctx=4096,  # Context window size (adjust based on model)
    streaming=True,  # Enable streaming for incremental responses
    # Additional options for better tool calling:
    # num_predict=512,  # Max tokens to generate
    # top_p=0.9,  # Nucleus sampling
    # repeat_penalty=1.1,  # Reduce repetition
)


# Set up memory
checkpointer = InMemorySaver()

# Create agent
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    checkpointer=checkpointer,
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
            message_content = str(extracted['message_content'])
            print(f"{colorize('Agent Response:', Colors.BRIGHT_GREEN, bold=True)} {colorize(message_content, Colors.BRIGHT_WHITE)}")
        else:
            print_warning("No response content available")


def display_memory(checkpointer, config: dict):
    """
    Display the conversation history stored in memory.
    
    Args:
        checkpointer: The InMemorySaver checkpointer instance
        config: Agent configuration with thread_id
    """
    try:
        # Retrieve the checkpoint from memory
        checkpoint = checkpointer.get(config)
        
        if checkpoint is None:
            print_warning("No conversation history found in memory.")
            return
        
        # Handle different checkpoint structures
        # Checkpoint might be a dict with channel_values or an object with channel_values attribute
        if isinstance(checkpoint, dict):
            channel_values = checkpoint.get("channel_values", {})
        elif hasattr(checkpoint, "channel_values"):
            channel_values = checkpoint.channel_values
        else:
            print_warning("Unexpected checkpoint structure.")
            return
        
        # Extract messages from channel_values
        if isinstance(channel_values, dict):
            messages = channel_values.get("messages", [])
        elif hasattr(channel_values, "messages"):
            messages = channel_values.messages
        else:
            messages = []
        
        if not messages:
            print_warning("Memory is empty - no messages stored yet.")
            return
        
        print_header("Conversation Memory", "💾")
        thread_id = config.get("configurable", {}).get("thread_id", "unknown")
        print(f"{colorize('Thread ID:', Colors.CYAN, bold=True)} {colorize(str(thread_id), Colors.BRIGHT_WHITE)}")
        print(f"{colorize('Total Messages:', Colors.CYAN, bold=True)} {colorize(str(len(messages)), Colors.BRIGHT_WHITE)}")
        print()
        
        for idx, msg in enumerate(messages, 1):
            # Determine message type and role
            msg_type = type(msg).__name__
            
            # Check message type using various methods
            is_human = (msg_type == "HumanMessage" or 
                       (hasattr(msg, "type") and getattr(msg, "type", None) == "human") or
                       (isinstance(msg, dict) and msg.get("type") == "human"))
            
            is_ai = (msg_type == "AIMessage" or 
                    (hasattr(msg, "type") and getattr(msg, "type", None) == "ai") or
                    (isinstance(msg, dict) and msg.get("type") == "ai"))
            
            is_tool = (msg_type == "ToolMessage" or 
                      (hasattr(msg, "type") and getattr(msg, "type", None) == "tool") or
                      (isinstance(msg, dict) and msg.get("type") == "tool"))
            
            if is_human:
                role = "User"
                emoji = "👤"
                color = Colors.BRIGHT_BLUE
            elif is_ai:
                role = "Agent"
                emoji = "🤖"
                color = Colors.BRIGHT_GREEN
            elif is_tool:
                role = "Tool"
                emoji = "🔧"
                color = Colors.BRIGHT_MAGENTA
            else:
                role = msg_type
                emoji = "📝"
                color = Colors.BRIGHT_WHITE
            
            # Extract content
            if hasattr(msg, "content"):
                content = str(msg.content) if msg.content is not None else ""
            elif isinstance(msg, dict):
                content = str(msg.get("content", ""))
            else:
                content = str(msg)
            
            # Extract tool calls if present
            tool_calls = None
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_calls = msg.tool_calls
            elif isinstance(msg, dict) and "tool_calls" in msg and msg["tool_calls"]:
                tool_calls = msg["tool_calls"]
            
            # Print message header
            print_section(f"Message {idx}: {role}", emoji)
            
            # Print tool calls if any
            if tool_calls:
                print(f"{colorize('Tool Calls:', Colors.MAGENTA, bold=True)}")
                for tool_call in tool_calls:
                    if isinstance(tool_call, dict):
                        tool_name = tool_call.get("name", "unknown")
                        tool_args = tool_call.get("args", {})
                    else:
                        tool_name = getattr(tool_call, "name", "unknown")
                        tool_args = getattr(tool_call, "args", {})
                    
                    print(f"  {colorize('•', Colors.BRIGHT_MAGENTA)} {colorize(tool_name, Colors.BRIGHT_MAGENTA, bold=True)}")
                    if tool_args:
                        print(f"    {colorize('Args:', Colors.MAGENTA)} {colorize(str(tool_args), Colors.WHITE)}")
                print()
            
            # Print content (truncate if too long)
            if content:
                if len(content) > 500:
                    print(f"{colorize('Content:', color, bold=True)} {colorize(content[:500] + '...', Colors.WHITE)}")
                    print(f"{colorize('(truncated, full length: ' + str(len(content)) + ' chars)', Colors.DIM)}")
                else:
                    print(f"{colorize('Content:', color, bold=True)} {colorize(content, Colors.WHITE)}")
            else:
                print(f"{colorize('Content:', color, bold=True)} {colorize('(empty)', Colors.DIM)}")
            
            print()  # Spacing between messages
        
        print_success(f"End of memory ({len(messages)} messages)", "✅")
        
    except Exception as e:
        print_error(f"Error retrieving memory: {str(e)}")
        import traceback
        if os.getenv("DEBUG", "").lower() == "true":
            print_error(f"Traceback: {traceback.format_exc()}")


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
    tool_results_shown = set()  # Track which tool results we've shown
    last_messages_count = 0  # Track message count to detect new tool results
    
    for token, metadata in agent.stream(
        {"messages": messages},
        config=config,
        context=context,
        stream_mode="messages"  # Stream LLM tokens as they're generated
    ):
        # Extract node information from metadata
        node = metadata.get("langgraph_node", "unknown")
        
        # Check if token contains messages list (for detecting new tool results)
        token_messages = None
        if isinstance(token, dict) and "messages" in token:
            token_messages = token["messages"]
        elif hasattr(token, "messages"):
            token_messages = token.messages
        
        # Check for new ToolMessages in the messages list
        if token_messages and len(token_messages) > last_messages_count:
            # New messages added - check for ToolMessages
            for msg in token_messages[last_messages_count:]:
                msg_type = type(msg).__name__ if hasattr(type(msg), "__name__") else str(type(msg))
                is_tool_msg = (msg_type == "ToolMessage" or 
                              (hasattr(msg, "type") and getattr(msg, "type", None) == "tool") or
                              (isinstance(msg, dict) and msg.get("type") == "tool"))
                
                if is_tool_msg:
                    # Extract tool name and content
                    if hasattr(msg, "name"):
                        tool_name = msg.name
                    elif isinstance(msg, dict):
                        tool_name = msg.get("name", "unknown")
                    else:
                        tool_name = "unknown"
                    
                    if hasattr(msg, "content"):
                        content = msg.content
                    elif isinstance(msg, dict):
                        content = msg.get("content", "")
                    else:
                        content = str(msg)
                    
                    # Show tool result
                    tool_id = f"{tool_name}_{str(content)[:50]}"
                    if tool_id not in tool_results_shown:
                        print_tool_call(f"{tool_name} (result)")
                        print(f"{colorize(str(content), Colors.BRIGHT_YELLOW)}")
                        tool_results_shown.add(tool_id)
            
            last_messages_count = len(token_messages)
        
        # Check if this token itself is a ToolMessage
        token_type = type(token).__name__ if hasattr(type(token), "__name__") else str(type(token))
        is_tool_message = (token_type == "ToolMessage" or 
                          (isinstance(token, dict) and token.get("type") == "tool") or
                          (hasattr(token, "type") and getattr(token, "type", None) == "tool"))
        
        # Handle tool execution results (if token is a ToolMessage)
        if is_tool_message or node in ["tools", "tool"]:
            # Extract tool name and content
            if hasattr(token, "name"):
                tool_name = token.name
            elif isinstance(token, dict):
                tool_name = token.get("name", "unknown")
            else:
                tool_name = None
            
            if hasattr(token, "content"):
                content = token.content
            elif isinstance(token, dict):
                content = token.get("content", "")
            else:
                content = str(token)
            
            # Show tool result if we haven't shown it yet
            tool_id = f"{tool_name}_{str(content)[:50]}"
            if tool_name and tool_id not in tool_results_shown:
                print_tool_call(f"{tool_name} (result)")
                print(f"{colorize(str(content), Colors.BRIGHT_YELLOW)}")
                tool_results_shown.add(tool_id)
            continue
        
        # Only process tokens from the model node for text streaming
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
                        text = str(text)
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
                    text = str(block.text)
                    if text:
                        print(f"{colorize(text, Colors.BRIGHT_WHITE)}", end="", flush=True)
    
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
    print(f"{colorize('Memory:', Colors.CYAN, bold=True)} {colorize('InMemorySaver (Short-term memory enabled)', Colors.BRIGHT_GREEN)}")
    
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
    print_info("Type '/memory' or '/history' to view conversation history.", "🔍")
    print()
    
    while True:
        # Get user input
        try:
            user_input = input(f"{colorize('You:', Colors.BRIGHT_BLUE, bold=True)} ").strip()
        except (UnicodeDecodeError, UnicodeError):
            # Handle encoding errors in input
            print_warning("Input encoding error. Please try again.")
            continue
        
        # Check for exit commands
        if user_input.lower() in ['exit', 'quit', 'q']:
            print()
            print_success("Goodbye! Thanks for using Weather Agent!", "👋")
            break
        
        # Skip empty input
        if not user_input:
            continue
        
        # Handle /memory and /history commands
        if user_input.lower() in ['/memory', '/history']:
            display_memory(checkpointer, config)
            print()  # Add spacing
            continue
        
        # Process user input with agent
        print(f"{colorize('Agent:', Colors.BRIGHT_GREEN, bold=True)} ", end="")
        stream_agent_response(
            agent=agent,
            messages=[{"role": "user", "content": user_input}],
            config=config,
            context=Context(user_id="1")
        )
        
        print()  # Add spacing between interactions


