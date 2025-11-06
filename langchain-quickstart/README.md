# Weather Agent - LangChain Quickstart

A simple LangChain agent for weather forecasting with LangSmith tracing support. Available in two versions:
- **weather_agent.py**: Uses OpenAI (GPT-4o-mini)
- **weather_agent_ollama.py**: Uses Ollama (local LLM)

## Prerequisites

- Docker and Docker Compose installed
- For Ollama version: Ollama running (locally or in another Docker container)
- For OpenAI version: OpenAI API key
- LangSmith account (optional but recommended for tracing) - Get your API key from [smith.langchain.com](https://smith.langchain.com)

## Quick Start

### Step 1: Set Up Environment Variables (Required)

**All environment variables are centralized in a single `.env` file** for easy management.

**Option A: Interactive Setup (Recommended)**
```bash
./setup_env.sh
```
This script will guide you through setting up your `.env` file interactively.

**Option B: Manual Setup**
```bash
# Copy the example file
cp .env.example .env

# Edit the .env file with your actual values
nano .env
# or
vim .env
```

**Required values in `.env`:**
- `OPENAI_API_KEY`: For `weather_agent.py` (OpenAI version)
- `OLLAMA_BASE_URL`: For `weather_agent_ollama.py` (Ollama version)
- `LANGSMITH_API_KEY`: For tracing (optional but recommended)

See `.env.example` for all available configuration options.

### Step 2: Run the Agents

### Option 1: Using Docker Compose (Recommended)

**All environment variables are automatically loaded from `.env` file!**

**For Ollama version:**
1. **Pull the Ollama model** (if not already done):
   ```bash
   ollama pull granite3.1-moe:3b
   # For better tool calling, consider larger models:
   # ollama pull llama3.2:3b    # Better tool calling support
   # ollama pull llama3.2:7b     # Even better tool calling
   # ollama pull mistral         # Good tool calling support
   ```
   
   **Note**: Smaller models (2b parameters) may have limited tool calling capabilities. The agent will work better with models that have 3b+ parameters.

2. **Run Ollama agent:**
   ```bash
   # Simple - uses .env file automatically
   docker compose up weather-agent-ollama --build

   # Override specific values if needed
   OLLAMA_MODEL=llama3.2:3b docker-compose up weather-agent-ollama --build
   ```

**For OpenAI version:**
```bash
# Simple - uses .env file automatically
docker-compose up weather-agent-openai --build
```

**Both agents automatically use:**
- Environment variables from `.env` file
- LangSmith tracing (if enabled in `.env`)
- All configuration from `.env` file

### Option 2: Using Convenience Script (Easiest)

**The `run_docker.sh` script automatically loads `.env` and runs the correct agent:**

```bash
# Run Ollama agent (loads from .env automatically)
./run_docker.sh ollama

# Run OpenAI agent (loads from .env automatically)
./run_docker.sh openai
```

### Option 3: Using Docker Build Directly

**For Ollama version:**
```bash
# Build the image
docker build --build-arg SCRIPT=weather_agent_ollama.py -t weather-agent-ollama:latest .

# Run with .env file (recommended)
docker run --rm -it --env-file .env \
  -e LANGSMITH_PROJECT="${LANGSMITH_PROJECT:-weather-agent-ollama}" \
  weather-agent-ollama:latest

# Or override specific values
docker run --rm -it --env-file .env \
  -e OLLAMA_MODEL=llama3.2:3b \
  weather-agent-ollama:latest
```

**For OpenAI version:**
```bash
# Build the image
docker build --build-arg SCRIPT=weather_agent.py -t weather-agent-openai:latest .

# Run with .env file (recommended)
docker run --rm -it --env-file .env \
  -e LANGSMITH_PROJECT="${LANGSMITH_PROJECT:-weather-agent-openai}" \
  weather-agent-openai:latest
```

## Configuration

### Centralized Environment Variables

**All environment variables are managed in a single `.env` file** for easy configuration. See `.env.example` for a complete template.

**OpenAI Configuration (for weather_agent.py):**
- `OPENAI_API_KEY`: Your OpenAI API key (required for OpenAI version)

**Ollama Configuration (for weather_agent_ollama.py):**
- `OLLAMA_BASE_URL`: Ollama service URL (default: `http://host.docker.internal:11434`)
  - For Docker: `http://host.docker.internal:11434`
  - For localhost: `http://localhost:11434`
  - For Docker network: `http://ollama:11434`
- `OLLAMA_MODEL`: Model name (default: `granite3.1-moe:3b`)

**LangSmith Tracing (Optional but Recommended):**
- `LANGSMITH_TRACING`: Enable tracing (set to `true` to enable)
- `LANGSMITH_API_KEY`: Your LangSmith API key (get from [smith.langchain.com](https://smith.langchain.com))
- `LANGSMITH_PROJECT`: Base project name for organizing traces (default: `weather-agent`)
- `LANGSMITH_PROJECT_OLLAMA`: Override project name for Ollama agent (optional)
- `LANGSMITH_PROJECT_OPENAI`: Override project name for OpenAI agent (optional)

**Benefits of using `.env` file:**
- ✅ Single point of configuration for both agents
- ✅ No need to pass environment variables manually
- ✅ Easy to manage and update
- ✅ Secure (`.env` is git-ignored)
- ✅ Works with Docker Compose and Docker directly

### Setting Up LangSmith Tracing

1. **Get your API key:**
   - Sign up at [smith.langchain.com](https://smith.langchain.com)
   - Navigate to Settings → API Keys
   - Create a new API key

2. **Enable tracing in `.env` file:**
   ```bash
   # Edit .env file
   nano .env
   
   # Set these values:
   LANGSMITH_TRACING=true
   LANGSMITH_API_KEY=your-actual-api-key-here
   LANGSMITH_PROJECT=weather-agent
   ```

3. **Or use the setup script:**
   ```bash
   ./setup_env.sh
   # Follow the prompts to set up LangSmith tracing
   ```

4. **View traces:**
   - Go to [smith.langchain.com](https://smith.langchain.com)
   - Navigate to your project (from `LANGSMITH_PROJECT` in `.env`) to see traces, metrics, and debugging information
   - Traces are automatically sent when `LANGSMITH_TRACING=true` in `.env`

## Docker Networking

If Ollama is running:
- **On host machine**: Use `http://host.docker.internal:11434` (default)
- **In another Docker container**: Use the container name, e.g., `http://ollama:11434`
- **On localhost**: Use `http://localhost:11434` (only works if not in Docker)

## What It Does

The agent demonstrates:
- LangChain agent with Ollama (local LLM)
- Tool calling (weather and location tools)
- Structured output using ToolStrategy
- Conversational memory
- System prompts
- **LangSmith tracing and observability** - Monitor, debug, and trace agent execution

## Running Without Docker

You can also run the agents directly:

**With OpenAI (weather_agent.py):**
```bash
pip install -r requirements.txt
export OPENAI_API_KEY="<your-openai-api-key>"
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="<your-langsmith-api-key>"
export LANGSMITH_PROJECT="weather-agent-openai"
python weather_agent.py
```

**With Ollama (weather_agent_ollama.py):**
```bash
pip install -r requirements.txt
export OLLAMA_BASE_URL="http://localhost:11434"
export OLLAMA_MODEL="granite3.1-moe:3b"
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="<your-langsmith-api-key>"
export LANGSMITH_PROJECT="weather-agent-ollama"
python weather_agent_ollama.py
```
