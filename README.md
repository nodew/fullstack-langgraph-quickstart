# Gemini Fullstack LangGraph Quickstart

This project demonstrates a fullstack application using a React frontend and a LangGraph-powered backend agent. The agent is designed to perform comprehensive research on a user's query by dynamically generating search terms, querying the web using Google Search, reflecting on the results to identify knowledge gaps, and iteratively refining its search until it can provide a well-supported answer with citations. This application serves as an example of building research-augmented conversational AI using LangGraph and Google's Gemini models.

![Gemini Fullstack LangGraph](./app.png)

## Features

- 💬 Fullstack application with a React frontend and LangGraph backend.
- 🧠 Powered by a LangGraph agent for advanced research and conversational AI.
- 🔍 Dynamic search query generation using multiple LLM providers (Gemini, OpenAI, Claude).
- 🌐 Integrated web research via Google Search API.
- 🤔 Reflective reasoning to identify knowledge gaps and refine searches.
- 📄 Generates answers with citations from gathered sources.
- ⚡ Multi-provider support for cost optimization and performance tuning.
- 🔄 Hot-reloading for both frontend and backend development during development.

## Project Structure

The project is divided into two main directories:

-   `frontend/`: Contains the React application built with Vite.
-   `backend/`: Contains the LangGraph/FastAPI application, including the research agent logic.

## Getting Started: Development and Local Testing

Follow these steps to get the application running locally for development and testing.

**1. Prerequisites:**

-   Node.js and npm (or yarn/pnpm)
-   Python 3.8+
-   **`GEMINI_API_KEY`**: The backend agent requires a Google Gemini API key.
    1.  Navigate to the `backend/` directory.
    2.  Create a file named `.env` by copying the `backend/.env.example` file.
    3.  Open the `.env` file and add your Gemini API key: `GEMINI_API_KEY="YOUR_ACTUAL_API_KEY"`

**2. Install Dependencies:**

**Backend:**

```bash
cd backend
pip install .
```

**Frontend:**

```bash
cd frontend
npm install
```

**3. Run Development Servers:**

**Backend & Frontend:**

```bash
make dev
```
This will run the backend and frontend development servers.    Open your browser and navigate to the frontend development server URL (e.g., `http://localhost:5173/app`).

_Alternatively, you can run the backend and frontend development servers separately. For the backend, open a terminal in the `backend/` directory and run `langgraph dev`. The backend API will be available at `http://127.0.0.1:2024`. It will also open a browser window to the LangGraph UI. For the frontend, open a terminal in the `frontend/` directory and run `npm run dev`. The frontend will be available at `http://localhost:5173`._

## Multi-Provider Support

This application now supports multiple LLM providers for enhanced flexibility and cost optimization:

### Supported Providers
- **Google Gemini** (default): `gemini-2.0-flash`, `gemini-2.5-flash-preview-04-17`, `gemini-2.5-pro-preview-05-06`
- **OpenAI**: `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`
- **Azure OpenAI**: `gpt-4o`, `gpt-4o-mini`, `gpt-35-turbo` (using your Azure deployments)
- **Anthropic Claude**: `claude-3-5-sonnet-20241022`, `claude-3-5-haiku-20241022`

### Configuration
Add API keys to your `.env` file:
```env
GEMINI_API_KEY=your_gemini_key     # Optional
OPENAI_API_KEY=your_openai_key     # Optional
ANTHROPIC_API_KEY=your_claude_key  # Optional

# Azure OpenAI (Optional)
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_KEY=your_azure_openai_key
AZURE_OPENAI_API_VERSION=2024-02-01
```

You can configure different providers for each step of the research process. For detailed setup instructions and configuration examples, see [`backend/MULTI_PROVIDER_SETUP.md`](backend/MULTI_PROVIDER_SETUP.md).

For Azure OpenAI specific setup, see [`backend/AZURE_OPENAI_SETUP.md`](backend/AZURE_OPENAI_SETUP.md).

### Quick Test
Run the multi-provider test script:
```bash
cd backend
python test_multi_provider.py
```

## How the Backend Agent Works (High-Level)

The core of the backend is a LangGraph agent defined in `backend/src/agent/graph.py`. It follows these steps:

![Agent Flow](./agent.png)

1.  **Generate Initial Queries:** Based on your input, it generates a set of initial search queries using a Gemini model.
2.  **Web Research:** For each query, it uses the Gemini model with the Google Search API to find relevant web pages.
3.  **Reflection & Knowledge Gap Analysis:** The agent analyzes the search results to determine if the information is sufficient or if there are knowledge gaps. It uses a Gemini model for this reflection process.
4.  **Iterative Refinement:** If gaps are found or the information is insufficient, it generates follow-up queries and repeats the web research and reflection steps (up to a configured maximum number of loops).
5.  **Finalize Answer:** Once the research is deemed sufficient, the agent synthesizes the gathered information into a coherent answer, including citations from the web sources, using a Gemini model.

## Deployment

In production, the backend server serves the optimized static frontend build. LangGraph requires a Redis instance and a Postgres database. Redis is used as a pub-sub broker to enable streaming real time output from background runs. Postgres is used to store assistants, threads, runs, persist thread state and long term memory, and to manage the state of the background task queue with 'exactly once' semantics. For more details on how to deploy the backend server, take a look at the [LangGraph Documentation](https://langchain-ai.github.io/langgraph/concepts/deployment_options/). Below is an example of how to build a Docker image that includes the optimized frontend build and the backend server and run it via `docker-compose`.

_Note: For the docker-compose.yml example you need a LangSmith API key, you can get one from [LangSmith](https://smith.langchain.com/settings)._

_Note: If you are not running the docker-compose.yml example or exposing the backend server to the public internet, you update the `apiUrl` in the `frontend/src/App.tsx` file your host. Currently the `apiUrl` is set to `http://localhost:8123` for docker-compose or `http://localhost:2024` for development._

**1. Build the Docker Image:**

   Run the following command from the **project root directory**:
   ```bash
   docker build -t gemini-fullstack-langgraph -f Dockerfile .
   ```
**2. Run the Production Server:**

   ```bash
   GEMINI_API_KEY=<your_gemini_api_key> LANGSMITH_API_KEY=<your_langsmith_api_key> docker-compose up
   ```

Open your browser and navigate to `http://localhost:8123/app/` to see the application. The API will be available at `http://localhost:8123`.

## Technologies Used

- [React](https://reactjs.org/) (with [Vite](https://vitejs.dev/)) - For the frontend user interface.
- [Tailwind CSS](https://tailwindcss.com/) - For styling.
- [Shadcn UI](https://ui.shadcn.com/) - For components.
- [LangGraph](https://github.com/langchain-ai/langgraph) - For building the backend research agent.
- [Google Gemini](https://ai.google.dev/models/gemini) - Default LLM provider and required for Google Search.
- [OpenAI](https://openai.com/) - Optional LLM provider for query generation, reflection, and answer synthesis.
- [Anthropic Claude](https://anthropic.com/) - Optional LLM provider for query generation, reflection, and answer synthesis.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.
