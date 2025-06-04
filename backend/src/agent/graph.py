import os

from agent.tools_and_schemas import SearchQueryList, Reflection
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
from google.genai import Client

from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
    generic_web_search_instructions,
)
from agent.models import get_llm
from agent.web_search import perform_web_search_with_llm
from agent.utils import (
    get_citations,
    get_research_topic,
    insert_citation_markers,
    resolve_urls,
)

load_dotenv()

# API key validation and client setup for Google native search
# If GEMINI_API_KEY is available, we'll use Google's native search with grounding
# Otherwise, fallback to our custom DuckDuckGo implementation
gemini_api_key = os.getenv("GEMINI_API_KEY")
if gemini_api_key:
    genai_client = Client(api_key=gemini_api_key)
else:
    genai_client = None

# Nodes
def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates a search queries based on the User's question.

    Uses the configured LLM to create optimized search queries for web research based on
    the User's question. Works with any LLM provider (Gemini, OpenAI, Anthropic).

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated query
    """
    configurable = Configuration.from_runnable_config(config)

    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # init query generator model
    llm = get_llm(
        model_name=configurable.query_generator_model,
        provider=configurable.query_generator_provider,
        temperature=1.0,
        max_retries=2,
    )
    structured_llm = llm.with_structured_output(SearchQueryList)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    # Generate the search queries
    result = structured_llm.invoke(formatted_prompt)
    return {"query_list": result.query}


def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["query_list"])
    ]


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that performs web research using hybrid approach.

    Uses Google's native search API with grounding metadata when GEMINI_API_KEY is available,
    otherwise falls back to DuckDuckGo search with any configured LLM.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including sources_gathered, search_query, and web_research_result
    """
    configurable = Configuration.from_runnable_config(config)
    
    # Check if we can use Google's native search
    if genai_client is not None:
        # Use Google's native search with grounding metadata
        formatted_prompt = web_searcher_instructions.format(
            current_date=get_current_date(),
            research_topic=state["search_query"],
        )

        try:
            # Uses the google genai client with native search capabilities
            response = genai_client.models.generate_content(
                model=configurable.query_generator_model,
                contents=formatted_prompt,
                config={
                    "tools": [{"google_search": {}}],
                    "temperature": 0,
                },
            )
            
            # Process Google search results with grounding metadata
            resolved_urls = resolve_urls(
                response.candidates[0].grounding_metadata.grounding_chunks, state["id"]
            )
            citations = get_citations(response, resolved_urls)
            modified_text = insert_citation_markers(response.text, citations)
            sources_gathered = [item for citation in citations for item in citation["segments"]]

            return {
                "sources_gathered": sources_gathered,
                "search_query": [state["search_query"]],
                "web_research_result": [modified_text],
            }
            
        except Exception as e:
            print(f"⚠️ Google native search failed ({e}), falling back to custom search")
            # Fall through to custom search implementation
    
    # Fallback to custom DuckDuckGo search implementation
    print(f"🔍 Using custom DuckDuckGo search for: {state['search_query']}")
    
    # Get the appropriate LLM for web research processing
    llm = get_llm(
        model_name=configurable.query_generator_model,
        provider=configurable.query_generator_provider,
        temperature=0,
        max_retries=2,
    )
    
    # Perform web search with the generic approach
    search_results = perform_web_search_with_llm(
        search_query=state["search_query"],
        llm=llm,
        search_prompt_template=generic_web_search_instructions,
        max_results=5
    )
    
    return {
        "sources_gathered": search_results["sources_gathered"],
        "search_query": [search_results["search_query"]],
        "web_research_result": [search_results["web_research_result"]],
    }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    potential follow-up queries. Uses structured output to extract
    the follow-up query in JSON format.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """
    configurable = Configuration.from_runnable_config(config)
    # Increment the research loop count and get the reasoning model
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model") or configurable.reasoning_model

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    # init reflection model
    llm = get_llm(
        model_name=reasoning_model,
        provider=configurable.reflection_provider,
        temperature=1.0,
        max_retries=2,
    )
    result = llm.with_structured_output(Reflection).invoke(formatted_prompt)

    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"]),
    }


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_research_loops setting

    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_summary")
    """
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


def finalize_answer(state: OverallState, config: RunnableConfig):
    """LangGraph node that finalizes the research summary.

    Prepares the final output by deduplicating and formatting sources, then
    combining them with the running summary to create a well-structured
    research report with proper citations.

    Args:
        state: Current graph state containing the running summary and sources gathered

    Returns:
        Dictionary with state update, including running_summary key containing the formatted final summary with sources
    """
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.reasoning_model

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )

    # init answer model
    llm = get_llm(
        model_name=configurable.answer_model,
        provider=configurable.answer_provider,
        temperature=0,
        max_retries=2,
    )
    result = llm.invoke(formatted_prompt)

    # Process sources for the final output - handle both Google grounding and custom formats
    unique_sources = []
    seen_urls = set()
    
    for source in state["sources_gathered"]:
        # Handle both Google grounding format and custom format
        if isinstance(source, dict):
            # Custom format: {"title", "url", "short_url", "snippet"}
            if "url" in source:
                url = source["url"]
                if url not in seen_urls:
                    seen_urls.add(url)
                    unique_sources.append(source)
            # Google grounding format: {"label", "short_url", "value"}  
            elif "value" in source:
                url = source["value"]
                if url not in seen_urls:
                    seen_urls.add(url)
                    # Replace short URLs with original URLs in the result content
                    if source["short_url"] in result.content:
                        result.content = result.content.replace(
                            source["short_url"], source["value"]
                        )
                    unique_sources.append(source)

    return {
        "messages": [AIMessage(content=result.content)],
        "sources_gathered": unique_sources,
    }


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes we will cycle between
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# Set the entrypoint as `generate_query`
# This means that this node is the first one called
builder.add_edge(START, "generate_query")
# Add conditional edge to continue with search queries in a parallel branch
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
# Reflect on the web research
builder.add_edge("web_research", "reflection")
# Evaluate the research
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
# Finalize the answer
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="pro-search-agent")
