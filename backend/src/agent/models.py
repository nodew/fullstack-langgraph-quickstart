"""Model utility functions for different LLM providers."""

import os
from typing import Any, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama


def get_llm(model_name: str, provider: str, **kwargs) -> Any:
    """
    Get an LLM instance based on the provider and model name.

    Args:
        model_name: The name of the model to use
        provider: The provider ('gemini', 'openai', 'azure_openai', 'anthropic', or 'ollama')
        **kwargs: Additional arguments to pass to the LLM constructor

    Returns:
        An LLM instance for the specified provider

    Raises:
        ValueError: If the provider is not supported or API key is missing
    """
    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set")

        return ChatGoogleGenerativeAI(
            model=model_name,
            api_key=api_key,
            **kwargs
        )

    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set")

        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            **kwargs
        )

    elif provider == "azure_openai":
        # Required Azure OpenAI environment variables
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

        if not azure_endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT is not set")
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY is not set")

        return AzureChatOpenAI(
            deployment_name=model_name,  # This should be the deployment name in Azure
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
            **kwargs
        )

    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")

        return ChatAnthropic(
            model=model_name,
            api_key=api_key,
            **kwargs
        )

    elif provider == "ollama":
        # Ollama typically runs locally and doesn't require an API key
        # Get the base URL from environment variable, default to localhost
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        return ChatOllama(
            model=model_name,
            base_url=base_url,
            **kwargs
        )

    else:
        raise ValueError(f"Unsupported provider: {provider}. Supported providers: gemini, openai, azure_openai, anthropic, ollama")


def get_supported_models() -> Dict[str, list]:
    """
    Get default model names for each provider.

    Returns:
        Dictionary mapping providers to their default models for different tasks
    """
    return {
        "gemini": [
            {
                "name": "gemini-2.0-flash",
                "displayName": "Gemini 2.0 Flash",
            },
            {
                "name": "gemini-2.5-flash-preview-04-17",
                "displayName": "Gemini 2.5 Flash",
            },
            {
                "name": "gemini-2.5-pro-preview-05-06",
                "displayName": "Gemini 2.5 Pro",
            }
        ],
        "openai": [
            {
                "name": "gpt-4o-mini",
                "displayName": "GPT-4o Mini",
            },
            {
                "name": "gpt-4o",
                "displayName": "GPT-4o",
            },
        ],
        "anthropic": [
            {
                "name": "claude-3-5-haiku-20241022",
                "displayName": "Claude 3.5 Haiku",
            },
            {
                "name": "claude-3-5-sonnet-20241022",
                "displayName": "Claude 3.5 Sonnet",
            }
        ],
        "azure_openai": [
            {
                "name": "gpt-35-turbo",
                "displayName": "GPT-3.5 Turbo",
            },
            {
                "name": "gpt-4",
                "displayName": "GPT-4",
            },
        ],
        "ollama": [
            {
                "name": "deepseek-r1:14b",
                "displayName": "Deepseek R1 14B",
            },
            {
                "name": "qwen3:14b",
                "displayName": "Qwen 3 14B",
            }
        ]
    }
