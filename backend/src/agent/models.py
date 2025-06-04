"""Model utility functions for different LLM providers."""

import os
from typing import Any, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_anthropic import ChatAnthropic


def get_llm(model_name: str, provider: str, **kwargs) -> Any:
    """
    Get an LLM instance based on the provider and model name.
    
    Args:
        model_name: The name of the model to use
        provider: The provider ('gemini', 'openai', or 'anthropic')
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
    
    else:
        raise ValueError(f"Unsupported provider: {provider}. Supported providers: gemini, openai, azure_openai, anthropic")


def get_supported_models() -> Dict[str, list]:
    """
    Get default model names for each provider.
    
    Returns:
        Dictionary mapping providers to their default models for different tasks
    """
    return {
        "gemini": [
            "gemini-2.0-flash", 
            "gemini-2.5-flash-preview-04-17",
            "gemini-2.5-pro-preview-05-06"
        ],
        "openai": [
            "gpt-4o-mini",
            "gpt-4o",
        ],
        "anthropic": [
            "claude-3-5-haiku-20241022",
            "claude-3-5-sonnet-20241022",
        ],
        "azure_openai": [
            "gpt-4o-mini",  # Default deployment names
            "gpt-4o",
        ]
    }
