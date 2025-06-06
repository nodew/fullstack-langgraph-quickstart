import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

type LlmProvider = {
  provider: string;
  models: [{
    name: string;
    displayName: string;
  }];
};

interface LlmContextType {
  effort: string;
  provider: string;
  model: string;
  llmProviders: LlmProvider[];
  updateEffort: (effort: string) => void;
  updateProvider: (provider: string) => void;
  updateModel: (model: string) => void;
}

const LlmContext = createContext<LlmContextType | undefined>(undefined);

const apiUrl = import.meta.env.DEV
  ? "http://localhost:2024"
  : "http://localhost:8123";

interface LlmProviderProps {
  children: ReactNode;
}

const DEFAULT_EFFORT = 'medium'; // Default effort level
const DEFAULT_PROVIDER = 'gemini';
const DEFAULT_MODEL = 'gemini-2.5-flash-preview-04-17';

export const LlmProvider: React.FC<LlmProviderProps> = ({ children }) => {
  const [effort, setEffort] = useState<string>(DEFAULT_EFFORT);
  const [provider, setProvider] = useState<string>(DEFAULT_PROVIDER);
  const [model, setModel] = useState<string>(DEFAULT_MODEL);
  const [llmProviders, setLlmProviders] = useState<LlmProvider[]>([]);

  useEffect(() => {
    const fetchLlmProviders = async () => {
      try {
        const response = await fetch(`${apiUrl}/api/providers`);
        if (!response.ok) {
          throw new Error("Failed to fetch LLM providers");
        }
        const providers = await response.json() as LlmProvider[];
        setLlmProviders(providers);

        if (providers.length > 0) {
          const defaultProvider = providers.find(p => p.provider === DEFAULT_PROVIDER) || providers[0];
          setProvider(defaultProvider.provider);
          if (defaultProvider.models.length > 0) {
            const defaultModel = defaultProvider.models.find(m => m.name === DEFAULT_MODEL) || defaultProvider.models[0];
            setModel(defaultModel.name);
          }
        }

      } catch (error) {
        console.error("Error fetching LLM providers:", error);
      }
    };

    fetchLlmProviders();
  }, []);

  useEffect(() => {
    // Ensure the model is set to a valid one when the provider changes
    const currentProvider = llmProviders.find(p => p.provider === provider);
    
    if (currentProvider) {
      if (currentProvider.models.length > 0) {
        const defaultModel = currentProvider.models.find(m => m.name === model) || currentProvider.models[0];
        setModel(defaultModel.name);
      }
    }
  }, [provider]);

  const contextValue: LlmContextType = {
    effort,
    provider,
    model,
    llmProviders,
    updateEffort: setEffort,
    updateProvider: setProvider,
    updateModel: setModel,
  };

  return (
    <LlmContext.Provider value={contextValue}>
      {children}
    </LlmContext.Provider>
  );
};

export const useLlmContext = (): LlmContextType => {
  const context = useContext(LlmContext);
  if (context === undefined) {
    throw new Error('useLlmContext must be used within an LlmProvider');
  }
  return context;
};
