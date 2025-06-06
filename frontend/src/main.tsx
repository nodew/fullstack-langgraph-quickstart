import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import "./global.css";
import App from "./App.tsx";
import { LlmProvider } from "@/contexts/LlmContext";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <LlmProvider>
        <App />
      </LlmProvider>
    </BrowserRouter>
  </StrictMode>
);
