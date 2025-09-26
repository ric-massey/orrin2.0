// UI/goals-dashboard/src/main.tsx
import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";

const rootEl = document.getElementById("root");
if (!rootEl) throw new Error("No #root; check index.html.");
createRoot(rootEl).render(<App />);
