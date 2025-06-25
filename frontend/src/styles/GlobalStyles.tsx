import { createGlobalStyle } from 'styled-components';

export const GlobalStyles = createGlobalStyle`
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  @import url('https://fonts.googleapis.com/css2?family=Moirai+One&family=Noto+Sans+Mono:wght@300;400;500&family=Fira+Code:wght@300;400;500&display=swap');

  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }

  body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    background: #1a1a1a;
    color: #ffffff;
    overflow: hidden;
    /* Enable ligatures globally for better typography */
    font-feature-settings: "liga" on, "calt" on;
    text-rendering: optimizeLegibility;
  }

  #root {
    height: 100vh;
    width: 100vw;
  }

  /* Enhanced scrollbar styling - default fallback */
  ::-webkit-scrollbar {
    width: 8px;
    height: 8px;
    background: transparent;
  }

  ::-webkit-scrollbar-track {
    background: rgba(42, 42, 42, 0.5);
    border-radius: 4px;
  }

  ::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #555 0%, #444 100%);
    border-radius: 4px;
    border: 1px solid rgba(42, 42, 42, 0.8);
    cursor: pointer;
  }

  ::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, #00ffff 0%, #00cccc 100%);
  }

  ::-webkit-scrollbar-corner {
    background: transparent;
  }

  /* Input and button resets */
  input, button, textarea {
    font-family: inherit;
    font-size: inherit;
  }

  button {
    cursor: pointer;
    border: none;
    background: none;
  }

  input {
    outline: none;
  }

  /* Selection styling */
  ::selection {
    background: rgba(0, 255, 255, 0.3);
    color: #ffffff;
  }
`; 