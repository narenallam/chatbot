/**
 * Application configuration
 * All environment variables should be accessed through this file
 */

// API Configuration
export const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
export const API_TIMEOUT = parseInt(process.env.REACT_APP_API_TIMEOUT || '30000', 10);
export const WS_URL = process.env.REACT_APP_WS_URL || 'ws://localhost:8000';

// Application Settings
export const APP_NAME = process.env.REACT_APP_APP_NAME || 'Personal Assistant AI Chatbot';
export const APP_VERSION = process.env.REACT_APP_APP_VERSION || '1.0.0';
export const ENVIRONMENT = process.env.REACT_APP_ENVIRONMENT || 'development';

// Feature Flags
export const FEATURES = {
  webSearch: process.env.REACT_APP_ENABLE_WEB_SEARCH === 'true',
  documentUpload: process.env.REACT_APP_ENABLE_DOCUMENT_UPLOAD === 'true',
  chatHistory: process.env.REACT_APP_ENABLE_CHAT_HISTORY === 'true',
  darkMode: process.env.REACT_APP_ENABLE_DARK_MODE === 'true',
  analytics: process.env.REACT_APP_ENABLE_ANALYTICS === 'true',
};

// UI Configuration
export const UI_CONFIG = {
  defaultTheme: process.env.REACT_APP_DEFAULT_THEME || 'light',
  sidebarDefaultOpen: process.env.REACT_APP_SIDEBAR_DEFAULT_OPEN === 'true',
  consoleDefaultOpen: process.env.REACT_APP_CONSOLE_DEFAULT_OPEN === 'true',
};

// File Upload Configuration
export const FILE_UPLOAD_CONFIG = {
  maxFileSizeMB: parseInt(process.env.REACT_APP_MAX_FILE_SIZE_MB || '500', 10),
  allowedFileTypes: (process.env.REACT_APP_ALLOWED_FILE_TYPES || '.pdf,.txt,.docx,.doc,.rtf,.odt,.csv,.xlsx,.xls,.pptx,.ppt,.png,.jpg,.jpeg,.gif,.bmp,.tiff,.svg,.md,.html,.htm,.xml,.json').split(','),
};

// Chat Configuration
export const CHAT_CONFIG = {
  maxMessageLength: parseInt(process.env.REACT_APP_MAX_MESSAGE_LENGTH || '10000', 10),
  autoScrollChat: process.env.REACT_APP_AUTO_SCROLL_CHAT !== 'false',
  showTimestamps: process.env.REACT_APP_SHOW_TIMESTAMPS !== 'false',
  messageSoundEnabled: process.env.REACT_APP_MESSAGE_SOUND_ENABLED === 'true',
};

// Document Preview Configuration
export const PREVIEW_CONFIG = {
  maxPages: parseInt(process.env.REACT_APP_PREVIEW_MAX_PAGES || '50', 10),
  defaultZoom: parseFloat(process.env.REACT_APP_PREVIEW_DEFAULT_ZOOM || '1.0'),
};

// WebSocket Configuration
export const WS_CONFIG = {
  reconnectInterval: parseInt(process.env.REACT_APP_WS_RECONNECT_INTERVAL || '5000', 10),
  maxReconnectAttempts: parseInt(process.env.REACT_APP_WS_MAX_RECONNECT_ATTEMPTS || '10', 10),
};

// Search Configuration
export const SEARCH_CONFIG = {
  debounceMs: parseInt(process.env.REACT_APP_SEARCH_DEBOUNCE_MS || '300', 10),
  minSearchLength: parseInt(process.env.REACT_APP_MIN_SEARCH_LENGTH || '2', 10),
};

// Polling Configuration
export const POLLING_CONFIG = {
  healthCheckInterval: parseInt(process.env.REACT_APP_HEALTH_CHECK_INTERVAL || '30000', 10),
  documentRefreshInterval: parseInt(process.env.REACT_APP_DOCUMENT_REFRESH_INTERVAL || '60000', 10),
};

// Performance Configuration
export const PERFORMANCE_CONFIG = {
  lazyLoadEnabled: process.env.REACT_APP_LAZY_LOAD_ENABLED !== 'false',
  virtualScrollEnabled: process.env.REACT_APP_VIRTUAL_SCROLL_ENABLED !== 'false',
};

// External Resources
export const EXTERNAL_RESOURCES = {
  fontUrl: process.env.REACT_APP_FONT_URL || 'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap',
  iconFontUrl: process.env.REACT_APP_ICON_FONT_URL || 'https://fonts.googleapis.com/css2?family=Material+Icons&display=swap',
};

// Logging Configuration
export const LOG_CONFIG = {
  level: process.env.REACT_APP_LOG_LEVEL || 'info',
  enableConsoleLogs: process.env.REACT_APP_ENABLE_CONSOLE_LOGS !== 'false',
};

// Development Tools
export const DEV_CONFIG = {
  enableDevTools: process.env.REACT_APP_ENABLE_DEV_TOOLS === 'true',
  mockApi: process.env.REACT_APP_MOCK_API === 'true',
};

// Helper function to build API endpoints
export const buildApiUrl = (path: string): string => {
  const baseUrl = API_URL.replace(/\/$/, ''); // Remove trailing slash
  const cleanPath = path.startsWith('/') ? path : `/${path}`;
  return `${baseUrl}${cleanPath}`;
};

// Helper function to build WebSocket URLs
export const buildWsUrl = (path: string): string => {
  const baseUrl = WS_URL.replace(/\/$/, ''); // Remove trailing slash
  const cleanPath = path.startsWith('/') ? path : `/${path}`;
  return `${baseUrl}${cleanPath}`;
};

// Export all configuration as a single object for convenience
export const config = {
  api: {
    url: API_URL,
    timeout: API_TIMEOUT,
    wsUrl: WS_URL,
  },
  app: {
    name: APP_NAME,
    version: APP_VERSION,
    environment: ENVIRONMENT,
  },
  features: FEATURES,
  ui: UI_CONFIG,
  fileUpload: FILE_UPLOAD_CONFIG,
  chat: CHAT_CONFIG,
  preview: PREVIEW_CONFIG,
  ws: WS_CONFIG,
  search: SEARCH_CONFIG,
  polling: POLLING_CONFIG,
  performance: PERFORMANCE_CONFIG,
  external: EXTERNAL_RESOURCES,
  log: LOG_CONFIG,
  dev: DEV_CONFIG,
  buildApiUrl,
  buildWsUrl,
};

export default config;