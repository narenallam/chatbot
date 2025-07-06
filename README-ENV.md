# Environment Variables Configuration Guide

This document explains how to configure environment variables for the Personal Assistant AI Chatbot application.

## Overview

The application uses environment variables to manage configuration, ensuring no hardcoded values exist in the codebase. This approach provides:
- Security: Sensitive information like API keys are not committed to version control
- Flexibility: Easy configuration changes without code modifications
- Portability: Different configurations for development, staging, and production

## Setup Instructions

### Backend Configuration

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

3. Edit the `.env` file and update the values according to your setup.

#### Key Backend Variables:

- **LLM Configuration**:
  - `LLM_PROVIDER`: Choose between "ollama" (local) or "openai"
  - `OLLAMA_BASE_URL`: URL for Ollama server (default: http://localhost:11434)
  - `OPENAI_API_KEY`: Your OpenAI API key (required if using OpenAI)

- **Web Search**:
  - `SERPAPI_API_KEY`: API key for SerpAPI web search
  - `BRAVE_API_KEY`: API key for Brave Search (optional)
  - `GOOGLE_API_KEY`: Google Custom Search API key (optional)

- **Database**:
  - `DATABASE_URL`: SQLite database path
  - `CHROMA_DB_PATH`: Vector database storage path

- **Security**:
  - `SECRET_KEY`: Generate a secure secret key for JWT tokens
  - `ADMIN_PASSWORD`: Change the default admin password

### Frontend Configuration

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

3. Edit the `.env` file and update the values.

#### Key Frontend Variables:

- **API Configuration**:
  - `REACT_APP_API_URL`: Backend API URL (default: http://localhost:8000)
  - `REACT_APP_WS_URL`: WebSocket URL (default: ws://localhost:8000)

- **Feature Flags**:
  - `REACT_APP_ENABLE_WEB_SEARCH`: Enable/disable web search
  - `REACT_APP_ENABLE_DOCUMENT_UPLOAD`: Enable/disable document uploads
  - `REACT_APP_ENABLE_CHAT_HISTORY`: Enable/disable chat history

## Production Deployment

For production deployments:

1. **Security**:
   - Generate a strong `SECRET_KEY` using: `openssl rand -hex 32`
   - Use HTTPS URLs for all endpoints
   - Store sensitive API keys in a secure secrets management system

2. **Performance**:
   - Set `DEBUG=false` in backend
   - Set `GENERATE_SOURCEMAP=false` in frontend
   - Configure appropriate worker counts and batch sizes

3. **CORS**:
   - Update `CORS_ORIGINS` to include your production domain

4. **Database**:
   - Consider using PostgreSQL instead of SQLite for production
   - Update `DATABASE_URL` accordingly

## Environment Variable Reference

### Backend Variables

See `backend/.env.example` for the complete list with descriptions.

### Frontend Variables

See `frontend/.env.example` for the complete list with descriptions.

## Docker Configuration

If using Docker, you can pass environment variables through:

1. Docker Compose `.env` file
2. `environment` section in docker-compose.yml
3. `--env-file` flag when running containers

Example docker-compose.yml snippet:
```yaml
services:
  backend:
    env_file:
      - ./backend/.env
    environment:
      - ENVIRONMENT=production
  
  frontend:
    env_file:
      - ./frontend/.env
    environment:
      - REACT_APP_API_URL=https://api.yourdomain.com
```

## Troubleshooting

1. **API Keys Not Working**:
   - Ensure no quotes around API key values in .env files
   - Check that the .env file is in the correct directory
   - Restart the application after changing environment variables

2. **CORS Issues**:
   - Verify CORS_ORIGINS includes your frontend URL
   - Use JSON array format: `["http://localhost:3000"]`

3. **WebSocket Connection Failed**:
   - Ensure WS_URL protocol matches (ws:// for http://, wss:// for https://)
   - Check firewall/proxy settings

## Security Best Practices

1. Never commit `.env` files to version control
2. Add `.env` to `.gitignore`
3. Use different API keys for development and production
4. Rotate API keys regularly
5. Use environment-specific secret management services in production

## Support

For issues or questions about configuration, please check:
- The `.env.example` files for documentation
- Application logs for configuration errors
- GitHub issues for known problems