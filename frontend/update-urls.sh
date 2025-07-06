#!/bin/bash

# Script to update hardcoded URLs in frontend code

echo "Updating hardcoded URLs in frontend..."

# Update App.tsx
sed -i '' "s|const BACKEND_URL = 'http://localhost:8000';|const BACKEND_URL = config.api.url;|g" src/App.tsx
sed -i '' "s|new WebSocket('ws://localhost:8000/ws/logs')|new WebSocket(config.buildWsUrl('/ws/logs'))|g" src/App.tsx

# Update Console.tsx
sed -i '' "s|ws://localhost:8000/ws/logs|' + config.buildWsUrl('/ws/logs') + '|g" src/components/Console.tsx
sed -i '' "s|new WebSocket('ws://localhost:8000/ws/logs')|new WebSocket(config.buildWsUrl('/ws/logs'))|g" src/components/Console.tsx

# Update FileUploadPanel.tsx 
sed -i '' "s|ws://localhost:8000/ws/logs|' + config.buildWsUrl('/ws/logs') + '|g" src/components/FileUploadPanel.tsx
sed -i '' "s|new WebSocket('ws://localhost:8000/ws/logs')|new WebSocket(config.buildWsUrl('/ws/logs'))|g" src/components/FileUploadPanel.tsx
sed -i '' "s|http://localhost:8000/api/documents/original/|' + config.buildApiUrl('/api/documents/original/') + '|g" src/components/FileUploadPanel.tsx

# Update DocumentPreview.tsx
sed -i '' "s|http://localhost:8000|' + config.api.url + '|g" src/components/DocumentPreview.tsx
sed -i '' "s|process.env.REACT_APP_API_URL || 'http://localhost:8000'|config.api.url|g" src/components/DocumentPreview.tsx

# Update DocumentDetails.tsx
sed -i '' "s|http://localhost:8000/api/documents/original/|' + config.buildApiUrl('/api/documents/original/') + '|g" src/components/DocumentDetails.tsx

# Update GlobalStyles.tsx
sed -i '' "s|https://fonts.googleapis.com/css2\\?family=Inter:wght@300;400;500;600;700&display=swap|' + config.external.fontUrl + '|g" src/styles/GlobalStyles.tsx

echo "URLs updated. Don't forget to add the import statement for config in each file!"
echo ""
echo "Add this import to the top of each file that uses config:"
echo "import { config } from '../config/config';"
echo ""
echo "For components in subdirectories, adjust the path accordingly."