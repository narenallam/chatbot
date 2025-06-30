import React, { useState, useEffect, useRef } from 'react';
import styled from 'styled-components';
import { Terminal, X, Minimize2, Maximize2, RotateCcw } from 'lucide-react';


interface LogEntry {
  type: string;
  level: string;
  message: string;
  timestamp: string;
  details?: any;
}



interface ProcessingFile {
  name: string;
  size: number;
  status: 'uploading' | 'processing' | 'completed' | 'error';
  stages: {
    uploaded: boolean;
    extracted: boolean;
    chunked: boolean;
    embedded: boolean;
    stored: boolean;
    ready: boolean;
  };
  currentStage?: string;
  statistics?: {
    chunks?: number;
    characters?: number;
    processingTime?: number;
  };
  error?: string;
}



interface ConsoleProps {
  isVisible: boolean;
  onToggle: () => void;
}

const ConsoleContainer = styled.div<{ $isVisible: boolean; $isMinimized: boolean; $isMaximized: boolean; $height: number }>`
  position: fixed;
  bottom: ${props => props.$isVisible ? '0' : `-${props.$height}px`};
  left: 0;
  right: 0;
  width: 100%;
  height: ${props => 
    props.$isMinimized ? '40px' : 
    props.$isMaximized ? '100vh' : `${props.$height}px`
  };
  background: #0a0a0a;
  border: 1px solid #333;
  border-bottom: none;
  border-left: none;
  border-right: none;
  border-radius: ${props => props.$isMaximized ? '0' : '0'};
  display: flex;
  flex-direction: column;
  z-index: 1000;
  transition: ${props => props.$isMaximized || props.$isMinimized ? 'all 0.3s ease' : 'bottom 0.3s ease'};
  box-shadow: none;
  resize: none;
`;

const ConsoleResizeHandle = styled.div`
  position: absolute;
  top: -4px;
  left: 0;
  right: 0;
  height: 8px;
  background: transparent;
  cursor: ns-resize;
  z-index: 1001;
  
  &:hover::after {
    content: '';
    position: absolute;
    top: 2px;
    left: 50%;
    transform: translateX(-50%);
    width: 40px;
    height: 4px;
    background: #00ffff;
    border-radius: 2px;
    opacity: 0.8;
  }
  
  &:active::after {
    background: #00cccc;
    opacity: 1;
  }
`;

const ConsoleHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  background: #1a1a1a;
  border-bottom: 1px solid #333;
  border-radius: 8px 8px 0 0;
  cursor: pointer;
`;

const ConsoleTitle = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  color: #00ffff;
  font-size: 0.8rem;
  font-weight: 600;
`;

const ConsoleControls = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
`;

const ControlButton = styled.button<{ $variant?: 'minimize' | 'maximize' | 'close' | 'clear' }>`
  background: none;
  border: none;
  color: #888;
  cursor: pointer;
  padding: 6px;
  border-radius: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 24px;
  height: 24px;
  
  &:hover {
    color: #fff;
    background: ${props => {
      switch (props.$variant) {
        case 'close': return '#ff5252';
        case 'minimize': return '#ffc107';
        case 'maximize': return '#4caf50';
        case 'clear': return '#2196f3';
        default: return '#333';
      }
    }};
  }
`;

const ConsoleContent = styled.div<{ $isMinimized: boolean }>`
  flex: 1;
  overflow: hidden;
  display: ${props => props.$isMinimized ? 'none' : 'flex'};
  flex-direction: column;
`;



const LogSection = styled.div`
  height: 100%;
  display: flex;
  flex-direction: column;
  overflow: hidden;
`;

const FileStatusSection = styled.div`
  height: 100%;
  display: flex;
  flex-direction: column;
  background: #0f0f0f;
  overflow: hidden;
`;

const ProcessingDetailsSection = styled.div`
  height: 50%;
  border-top: 1px solid #333;
  overflow-y: auto;
  padding: 8px;
`;

const FileProcessingSection = styled.div`
  height: 100%;
  display: flex;
  flex-direction: column;
  background: #0f0f0f;
  overflow: hidden;
`;

const ProcessingSectionHeader = styled.div`
  padding: 8px 12px;
  background: #1a1a1a;
  border-bottom: 1px solid #333;
`;

const ProcessingSectionTitle = styled.div`
  color: #00ffff;
  font-size: 0.8rem;
  font-weight: 600;
`;

const ProcessingFilesList = styled.div`
  height: 60%;
  overflow-y: auto;
  padding: 8px;
`;



const EmptyState = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 200px;
  text-align: center;
  color: #666;
`;

const EmptyStateIcon = styled.div`
  font-size: 2rem;
  margin-bottom: 8px;
  opacity: 0.6;
`;

const EmptyStateText = styled.div`
  font-size: 0.8rem;
  font-weight: 500;
  margin-bottom: 4px;
`;

const EmptyStateHint = styled.div`
  font-size: 0.7rem;
  opacity: 0.8;
  line-height: 1.3;
`;

/* Add spinning animation for processing indicators */
const SpinningIcon = styled.span`
  display: inline-block;
  animation: spin 1s linear infinite;
  
  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
`;

const ProcessingFileItem = styled.div`
  background: rgba(255, 255, 255, 0.02);
  border-radius: 6px;
  padding: 12px;
  margin-bottom: 8px;
  border-left: 3px solid #00ffff;
`;

const ProcessingFileName = styled.div`
  color: #fff;
  font-size: 0.75rem;
  font-weight: 600;
  margin-bottom: 8px;
  display: flex;
  align-items: center;
  gap: 8px;
`;

const ProcessingStages = styled.div`
  display: flex;
  flex-direction: column;
  gap: 4px;
  margin-bottom: 8px;
`;

const ProcessingStageItem = styled.div<{ $completed: boolean; $active: boolean }>`
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.7rem;
  color: ${props => props.$completed ? '#00ff00' : props.$active ? '#00ffff' : '#666'};
  padding: 2px 0;
`;

const StageIcon = styled.div<{ $completed: boolean; $active: boolean }>`
  width: 16px;
  height: 16px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.6rem;
  background: ${props => 
    props.$completed ? '#00ff00' : 
    props.$active ? '#00ffff' : '#333'
  };
  color: ${props => props.$completed || props.$active ? '#000' : '#666'};
`;

const ProcessingStats = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  font-size: 0.65rem;
  color: #888;
`;

const StatItem = styled.div`
  display: flex;
  align-items: center;
  gap: 4px;
`;

const StatValue = styled.span`
  color: #00ffff;
  font-weight: 500;
`;



const SectionHeader = styled.div`
  padding: 8px 12px;
  background: #1a1a1a;
  border-bottom: 1px solid #333;
  color: #00ffff;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
`;

const FileStatusList = styled.div`
  height: 50%;
  overflow-y: auto;
  padding: 8px;
  
  &::-webkit-scrollbar {
    width: 6px;
    background: #0f0f0f;
  }
  
  &::-webkit-scrollbar-thumb {
    background: #333;
    border-radius: 3px;
  }
`;

const FileStatusItem = styled.div<{ $status: 'uploading' | 'processing' | 'completed' | 'error' }>`
  padding: 8px;
  margin-bottom: 4px;
  border-radius: 4px;
  background: rgba(255, 255, 255, 0.02);
  border-left: 3px solid ${props => {
    switch (props.$status) {
      case 'uploading': return '#ffaa00';
      case 'processing': return '#00ffff';
      case 'completed': return '#00ff00';
      case 'error': return '#ff4444';
      default: return '#666';
    }
  }};
`;



const ProgressBar = styled.div<{ $progress: number }>`
  width: 100%;
  height: 2px;
  background: #333;
  border-radius: 1px;
  overflow: hidden;
  margin-top: 4px;
  
  &::after {
    content: '';
    display: block;
    height: 100%;
    width: ${props => props.$progress}%;
    background: linear-gradient(90deg, #00ffff, #00ff7f);
    border-radius: 1px;
    transition: width 0.3s ease;
  }
`;

const LogContainer = styled.div`
  flex: 1;
  overflow-y: auto;
  overflow-x: auto;
  padding: 8px;
  font-family: 'Noto Sans Mono', 'Fira Code', 'Monaco', 'Consolas', monospace;
  font-weight: 300;
  font-size: 0.75rem;
  line-height: 1.4;
  
  /* Custom scrollbar styling */
  &::-webkit-scrollbar {
    width: 8px;
    height: 8px;
    background: #0a0a0a;
  }
  
  &::-webkit-scrollbar-track {
    background: #0a0a0a;
    border-radius: 4px;
    margin: 2px;
  }
  
  &::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #444 0%, #333 100%);
    border-radius: 4px;
    border: 1px solid #0a0a0a;
    cursor: pointer;
  }
  
  &::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, #00ffff 0%, #00cccc 100%);
  }
  
  &::-webkit-scrollbar-thumb:active {
    background: linear-gradient(180deg, #00cccc 0%, #009999 100%);
  }
  
  /* Corner where scrollbars meet */
  &::-webkit-scrollbar-corner {
    background: #0a0a0a;
  }
`;

const LogEntry = styled.div<{ $level: string }>`
  padding: 2px 0;
  color: ${props => {
    switch (props.$level) {
      case 'error': return '#ff4444';
      case 'success': return '#00ff00';
      case 'warning': return '#ffaa00';
      case 'info': return '#00ffff';
      default: return '#ccc';
    }
  }};
  
  &:hover {
    background: rgba(255, 255, 255, 0.05);
  }
`;

const LogTimestamp = styled.span`
  color: #666;
  margin-right: 8px;
`;

const LogMessage = styled.span`
  color: inherit;
`;

const ConnectionStatus = styled.div<{ $connected: boolean }>`
  display: flex;
  align-items: center;
  gap: 4px;
  color: ${props => props.$connected ? '#00ff00' : '#ff4444'};
  font-size: 0.7rem;
`;

const StatusDot = styled.div<{ $connected: boolean }>`
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: ${props => props.$connected ? '#00ff00' : '#ff4444'};
`;

const DocumentsSection = styled.div`
  height: 40%;
  display: flex;
  flex-direction: column;
  border-top: 1px solid #333;
`;

const DocumentsSectionHeader = styled.div`
  padding: 8px 12px;
  background: #1a1a1a;
  border-bottom: 1px solid #333;
  color: #00ffff;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
`;

const DocumentsList = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: 8px;
  
  &::-webkit-scrollbar {
    width: 6px;
    background: #0f0f0f;
  }
  
  &::-webkit-scrollbar-thumb {
    background: #333;
    border-radius: 3px;
  }
`;

const DocumentItem = styled.div`
  background: rgba(255, 255, 255, 0.02);
  border-radius: 4px;
  padding: 8px;
  margin-bottom: 6px;
  display: flex;
  align-items: center;
  gap: 8px;
  border-left: 3px solid #666;
`;

const DocumentIcon = styled.div<{ $status: 'processing' | 'ready' | 'error' }>`
  width: 16px;
  height: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: ${props => {
    switch (props.$status) {
      case 'ready': return '#00ff00';
      case 'error': return '#ff4444';
      default: return '#ffaa00';
    }
  }};
`;

const DocumentInfo = styled.div`
  flex: 1;
  min-width: 0;
`;

const DocumentName = styled.div`
  color: #fff;
  font-size: 0.75rem;
  font-weight: 500;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
`;

const DocumentMeta = styled.div`
  color: #888;
  font-size: 0.7rem;
  margin-top: 2px;
`;

export const Console: React.FC<ConsoleProps> = ({ isVisible, onToggle }) => {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [processingFiles, setProcessingFiles] = useState<ProcessingFile[]>([]);
  const [ws, setWs] = useState<WebSocket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
  const [isMaximized, setIsMaximized] = useState(false);
  const [consoleHeight, setConsoleHeight] = useState(300);
  const [isResizing, setIsResizing] = useState(false);
  const logContainerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs]);

  // WebSocket connection
  useEffect(() => {
    if (!isVisible) return;

    const connectWebSocket = () => {
      const websocket = new WebSocket('ws://localhost:8000/ws/logs');
      
      websocket.onopen = () => {
        console.log('Console WebSocket connected');
        setIsConnected(true);
        setWs(websocket);
      };

      websocket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setLogs(prev => [...prev, data]);
          
          // Debug: log all messages to console for troubleshooting
          console.log('WebSocket message:', data.message);
          
          // Update file status based on log messages
          // Starting batch upload
          if (data.message.includes('Starting upload batch') || data.message.includes('📁 Starting upload batch')) {
            const countMatch = data.message.match(/with (\d+) files/);
            if (countMatch) {
              const fileCount = parseInt(countMatch[1]);
              setProcessingFiles(prev => []); // Clear previous processing files
            }
          }
          
          // Detect file processing from various message patterns
          let detectedFileName: string | null = null;
          let detectedSize: string | null = null;
          
          // Pattern 1: "📄 Processing file 1/1: filename.pdf"
          const processingMatch1 = data.message.match(/📄 Processing file \d+\/\d+: (.+)/);
          if (processingMatch1) {
            detectedFileName = processingMatch1[1];
          }
          
          // Pattern 2: "filename.pdf using enhanced_ocr_service (7.2MB)"
          const processingMatch2 = data.message.match(/^([^\/\s]+\.[a-zA-Z0-9]+)\s+using\s+\w+.*?\(([0-9.]+[KMGT]?B)\)/);
          if (processingMatch2) {
            detectedFileName = processingMatch2[1];
            detectedSize = processingMatch2[2];
          }
          
          // Pattern 3: "Processing filename.pdf"
          const processingMatch3 = data.message.match(/Processing\s+([^\/\s]+\.[a-zA-Z0-9]+)/);
          if (processingMatch3) {
            detectedFileName = processingMatch3[1];
          }
          
          // If we detected a file being processed, add it to our tracking
          if (detectedFileName) {
            // Only add to processingFiles (detailed tracking), skip fileStatuses to avoid duplicates
            setProcessingFiles(prev => {
              const existing = prev.find(f => f.name === detectedFileName);
              if (!existing) {
                return [...prev, {
                  name: detectedFileName!,
                  size: 0, // We'll update this when we get size info
                  status: 'processing' as const,
                  stages: {
                    uploaded: true,  // Start with uploaded since we detected processing
                    extracted: false,
                    chunked: false,
                    embedded: false,
                    stored: false,
                    ready: false
                  },
                  currentStage: 'Uploaded'
                }];
              } else {
                return prev.map(f => f.name === detectedFileName ? { 
                  ...f, 
                  stages: { ...f.stages, uploaded: true },
                  currentStage: 'Uploaded'
                } : f);
              }
            });
          }
          
          // Reading file content
          if (data.message.includes('📥 Reading file content from')) {
            const fileMatch = data.message.match(/from (.+)/);
            if (fileMatch) {
              const fileName = fileMatch[1];
              setProcessingFiles(prev => 
                prev.map(f => f.name === fileName ? { 
                  ...f, 
                  currentStage: 'Reading file...' 
                } : f)
              );
            }
          }
          
          // File read successfully
          if (data.message.includes('✅ Read') && data.message.includes('KB from')) {
            const fileMatch = data.message.match(/from (.+)/);
            if (fileMatch) {
              const fileName = fileMatch[1];
              setProcessingFiles(prev => 
                prev.map(f => f.name === fileName ? { 
                  ...f, 
                  stages: { ...f.stages, uploaded: true },
                  currentStage: 'File loaded'
                } : f)
              );
            }
          }
          
          // Processing type selection
          if (data.message.includes('🔍 Using enhanced OCR processing') || data.message.includes('⚡ Using standard processing')) {
            const fileMatch = data.message.match(/for (.+?) \(/);
            if (fileMatch) {
              const fileName = fileMatch[1];
              const isOCR = data.message.includes('enhanced OCR');
              setProcessingFiles(prev => 
                prev.map(f => f.name === fileName ? { 
                  ...f, 
                  currentStage: isOCR ? 'OCR processing...' : 'Text extraction...'
                } : f)
              );
            }
          }
          
          // Stage 2: Extracting text (match various patterns)
          if (data.message.includes('Extracting text') || 
              data.message.includes('text extraction') ||
              data.message.includes('🔍 Starting') ||
              data.message.includes('enhanced_ocr') ||
              data.message.includes('enhanced OCR') ||
              data.message.includes('standard processing') ||
              data.message.includes('Extracted') && data.message.includes('characters')) {
            setProcessingFiles(prev => 
              prev.map(f => f.status === 'processing' && !f.stages.extracted ? { 
                ...f, 
                stages: { ...f.stages, extracted: true },
                currentStage: 'Extracted Text'
              } : f)
            );
          }
           
          // Stage 3: Creating chunks (match broader patterns)
          if (data.message.includes('Creating semantic chunks') || 
              data.message.includes('Created') && data.message.includes('chunks') ||
              data.message.includes('text chunks') ||
              data.message.includes('📝 Created') ||
              data.message.includes('✂️ Creating') ||
              data.message.includes('chunk')) {
            setProcessingFiles(prev => 
              prev.map(f => f.status === 'processing' && !f.stages.chunked ? { 
                ...f, 
                stages: { ...f.stages, chunked: true },
                currentStage: 'Created Chunks'
              } : f)
            );
          }
          
          // Stage 4: Generating embeddings (match broader patterns)
          if (data.message.includes('Generating AI embeddings') || 
              data.message.includes('Generated embeddings') ||
              data.message.includes('🧠 Generating') ||
              data.message.includes('⚡ Generated embeddings') ||
              data.message.includes('embedding')) {
            setProcessingFiles(prev => 
              prev.map(f => f.status === 'processing' && !f.stages.embedded ? { 
                ...f, 
                stages: { ...f.stages, embedded: true },
                currentStage: 'Generated Embeddings'
              } : f)
            );
          }
          
          // Stage 5: Storing in Vector DB (match broader patterns)
          if (data.message.includes('Saving document metadata') || 
              data.message.includes('stored in vector database') ||
              data.message.includes('💾 Saving') ||
              data.message.includes('vector') && data.message.includes('database') ||
              data.message.includes('chromadb') ||
              data.message.includes('Adding to vector')) {
            setProcessingFiles(prev => 
              prev.map(f => f.status === 'processing' && !f.stages.stored ? { 
                ...f, 
                stages: { ...f.stages, stored: true },
                currentStage: 'Stored in Vector DB'
              } : f)
            );
          }
          
          // Stage 6: Successfully processed / Ready for AI
          if (data.message.includes('✅ Successfully processed') ||
              data.message.includes('🎉 Upload batch completed') ||
              data.message.includes('Upload batch completed') ||
              data.message.includes('Processing completed')) {
            setProcessingFiles(prev => 
              prev.map(f => f.status === 'processing' ? { 
                ...f, 
                status: 'completed',
                stages: { 
                  uploaded: true,
                  extracted: true,
                  chunked: true,
                  embedded: true,
                  stored: true,
                  ready: true 
                },
                currentStage: 'Ready for AI'
              } : f)
            );
            
            // Remove completed files after 3 seconds
            setTimeout(() => {
              setProcessingFiles(prev => prev.filter(f => f.status !== 'completed'));
            }, 3000);
          }
          
          // Error processing
          if (data.message.includes('❌ Failed to process') || 
              data.message.includes('Error') || 
              data.message.includes('failed')) {
            setProcessingFiles(prev => 
              prev.map(f => f.status === 'processing' ? { 
                ...f, 
                status: 'error',
                error: 'Processing failed',
                currentStage: 'Error occurred'
              } : f)
            );
          }
          
          // Fallback: if we see any .pdf, .docx, etc. file extensions in logs but haven't detected them yet
          if (!detectedFileName) {
            const extensionMatch = data.message.match(/([a-zA-Z0-9_-]+\.(pdf|docx|txt|md|html))/i);
            if (extensionMatch) {
              const fallbackFileName = extensionMatch[1];
              // Only add if we don't already have this file
              const existing = processingFiles.find(f => f.name === fallbackFileName);
              if (!existing) {
                setProcessingFiles(prev => [...prev, {
                  name: fallbackFileName,
                  size: 0,
                  status: 'processing' as const,
                  stages: {
                    uploaded: true,
                    extracted: false,
                    chunked: false,
                    embedded: false,
                    stored: false,
                    ready: false
                  },
                  currentStage: 'Processing...'
                }]);
              }
            }
          }
          
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      websocket.onclose = () => {
        console.log('Console WebSocket disconnected');
        setIsConnected(false);
        setWs(null);
        
        // Attempt to reconnect after 3 seconds
        setTimeout(connectWebSocket, 3000);
      };

      websocket.onerror = (error) => {
        console.error('Console WebSocket error:', error);
        setIsConnected(false);
      };
    };

    connectWebSocket();

    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, [isVisible]);

  // Send heartbeat every 30 seconds
  useEffect(() => {
    if (!ws || !isConnected) return;

    const heartbeat = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send('ping');
      }
    }, 30000);

    return () => clearInterval(heartbeat);
  }, [ws, isConnected]);

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  const clearLogs = () => {
    setLogs([]);
  };

  const toggleMinimize = () => {
    setIsMinimized(!isMinimized);
    if (isMaximized) setIsMaximized(false);
  };

  const toggleMaximize = () => {
    setIsMaximized(!isMaximized);
    if (isMinimized) setIsMinimized(false);
  };

  // Handle resize functionality
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing) return;
      
      const newHeight = window.innerHeight - e.clientY;
      const minHeight = 100;
      const maxHeight = window.innerHeight - 100;
      
      setConsoleHeight(Math.max(minHeight, Math.min(maxHeight, newHeight)));
    };

    const handleMouseUp = () => {
      setIsResizing(false);
    };

    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = 'ns-resize';
      document.body.style.userSelect = 'none';
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
  }, [isResizing]);

  const handleResizeStart = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
  };

  if (!isVisible) return null;

  return (
    <ConsoleContainer $isVisible={isVisible} $isMinimized={isMinimized} $isMaximized={isMaximized} $height={consoleHeight}>
      {!isMinimized && !isMaximized && (
        <ConsoleResizeHandle onMouseDown={handleResizeStart} />
      )}
      <ConsoleHeader onClick={!isMaximized ? toggleMinimize : undefined}>
        <ConsoleTitle>
          <Terminal size={14} />
          Console
          <ConnectionStatus $connected={isConnected}>
            <StatusDot $connected={isConnected} />
            {isConnected ? 'Connected' : 'Disconnected'}
          </ConnectionStatus>
        </ConsoleTitle>
        
        <ConsoleControls>
          <ControlButton 
            $variant="clear"
            onClick={(e) => { e.stopPropagation(); clearLogs(); }} 
            title="Clear logs"
          >
            <RotateCcw size={12} />
          </ControlButton>
          <ControlButton 
            $variant="minimize"
            onClick={(e) => { e.stopPropagation(); toggleMinimize(); }} 
            title={isMinimized ? "Restore" : "Minimize"}
          >
            <Minimize2 size={12} />
          </ControlButton>
          <ControlButton 
            $variant="maximize"
            onClick={(e) => { e.stopPropagation(); toggleMaximize(); }} 
            title={isMaximized ? "Restore" : "Maximize"}
          >
            <Maximize2 size={12} />
          </ControlButton>
          <ControlButton 
            $variant="close"
            onClick={(e) => { e.stopPropagation(); onToggle(); }} 
            title="Close console"
          >
            <X size={12} />
          </ControlButton>
        </ConsoleControls>
      </ConsoleHeader>
      
      <ConsoleContent $isMinimized={isMinimized}>
        <LogSection>
          <LogContainer ref={logContainerRef}>
            {logs.length === 0 ? (
              <LogEntry $level="info">
                <LogMessage>Console ready. Waiting for processing events...</LogMessage>
              </LogEntry>
            ) : (
              logs.map((log, index) => (
                <LogEntry key={index} $level={log.level}>
                  <LogTimestamp>{formatTimestamp(log.timestamp)}</LogTimestamp>
                  <LogMessage>{log.message}</LogMessage>
                </LogEntry>
              ))
            )}
          </LogContainer>
        </LogSection>
      </ConsoleContent>
    </ConsoleContainer>
  );
}; 