import React, { useCallback, useState, useRef, useEffect } from 'react';
import styled from 'styled-components';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, CheckCircle, AlertCircle, Loader, RotateCcw, FileImage, FileSpreadsheet, Presentation, Eye, EyeOff, Download, ExternalLink, Copy, Info, XCircle } from 'lucide-react';
import { DocumentDetails } from './DocumentDetails';

interface UploadedDocument {
  id: string;
  name: string;
  size: number;
  type: string;
  uploadedAt: Date | string;
  status: 'processing' | 'ready' | 'error';
  // Enhanced fields from the new document service
  fullFileName?: string;
  fileHash?: string;
  newFileName?: string;
  fileDataHash?: string;
  contentType?: string;
  metadata?: any;
  errorMessage?: string;
}

interface FileProgress {
  filename: string;
  size: number;
  status: 'queued' | 'uploading' | 'processing' | 'completed' | 'error';
  progress: number; // 0-100
  currentStage: string;
  error?: string;
  startTime?: Date;
  endTime?: Date;
}

interface FileUploadPanelProps {
  onFileUpload: (files: File[]) => Promise<any>;
  onRegisterCallback?: (callback: (file: File, result: any) => void) => void;
  uploadedDocuments?: UploadedDocument[];
  onRefreshDocuments?: () => Promise<void>;
  logToConsole?: (message: string, level?: string) => void;
}

const PanelContainer = styled.div`
  background: #1e1e1e;
  border-right: 1px solid #333;
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow-y: auto;
`;

const Header = styled.div`
  padding: 20px;
  border-bottom: 1px solid #333;
`;

const Title = styled.h3`
  color: #00ffff;
  font-size: 1rem;
  font-weight: 600;
  margin: 0;
  display: flex;
  align-items: center;
  gap: 8px;
`;

const DropzoneContainer = styled.div<{ $isDragActive: boolean }>`
  margin: 20px;
  padding: 30px 20px;
  border: 2px dashed ${props => props.$isDragActive ? '#888888' : '#88888880'};
  border-radius: 17px;
  background: ${props => props.$isDragActive ? 
    'linear-gradient(135deg, rgba(136, 136, 136, 0.08) 0%, rgba(136, 136, 136, 0.08) 100%)' : 
    'linear-gradient(135deg, rgba(136, 136, 136, 0.03) 0%, rgba(136, 136, 136, 0.03) 100%)'
  };
  text-align: center;
  cursor: pointer;
  transition: all 0.2s ease;
  box-shadow: none;
  
  &:hover {
    border-color: #888888;
    background: linear-gradient(135deg, rgba(136, 136, 136, 0.06) 0%, rgba(136, 136, 136, 0.06) 100%);
    box-shadow: none;
  }
`;

const DropzoneContent = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
`;

const UploadIcon = styled(Upload)<{ $isDragActive: boolean }>`
  width: 32px;
  height: 32px;
  color: ${props => props.$isDragActive ? '#ffffff' : '#888888'};
  transition: all 0.3s ease;
  filter: none;
`;

const DropzoneText = styled.div`
  color: #888888;
  font-size: 0.9rem;
  line-height: 1.4;
  text-shadow: none;
`;

const DropzoneHint = styled.div`
  color: #666666;
  font-size: 0.75rem;
  margin-top: 4px;
  opacity: 0.8;
`;

const DocumentsStyle = styled.div`
  color: #28b80f;
  font-size: 0.9rem;
  margin-top: 8px;
  opacity: 0.9;
  font-weight: 500;
`;
const DocumentsSection = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
`;

const DocumentsHeader = styled.div`
  padding: 16px 20px 8px;
  border-bottom: 1px solid #333;
  display: flex;
  align-items: center;
  justify-content: space-between;
`;

const DocumentsTitle = styled.h3`
  color: #28b80f;
  font-size: 0.9rem;
  font-weight: 500;
  margin: 0;
  display: flex;
  align-items: center;
  gap: 8px;
`;

const RefreshButton = styled.button`
  background: none;
  border: 1px solid #444;
  color: #888;
  padding: 6px 8px;
  border-radius: 6px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 0.75rem;
  transition: all 0.2s ease;

  &:hover {
    border-color: #00ffff;
    color: #00ffff;
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

const DocumentsList = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: 8px 20px 20px;
  
  /* Custom scrollbar styling */
  &::-webkit-scrollbar {
    width: 8px;
    background: #1a1a1a;
  }
  
  &::-webkit-scrollbar-track {
    background: #1a1a1a;
    border-radius: 4px;
    margin: 4px;
  }
  
  &::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #444 0%, #333 100%);
    border-radius: 4px;
    border: 1px solid #1a1a1a;
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
    background: #1a1a1a;
  }
`;

const DocumentItem = styled.div`
  background: rgba(255, 255, 255, 0.03);
  border-radius: 8px;
  margin-bottom: 8px;
  border: 1px solid #333;
  padding: 12px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  transition: all 0.2s ease;
  
  &:hover {
    background: rgba(255, 255, 255, 0.06);
    border-color: #444;
  }
`;

const DocumentHeader = styled.div`
  display: flex;
  align-items: flex-start;
  gap: 12px;
  cursor: pointer;
`;

const DocumentIcon = styled.div<{ $status: 'processing' | 'ready' | 'error' }>`
  width: 20px;
  height: 20px;
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
  flex-shrink: 0;
  margin-top: 2px;
`;

const DocumentInfo = styled.div`
  flex: 1;
  min-width: 0;
`;

const DocumentActions = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  flex-shrink: 0;
`;

const ActionButton = styled.button`
  background: none;
  border: 1px solid #444;
  color: #888;
  padding: 6px;
  border-radius: 4px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s ease;
  width: 28px;
  height: 28px;

  &:hover {
    border-color: #00ffff;
    color: #00ffff;
    background: rgba(0, 255, 255, 0.1);
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    &:hover {
      border-color: #444;
      color: #888;
      background: none;
    }
  }
`;

const DocumentPreview = styled.div`
  border-top: 1px solid #333;
  padding-top: 12px;
  margin-top: 0;
`;

const PreviewContainer = styled.div`
  width: 100%;
  height: 200px;
  border: 1px solid #444;
  border-radius: 6px;
  background: #0f0f0f;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  position: relative;
`;

const PreviewImage = styled.img`
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
`;

const PreviewPDF = styled.iframe`
  width: 100%;
  height: 100%;
  border: none;
  background: white;
`;

const PreviewError = styled.div`
  color: #ff6666;
  font-size: 0.8rem;
  text-align: center;
  padding: 20px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
`;

const PreviewLoading = styled.div`
  color: #888;
  font-size: 0.8rem;
  text-align: center;
  padding: 20px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
`;

const PreviewPlaceholder = styled.div`
  color: #666;
  font-size: 0.8rem;
  text-align: center;
  padding: 20px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
`;

const DocumentName = styled.div`
  color: #fff;
  font-size: 0.85rem;
  font-weight: 500;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  margin-bottom: 4px;
`;

const DocumentMeta = styled.div`
  color: #888;
  font-size: 0.7rem;
  line-height: 1.3;
`;

const DocumentError = styled.div`
  color: #ff6666;
  font-size: 0.7rem;
  margin-top: 4px;
  font-style: italic;
`;

const FileTypeIcon = styled.div`
  width: 16px;
  height: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #888;
  margin-right: 6px;
`;

const EmptyDocuments = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 120px;
  text-align: center;
  color: #666;
`;

const EmptyDocumentsIcon = styled.div`
  font-size: 1.5rem;
  margin-bottom: 8px;
  opacity: 0.6;
`;

const EmptyDocumentsText = styled.div`
  font-size: 0.8rem;
  font-weight: 500;
`;

const ProgressSection = styled.div`
  margin: 12px 20px;
  border: 1px solid #333;
  border-radius: 8px;
  background: #1a1a1a;
  max-height: 300px;
  overflow-y: auto;
`;

const ProgressHeader = styled.div`
  padding: 12px;
  border-bottom: 1px solid #333;
  background: #222;
  border-radius: 8px 8px 0 0;
`;

const ProgressTitle = styled.h4`
  color: #00ffff;
  font-size: 0.85rem;
  font-weight: 600;
  margin: 0;
  display: flex;
  align-items: center;
  gap: 8px;
`;

const ProgressList = styled.div`
  padding: 8px;
`;

const ProgressItem = styled.div`
  margin-bottom: 12px;
  padding: 12px;
  background: #0f0f0f;
  border-radius: 6px;
  border: 1px solid #333;
`;

const ProgressItemHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
`;

const ProgressFileName = styled.div`
  color: #fff;
  font-size: 0.8rem;
  font-weight: 500;
  display: flex;
  align-items: center;
  gap: 8px;
  flex: 1;
  min-width: 0;
`;

const ProgressFileNameText = styled.span`
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
`;

const ProgressPercentage = styled.div<{ $status: string }>`
  color: ${props => {
    switch (props.$status) {
      case 'completed': return '#4ade80';
      case 'error': return '#ef4444';
      case 'processing': return '#3b82f6';
      case 'uploading': return '#f59e0b';
      default: return '#6b7280';
    }
  }};
  font-size: 0.8rem;
  font-weight: 600;
  min-width: 45px;
  text-align: right;
`;

const ProgressBarContainer = styled.div`
  width: 100%;
  height: 6px;
  background: #333;
  border-radius: 3px;
  overflow: hidden;
  margin-bottom: 6px;
`;

const ProgressBar = styled.div<{ $progress: number; $status: string }>`
  height: 100%;
  background: ${props => {
    switch (props.$status) {
      case 'completed': return 'linear-gradient(90deg, #4ade80, #22c55e)';
      case 'error': return 'linear-gradient(90deg, #ef4444, #dc2626)';
      case 'processing': return 'linear-gradient(90deg, #3b82f6, #2563eb)';
      case 'uploading': return 'linear-gradient(90deg, #f59e0b, #d97706)';
      default: return '#6b7280';
    }
  }};
  width: ${props => Math.max(0, Math.min(100, props.$progress))}%;
  transition: width 0.3s ease;
  border-radius: 3px;
  position: relative;
  
  &::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    animation: ${props => props.$status === 'processing' || props.$status === 'uploading' ? 'shimmer 2s infinite' : 'none'};
  }
  
  @keyframes shimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
  }
`;

const ProgressStage = styled.div`
  color: #888;
  font-size: 0.75rem;
  display: flex;
  align-items: center;
  gap: 6px;
`;

const ProgressError = styled.div`
  color: #ef4444;
  font-size: 0.75rem;
  margin-top: 4px;
  padding: 6px 8px;
  background: rgba(239, 68, 68, 0.1);
  border-radius: 4px;
  border: 1px solid rgba(239, 68, 68, 0.2);
`;

const ProgressStats = styled.div`
  padding: 8px 12px;
  background: #222;
  border-top: 1px solid #333;
  border-radius: 0 0 8px 8px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.75rem;
  color: #888;
`;

const ProgressStatsText = styled.span`
  color: #888;
`;

const ProgressStatsCount = styled.span<{ $status: string }>`
  color: ${props => {
    switch (props.$status) {
      case 'completed': return '#4ade80';
      case 'error': return '#ef4444';
      case 'processing': return '#3b82f6';
      default: return '#888';
    }
  }};
  font-weight: 600;
`;

const DuplicateBadge = styled.span`
  display: inline-flex;
  align-items: center;
  gap: 4px;
  background: #222;
  color: #f59e42;
  border: 1px solid #f59e42;
  border-radius: 8px;
  font-size: 0.7rem;
  font-weight: 600;
  padding: 2px 8px;
  margin-left: 8px;
`;

const DuplicateNotification = styled.div`
  background: #222;
  color: #f59e42;
  border: 1px solid #f59e42;
  border-radius: 8px;
  padding: 10px 18px;
  margin: 12px 0;
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.95rem;
`;

// Snackbar stack container for toast style
const SnackbarStackContainer = styled.div`
  position: fixed;
  left: 50%;
  bottom: 32px;
  transform: translateX(-50%);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  z-index: 2000;
  pointer-events: none;
`;

// Snackbar styled component (toast style)
const Snackbar = styled.div<{ $visible: boolean }>`
  background: linear-gradient(90deg, #3a0d0d 0%, #2a0000 100%);
  color: #ffbdbd;
  border: none;
  border-radius: 18px;
  padding: 10px 22px 10px 18px;
  font-size: 0.665rem;
  font-weight: 400;
  box-shadow: 0 4px 24px #00000040;
  opacity: ${props => props.$visible ? 1 : 0};
  min-width: 220px;
  max-width: 900px;
  width: fit-content;
  height: fit-content;
  display: flex;
  align-items: center;
  justify-content: flex-start;
  text-align: left;
  word-break: break-word;
  white-space: normal;
  margin: 0 auto;
  pointer-events: auto;
`;

// Yellow for 'Duplicate file!'
const Highlight = styled.span`
  color: #ffe066;
  font-weight: 300;
`;

// File name span for white color (no bold)
const FileName = styled.span`
  color: #fff;
  font-weight: 200;
`;

// Cancel button
const SnackbarClose = styled.button`
  background: none;
  border: none;
  color: #fff;
  font-size: 1.3rem;
  margin-left: 18px;
  cursor: pointer;
  padding: 0 6px;
  border-radius: 50%;
  transition: background 0.2s;
  align-self: center;
  display: flex;
  align-items: center;
  height: 100%;
  &:hover {
    background: #ff3b3b33;
  }
`;

// Utility: Compute SHA-256 hash of a File
async function computeSHA256(file: File): Promise<string> {
  const arrayBuffer = await file.arrayBuffer();
  const hashBuffer = await window.crypto.subtle.digest('SHA-256', arrayBuffer);
  return Array.from(new Uint8Array(hashBuffer)).map(b => b.toString(16).padStart(2, '0')).join('');
}

export const FileUploadPanel: React.FC<FileUploadPanelProps> = ({ 
  onFileUpload,
  uploadedDocuments = [],
  onRefreshDocuments,
  logToConsole
}) => {
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [selectedDocument, setSelectedDocument] = useState<UploadedDocument | null>(null);
  const [fileProgressList, setFileProgressList] = useState<FileProgress[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [expandedPreviews, setExpandedPreviews] = useState<Set<string>>(new Set());
  const [previewStates, setPreviewStates] = useState<Map<string, { loading: boolean; error: string | null; url: string | null }>>(new Map());
  const wsRef = useRef<WebSocket | null>(null);
  const [duplicateFiles, setDuplicateFiles] = useState<string[]>([]);
  const [snackbarStack, setSnackbarStack] = useState<{ id: number, message: React.ReactNode }[]>([]);

  // WebSocket connection for real-time progress
  useEffect(() => {
    const connectWebSocket = () => {
      const wsUrl = `ws://localhost:8000/ws/logs`;
      const ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        console.log('Progress WebSocket connected');
        wsRef.current = ws;
      };
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'log' && data.message) {
            updateProgressFromLogMessage(data);
          }
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };
      
      ws.onclose = () => {
        console.log('Progress WebSocket disconnected');
        wsRef.current = null;
        // Reconnect after 2 seconds
        setTimeout(connectWebSocket, 2000);
      };
      
      ws.onerror = (error) => {
        console.error('Progress WebSocket error:', error);
      };
    };

    connectWebSocket();
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  // Cleanup object URLs when component unmounts
  useEffect(() => {
    return () => {
      previewStates.forEach((state) => {
        if (state.url) {
          URL.revokeObjectURL(state.url);
        }
      });
    };
  }, [previewStates]);

  const updateProgressFromLogMessage = (logData: any) => {
    const message = logData.message;
    const details = logData.details || {};
    
    // Parse different types of log messages to update progress
    if (message.includes('📁 Starting upload batch')) {
      // Reset and initialize progress for new batch
      const fileCount = details.file_count || 0;
      setIsUploading(true);
      setFileProgressList([]);
    } else if (message.includes('📄 Analyzing file')) {
      // File analysis stage
      const filename = details.filename;
      if (filename) {
        setFileProgressList(prev => {
          const existing = prev.find(f => f.filename === filename);
          if (existing) {
            return prev.map(f => f.filename === filename ? {
              ...f,
              status: 'uploading',
              progress: 10,
              currentStage: 'Analyzing file...'
            } : f);
          } else {
            return [...prev, {
              filename,
              size: 0,
              status: 'uploading',
              progress: 10,
              currentStage: 'Analyzing file...',
              startTime: new Date()
            }];
          }
        });
      }
    } else if (message.includes('✅ Prepared')) {
      // File preparation complete
      const filename = message.match(/Prepared (.+?) \(/)?.[1];
      if (filename) {
        setFileProgressList(prev => 
          prev.map(f => f.filename === filename ? {
            ...f,
            status: 'processing',
            progress: 25,
            currentStage: 'File prepared, starting processing...'
          } : f)
        );
      }
    } else if (message.includes('🔍 Processing') && message.includes('service')) {
      // Processing started
      const filename = details.filename;
      if (filename) {
        setFileProgressList(prev => 
          prev.map(f => f.filename === filename ? {
            ...f,
            status: 'processing',
            progress: 40,
            currentStage: 'Processing document content...'
          } : f)
        );
      }
    } else if (message.includes('Extracting text from')) {
      // Text extraction
      const filename = message.match(/from (.+?) using/)?.[1];
      if (filename) {
        setFileProgressList(prev => 
          prev.map(f => f.filename === filename ? {
            ...f,
            progress: 60,
            currentStage: 'Extracting text content...'
          } : f)
        );
      }
    } else if (message.includes('Creating chunks')) {
      // Chunking stage
      const filename = message.match(/for (.+?)$/)?.[1] || message.match(/Creating chunks for (.+?)(?:\s|$)/)?.[1];
      if (filename) {
        setFileProgressList(prev => 
          prev.map(f => f.filename === filename ? {
            ...f,
            progress: 75,
            currentStage: 'Creating text chunks...'
          } : f)
        );
      }
    } else if (message.includes('Generated embeddings')) {
      // Embeddings generation
      const filename = message.match(/for (.+?)$/)?.[1];
      if (filename) {
        setFileProgressList(prev => 
          prev.map(f => f.filename === filename ? {
            ...f,
            progress: 90,
            currentStage: 'Generating embeddings...'
          } : f)
        );
      }
    } else if (message.includes('Successfully processed')) {
      // Processing complete
      const filename = details.filename || message.match(/processed (.+?)(?:\s|$)/)?.[1];
      if (filename) {
        setFileProgressList(prev => 
          prev.map(f => f.filename === filename ? {
            ...f,
            status: 'completed',
            progress: 100,
            currentStage: 'Processing completed',
            endTime: new Date()
          } : f)
        );
      }
    } else if (message.includes('❌ Failed to process')) {
      // Processing error
      const filename = details.filename || message.match(/process (.+?):/)?.[1];
      const error = details.error || message.split(': ').slice(1).join(': ');
      if (filename) {
        setFileProgressList(prev => 
          prev.map(f => f.filename === filename ? {
            ...f,
            status: 'error',
            currentStage: 'Processing failed',
            error: error,
            endTime: new Date()
          } : f)
        );
      }
    } else if (message.includes('🎉 Upload batch completed')) {
      // Batch complete
      setTimeout(() => {
        setIsUploading(false);
        setFileProgressList([]);
      }, 3000); // Clear progress after 3 seconds
    }
  };

  const showSnackbar = (message: React.ReactNode) => {
    setSnackbarStack(stack => [...stack, { id: Date.now() + Math.random(), message }]);
  };

  const handleDrop = useCallback(async (acceptedFiles: File[]) => {
    const nonDuplicateFiles: File[] = [];
    for (const file of acceptedFiles) {
      // Step 1: Check name+size
      const nameSizeDuplicate = uploadedDocuments.find(
        doc => doc.name === file.name && doc.size === file.size
      );
      if (nameSizeDuplicate) {
        const msg = <span><Highlight>Duplicate file!</Highlight> <FileName>"{file.name}"</FileName></span>;
        showSnackbar(msg);
        logToConsole && logToConsole(`Duplicate file: "${file.name}"`, 'error');
        continue;
      }
      // Step 2: Check hash
      const fileHash = await computeSHA256(file);
      const hashDuplicate = uploadedDocuments.find(
        doc => doc.fileHash === fileHash
      );
      if (hashDuplicate) {
        const msg = <span><Highlight>Duplicate file!</Highlight> <FileName>"{file.name}"</FileName> == <FileName>"{hashDuplicate.name}"</FileName></span>;
        showSnackbar(msg);
        logToConsole && logToConsole(`Duplicate file! "${file.name}" == "${hashDuplicate.name}"`, 'error');
        continue;
      }
      nonDuplicateFiles.push(file);
    }
    if (nonDuplicateFiles.length > 0) {
      const results = await onFileUpload(nonDuplicateFiles);
      if (Array.isArray(results)) {
        results.forEach(result => {
          if (result.status === 'duplicate') {
            // Backend duplicate: try to find which file matches by hash
            const backendHashDuplicate = uploadedDocuments.find(
              doc => doc.fileHash === result.fileHash
            );
            if (backendHashDuplicate) {
              const msg = <span><Highlight>Duplicate file!</Highlight> <FileName>"{result.name}"</FileName> == <FileName>"{backendHashDuplicate.name}"</FileName></span>;
              showSnackbar(msg);
              logToConsole && logToConsole(`Duplicate file! "${result.name}" == "${backendHashDuplicate.name}"`, 'error');
            } else {
              const msg = <span><Highlight>Duplicate file!</Highlight> <FileName>"{result.name}"</FileName></span>;
              showSnackbar(msg);
              logToConsole && logToConsole(`Duplicate file: "${result.name}"`, 'error');
            }
          }
        });
      }
    }
  }, [uploadedDocuments, onFileUpload, logToConsole]);

  const handleRefresh = async () => {
    if (!onRefreshDocuments || isRefreshing) return;
    
    setIsRefreshing(true);
    try {
      await onRefreshDocuments();
    } catch (error) {
      console.error('Refresh failed:', error);
    } finally {
      setIsRefreshing(false);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: handleDrop,
    accept: {
      // Enhanced document service supported file types
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'application/vnd.openxmlformats-officedocument.presentationml.presentation': ['.pptx'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'image/png': ['.png'],
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/heic': ['.heic'],
      'text/markdown': ['.md'],
      'text/plain': ['.txt', '.c', '.cpp', '.java', '.cs', '.js', '.ts', '.go', '.rs', '.php', '.pl', '.rb', '.py', '.swift', '.kt', '.scala', '.sh', '.bat', '.ps1', '.html', '.css', '.json', '.xml', '.yaml', '.yml', '.toml'],
      'application/json': ['.json'],
      'application/xml': ['.xml'],
      'text/x-python': ['.py'],
      'text/x-c': ['.c'],
      'text/x-c++': ['.cpp'],
      'text/x-java': ['.java'],
      'text/x-csharp': ['.cs'],
      'text/x-go': ['.go'],
      'text/x-rustsrc': ['.rs'],
      'text/x-php': ['.php'],
      'text/x-perl': ['.pl'],
      'text/x-ruby': ['.rb'],
      'text/x-shellscript': ['.sh'],
      'text/x-swift': ['.swift'],
      'text/x-kotlin': ['.kt'],
      'text/x-scala': ['.scala'],
      'text/css': ['.css'],
      'text/html': ['.html'],
      'application/x-sh': ['.sh'],
      'application/x-bat': ['.bat'],
      'application/x-powershell': ['.ps1'],
      'application/x-toml': ['.toml'],
      'application/x-yaml': ['.yaml', '.yml']
    },
    multiple: true,
    noClick: false,
  });

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
  };

  const formatUploadDate = (uploadedAt: Date | string | undefined): string => {
    if (!uploadedAt) return 'Unknown date';
    
    try {
      const date = uploadedAt instanceof Date ? uploadedAt : new Date(uploadedAt);
      return date.toLocaleDateString();
    } catch (error) {
      return 'Unknown date';
    }
  };

  const getFileTypeIcon = (fileName: string | undefined) => {
    if (!fileName) return <FileText size={16} />;
    const ext = fileName.toLowerCase().split('.').pop();
    switch (ext) {
      case 'pdf':
      case 'docx':
        return <FileText size={16} />;
      case 'pptx':
        return <Presentation size={16} />;
      case 'xlsx':
        return <FileSpreadsheet size={16} />;
      case 'png':
      case 'jpg':
      case 'jpeg':
      case 'heic':
        return <FileImage size={16} />;
      case 'md':
        return <FileText size={16} />;
      case 'c':
      case 'cpp':
      case 'java':
      case 'cs':
      case 'js':
      case 'ts':
      case 'go':
      case 'rs':
      case 'php':
      case 'pl':
      case 'rb':
      case 'py':
      case 'swift':
      case 'kt':
      case 'scala':
      case 'sh':
      case 'bat':
      case 'ps1':
      case 'html':
      case 'css':
      case 'json':
      case 'xml':
      case 'yaml':
      case 'yml':
      case 'toml':
        return <FileText size={16} />;
      default:
        return <FileText size={16} />;
    }
  };

  const getFileTypeIconColor = (fileName: string | undefined) => {
    if (!fileName) return '#888';
    const ext = fileName.toLowerCase().split('.').pop();
    switch (ext) {
      case 'pdf': return '#ff4444'; // Red
      case 'docx': return '#a259e6'; // Purple
      case 'pptx': return '#ff9900'; // Orange
      case 'xlsx': return '#4ade80'; // Green
      case 'png':
      case 'jpg':
      case 'jpeg':
      case 'heic':
      case 'bmp':
      case 'gif':
      case 'tiff': return '#38bdf8'; // Blue
      case 'md':
      case 'txt':
      case 'c':
      case 'cpp':
      case 'java':
      case 'cs':
      case 'js':
      case 'ts':
      case 'go':
      case 'rs':
      case 'php':
      case 'pl':
      case 'rb':
      case 'py':
      case 'swift':
      case 'kt':
      case 'scala':
      case 'sh':
      case 'bat':
      case 'ps1':
      case 'html':
      case 'css':
      case 'json':
      case 'xml':
      case 'yaml':
      case 'yml':
      case 'toml': return '#888'; // Gray
      default: return '#888';
    }
  };

  const getFileTypeName = (fileName: string | undefined) => {
    if (!fileName) return 'Document';
    const ext = fileName.toLowerCase().split('.').pop();
    switch (ext) {
      case 'pdf': return 'PDF Document';
      case 'docx': return 'Word Document';
      case 'pptx': return 'PowerPoint Presentation';
      case 'xlsx': return 'Excel Spreadsheet';
      case 'png': return 'PNG Image';
      case 'jpg':
      case 'jpeg': return 'JPEG Image';
      case 'heic': return 'HEIC Image';
      case 'md': return 'Markdown File';
      case 'c': return 'C Source File';
      case 'cpp': return 'C++ Source File';
      case 'java': return 'Java Source File';
      case 'cs': return 'C# Source File';
      case 'js': return 'JavaScript File';
      case 'ts': return 'TypeScript File';
      case 'go': return 'Go Source File';
      case 'rs': return 'Rust Source File';
      case 'php': return 'PHP File';
      case 'pl': return 'Perl File';
      case 'rb': return 'Ruby File';
      case 'py': return 'Python File';
      case 'swift': return 'Swift File';
      case 'kt': return 'Kotlin File';
      case 'scala': return 'Scala File';
      case 'sh': return 'Shell Script';
      case 'bat': return 'Batch Script';
      case 'ps1': return 'PowerShell Script';
      case 'html': return 'HTML File';
      case 'css': return 'CSS File';
      case 'json': return 'JSON File';
      case 'xml': return 'XML File';
      case 'yaml':
      case 'yml': return 'YAML File';
      case 'toml': return 'TOML File';
      default: return 'Document';
    }
  };

  const handleDocumentClick = (document: UploadedDocument) => {
    setSelectedDocument(document);
  };

  const handleCloseDetails = () => {
    setSelectedDocument(null);
  };

  const getCompletedCount = () => fileProgressList.filter(f => f.status === 'completed').length;
  const getErrorCount = () => fileProgressList.filter(f => f.status === 'error').length;
  const getProcessingCount = () => fileProgressList.filter(f => f.status === 'processing' || f.status === 'uploading').length;

  const togglePreview = (docId: string) => {
    setExpandedPreviews(prev => {
      const newSet = new Set(prev);
      if (newSet.has(docId)) {
        newSet.delete(docId);
        // Clear preview state when closing
        setPreviewStates(prevStates => {
          const newStates = new Map(prevStates);
          newStates.delete(docId);
          return newStates;
        });
      } else {
        newSet.add(docId);
        loadPreview(docId);
      }
      return newSet;
    });
  };

  const loadPreview = async (docId: string) => {
    // Set loading state
    setPreviewStates(prev => new Map(prev).set(docId, { loading: true, error: null, url: null }));

    try {
      const response = await fetch(`http://localhost:8000/api/documents/original/${docId}`);
      if (response.ok) {
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        setPreviewStates(prev => new Map(prev).set(docId, { loading: false, error: null, url }));
      } else {
        throw new Error(`Failed to load preview: ${response.status}`);
      }
    } catch (error) {
      setPreviewStates(prev => new Map(prev).set(docId, { loading: false, error: (error as Error).message || 'Failed to load preview', url: null }));
    }
  };

  const downloadDocument = async (docId: string, filename: string) => {
    try {
      const response = await fetch(`http://localhost:8000/api/documents/original/${docId}`);
      if (response.ok) {
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      } else {
        throw new Error(`Failed to download: ${response.status}`);
      }
    } catch (error) {
      console.error('Download failed:', error);
    }
  };

  const openInNewTab = async (docId: string) => {
    const previewUrl = `http://localhost:8000/api/documents/original/${docId}`;
    window.open(previewUrl, '_blank');
  };

  const isImageFile = (filename: string): boolean => {
    const ext = filename.toLowerCase().split('.').pop();
    return ['png', 'jpg', 'jpeg', 'heic', 'bmp', 'gif', 'tiff'].includes(ext || '');
  };

  const isPDFFile = (filename: string): boolean => {
    const ext = filename.toLowerCase().split('.').pop();
    return ext === 'pdf';
  };

  const renderPreview = (doc: UploadedDocument) => {
    const previewState = previewStates.get(doc.id);
    
    if (previewState?.loading) {
      return (
        <PreviewLoading>
          <Loader size={24} />
          Loading preview...
        </PreviewLoading>
      );
    }

    if (previewState?.error) {
      return (
        <PreviewError>
          <AlertCircle size={24} />
          {previewState.error}
        </PreviewError>
      );
    }

    if (previewState?.url) {
      if (isImageFile(doc.name)) {
        return <PreviewImage src={previewState.url} alt={doc.name} />;
      } else if (isPDFFile(doc.name)) {
        return <PreviewPDF src={previewState.url} title={doc.name} />;
      } else {
        return (
          <PreviewPlaceholder>
            <FileText size={24} />
            Preview not available for this file type
            <small>Click "Open" to view in new tab</small>
          </PreviewPlaceholder>
        );
      }
    }

    return (
      <PreviewPlaceholder>
        <FileText size={24} />
        Click the eye icon to load preview
      </PreviewPlaceholder>
    );
  };

  const canPreview = (doc: UploadedDocument) => {
    const name = doc.metadata?.converted_png || doc.name;
    const ext = name.toLowerCase().split('.').pop();
    return ['pdf', 'png', 'jpg', 'jpeg', 'bmp', 'gif', 'tiff'].includes(ext);
  };

  const closeSnackbar = (id: number) => {
    setSnackbarStack(stack => stack.filter(snack => snack.id !== id));
  };

  return (
    <PanelContainer>
      <DropzoneContainer {...getRootProps()} $isDragActive={isDragActive}>
        <input {...getInputProps()} />
        <DropzoneContent>
          <UploadIcon $isDragActive={isDragActive} />
          <DropzoneText>
            {isDragActive ? 
              "Drop files here..." : 
              "Drop files here or click to browse"
            }
          </DropzoneText>
          <DropzoneHint>
            Supports PDF, DOCX, PPTX, XLSX, PNG, JPG, HEIC, MD, and major code files (C, C++, Java, JS, Python, Go, Rust, PHP, etc) • Max 500MB
          </DropzoneHint>
        </DropzoneContent>
      </DropzoneContainer>

      {/* Real-time Progress Section */}
      {isUploading && fileProgressList.length > 0 && (
        <ProgressSection>
          <ProgressHeader>
            <ProgressTitle>
              <Upload size={16} />
              Processing Files ({fileProgressList.length})
            </ProgressTitle>
          </ProgressHeader>
          <ProgressList>
            {fileProgressList.map((file, index) => (
              <ProgressItem key={`${file.filename}-${index}`}>
                <ProgressItemHeader>
                  <ProgressFileName>
                    {getFileTypeIcon(file.filename)}
                    <ProgressFileNameText title={file.filename}>
                      {file.filename}
                      {duplicateFiles.includes(file.filename) && (
                        <DuplicateBadge>
                          <Info size={12} /> Duplicate
                        </DuplicateBadge>
                      )}
                    </ProgressFileNameText>
                  </ProgressFileName>
                  <ProgressPercentage $status={file.status}>
                    {file.progress}%
                  </ProgressPercentage>
                </ProgressItemHeader>
                <ProgressBarContainer>
                  <ProgressBar $progress={file.progress} $status={file.status} />
                </ProgressBarContainer>
                <ProgressStage>
                  {file.status === 'processing' && <Loader size={12} />}
                  {file.status === 'uploading' && <Upload size={12} />}
                  {file.status === 'completed' && <CheckCircle size={12} />}
                  {file.status === 'error' && <AlertCircle size={12} />}
                  {file.currentStage}
                  {file.size > 0 && ` • ${formatFileSize(file.size)}`}
                </ProgressStage>
                {file.error && (
                  <ProgressError>
                    {file.error}
                  </ProgressError>
                )}
              </ProgressItem>
            ))}
          </ProgressList>
          <ProgressStats>
            <ProgressStatsText>
              <ProgressStatsCount $status="completed">{getCompletedCount()}</ProgressStatsCount> completed • 
              <ProgressStatsCount $status="processing"> {getProcessingCount()}</ProgressStatsCount> processing • 
              <ProgressStatsCount $status="error">{getErrorCount()}</ProgressStatsCount> failed
            </ProgressStatsText>
                     </ProgressStats>
         </ProgressSection>
       )}

      {/* Documents Section */}
      <DocumentsSection>
        <DocumentsHeader>
          <DocumentsTitle>
            <FileText size={16} />
            <DocumentsStyle>
            Documents ({uploadedDocuments.length})
            </DocumentsStyle>
          </DocumentsTitle>
          <RefreshButton onClick={handleRefresh} disabled={!onRefreshDocuments || isRefreshing}>
            {isRefreshing ? <Loader size={12} /> : <RotateCcw size={12} />}
            Refresh
          </RefreshButton>
        </DocumentsHeader>
        
        <DocumentsList>
          {uploadedDocuments.length > 0 ? (
            uploadedDocuments.map((doc) => {
              const iconColor = getFileTypeIconColor(doc.metadata?.converted_png || doc.name);
              return (
                <DocumentItem key={doc.id}>
                  <DocumentHeader onClick={() => handleDocumentClick(doc)}>
                    <span style={{ display: 'flex', alignItems: 'center', marginRight: 3 }}>
                      <FileTypeIcon>
                        {React.cloneElement(getFileTypeIcon(doc.metadata?.converted_png || doc.name), { color: iconColor })}
                      </FileTypeIcon>
                    </span>
                    <DocumentInfo>
                      <DocumentName>
                        {doc.name || 'Unknown Document'}
                      </DocumentName>
                      <DocumentMeta>
                        {formatFileSize(doc.size || 0)} • {getFileTypeName(doc.name)} • {formatUploadDate(doc.uploadedAt)}
                      </DocumentMeta>
                      {doc.status === 'error' && doc.errorMessage && (
                        <DocumentError>
                          Error: {doc.errorMessage}
                        </DocumentError>
                      )}
                    </DocumentInfo>
                  </DocumentHeader>
                </DocumentItem>
              );
            })
          ) : (
            <EmptyDocuments>
              <EmptyDocumentsIcon>
                <FileText size={24} color="#666" />
              </EmptyDocumentsIcon>
              <EmptyDocumentsText>No documents uploaded</EmptyDocumentsText>
            </EmptyDocuments>
          )}
        </DocumentsList>
      </DocumentsSection>

      {/* Document Details Modal */}
      {selectedDocument && (
        <DocumentDetails 
          document={selectedDocument} 
          onClose={handleCloseDetails} 
        />
      )}

      {/* Add CSS animation for spinning */}
      <style>
        {`
          @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
          }
        `}
      </style>

      {duplicateFiles.length > 0 && (
        <DuplicateNotification>
          <Info size={18} color="#f59e42" />
          {duplicateFiles.length === 1
            ? `File already exists and was not uploaded again: ${duplicateFiles[0]}`
            : `Some files already exist and were not uploaded again: ${duplicateFiles.join(', ')}`}
        </DuplicateNotification>
      )}

      {/* Render stacked Snackbars vertically */}
      <SnackbarStackContainer>
        {snackbarStack.map(snack => (
          <Snackbar key={snack.id} $visible={true}>
            {snack.message}
            <SnackbarClose onClick={() => closeSnackbar(snack.id)} aria-label="Close">×</SnackbarClose>
          </Snackbar>
        ))}
      </SnackbarStackContainer>
    </PanelContainer>
  );
}; 