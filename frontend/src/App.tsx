import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { Terminal } from 'lucide-react';
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels';
import { FileUploadPanel } from './components/FileUploadPanel';
import { ChatPanel } from './components/ChatPanel';
import { ConversationPanel } from './components/ConversationPanel';
import { StatusPanel } from './components/StatusPanel';
import { Console } from './components/Console';
import { GlobalStyles } from './styles/GlobalStyles';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  neonColor?: string;
  sources?: DocumentSource[];
}

export interface DocumentSource {
  document_id: string;
  filename: string;
  chunk_text: string;
  similarity_score: number;
}

export interface Conversation {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: Date;
  updatedAt: Date;
}

export interface UploadedDocument {
  id: string;
  name: string;
  size: number;
  type: string;
  uploadedAt: Date;
  status: 'processing' | 'ready' | 'error';
  fullFileName?: string;
  fileHash?: string;
  newFileName?: string;
  fileDataHash?: string;
  errorMessage?: string;
}

export interface SystemStatus {
  backend: 'online' | 'offline';
  aiModels: 'online' | 'offline';
  documents: number;
  uploadedDocuments: number;
  cpuCores: number;
  cpuUsage?: number;
  isProcessing?: boolean;
}

interface FileUploadResult {
  file: File;
  success: boolean;
  response?: any;
  error?: string;
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  sources?: DocumentSource[];
}

interface Document {
  id: number;
  filename: string;
  file_type: string;
  upload_date: string;
  file_size: number;
  file_hash?: string;
}

interface ChatPanelProps {
  messages: Message[];
  onSendMessage: (message: string) => void;
  isStreaming: boolean;
}

interface ConversationPanelProps {
  conversations: Array<{ id: string; title: string; lastMessage: Date }>;
  activeConversationId: string | null;
  onSelectConversation: (id: string) => void;
  onNewConversation: () => void;
  onDeleteConversation: (id: string) => void;
}

interface StatusPanelProps {
  systemStatus: string;
  uploadedDocuments: Document[];
  onRefreshDocuments: () => void;
  onDeleteDocument: (id: number) => void;
}

const AppContainer = styled.div`
  display: flex;
  flex-direction: column;
  height: 100vh;
  background: #1a1a1a;
  color: #ffffff;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
`;

const Header = styled.header`
  background: #0a0a0a;
  padding: 15px 20px;
  border-bottom: 1px solid #333;
  z-index: 100;
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const Title = styled.h1`
  margin: 0;
  font-size: 1.8rem;
  font-weight: 600;
  color: #ffffff;
  font-family: 'Moirai One', serif;
  letter-spacing: 1px;
`;

const ClearChatButton = styled.button`
  background: linear-gradient(135deg, rgba(255, 20, 147, 0.2) 0%, rgba(255, 0, 128, 0.2) 100%);
  border: 1px solid #ff1493;
  color: #ff69b4;
  padding: 8px 12px;
  border-radius: 25px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.8rem;
  transition: all 0.2s ease;
  box-shadow: none;
  
  &:hover:not(:disabled) {
    color: #ffffff;
    border-color: #ff69b4;
    background: linear-gradient(135deg, rgba(255, 20, 147, 0.4) 0%, rgba(255, 0, 128, 0.4) 100%);
    box-shadow: none;
  }
  
  &:disabled {
    opacity: 0.3;
    cursor: not-allowed;
    box-shadow: none;
  }
`;

const ConsoleToggle = styled.button`
  background: linear-gradient(135deg, rgba(0, 255, 255, 0.1) 0%, rgba(0, 200, 255, 0.1) 100%);
  border: 1px solid #00ffff;
  color: #00ffff;
  padding: 8px 12px;
  border-radius: 25px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.8rem;
  transition: all 0.2s ease;
  box-shadow: none;
  
  &:hover {
    color: #ffffff;
    border-color: #00ffff;
    background: linear-gradient(135deg, rgba(0, 255, 255, 0.2) 0%, rgba(0, 200, 255, 0.2) 100%);
    box-shadow: none;
  }
  
  &.active {
    color: #ffffff;
    border-color: #00ffff;
    background: linear-gradient(135deg, rgba(0, 255, 255, 0.3) 0%, rgba(0, 200, 255, 0.3) 100%);
    box-shadow: none;
  }
`;

const MainContent = styled.div`
  height: calc(100vh - 70px);
  overflow: hidden;
`;

const ResizeHandle = styled(PanelResizeHandle)`
  background: #333;
  width: 2px;
  cursor: col-resize;
  transition: background-color 0.2s ease;
  
  &:hover {
    background: #00ffff;
  }
  
  &[data-resize-handle-active] {
    background: #00ffff;
  }
`;

const PanelContainer = styled(Panel)`
  overflow: hidden;
  display: flex;
  flex-direction: column;
`;

const BACKEND_URL = 'http://localhost:8000';

const NEON_COLORS = [
  '#00ffff', // Cyan
  '#ff1493', // Deep Pink
  '#00ff00', // Lime
  '#ff6600', // Orange
  '#9932cc', // Purple
  '#ffff00', // Yellow
  '#ff69b4', // Hot Pink
  '#00ff7f', // Spring Green
];

function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null);
  const [uploadedDocuments, setUploadedDocuments] = useState<UploadedDocument[]>([]);
  const [systemStatus, setSystemStatus] = useState<SystemStatus>({
    backend: 'offline',
    aiModels: 'offline',
    documents: 0,
    uploadedDocuments: 0,
    cpuCores: navigator.hardwareConcurrency || 8,
    cpuUsage: 0,
    isProcessing: false
  });
  const [fileUploadCallback, setFileUploadCallback] = useState<((file: File, result: any) => void) | null>(null);
  const [isConsoleVisible, setIsConsoleVisible] = useState(false);
  const [isStreamingResponse, setIsStreamingResponse] = useState(false);

  // Add cache busting utility
  const addCacheBuster = (url: string): string => {
    const separator = url.includes('?') ? '&' : '?';
    return `${url}${separator}_t=${Date.now()}&_r=${Math.random()}`;
  };

  // Add no-cache headers to fetch requests
  const fetchWithNoCache = async (url: string, options: RequestInit = {}): Promise<Response> => {
    const bustedUrl = addCacheBuster(url);
    const headers = {
      'Cache-Control': 'no-cache, no-store, must-revalidate',
      'Pragma': 'no-cache',
      'Expires': '0',
      ...options.headers,
    };
    
    return fetch(bustedUrl, {
      ...options,
      headers,
    });
  };

  // Persistent storage functions
  const saveToStorage = () => {
    localStorage.setItem('ai-mate-conversations', JSON.stringify(conversations));
    localStorage.setItem('ai-mate-documents', JSON.stringify(uploadedDocuments));
    localStorage.setItem('ai-mate-current-conversation', currentConversationId || '');
  };

  const loadFromStorage = () => {
    try {
      // Normal loading from localStorage without system reset check
      const savedConversations = localStorage.getItem('ai-mate-conversations');
      const savedDocuments = localStorage.getItem('ai-mate-documents');
      const savedCurrentConversation = localStorage.getItem('ai-mate-current-conversation');

      if (savedConversations) {
        const parsedConversations = JSON.parse(savedConversations).map((conv: any) => ({
          ...conv,
          createdAt: new Date(conv.createdAt),
          updatedAt: new Date(conv.updatedAt),
          messages: conv.messages.map((msg: any) => ({
            ...msg,
            timestamp: new Date(msg.timestamp)
          }))
        }));
        setConversations(parsedConversations);
      }

      if (savedDocuments) {
        const parsedDocuments = JSON.parse(savedDocuments).map((doc: any) => ({
          ...doc,
          uploadedAt: new Date(doc.uploadedAt)
        }));
        setUploadedDocuments(parsedDocuments);
      }

      if (savedCurrentConversation) {
        setCurrentConversationId(savedCurrentConversation);
        // Find the conversation in the already loaded conversations
        const currentConv = JSON.parse(savedConversations || '[]').find((c: any) => c.id === savedCurrentConversation);
        if (currentConv) {
          const parsedMessages = currentConv.messages.map((msg: any) => ({
            ...msg,
            timestamp: new Date(msg.timestamp)
          }));
          setMessages(parsedMessages);
        }
      }
    } catch (error) {
      console.error('Error loading from storage:', error);
      // Clear corrupted storage data
      localStorage.removeItem('ai-mate-conversations');
      localStorage.removeItem('ai-mate-documents');
      localStorage.removeItem('ai-mate-current-conversation');
    }
  };

  const getRandomNeonColor = () => {
    return NEON_COLORS[Math.floor(Math.random() * NEON_COLORS.length)];
  };

  const createNewConversation = () => {
    const newConversation: Conversation = {
      id: Date.now().toString(),
      title: 'New Conversation',
      messages: [],
      createdAt: new Date(),
      updatedAt: new Date()
    };
    setConversations(prev => [newConversation, ...prev]);
    setCurrentConversationId(newConversation.id);
    setMessages([]);
    return newConversation;
  };

  const updateCurrentConversation = (newMessages: ChatMessage[]) => {
    let conversationId = currentConversationId;
    
    // Create conversation if it doesn't exist, but don't reset messages
    if (!conversationId) {
      const newConversation: Conversation = {
        id: Date.now().toString(),
        title: 'New Conversation',
        messages: newMessages, // Use the provided messages instead of empty array
        createdAt: new Date(),
        updatedAt: new Date()
      };
      setConversations(prev => [newConversation, ...prev]);
      setCurrentConversationId(newConversation.id);
      conversationId = newConversation.id;
    } else {
      // Update existing conversation
      setConversations(prev => prev.map(conv => {
        if (conv.id === conversationId) {
          const title = newMessages.length > 0 ? 
            newMessages[0].content.slice(0, 50) + (newMessages[0].content.length > 50 ? '...' : '') :
            'New Conversation';
          return {
            ...conv,
            title,
            messages: newMessages,
            updatedAt: new Date()
          };
        }
        return conv;
      }));
    }
  };

  const loadConversation = (conversationId: string) => {
    const conversation = conversations.find(c => c.id === conversationId);
    if (conversation) {
      setCurrentConversationId(conversationId);
      setMessages(conversation.messages);
    }
  };

  const checkSystemStatus = async (skipDuringStreaming = true) => {
    // Skip status checks during streaming to avoid conflicts
    if (skipDuringStreaming && isStreamingResponse) {
      return;
    }

    try {
      const response = await fetchWithNoCache(`${BACKEND_URL}/health`);
      const isBackendOnline = response.ok;

      let aiModelsOnline = false;
      let documentCount = 0;

      if (isBackendOnline) {
        try {
          // Check if AI models are accessible
          const chatResponse = await fetchWithNoCache(`${BACKEND_URL}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              message: "test",
              conversation_id: "health_check",
              use_context: false
            })
          });
          aiModelsOnline = chatResponse.ok;

          // Get document count
          const docsResponse = await fetchWithNoCache(`${BACKEND_URL}/api/documents/`);
          if (docsResponse.ok) {
            const docsData = await docsResponse.json();
            documentCount = docsData.total || 0;
          }
        } catch (error) {
          // AI models or documents endpoint not accessible
        }
      }

      setSystemStatus(prev => ({
        ...prev,
        backend: isBackendOnline ? 'online' : 'offline',
        aiModels: aiModelsOnline ? 'online' : 'offline',
        documents: documentCount,
        uploadedDocuments: uploadedDocuments.length,
        cpuCores: navigator.hardwareConcurrency || 8
      }));
    } catch (error) {
      console.error('Status check failed:', error);
    }
  };

  const loadDocumentsFromBackend = async () => {
    try {
      const response = await fetchWithNoCache(`${BACKEND_URL}/api/documents/`);
      if (response.ok) {
        const data = await response.json();
        const backendDocs = data.documents.map((doc: any) => ({
          id: doc.id,
          name: doc.name,  // API now returns 'name' field correctly mapped
          size: doc.size,
          type: doc.type,
          uploadedAt: new Date(doc.uploadedAt),
          status: doc.status || 'ready' as const,
          // Enhanced fields from the new document service
          fullFileName: doc.fullFileName,
          fileHash: doc.fileHash,
          newFileName: doc.newFileName,
          fileDataHash: doc.fileDataHash,
          contentType: doc.contentType,
          metadata: doc.metadata
        }));
        setUploadedDocuments(backendDocs);
      }
    } catch (error) {
      console.error('Failed to load documents from backend:', error);
    }
  };

  const refreshDocuments = async () => {
    await loadDocumentsFromBackend();
    await checkSystemStatus(false);
  };

  const sendMessage = async (message: string) => {
    if (!message.trim()) return;
    
    console.log('🚀 Sending message:', message);
    console.log('📝 Current messages count:', messages.length);

    // Set streaming state to prevent UI conflicts
    setIsStreamingResponse(true);

    // Ensure we have a conversation ID before sending message
    let conversationId = currentConversationId;
    if (!conversationId) {
      const newConversation: Conversation = {
        id: Date.now().toString(),
        title: message.slice(0, 50) + (message.length > 50 ? '...' : ''),
        messages: [],
        createdAt: new Date(),
        updatedAt: new Date()
      };
      setConversations(prev => [newConversation, ...prev]);
      setCurrentConversationId(newConversation.id);
      conversationId = newConversation.id;
    }

    // Add user message
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: message,
      timestamp: new Date(),
      neonColor: getRandomNeonColor()
    };

    // Create AI message placeholder
    const aiMessageId = (Date.now() + 1).toString();
    const aiMessage: ChatMessage = {
      id: aiMessageId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      neonColor: getRandomNeonColor()
    };

    // Add both messages at once
    const newMessages = [...messages, userMessage, aiMessage];
    console.log('💬 Adding messages, new count will be:', newMessages.length);
    setMessages(newMessages);

    try {
      const response = await fetch(`${BACKEND_URL}/api/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message,
          conversation_id: conversationId,
          use_context: true,
          temperature: 0.7
        })
      });

      if (!response.ok) {
        throw new Error('Failed to get AI response');
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No response body reader');
      }

      const decoder = new TextDecoder();
      let buffer = '';
      let currentContent = '';
      let documentSources: DocumentSource[] = [];
      
      // Immediate update function for real-time typewriter effect
      const updateContent = () => {
        setMessages(prev => prev.map(msg => 
          msg.id === aiMessageId 
            ? { ...msg, content: currentContent }
            : msg
        ));
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.type === 'content') {
                currentContent += data.content;
                // Update immediately for real-time typewriter effect
                updateContent();
              } else if (data.type === 'sources') {
                // Handle document sources
                documentSources = data.sources || [];
              } else if (data.type === 'end') {
                // Streaming finished
                break;
              } else if (data.type === 'error') {
                throw new Error(data.error);
              }
            } catch (parseError) {
              console.error('Error parsing stream data:', parseError);
            }
          }
        }
      }

      // Streaming complete - ensure final update

      // Update with the complete response
      setMessages(prev => {
        const finalMessages = prev.map(msg => 
          msg.id === aiMessageId 
            ? { ...msg, content: currentContent || 'No response received', sources: documentSources }
            : msg
        );
        updateCurrentConversation(finalMessages);
        return finalMessages;
      });

    } catch (error) {
      const errorMessage: ChatMessage = {
        id: aiMessageId,
        role: 'assistant',
        content: `❌ Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: new Date(),
        neonColor: '#ff4444'
      };
      setMessages(prev => {
        const finalMessages = prev.map(msg => 
          msg.id === aiMessageId ? errorMessage : msg
        );
        updateCurrentConversation(finalMessages);
        return finalMessages;
      });
    } finally {
      // Clear streaming state
      setIsStreamingResponse(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
    setCurrentConversationId(null);
  };

  const handleFileUpload = async (files: File[]): Promise<FileUploadResult[]> => {
    const results: FileUploadResult[] = [];

    // Show console when upload starts
    setIsConsoleVisible(true);
    
    // Set processing state and start CPU monitoring
    setSystemStatus(prev => ({ ...prev, isProcessing: true, cpuUsage: 0 }));
    
    // Simulate CPU usage during processing (in a real app, this would come from the backend)
    const cpuMonitorInterval = setInterval(() => {
      const cpuUsage = Math.floor(Math.random() * 40) + 30; // 30-70% usage during processing
      setSystemStatus(prev => ({ ...prev, cpuUsage }));
    }, 500);

    // Send all files in a single request (as the backend expects)
    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file); // Use 'files' (plural) to match backend
    });

    try {
      const response = await fetch(`${BACKEND_URL}/api/upload/documents`, {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const data = await response.json();
        
        // Process each file result from the upload response
        if (data.files && Array.isArray(data.files)) {
          data.files.forEach((fileResult: any) => {
            const matchingFile = files.find(f => f.name === fileResult.filename);
            if (matchingFile) {
              results.push({
                file: matchingFile,
                success: fileResult.status === 'success',
                response: fileResult
              });
              
              // Add to uploaded documents list
              if (fileResult.status === 'success') {
                const newDoc: UploadedDocument = {
                  id: fileResult.document_id?.toString() || Date.now().toString() + Math.random().toString(36),
                  name: matchingFile.name,
                  size: matchingFile.size,
                  type: matchingFile.type,
                  uploadedAt: new Date(),
                  status: 'ready',
                  // Enhanced fields from the new document service
                  fullFileName: fileResult.full_file_name,
                  fileHash: fileResult.file_hash,
                  newFileName: fileResult.new_file_name,
                  fileDataHash: fileResult.file_data_hash
                };
                setUploadedDocuments(prev => [newDoc, ...prev]);
              } else {
                // Handle error case
                const errorDoc: UploadedDocument = {
                  id: Date.now().toString() + Math.random().toString(36),
                  name: matchingFile.name,
                  size: matchingFile.size,
                  type: matchingFile.type,
                  uploadedAt: new Date(),
                  status: 'error',
                  errorMessage: fileResult.message || 'Processing failed'
                };
                setUploadedDocuments(prev => [errorDoc, ...prev]);
              }
              
              // Notify the FileUploadPanel about the result
              if (fileUploadCallback) {
                fileUploadCallback(matchingFile, fileResult);
              }
            }
          });
        }
        
        // Note: System status will be updated by the periodic check
      } else {
        const errorData = await response.json().catch(() => ({ message: 'Upload failed' }));
        
        // If upload request failed, mark all files as failed
        files.forEach(file => {
          results.push({
            file,
            success: false,
            error: errorData.message || 'Upload failed'
          });
          
          if (fileUploadCallback) {
            fileUploadCallback(file, { status: 'error', message: errorData.message || 'Upload failed' });
          }
        });
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Upload failed';
      
      // If network error, mark all files as failed  
      files.forEach(file => {
        results.push({
          file,
          success: false,
          error: errorMessage
        });
        
        if (fileUploadCallback) {
          fileUploadCallback(file, { status: 'error', message: errorMessage });
        }
      });
      
      console.error('Upload failed:', error);
    } finally {
      // Stop CPU monitoring and reset processing state
      clearInterval(cpuMonitorInterval);
      setSystemStatus(prev => ({ ...prev, isProcessing: false, cpuUsage: 0 }));
    }
    
    return results;
  };

  // Check system status on mount and periodically
  // Load data on component mount
  useEffect(() => {
    loadFromStorage();
    checkSystemStatus(false); // Initial status check
    
    // Only load documents if not streaming to avoid conflicts
    if (!isStreamingResponse) {
      loadDocumentsFromBackend();
    }
    
    const interval = setInterval(() => {
      checkSystemStatus(); // Subsequent checks respect streaming state
    }, 60000); // Every 60 seconds (reduced frequency)
    return () => clearInterval(interval);
  }, []); // Empty dependency array to prevent infinite loop

  // Save data whenever conversations or documents change (but not during streaming)
  useEffect(() => {
    if (!isStreamingResponse && (conversations.length > 0 || uploadedDocuments.length > 0)) {
      saveToStorage();
    }
  }, [conversations, uploadedDocuments, currentConversationId, isStreamingResponse]); // Remove saveToStorage from dependencies

  return (
    <AppContainer>
      <GlobalStyles />
      
      <Header>
        <Title>AI MATE</Title>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <ClearChatButton onClick={clearChat} disabled={messages.length === 0}>
            Clear Chat
          </ClearChatButton>
          <ConsoleToggle 
            className={isConsoleVisible ? 'active' : ''} 
            onClick={() => setIsConsoleVisible(!isConsoleVisible)}
            title="Toggle processing console"
          >
            <Terminal size={14} />
            Console
          </ConsoleToggle>
          <StatusPanel status={systemStatus} />
        </div>
      </Header>

      <MainContent>
        <PanelGroup direction="horizontal">
          <PanelContainer defaultSize={25} minSize={20} maxSize={35}>
            <FileUploadPanel 
              onFileUpload={handleFileUpload} 
              onRegisterCallback={setFileUploadCallback}
              uploadedDocuments={uploadedDocuments}
              onRefreshDocuments={refreshDocuments}
            />
          </PanelContainer>
          
          <ResizeHandle />
          
          <PanelContainer defaultSize={55} minSize={30}>
            <ChatPanel 
              messages={messages} 
              onSendMessage={sendMessage}
              onFileUpload={handleFileUpload}
              isLoading={false}
            />
          </PanelContainer>
          
          <ResizeHandle />
          
          <PanelContainer defaultSize={20} minSize={15} maxSize={30}>
            <ConversationPanel 
              conversations={conversations}
              currentConversationId={currentConversationId}
              onLoadConversation={loadConversation}
              onNewConversation={createNewConversation}
              onClearChat={clearChat}
            />
          </PanelContainer>
        </PanelGroup>
      </MainContent>

      <Console 
        isVisible={isConsoleVisible} 
        onToggle={() => setIsConsoleVisible(false)}
      />
    </AppContainer>
  );
}

export default App;
