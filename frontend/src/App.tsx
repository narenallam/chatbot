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

interface FileUploadPanelProps {
  onFileUpload: (files: File[]) => Promise<FileUploadResult[]>;
  onRegisterCallback?: (callback: (file: File, result: any) => void) => void;
  onUploadStart?: () => void;
  uploadedDocuments?: UploadedDocument[];
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
  box-shadow: 0 0 10px rgba(255, 20, 147, 0.3);
  
  &:hover:not(:disabled) {
    color: #ffffff;
    border-color: #ff69b4;
    background: linear-gradient(135deg, rgba(255, 20, 147, 0.4) 0%, rgba(255, 0, 128, 0.4) 100%);
    box-shadow: 0 0 20px rgba(255, 20, 147, 0.6);
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
  box-shadow: 0 0 10px rgba(0, 255, 255, 0.3);
  
  &:hover {
    color: #ffffff;
    border-color: #00ffff;
    background: linear-gradient(135deg, rgba(0, 255, 255, 0.2) 0%, rgba(0, 200, 255, 0.2) 100%);
    box-shadow: 0 0 20px rgba(0, 255, 255, 0.6);
  }
  
  &.active {
    color: #ffffff;
    border-color: #00ffff;
    background: linear-gradient(135deg, rgba(0, 255, 255, 0.3) 0%, rgba(0, 200, 255, 0.3) 100%);
    box-shadow: 0 0 25px rgba(0, 255, 255, 0.8);
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
    if (!currentConversationId) {
      const conversation = createNewConversation();
      setCurrentConversationId(conversation.id);
    }

    setConversations(prev => prev.map(conv => {
      if (conv.id === currentConversationId) {
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
  };

  const loadConversation = (conversationId: string) => {
    const conversation = conversations.find(c => c.id === conversationId);
    if (conversation) {
      setCurrentConversationId(conversationId);
      setMessages(conversation.messages);
    }
  };

  const checkSystemStatus = async () => {
    try {
      // Check backend
      const backendResponse = await fetch(`${BACKEND_URL}/health`, {
        method: 'GET',
        signal: AbortSignal.timeout(3000)
      });
      const backendStatus = backendResponse.ok ? 'online' : 'offline';

      // Check AI models
      let aiStatus: 'online' | 'offline' = 'offline';
      try {
        const aiResponse = await fetch(`${BACKEND_URL}/api/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: 'Hi',
            conversation_id: 'health_test',
            use_context: false,
            temperature: 0.1
          }),
          signal: AbortSignal.timeout(5000)
        });
        aiStatus = aiResponse.ok ? 'online' : 'offline';
      } catch {
        aiStatus = 'offline';
      }

      // Check documents
      let docCount = 0;
      try {
        const docsResponse = await fetch(`${BACKEND_URL}/api/documents`, {
          signal: AbortSignal.timeout(3000)
        });
        if (docsResponse.ok) {
          const docs = await docsResponse.json();
          docCount = Array.isArray(docs) ? docs.length : 0;
        }
      } catch {
        docCount = 0;
      }

      setSystemStatus(prev => ({
        ...prev,
        backend: backendStatus,
        aiModels: aiStatus,
        documents: docCount,
        uploadedDocuments: uploadedDocuments.length,
        cpuCores: navigator.hardwareConcurrency || 8
      }));
    } catch (error) {
      console.error('Status check failed:', error);
    }
  };

  const sendMessage = async (message: string) => {
    if (!message.trim()) return;

    // Add user message
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: message,
      timestamp: new Date(),
      neonColor: getRandomNeonColor()
    };
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);

    try {
      const response = await fetch(`${BACKEND_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message,
          conversation_id: 'main_chat',
          use_context: true,
          temperature: 0.7
        })
      });

      if (response.ok) {
        const data = await response.json();
        const aiMessage: ChatMessage = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: data.message || 'No response received',
          timestamp: new Date(),
          neonColor: getRandomNeonColor()
        };
        const finalMessages = [...newMessages, aiMessage];
        setMessages(finalMessages);
        updateCurrentConversation(finalMessages);
      } else {
        throw new Error('Failed to get AI response');
      }
    } catch (error) {
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `❌ Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: new Date(),
        neonColor: '#ff4444'
      };
      const finalMessages = [...newMessages, errorMessage];
      setMessages(finalMessages);
      updateCurrentConversation(finalMessages);
    }
  };

  const clearChat = () => {
    setMessages([]);
    setCurrentConversationId(null);
  };

  const clearAllData = () => {
    if (window.confirm('⚠️ This will clear all conversations and document history from your browser. Are you sure?')) {
      // Clear state
      setMessages([]);
      setConversations([]);
      setUploadedDocuments([]);
      setCurrentConversationId(null);
      
      // Clear localStorage
      localStorage.removeItem('ai-mate-conversations');
      localStorage.removeItem('ai-mate-documents');
      localStorage.removeItem('ai-mate-current-conversation');
      
      console.log('🧹 All frontend data cleared');
    }
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
      const response = await fetch(`${BACKEND_URL}/api/upload`, {
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
                  id: Date.now().toString() + Math.random().toString(36),
                  name: matchingFile.name,
                  size: matchingFile.size,
                  type: matchingFile.type,
                  uploadedAt: new Date(),
                  status: 'ready'
                };
                setUploadedDocuments(prev => [newDoc, ...prev]);
              }
              
              // Notify the FileUploadPanel about the result
              if (fileUploadCallback) {
                fileUploadCallback(matchingFile, fileResult);
              }
            }
          });
        }
        
        // Refresh system status to update document count
        await checkSystemStatus();
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
    checkSystemStatus();
    const interval = setInterval(checkSystemStatus, 30000); // Every 30 seconds
    return () => clearInterval(interval);
  }, []);

  // Save data whenever conversations or documents change
  useEffect(() => {
    if (conversations.length > 0 || uploadedDocuments.length > 0) {
      saveToStorage();
    }
  }, [conversations, uploadedDocuments, currentConversationId]);

  return (
    <AppContainer>
      <GlobalStyles />
      
      <Header>
        <Title>AI MATE</Title>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <ClearChatButton onClick={clearChat} disabled={messages.length === 0}>
            Clear Chat
          </ClearChatButton>
          <ClearChatButton 
            onClick={clearAllData} 
            disabled={conversations.length === 0 && uploadedDocuments.length === 0}
            style={{ 
              background: 'linear-gradient(135deg, rgba(255, 69, 0, 0.2) 0%, rgba(255, 99, 71, 0.2) 100%)',
              borderColor: '#ff4500',
              color: '#ff6347'
            }}
          >
            Clear All Data
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
