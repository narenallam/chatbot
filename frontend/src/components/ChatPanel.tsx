import React, { useState, useRef, useEffect, useCallback } from 'react';
import styled from 'styled-components';
import { Send, Bot, User, Upload } from 'lucide-react';
import { useDropzone } from 'react-dropzone';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { ChatMessage } from '../App';
import { DocumentPreview } from './DocumentPreview';
import { DocumentSources } from './DocumentSources';

interface ChatPanelProps {
  messages: ChatMessage[];
  onSendMessage: (message: string) => void;
  onFileUpload: (files: File[]) => Promise<any>;
  isLoading: boolean;
  contextInfo?: { model_name: string; context_window: number; buffer_size: number } | null;
}

const Container = styled.div`
  display: flex;
  flex-direction: column;
  height: 100%;
  background: #1a1a1a;
  border-radius: 8px;
  border: 1px solid #333;
  overflow: hidden;
`;

const MessagesContainer = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  scroll-behavior: smooth;
  display: flex;
  flex-direction: column;
  
  &::-webkit-scrollbar {
    width: 8px;
  }
  
  &::-webkit-scrollbar-track {
    background: #2a2a2a;
    border-radius: 4px;
  }
  
  &::-webkit-scrollbar-thumb {
    background: #555;
    border-radius: 4px;
  }
  
  &::-webkit-scrollbar-thumb:hover {
    background: #777;
  }
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

const MessageBubble = styled.div<{ $isUser: boolean }>`
  display: flex;
  align-items: flex-start;
  gap: 12px;
  margin: ${props => props.$isUser ? '2px 0' : '8px 0'};
  width: 100%;
  flex-direction: ${props => props.$isUser ? 'row-reverse' : 'row'};
  justify-content: ${props => props.$isUser ? 'flex-end' : 'flex-start'};
`;

const Avatar = styled.div<{ $isUser: boolean }>`
  width: 36px;
  height: 36px;
  border-radius: 50%;
  background: transparent;
  border: 1px solid ${props => props.$isUser ? '#00ffff' : '#ff1493'};
  display: flex;
  align-items: center;
  justify-content: center;
  color: ${props => props.$isUser ? '#00ffff' : '#ff1493'};
  font-size: 16px;
  flex-shrink: 0;
  box-shadow: none;
`;

const MessageContent = styled.div<{ $isUser: boolean; $neonColor?: string }>`
  background: linear-gradient(135deg, rgba(45, 45, 45, 0.7) 0%, rgba(35, 35, 35, 0.8) 100%);
  border: 0.5px solid rgb(42, 42, 42);
  border-radius: ${props => props.$isUser ? '22px 8px 22px 22px' : '17px'};
  overflow: hidden;
  color: #fff;
  font-size: ${props => props.$isUser ? '0.95rem' : '0.9rem'};
  line-height: ${props => props.$isUser ? '1.2' : '1.6'};
  word-wrap: break-word;
  box-shadow: none;
  position: relative;
  transition: all 0.2s ease;
  display: flex;
  flex-direction: column;
  align-items: stretch;
  margin-left: ${props => props.$isUser ? 'auto' : '0'};
  margin-right: ${props => props.$isUser ? '0' : 'auto'};
  max-width: ${props => props.$isUser ? '60%' : '75%'};
  min-width: 36px;
  width: ${props => props.$isUser ? 'fit-content' : '75%'};
  padding: ${props => props.$isUser ? '6px 16px' : '4px 22px 8px 22px'};
`;

const MessageCopyButton = styled.button`
  background: linear-gradient(135deg, rgba(35, 35, 40, 0.7) 0%, rgba(25, 25, 30, 0.8) 100%);
  border: none;
  color: #999;
  cursor: pointer;
  padding: 6px 10px;
  border-radius: 6px;
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.7rem;
  font-weight: 500;
  transition: all 0.3s ease;
  backdrop-filter: blur(8px);
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
  
  &:hover {
    color: #ffffff;
    background: linear-gradient(135deg, rgba(0, 255, 255, 0.15) 0%, rgba(0, 200, 255, 0.2) 100%);
    box-shadow: 0 3px 12px rgba(0, 255, 255, 0.15);
    transform: translateY(-0.5px);
  }
  
  &:active {
    transform: translateY(0);
    box-shadow: 0 2px 6px rgba(0, 255, 255, 0.2);
  }
`;

const MessageCopyIcon = styled.div`
  position: relative;
  width: 14px;
  height: 14px;
  
  &::before, &::after {
    content: '';
    position: absolute;
    border: 1.5px solid currentColor;
    border-radius: 4px;
  }
  
  &::before {
    width: 9px;
    height: 9px;
    top: 0;
    left: 2.5px;
  }
  
  &::after {
    width: 9px;
    height: 9px;
    top: 2.5px;
    left: 0;
    background: #1a1a1a;
  }
`;

const MessageBody = styled.div`
  padding: 0;
`;

const MessageTime = styled.div`
  font-size: 0.7rem;
  color: #666;
  margin-top: 12px;
  text-align: right;
  padding-top: 8px;
  border-top: 1px solid rgba(255, 255, 255, 0.05);
`;

const MarkdownContent = styled.div`
  color: #e0e0e0;
  line-height: 1.8;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
  font-feature-settings: "liga" 1, "calt" 1;
  
  h1, h2, h3, h4, h5, h6 {
    color: #ffffff;
    margin: 24px 0 16px 0;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 8px;
    letter-spacing: -0.02em;
    
    &:first-child {
      margin-top: 0;
    }
    
    &::before {
      font-size: 0.9em;
    }
  }
  
  h1 { 
    font-size: 1.8em;
    &::before { content: '🎯'; }
  }
  h2 { 
    font-size: 1.5em;
    &::before { content: '📋'; }
  }
  h3 { 
    font-size: 1.3em;
    &::before { content: '🔹'; }
  }
  h4 { 
    font-size: 1.1em;
    &::before { content: '▶️'; }
  }
  h5 { 
    font-size: 1em;
    &::before { content: '🔸'; }
  }
  h6 { 
    font-size: 0.95em;
    &::before { content: '•'; }
  }
  
  p {
    margin: 16px 0;
    color: #e8e8e8;
    font-weight: 400;
    line-height: 1.7;
  }
  
  ul, ol {
    margin: 16px 0;
    padding-left: 24px;
  }
  
  li {
    margin: 8px 0;
    color: #e8e8e8;
    line-height: 1.6;
    
    &::marker {
      color: #00ffff;
      font-weight: 600;
    }
  }
  
  blockquote {
    border-left: 4px solid #00ffff;
    background: rgba(0, 255, 255, 0.05);
    margin: 20px 0;
    padding: 16px 20px;
    border-radius: 0 8px 8px 0;
    color: #d0d0d0;
    font-style: italic;
    font-weight: 400;
  }
  
  code:not(.custom-code-block code) {
    background: linear-gradient(135deg, rgba(40, 40, 45, 0.8) 0%, rgba(30, 30, 35, 0.9) 100%);
    color: #e8e8e8;
    padding: 4px 8px;
    border-radius: 6px;
    font-family: 'Noto Sans Mono', 'Fira Code', 'JetBrains Mono', 'Monaco', 'Consolas', monospace !important;
    font-weight: 300 !important;
    font-size: 0.9em;
    backdrop-filter: blur(8px);
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
    font-feature-settings: "liga" 1, "calt" 1;
  }

  
  table {
    border-collapse: collapse;
    width: 100%;
    margin: 20px 0;
    border: 1px solid #404040;
    border-radius: 10px;
    overflow: hidden;
    font-size: 0.95em;
  }
  
  th, td {
    border: 1px solid #404040;
    padding: 14px 16px;
    text-align: left;
  }
  
  th {
    background: rgba(0, 255, 255, 0.12);
    color: #ffffff;
    font-weight: 700;
    font-size: 0.9em;
  }
  
  tr:nth-child(even) {
    background: rgba(255, 255, 255, 0.03);
  }
  
  a {
    color: #4a9eff;
    text-decoration: none;
    font-weight: 500;
    
    &:hover {
      text-decoration: underline;
      color: #6ab7ff;
    }
  }
  
  strong {
    color: #ffffff;
    font-weight: 700;
  }
  
  em {
    color: #d8d8d8;
    font-style: italic;
    font-weight: 400;
  }
  
  hr {
    border: none;
    height: 2px;
    background: linear-gradient(90deg, transparent, #404040, transparent);
    margin: 32px 0;
    border-radius: 1px;
  }
`;

const InputContainer = styled.div<{ $isDragActive?: boolean }>`
  padding: 20px;
  background: ${props => props.$isDragActive ? 
    'linear-gradient(135deg, rgba(0, 255, 255, 0.08) 0%, rgba(0, 200, 255, 0.08) 100%)' :
    'linear-gradient(135deg, rgba(0, 255, 255, 0.02) 0%, rgba(0, 200, 255, 0.02) 100%)'
  };
  box-shadow: none;
  min-height: 176px;
  transition: all 0.2s ease;
  
  ${props => props.$isDragActive && `
    border: 2px dashed #00ffff;
    border-radius: 12px;
    box-shadow: none;
  `}
`;

const InputWrapper = styled.div`
  display: flex;
  gap: 12px;
  align-items: flex-end;
  position: relative;
`;

const DragOverlay = styled.div<{ $isDragActive: boolean }>`
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 255, 255, 0.1);
  border: 2px dashed #00ffff;
  border-radius: 12px;
  display: ${props => props.$isDragActive ? 'flex' : 'none'};
  align-items: center;
  justify-content: center;
  flex-direction: column;
  gap: 12px;
  z-index: 10;
  pointer-events: none;
`;

const DragText = styled.div`
  color: #00ffff;
  font-size: 1.1rem;
  font-weight: 600;
  text-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
`;

const DragHint = styled.div`
  color: #00cccc;
  font-size: 0.9rem;
  opacity: 0.8;
`;

const MessageInput = styled.textarea`
  flex: 1;
  background: linear-gradient(135deg, rgba(136, 136, 136, 0.05) 0%, rgba(136, 136, 136, 0.05) 100%);
  border: 1px solid #888888;
  border-radius: 17px;
  padding: 16px 20px;
  color: #ffffff;
  font-size: 0.9rem;
  resize: none;
  min-height: 176px;
  max-height: 400px;
  font-family: inherit;
  transition: all 0.2s ease;
  box-shadow: none;
  
  &:focus {
    outline: none;
    border-color: #888888;
    box-shadow: none;
    background: linear-gradient(135deg, rgba(136, 136, 136, 0.1) 0%, rgba(136, 136, 136, 0.1) 100%);
  }
  
  &::placeholder {
    color: #888888;
    opacity: 0.7;
  }
`;

const SendButton = styled.button<{ $disabled: boolean }>`
  background: ${props => props.$disabled ? 
    'linear-gradient(135deg, rgba(102, 102, 102, 0.2) 0%, rgba(68, 68, 68, 0.2) 100%)' : 
    'linear-gradient(135deg, #00ffff 0%, #00cccc 100%)'
  };
  color: ${props => props.$disabled ? '#666' : '#000000'};
  border: ${props => props.$disabled ? '1px solid #444' : '1px solid #00ffff'};
  border-radius: 50%;
  width: 54px;
  height: 54px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: ${props => props.$disabled ? 'not-allowed' : 'pointer'};
  transition: all 0.2s ease;
  box-shadow: none;
  
  &:hover:not(:disabled) {
    transform: scale(1.1);
    box-shadow: none;
    background: linear-gradient(135deg, #ffffff 0%, #00ffff 100%);
  }
  
  &:active:not(:disabled) {
    transform: scale(0.95);
  }
`;

const EmptyState = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: #888;
  text-align: center;
  gap: 16px;
  padding: 40px;
`;

const EmptyStateContent = styled.div`
  background: transparent;
  border-radius: 300px;
  padding: 60px 80px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 20px;
`;

const EmptyIcon = styled.div`
  width: 80px;
  height: 80px;
  border-radius: 50%;
  background: transparent;
  border: 3px solid #ff1493;
  box-shadow: 0 0 8px #ff149530, inset 0 0 8px #ff149315;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 8px;
  animation: neonPulse 3s ease-in-out infinite alternate;
  
  @keyframes neonPulse {
    from {
      box-shadow: 0 0 8px #ff149530, inset 0 0 8px #ff149315;
      border-color: #ff1493;
    }
    to {
      box-shadow: 0 0 12px #ff149540, inset 0 0 12px #ff149320;
      border-color: #ff69b4;
    }
  }
`;

const EmptyText = styled.div`
  font-size: 1.2rem;
  color: #ccc;
  font-weight: 600;
`;

const EmptyHint = styled.div`
  font-size: 0.9rem;
  color: #888;
  max-width: 400px;
  line-height: 1.5;
`;

const CopyNotification = styled.div<{ $visible: boolean }>`
  position: fixed;
  top: 20px;
  right: 20px;
  background: linear-gradient(135deg, #4caf50 0%, #45a049 100%);
  color: white;
  padding: 12px 20px;
  border-radius: 8px;
  box-shadow: none;
  z-index: 1000;
  transform: translateY(${props => props.$visible ? '0' : '-100px'});
  opacity: ${props => props.$visible ? '1' : '0'};
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.9rem;
  font-weight: 500;
`;

const CodeBlock = styled.div`
  background: #1e1e1e;
  border: 1px solid #444;
  border-radius: 8px;
  margin: 16px 0;
  overflow: hidden;
`;

const CodeHeader = styled.div`
  background: #2a2a2a;
  padding: 8px 12px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid #444;
`;

const CodeLanguage = styled.span`
  color: #00ffff;
  font-size: 0.75rem;
  font-weight: 500;
  text-transform: capitalize;
`;

const CopyButton = styled.button`
  background: linear-gradient(135deg, rgba(40, 40, 45, 0.8) 0%, rgba(30, 30, 35, 0.9) 100%);
  border: none;
  color: #aaa;
  cursor: pointer;
  padding: 8px 12px;
  border-radius: 8px;
  font-size: 12px;
  font-weight: 500;
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  gap: 8px;
  backdrop-filter: blur(10px);
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
  
  &:hover {
    color: #ffffff;
    background: linear-gradient(135deg, rgba(0, 255, 255, 0.2) 0%, rgba(0, 200, 255, 0.25) 100%);
    box-shadow: 0 4px 16px rgba(0, 255, 255, 0.2);
    transform: translateY(-1px);
  }
  
  &:active {
    transform: translateY(0);
    box-shadow: 0 2px 8px rgba(0, 255, 255, 0.3);
  }
`;

const CopyIcon = styled.div`
  position: relative;
  width: 16px;
  height: 16px;
  
  &::before, &::after {
    content: '';
    position: absolute;
    border: 2px solid currentColor;
    border-radius: 5px;
  }
  
  &::before {
    width: 11px;
    height: 11px;
    top: 0;
    left: 3px;
  }
  
  &::after {
    width: 11px;
    height: 11px;
    top: 3px;
    left: 0;
    background: #1e1e1e;
  }
`;

export const ChatPanel: React.FC<ChatPanelProps> = ({ messages, onSendMessage, onFileUpload, isLoading, contextInfo }) => {
  console.log('🎯 ChatPanel received messages:', messages.length, messages);
  
  const [inputValue, setInputValue] = useState('');
  const [copyNotification, setCopyNotification] = useState(false);
  const [previewDocument, setPreviewDocument] = useState<{ documentId: string; filename: string } | null>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (inputValue.trim() && !isLoading) {
      onSendMessage(inputValue.trim());
      setInputValue('');
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const adjustTextareaHeight = () => {
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
      inputRef.current.style.height = `${Math.min(inputRef.current.scrollHeight, 120)}px`;
    }
  };

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopyNotification(true);
      setTimeout(() => setCopyNotification(false), 2000);
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
  };

  const handleDocumentClick = (documentId: string, filename: string) => {
    setPreviewDocument({ documentId, filename });
  };

  const handleClosePreview = () => {
    setPreviewDocument(null);
  };

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (!acceptedFiles || acceptedFiles.length === 0) {
      return;
    }

    try {
      await onFileUpload(acceptedFiles);
    } catch (error) {
      console.error('Upload failed:', error);
    }
  }, [onFileUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/msword': ['.doc'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'application/vnd.ms-powerpoint': ['.ppt'],
      'application/vnd.openxmlformats-officedocument.presentationml.presentation': ['.pptx'],
      'application/vnd.ms-excel': ['.xls'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'text/plain': ['.txt'],
      'text/markdown': ['.md'],
      'text/html': ['.html'],
      'image/png': ['.png'],
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/heic': ['.heic'],
      'image/bmp': ['.bmp'],
      'image/gif': ['.gif'],
      'image/tiff': ['.tiff']
    },
    multiple: true,
    noClick: true // Prevent clicking on the entire area from opening file dialog
  });

  const handleFileInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (files) {
      onDrop(Array.from(files));
    }
  };

  // Add custom neon theme styles for syntax highlighting
  useEffect(() => {
    const style = document.createElement('style');
    style.textContent = `
      .token.keyword { color: #ff6b9d !important; font-weight: bold !important; text-shadow: 0 0 6px #ff6b9d60 !important; }
      .token.function { color: #00ffff !important; font-weight: bold !important; text-shadow: 0 0 6px #00ffff50 !important; }
      .token.string { color: #00ff7f !important; text-shadow: 0 0 4px #00ff7f40 !important; }
      .token.number { color: #ffff00 !important; text-shadow: 0 0 4px #ffff0040 !important; }
      .token.boolean { color: #ff6600 !important; text-shadow: 0 0 4px #ff660040 !important; }
      .token.variable { color: #9932cc !important; background: rgba(153, 50, 204, 0.15) !important; padding: 2px 4px !important; border-radius: 3px !important; text-shadow: 0 0 3px #9932cc40 !important; }
      .token.parameter { color: #9932cc !important; background: rgba(153, 50, 204, 0.15) !important; padding: 2px 4px !important; border-radius: 3px !important; text-shadow: 0 0 3px #9932cc40 !important; }
      .token.operator { color: #ff1493 !important; font-weight: bold !important; }
      .token.punctuation { color: #d4d4d4 !important; }
      .token.comment { color: #6a9955 !important; font-style: italic !important; opacity: 0.8 !important; }
      .token.class-name { color: #00ffff !important; font-weight: bold !important; text-shadow: 0 0 4px #00ffff40 !important; }
      .token.builtin { color: #ff69b4 !important; text-shadow: 0 0 4px #ff69b440 !important; }
      .token.property { color: #9932cc !important; background: rgba(153, 50, 204, 0.15) !important; padding: 2px 4px !important; border-radius: 3px !important; text-shadow: 0 0 3px #9932cc40 !important; }
      .token.attr-name { color: #ff6600 !important; text-shadow: 0 0 4px #ff660030 !important; }
      .token.selector { color: #00ff7f !important; text-shadow: 0 0 4px #00ff7f30 !important; }
      .token.tag { color: #ff6b9d !important; font-weight: bold !important; text-shadow: 0 0 4px #ff6b9d30 !important; }
      .token.attr-value { color: #00ff7f !important; text-shadow: 0 0 4px #00ff7f30 !important; }
      .token.namespace { color: #ff69b4 !important; opacity: 0.8 !important; }
      .token.regex { color: #ffff00 !important; text-shadow: 0 0 4px #ffff0030 !important; }
      .token.important { color: #ff6600 !important; font-weight: bold !important; text-shadow: 0 0 4px #ff660030 !important; }
    `;
    document.head.appendChild(style);
    
    return () => {
      if (document.head.contains(style)) {
        document.head.removeChild(style);
      }
    };
  }, []);

  const renderMarkdown = (content: string) => {
    return (
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
                    code({ node, className, children, ...props }: any) {
            const match = /language-(\w+)/.exec(className || '');
            const language = match ? match[1] : '';
            const isBlockCode = !props.inline;
            
            if (isBlockCode && language) {
              const codeContent = String(children).replace(/\n$/, '');
              
              return (
                <CodeBlock className="custom-code-block">
                  <CodeHeader>
                    <CodeLanguage>{language}</CodeLanguage>
                    <CopyButton onClick={() => copyToClipboard(codeContent)}>
                      <CopyIcon />
                      Copy
                    </CopyButton>
                  </CodeHeader>
                  <div style={{
                    background: '#1e1e1e',
                    padding: '16px',
                    fontSize: '0.85rem',
                    fontFamily: "'Noto Sans Mono', 'Fira Code', 'Monaco', 'Consolas', monospace",
                    fontWeight: 300,
                    lineHeight: 1.5,
                    overflow: 'auto'
                  }}>
                    <SyntaxHighlighter
                      style={tomorrow}
                      language={language}
                      PreTag="div"
                      customStyle={{
                        background: 'transparent',
                        margin: 0,
                        padding: 0,
                        fontSize: 'inherit',
                        fontFamily: 'inherit',
                        lineHeight: 'inherit'
                      }}
                    >
                      {codeContent}
                    </SyntaxHighlighter>
                  </div>
                </CodeBlock>
              );
            }
            
            // Inline code
            return (
              <code 
                className={className} 
                {...props}
                style={{
                  background: 'linear-gradient(135deg, rgba(40, 40, 45, 0.8) 0%, rgba(30, 30, 35, 0.9) 100%)',
                  padding: '4px 8px',
                  borderRadius: '6px',
                  fontSize: '0.85em',
                  fontFamily: "'Noto Sans Mono', 'Fira Code', 'Monaco', 'Consolas', monospace",
                  fontWeight: 300,
                  color: '#aaa',
                  backdropFilter: 'blur(8px)',
                  boxShadow: '0 2px 6px rgba(0, 0, 0, 0.2)'
                }}
              >
                {children}
              </code>
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>
    );
  };

  return (
    <Container>
      <MessagesContainer>
        {messages.length === 0 ? (
          <EmptyState>
            <EmptyStateContent>
              <EmptyIcon>
                <Bot size={32} color="#ff1493" />
              </EmptyIcon>
              <EmptyText>Welcome to AI MATE!</EmptyText>
              <EmptyHint>
                Ask questions about your uploaded documents, request analysis, coding or chat about anything else. 
              </EmptyHint>
            </EmptyStateContent>
          </EmptyState>
        ) : (
          messages.map((message) => (
            <MessageBubble key={message.id} $isUser={message.role === 'user'}>
              <Avatar $isUser={message.role === 'user'}>
                {message.role === 'user' ? <User size={18} /> : <Bot size={18} />}
              </Avatar>
              <MessageContent 
                $isUser={message.role === 'user'}
                $neonColor={message.neonColor}
              >
                <MessageBody>
                  <MarkdownContent style={{padding: message.role === 'user' ? '0' : '16px 16px 0 16px'}}>
                    {message.role === 'user' ? (
                      <p>{message.content}</p>
                    ) : (
                      renderMarkdown(message.content)
                    )}
                  </MarkdownContent>
                  
                  {/* Show document sources for AI messages */}
                  {message.role === 'assistant' && message.sources && message.sources.length > 0 && (
                    <DocumentSources 
                      sources={message.sources} 
                      onDocumentClick={handleDocumentClick}
                    />
                  )}
                  
                  {/* Only show copy button and time for assistant */}
                  {message.role === 'assistant' && (
                    <MessageTime style={{padding: '0 16px 12px 16px'}}>
                      <MessageCopyButton 
                        onClick={() => copyToClipboard(message.content)}
                        style={{
                          float: 'left',
                          marginRight: '8px',
                          marginTop: '0px'
                        }}
                        title="Copy message"
                      >
                        <MessageCopyIcon />
                        Copy
                      </MessageCopyButton>
                      {formatTime(message.timestamp)}
                    </MessageTime>
                  )}
                </MessageBody>
              </MessageContent>
            </MessageBubble>
          ))
        )}
        <div ref={messagesEndRef} />
      </MessagesContainer>

      <InputContainer {...getRootProps()} $isDragActive={isDragActive}>
        <DragOverlay $isDragActive={isDragActive}>
          <Upload size={48} color="#00ffff" />
          <DragText>Drop files here</DragText>
          <DragHint>Supports PDF, DOCX, PPT, XLS, TXT, MD, HTML, Images • Max 500MB</DragHint>
        </DragOverlay>
        
        <form onSubmit={handleSubmit}>
          <InputWrapper>
            <MessageInput
              ref={inputRef}
              value={inputValue}
              onChange={(e) => {
                setInputValue(e.target.value);
                adjustTextareaHeight();
              }}
              onKeyPress={handleKeyPress}
              placeholder="Type your message here..."
              disabled={isLoading}
              rows={1}
            />
            <SendButton 
              type="submit" 
              $disabled={!inputValue.trim() || isLoading}
            >
              <Send size={18} />
            </SendButton>
          </InputWrapper>
        </form>
        
        <input
          {...getInputProps()}
          ref={fileInputRef}
          onChange={handleFileInputChange}
          style={{ display: 'none' }}
        />
      </InputContainer>

      <CopyNotification $visible={copyNotification}>
        <span className="material-icons" style={{ fontSize: '18px' }}>check_circle</span>
        Copied to clipboard!
      </CopyNotification>

      {/* Document Preview Modal */}
      {previewDocument && (
        <DocumentPreview
          documentId={previewDocument.documentId}
          filename={previewDocument.filename}
          isOpen={!!previewDocument}
          onClose={handleClosePreview}
        />
      )}
    </Container>
  );
}; 