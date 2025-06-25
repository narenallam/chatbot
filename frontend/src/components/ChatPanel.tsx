import React, { useState, useRef, useEffect, useCallback } from 'react';
import styled from 'styled-components';
import { Send, Bot, User, Upload } from 'lucide-react';
import { useDropzone } from 'react-dropzone';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { tomorrow } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { ChatMessage } from '../App';

interface ChatPanelProps {
  messages: ChatMessage[];
  onSendMessage: (message: string) => void;
  onFileUpload: (files: File[]) => Promise<any>;
  isLoading: boolean;
}

const PanelContainer = styled.div`
  background: #1a1a1a;
  display: flex;
  flex-direction: column;
  height: 100%;
  border-right: 1px solid #333;
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

const MessagesContainer = styled.div`
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  
  /* Custom scrollbar styling */
  &::-webkit-scrollbar {
    width: 12px;
    background: #1a1a1a;
  }
  
  &::-webkit-scrollbar-track {
    background: #1a1a1a;
    border-radius: 6px;
    margin: 4px;
  }
  
  &::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #444 0%, #333 100%);
    border-radius: 6px;
    border: 2px solid #1a1a1a;
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

const MessageBubble = styled.div<{ $isUser: boolean }>`
  display: flex;
  align-items: flex-start;
  gap: 12px;
  margin: 8px 16px;
  ${props => props.$isUser && 'flex-direction: row-reverse;'}
`;

const Avatar = styled.div<{ $isUser: boolean }>`
  width: 36px;
  height: 36px;
  border-radius: 50%;
  background: ${props => props.$isUser ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' : 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)'};
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-size: 16px;
  flex-shrink: 0;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
`;

const MessageContent = styled.div<{ $isUser: boolean; $neonColor?: string }>`
  max-width: 75%;
  background: ${props => props.$isUser ? 
    'linear-gradient(135deg, rgba(45, 45, 45, 0.7) 0%, rgba(35, 35, 35, 0.8) 100%)' : 
    'linear-gradient(135deg, rgba(30, 30, 30, 0.8) 0%, rgba(25, 25, 25, 0.9) 100%)'
  };
  border: 0.5px solid ${props => props.$neonColor || (props.$isUser ? '#667eea' : '#ff6b6b')};
  border-radius: 12px;
  overflow: hidden;
  color: #fff;
  font-size: 0.9rem;
  line-height: 1.6;
  word-wrap: break-word;
  box-shadow: 0 0 8px ${props => (props.$neonColor || (props.$isUser ? '#667eea' : '#ff6b6b')) + '20'};
  position: relative;
  transition: all 0.2s ease;
  
  &:hover {
    box-shadow: 0 0 12px ${props => (props.$neonColor || (props.$isUser ? '#667eea' : '#ff6b6b')) + '30'};
    border-color: ${props => props.$neonColor || (props.$isUser ? '#667eea' : '#ff6b6b')};
  }
`;

const MessageHeader = styled.div<{ $isUser: boolean }>`
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px 8px;
  border-bottom: 1px solid ${props => props.$isUser ? 'rgba(102, 126, 234, 0.2)' : '#333'};
  background: ${props => props.$isUser ? 
    'rgba(102, 126, 234, 0.05)' : 
    'rgba(255, 255, 255, 0.02)'
  };
`;

const MessageRole = styled.div<{ $isUser: boolean }>`
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 0.75rem;
  font-weight: 600;
  color: ${props => props.$isUser ? '#8b9dc3' : '#ff6b6b'};
  text-transform: uppercase;
  letter-spacing: 0.5px;
`;

const CopyButton = styled.button`
  background: none;
  border: none;
  color: #888;
  cursor: pointer;
  padding: 4px;
  border-radius: 4px;
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 0.7rem;
  transition: all 0.2s ease;
  
  &:hover {
    color: #00ffff;
    background: rgba(0, 255, 255, 0.1);
  }
  
  .material-icons {
    font-size: 14px;
  }
`;

const MessageBody = styled.div`
  padding: 16px;
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
  line-height: 1.7;
  
  h1, h2, h3, h4, h5, h6 {
    color: #00ffff;
    margin: 20px 0 12px 0;
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: 8px;
    
    &:first-child {
      margin-top: 0;
    }
    
    &::before {
      font-family: 'Material Icons';
      font-size: 0.9em;
    }
  }
  
  h1::before { content: '🎯'; }
  h2::before { content: '📋'; }
  h3::before { content: '🔹'; }
  h4::before { content: '▶️'; }
  h5::before { content: '🔸'; }
  h6::before { content: '•'; }
  
  p {
    margin: 12px 0;
    color: #e0e0e0;
  }
  
  ul, ol {
    margin: 12px 0;
    padding-left: 20px;
  }
  
  li {
    margin: 6px 0;
    color: #e0e0e0;
    
    &::marker {
      color: #00ffff;
    }
  }
  
  blockquote {
    border-left: 4px solid #00ffff;
    background: rgba(0, 255, 255, 0.05);
    margin: 16px 0;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    color: #b0b0b0;
    font-style: italic;
  }
  
  code {
    background: rgba(255, 255, 255, 0.1);
    color: #ff9800;
    padding: 2px 6px;
    border-radius: 4px;
    font-family: 'Fira Code', 'Monaco', 'Consolas', monospace;
    font-size: 0.85em;
  }
  
  pre {
    background: #1e1e1e !important;
    border: 1px solid #333;
    border-radius: 8px;
    margin: 16px 0;
    overflow-x: auto;
    
    code {
      background: none;
      color: inherit;
      padding: 0;
    }
  }
  
  table {
    border-collapse: collapse;
    width: 100%;
    margin: 16px 0;
    border: 1px solid #333;
    border-radius: 8px;
    overflow: hidden;
  }
  
  th, td {
    border: 1px solid #333;
    padding: 12px;
    text-align: left;
  }
  
  th {
    background: rgba(0, 255, 255, 0.1);
    color: #00ffff;
    font-weight: 600;
  }
  
  tr:nth-child(even) {
    background: rgba(255, 255, 255, 0.02);
  }
  
  a {
    color: #00ffff;
    text-decoration: none;
    border-bottom: 1px dotted #00ffff;
    
    &:hover {
      border-bottom-style: solid;
    }
  }
  
  strong {
    color: #fff;
    font-weight: 600;
  }
  
  em {
    color: #ccc;
    font-style: italic;
  }
  
  hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #333, transparent);
    margin: 24px 0;
  }
`;

const InputContainer = styled.div<{ $isDragActive?: boolean }>`
  padding: 20px;
  background: ${props => props.$isDragActive ? 
    'linear-gradient(135deg, rgba(0, 255, 255, 0.08) 0%, rgba(0, 200, 255, 0.08) 100%)' :
    'linear-gradient(135deg, rgba(0, 255, 255, 0.02) 0%, rgba(0, 200, 255, 0.02) 100%)'
  };
  box-shadow: 0 -5px 15px rgba(0, 255, 255, 0.1);
  min-height: 176px;
  transition: all 0.2s ease;
  
  ${props => props.$isDragActive && `
    border: 2px dashed #00ffff;
    border-radius: 12px;
    box-shadow: 0 0 20px rgba(0, 255, 255, 0.3);
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
  background: linear-gradient(135deg, rgba(0, 255, 255, 0.05) 0%, rgba(0, 200, 255, 0.05) 100%);
  border: 1px solid #00ffff;
  border-radius: 12px;
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
    border-color: #00ffff;
    box-shadow: none;
    background: linear-gradient(135deg, rgba(0, 255, 255, 0.1) 0%, rgba(0, 200, 255, 0.1) 100%);
  }
  
  &::placeholder {
    color: #00cccc;
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
  box-shadow: ${props => props.$disabled ? 'none' : '0 0 15px rgba(0, 255, 255, 0.5)'};
  
  &:hover:not(:disabled) {
    transform: scale(1.1);
    box-shadow: 0 0 25px rgba(0, 255, 255, 0.8);
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
  box-shadow: 0 4px 12px rgba(76, 175, 80, 0.3);
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
  border: 1px solid #333;
  border-radius: 8px;
  margin: 12px 0;
  overflow: hidden;
`;

const CodeHeader = styled.div`
  background: #2d2d2d;
  padding: 8px 12px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid #333;
`;

const CodeLanguage = styled.span`
  color: #00ffff;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
`;

const CopyCodeButton = styled.button`
  background: none;
  border: none;
  color: #888;
  cursor: pointer;
  padding: 4px;
  border-radius: 4px;
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 0.7rem;
  transition: all 0.2s ease;
  
  &:hover {
    color: #00ffff;
    background: rgba(0, 255, 255, 0.1);
  }
`;

export const ChatPanel: React.FC<ChatPanelProps> = ({ messages, onSendMessage, onFileUpload, isLoading }) => {
  const [inputValue, setInputValue] = useState('');
  const [copyNotification, setCopyNotification] = useState(false);
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
      'text/plain': ['.txt'],
      'text/markdown': ['.md'],
      'text/html': ['.html']
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
                <CodeBlock>
                  <CodeHeader>
                    <CodeLanguage>{language}</CodeLanguage>
                    <CopyCodeButton onClick={() => copyToClipboard(codeContent)}>
                      <span className="material-icons" style={{ fontSize: '14px' }}>content_copy</span>
                    </CopyCodeButton>
                  </CodeHeader>
                  <SyntaxHighlighter
                    style={{
                      ...tomorrow,
                      // Override with neon theme
                      'pre[class*="language-"]': {
                        ...tomorrow['pre[class*="language-"]'],
                        background: '#1e1e1e',
                        margin: 0,
                        padding: '16px',
                        overflow: 'auto'
                      },
                      'code[class*="language-"]': {
                        ...tomorrow['code[class*="language-"]'],
                        fontFamily: "'Noto Sans Mono', 'Fira Code', 'Monaco', 'Consolas', monospace",
                        fontSize: '0.85rem',
                        lineHeight: 1.6
                      }
                    } as any}
                    language={language}
                    PreTag="div"
                    customStyle={{
                      background: '#1e1e1e',
                      margin: 0,
                      padding: '16px',
                      fontSize: '0.85rem',
                      fontFamily: "'Noto Sans Mono', 'Fira Code', 'Monaco', 'Consolas', monospace",
                      lineHeight: 1.6,
                      overflow: 'auto'
                    }}
                    codeTagProps={{
                      style: {
                        fontFamily: "'Noto Sans Mono', 'Fira Code', 'Monaco', 'Consolas', monospace",
                        fontSize: '0.85rem'
                      }
                    }}
                  >
                    {codeContent}
                  </SyntaxHighlighter>
                </CodeBlock>
              );
            }
            
            // Inline code with static cyan highlighting
            return (
              <code 
                className={className} 
                {...props}
                style={{
                  background: '#00ffff20',
                  padding: '3px 6px',
                  borderRadius: '4px',
                  fontSize: '0.85em',
                  fontFamily: "'Noto Sans Mono', 'Fira Code', 'Monaco', 'Consolas', monospace",
                  fontWeight: 500,
                  color: '#ffffff',
                  border: '1px solid #00ffff60',
                  boxShadow: '0 0 8px #00ffff30'
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
    <PanelContainer>

      <MessagesContainer>
        {messages.length === 0 ? (
          <EmptyState>
            <EmptyIcon>
              <Bot size={32} color="#ff1493" />
            </EmptyIcon>
            <EmptyText>Welcome to AI MATE!</EmptyText>
            <EmptyHint>
              Ask questions about your uploaded documents, request analysis, or chat about anything else. 
              I can format responses with headings, lists, code blocks, and more.
            </EmptyHint>
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
                  {message.role === 'assistant' && (
                    <CopyButton 
                      onClick={() => copyToClipboard(message.content)}
                      style={{
                        position: 'absolute',
                        top: '8px',
                        right: '8px',
                        opacity: 0,
                        transition: 'opacity 0.2s ease'
                      }}
                      className="copy-button"
                    >
                      <span className="material-icons" style={{ fontSize: '14px' }}>content_copy</span>
                    </CopyButton>
                  )}
                  <MarkdownContent>
                    {message.role === 'user' ? (
                      <p>{message.content}</p>
                    ) : (
                      renderMarkdown(message.content)
                    )}
                  </MarkdownContent>
                  <MessageTime>
                    {formatTime(message.timestamp)}
                  </MessageTime>
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
          <DragHint>Supports PDF, DOCX, TXT, MD, HTML • Max 500MB</DragHint>
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
              placeholder="Type your message or drag files here..."
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
    </PanelContainer>
  );
}; 