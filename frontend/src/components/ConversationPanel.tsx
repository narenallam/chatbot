import React from 'react';
import styled from 'styled-components';
import { MessageSquare, MessageCircle, Trash2, Clock, Plus } from 'lucide-react';

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  neonColor?: string;
}

interface Conversation {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: Date;
  updatedAt: Date;
}

interface ConversationPanelProps {
  conversations: Conversation[];
  currentConversationId: string | null;
  onLoadConversation: (conversationId: string) => void;
  onNewConversation: () => void;
  onClearChat: () => void;
}

const PanelContainer = styled.div`
  background: #1e1e1e;
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
  justify-content: center;
  gap: 8px;
`;

const Controls = styled.div`
  padding: 0 20px 20px;
  border-bottom: 1px solid #333;
`;

const ClearButton = styled.button`
  background: rgba(255, 68, 68, 0.1);
  border: 1px solid rgba(255, 68, 68, 0.3);
  color: #ff4444;
  border-radius: 25px;
  padding: 8px 12px;
  font-size: 0.8rem;
  cursor: pointer;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  gap: 6px;
  width: 100%;
  justify-content: center;
  
  &:hover {
    background: rgba(255, 68, 68, 0.2);
    border-color: rgba(255, 68, 68, 0.5);
  }
  
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

const HistoryContainer = styled.div`
  flex: 1;
  padding: 20px;
  overflow-y: auto;
  overflow-x: hidden;
  
  /* Custom scrollbar styling */
  &::-webkit-scrollbar {
    width: 8px;
    background: #1e1e1e;
  }
  
  &::-webkit-scrollbar-track {
    background: #1e1e1e;
    border-radius: 4px;
    margin: 4px;
  }
  
  &::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #444 0%, #333 100%);
    border-radius: 4px;
    border: 1px solid #1e1e1e;
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
    background: #1e1e1e;
  }
`;

const HistoryItem = styled.div`
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid #333;
  border-radius: 8px;
  padding: 12px;
  margin-bottom: 8px;
  cursor: pointer;
  transition: all 0.2s ease;
  
  &:hover {
    background: rgba(255, 255, 255, 0.05);
    border-color: #444;
  }
`;

const HistoryPreview = styled.div`
  color: #fff;
  font-size: 0.8rem;
  line-height: 1.3;
  margin-bottom: 6px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
`;

const HistoryMeta = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 0.7rem;
  color: #888;
`;

const MessageCount = styled.span`
  display: flex;
  align-items: center;
  gap: 4px;
`;

const TimeStamp = styled.span`
  display: flex;
  align-items: center;
  gap: 4px;
`;

const EmptyState = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: #666;
  text-align: center;
  gap: 12px;
`;

const EmptyIcon = styled(MessageSquare)`
  width: 32px;
  height: 32px;
  color: #444;
`;

const EmptyText = styled.div`
  font-size: 0.9rem;
  color: #555;
`;

const Stats = styled.div`
  padding: 20px;
  border-top: 1px solid #333;
  background: rgba(255, 255, 255, 0.02);
`;

const StatItem = styled.div`
  display: flex;
  justify-content: space-between;
  margin: 8px 0;
  font-size: 0.8rem;
`;

const StatLabel = styled.span`
  color: #888;
`;

const StatValue = styled.span`
  color: #00ffff;
  font-weight: 500;
`;

const NewConversationButton = styled.button`
  background: rgba(0, 255, 255, 0.1);
  border: 1px solid rgba(0, 255, 255, 0.3);
  color: #00ffff;
  border-radius: 25px;
  padding: 6px 10px;
  font-size: 0.75rem;
  cursor: pointer;
  transition: all 0.2s ease;
  display: flex;
  align-items: center;
  gap: 4px;
  width: 80%;
  justify-content: center;
  margin: 0 auto 8px auto;
  
  &:hover {
    background: rgba(0, 255, 255, 0.2);
    border-color: rgba(0, 255, 255, 0.5);
    box-shadow: 0 0 10px rgba(0, 255, 255, 0.3);
  }
`;

const DateHeader = styled.div`
  color: #888;
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin: 16px 0 8px 0;
  padding: 0 4px;
`;

const ConversationItem = styled.div<{ $isActive: boolean; $neonColor: string }>`
  background: ${props => props.$isActive ? 
    `linear-gradient(135deg, ${props.$neonColor}15 0%, ${props.$neonColor}05 100%)` : 
    'rgba(255, 255, 255, 0.03)'
  };
  border: 1px solid ${props => props.$isActive ? 
    `${props.$neonColor}80` : 
    '#333'
  };
  border-radius: 8px;
  padding: 12px;
  margin-bottom: 8px;
  cursor: pointer;
  transition: all 0.2s ease;
  box-shadow: ${props => props.$isActive ? 
    `0 0 15px ${props.$neonColor}30` : 
    'none'
  };
  
  &:hover {
    background: ${props => 
      `linear-gradient(135deg, ${props.$neonColor}20 0%, ${props.$neonColor}10 100%)`
    };
    border-color: ${props => `${props.$neonColor}60`};
    box-shadow: 0 0 12px ${props => `${props.$neonColor}25`};
  }
`;

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

export const ConversationPanel: React.FC<ConversationPanelProps> = ({ 
  conversations, 
  currentConversationId, 
  onLoadConversation, 
  onNewConversation, 
  onClearChat 
}) => {
  const groupConversationsByTime = () => {
    const groups: { [key: string]: Conversation[] } = {};
    
    conversations.forEach(conversation => {
      const date = conversation.updatedAt.toDateString();
      if (!groups[date]) {
        groups[date] = [];
      }
      groups[date].push(conversation);
    });
    
    return Object.entries(groups).sort(([a], [b]) => 
      new Date(b).getTime() - new Date(a).getTime()
    );
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const today = new Date();
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);
    
    if (date.toDateString() === today.toDateString()) {
      return 'Today';
    } else if (date.toDateString() === yesterday.toDateString()) {
      return 'Yesterday';
    } else {
      return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
    }
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  const groupedConversations = groupConversationsByTime();
  const totalConversations = conversations.length;
  const totalMessages = conversations.reduce((sum, conv) => sum + conv.messages.length, 0);
  const currentConversation = conversations.find(c => c.id === currentConversationId);

  return (
    <PanelContainer>
      <Header>
        <Title>
          <MessageCircle size={16} />
          Conversations
        </Title>
      </Header>
      
      <Controls>
        <NewConversationButton onClick={onNewConversation}>
          <Plus size={12} />
          New Chat
        </NewConversationButton>
      </Controls>

      <HistoryContainer>
        {groupedConversations.length === 0 ? (
          <EmptyState>
            <EmptyIcon />
            <EmptyText>No conversations yet</EmptyText>
          </EmptyState>
        ) : (
          groupedConversations.map(([date, convs], groupIndex) => (
            <div key={date}>
              <DateHeader>{formatDate(date)}</DateHeader>
              {convs.map((conversation, index) => (
                <ConversationItem 
                  key={conversation.id}
                  $isActive={conversation.id === currentConversationId}
                  $neonColor={NEON_COLORS[(groupIndex * convs.length + index) % NEON_COLORS.length]}
                  onClick={() => onLoadConversation(conversation.id)}
                >
                  <HistoryPreview>{conversation.title}</HistoryPreview>
                  <HistoryMeta>
                    <MessageCount>
                      <MessageSquare size={10} />
                      {conversation.messages.length}
                    </MessageCount>
                    <TimeStamp>
                      <Clock size={10} />
                      {formatTime(conversation.updatedAt)}
                    </TimeStamp>
                  </HistoryMeta>
                </ConversationItem>
              ))}
            </div>
          ))
        )}
      </HistoryContainer>

      {totalConversations > 0 && (
        <Stats>
          <StatItem>
            <StatLabel>Conversations:</StatLabel>
            <StatValue>{totalConversations}</StatValue>
          </StatItem>
          <StatItem>
            <StatLabel>Total Messages:</StatLabel>
            <StatValue>{totalMessages}</StatValue>
          </StatItem>
        </Stats>
      )}
    </PanelContainer>
  );
}; 