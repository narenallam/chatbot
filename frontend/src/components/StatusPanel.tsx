import React from 'react';
import styled from 'styled-components';
import { SystemStatus } from '../App';

interface StatusPanelProps {
  status: SystemStatus;
}

const PanelContainer = styled.div`
  display: flex;
  align-items: center;
  gap: 20px;
  font-size: 0.8rem;
`;

const StatusItem = styled.div`
  display: flex;
  align-items: center;
  gap: 6px;
  position: relative;

  &:not(:last-child)::after {
    content: '';
    position: absolute;
    right: -10px;
    width: 1px;
    height: 12px;
    background: #333;
  }
`;

const StatusLabel = styled.span`
  color: #ccc;
  font-size: 0.75rem;
`;

const StatusValue = styled.span`
  color: #00ffff;
  font-size: 0.75rem;
  font-weight: 500;
`;

const StatusDot = styled.span<{ status: 'online' | 'offline' }>`
  width: 6px;
  height: 6px;
  border-radius: 50%;
  display: inline-block;
  background: ${props => props.status === 'online' ? '#00ff00' : '#ff4444'};
  box-shadow: 0 0 4px ${props => props.status === 'online' ? '#00ff0066' : '#ff444466'};
`;

export const StatusPanel: React.FC<StatusPanelProps> = ({ status }) => {
  return (
    <PanelContainer>
      <StatusItem>
        <StatusDot status={status.backend} />
        <StatusLabel>Backend</StatusLabel>
        <StatusValue>
          {status.backend === 'online' ? 'Online' : 'Offline'}
        </StatusValue>
      </StatusItem>
      
      <StatusItem>
        <StatusDot status={status.aiModels} />
        <StatusLabel>AI Models</StatusLabel>
        <StatusValue>
          {status.aiModels === 'online' ? 'Ready' : 'Offline'}
        </StatusValue>
      </StatusItem>
      
      <StatusItem>
        <StatusDot status="online" />
        <StatusLabel>Documents</StatusLabel>
        <StatusValue>{status.documents}</StatusValue>
      </StatusItem>

      <StatusItem>
        <StatusDot status="online" />
        <StatusLabel>Uploaded</StatusLabel>
        <StatusValue>{status.uploadedDocuments}</StatusValue>
      </StatusItem>
      
      <StatusItem>
        <StatusDot status={status.isProcessing ? 'online' : 'offline'} />
        <StatusLabel>CPU Cores</StatusLabel>
        <StatusValue>
          {status.cpuCores}
          {status.isProcessing && status.cpuUsage ? ` (${status.cpuUsage}%)` : ''}
        </StatusValue>
      </StatusItem>
    </PanelContainer>
  );
}; 