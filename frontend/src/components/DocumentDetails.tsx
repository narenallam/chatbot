import React from 'react';
import styled from 'styled-components';
import { FileText, Hash, Calendar, HardDrive, FileImage, FileSpreadsheet, Presentation, CheckCircle, AlertCircle, Info } from 'lucide-react';

interface DocumentDetailsProps {
  document: {
    id: string;
    name: string;
    size: number;
    type: string;
    uploadedAt: Date | string;
    status: 'processing' | 'ready' | 'error';
    fullFileName?: string;
    fileHash?: string;
    newFileName?: string;
    fileDataHash?: string;
    contentType?: string;
    metadata?: any;
    errorMessage?: string;
  };
  onClose: () => void;
}

const Overlay = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.8);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
`;

const Modal = styled.div`
  background: #1e1e1e;
  border: 1px solid #333;
  border-radius: 12px;
  padding: 24px;
  max-width: 500px;
  width: 90%;
  max-height: 80vh;
  overflow-y: auto;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
`;

const Header = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 20px;
  padding-bottom: 16px;
  border-bottom: 1px solid #333;
`;

const Title = styled.h3`
  color: #00ffff;
  font-size: 1.1rem;
  font-weight: 600;
  margin: 0;
  display: flex;
  align-items: center;
  gap: 8px;
`;

const CloseButton = styled.button`
  background: none;
  border: none;
  color: #888;
  cursor: pointer;
  padding: 4px;
  border-radius: 4px;
  transition: all 0.2s ease;
  
  &:hover {
    color: #fff;
    background: rgba(255, 255, 255, 0.1);
  }
`;

const Content = styled.div`
  display: flex;
  flex-direction: column;
  gap: 16px;
`;

const Section = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
`;

const SectionTitle = styled.div`
  color: #888;
  font-size: 0.8rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  display: flex;
  align-items: center;
  gap: 6px;
`;

const InfoGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr;
  gap: 8px;
`;

const InfoItem = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background: rgba(255, 255, 255, 0.03);
  border-radius: 6px;
  border: 1px solid #333;
`;

const InfoLabel = styled.span`
  color: #ccc;
  font-size: 0.8rem;
  min-width: 80px;
`;

const InfoValue = styled.span`
  color: #fff;
  font-size: 0.8rem;
  font-weight: 500;
  flex: 1;
`;

const HashValue = styled.span`
  color: #00ffff;
  font-size: 0.7rem;
  font-family: monospace;
  background: rgba(0, 255, 255, 0.1);
  padding: 2px 6px;
  border-radius: 4px;
  border: 1px solid rgba(0, 255, 255, 0.2);
`;

const StatusBadge = styled.div<{ status: 'processing' | 'ready' | 'error' }>`
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 4px 8px;
  border-radius: 12px;
  font-size: 0.7rem;
  font-weight: 500;
  background: ${props => {
    switch (props.status) {
      case 'ready': return 'rgba(0, 255, 0, 0.1)';
      case 'error': return 'rgba(255, 68, 68, 0.1)';
      default: return 'rgba(255, 170, 0, 0.1)';
    }
  }};
  color: ${props => {
    switch (props.status) {
      case 'ready': return '#00ff00';
      case 'error': return '#ff4444';
      default: return '#ffaa00';
    }
  }};
  border: 1px solid ${props => {
    switch (props.status) {
      case 'ready': return 'rgba(0, 255, 0, 0.3)';
      case 'error': return 'rgba(255, 68, 68, 0.3)';
      default: return 'rgba(255, 170, 0, 0.3)';
    }
  }};
`;

const ErrorMessage = styled.div`
  color: #ff6666;
  font-size: 0.8rem;
  background: rgba(255, 102, 102, 0.1);
  border: 1px solid rgba(255, 102, 102, 0.3);
  border-radius: 6px;
  padding: 12px;
  display: flex;
  align-items: flex-start;
  gap: 8px;
`;

const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
};

const formatUploadDate = (uploadedAt: Date | string): string => {
  try {
    const date = uploadedAt instanceof Date ? uploadedAt : new Date(uploadedAt);
    return date.toLocaleString();
  } catch (error) {
    return 'Unknown date';
  }
};

const getFileTypeIcon = (fileName: string) => {
  const ext = fileName.toLowerCase().split('.').pop();
  switch (ext) {
    case 'pdf':
      return <FileText size={16} />;
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
    default:
      return <FileText size={16} />;
  }
};

const getFileTypeName = (fileName: string) => {
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
    default: return 'Document';
  }
};

export const DocumentDetails: React.FC<DocumentDetailsProps> = ({ document, onClose }) => {
  return (
    <Overlay onClick={onClose}>
      <Modal onClick={(e) => e.stopPropagation()}>
        <Header>
          <Title>
            {getFileTypeIcon(document.name)}
            Document Details
          </Title>
          <CloseButton onClick={onClose}>✕</CloseButton>
        </Header>
        
        <Content>
          <Section>
            <SectionTitle>
              <FileText size={14} />
              File Information
            </SectionTitle>
            <InfoGrid>
              <InfoItem>
                <InfoLabel>Name:</InfoLabel>
                <InfoValue>{document.name}</InfoValue>
              </InfoItem>
              <InfoItem>
                <InfoLabel>Type:</InfoLabel>
                <InfoValue>{getFileTypeName(document.name)}</InfoValue>
              </InfoItem>
              <InfoItem>
                <InfoLabel>Size:</InfoLabel>
                <InfoValue>{formatFileSize(document.size)}</InfoValue>
              </InfoItem>
              <InfoItem>
                <InfoLabel>Status:</InfoLabel>
                <StatusBadge status={document.status}>
                  {document.status === 'ready' && <CheckCircle size={12} />}
                  {document.status === 'error' && <AlertCircle size={12} />}
                  {document.status === 'processing' && <Info size={12} />}
                  {document.status.charAt(0).toUpperCase() + document.status.slice(1)}
                </StatusBadge>
              </InfoItem>
              <InfoItem>
                <InfoLabel>Uploaded:</InfoLabel>
                <InfoValue>{formatUploadDate(document.uploadedAt)}</InfoValue>
              </InfoItem>
            </InfoGrid>
          </Section>
          
          {document.newFileName && (
            <Section>
              <SectionTitle>
                <HardDrive size={14} />
                Storage Information
              </SectionTitle>
              <InfoGrid>
                <InfoItem>
                  <InfoLabel>Stored as:</InfoLabel>
                  <InfoValue>{document.newFileName}</InfoValue>
                </InfoItem>
                {document.fullFileName && (
                  <InfoItem>
                    <InfoLabel>Full name:</InfoLabel>
                    <InfoValue>{document.fullFileName}</InfoValue>
                  </InfoItem>
                )}
              </InfoGrid>
            </Section>
          )}
          
          {(document.fileHash || document.fileDataHash) && (
            <Section>
              <SectionTitle>
                <Hash size={14} />
                Hash Information
              </SectionTitle>
              <InfoGrid>
                {document.fileHash && (
                  <InfoItem>
                    <InfoLabel>File Hash:</InfoLabel>
                    <HashValue>{document.fileHash}</HashValue>
                  </InfoItem>
                )}
                {document.fileDataHash && (
                  <InfoItem>
                    <InfoLabel>Data Hash:</InfoLabel>
                    <HashValue>{document.fileDataHash}</HashValue>
                  </InfoItem>
                )}
              </InfoGrid>
            </Section>
          )}
          
          {document.status === 'error' && document.errorMessage && (
            <Section>
              <SectionTitle>
                <AlertCircle size={14} />
                Error Information
              </SectionTitle>
              <ErrorMessage>
                <AlertCircle size={16} />
                {document.errorMessage}
              </ErrorMessage>
            </Section>
          )}
        </Content>
      </Modal>
    </Overlay>
  );
}; 