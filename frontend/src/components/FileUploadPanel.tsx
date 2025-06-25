import React, { useCallback, useState } from 'react';
import styled from 'styled-components';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, CheckCircle, AlertCircle, Loader } from 'lucide-react';

interface UploadedDocument {
  id: string;
  name: string;
  size: number;
  type: string;
  uploadedAt: Date;
  status: 'processing' | 'ready' | 'error';
}

interface FileUploadPanelProps {
  onFileUpload: (files: File[]) => Promise<any>;
  onRegisterCallback?: (callback: (file: File, result: any) => void) => void;
  uploadedDocuments?: UploadedDocument[];
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
  border: 2px dashed ${props => props.$isDragActive ? '#00ffff' : '#00ffff80'};
  border-radius: 12px;
  background: ${props => props.$isDragActive ? 
    'linear-gradient(135deg, rgba(0, 255, 255, 0.08) 0%, rgba(0, 200, 255, 0.08) 100%)' : 
    'linear-gradient(135deg, rgba(0, 255, 255, 0.03) 0%, rgba(0, 200, 255, 0.03) 100%)'
  };
  text-align: center;
  cursor: pointer;
  transition: all 0.2s ease;
  box-shadow: ${props => props.$isDragActive ? 
    '0 0 12px rgba(0, 255, 255, 0.2)' : 
    '0 0 6px rgba(0, 255, 255, 0.1)'
  };
  
  &:hover {
    border-color: #00ffff;
    background: linear-gradient(135deg, rgba(0, 255, 255, 0.06) 0%, rgba(0, 200, 255, 0.06) 100%);
    box-shadow: 0 0 10px rgba(0, 255, 255, 0.15);
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
  color: ${props => props.$isDragActive ? '#ffffff' : '#00ffff'};
  transition: all 0.3s ease;
  filter: ${props => props.$isDragActive ? 
    'drop-shadow(0 0 10px rgba(0, 255, 255, 0.8))' : 
    'drop-shadow(0 0 5px rgba(0, 255, 255, 0.4))'
  };
`;

const DropzoneText = styled.div`
  color: #00ffff;
  font-size: 0.9rem;
  line-height: 1.4;
  text-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
`;

const DropzoneHint = styled.div`
  color: #00cccc;
  font-size: 0.75rem;
  margin-top: 4px;
  opacity: 0.8;
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
`;

const DocumentsTitle = styled.h3`
  color: #00ffff;
  font-size: 0.9rem;
  font-weight: 600;
  margin: 0;
  display: flex;
  align-items: center;
  gap: 8px;
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
  align-items: center;
  gap: 12px;
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
  flex-shrink: 0;
`;

const DocumentInfo = styled.div`
  flex: 1;
  min-width: 0;
`;

const DocumentName = styled.div`
  color: #fff;
  font-size: 0.85rem;
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



export const FileUploadPanel: React.FC<FileUploadPanelProps> = ({ 
  onFileUpload,
  uploadedDocuments = []
}) => {
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
    multiple: true
  });

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
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
            Supports PDF, DOCX, TXT, MD, HTML • Max 500MB
          </DropzoneHint>
          <DropzoneHint style={{ marginTop: '8px', fontSize: '0.7rem', opacity: 0.6 }}>
            📋 Processing details available in the Console
          </DropzoneHint>
        </DropzoneContent>
      </DropzoneContainer>

      {/* Documents Section */}
      <DocumentsSection>
        <DocumentsHeader>
          <DocumentsTitle>
            <FileText size={16} />
            Documents ({uploadedDocuments.length})
          </DocumentsTitle>
        </DocumentsHeader>
        
        <DocumentsList>
          {uploadedDocuments.length > 0 ? (
            uploadedDocuments.map((doc) => (
              <DocumentItem key={doc.id}>
                <DocumentIcon $status={doc.status}>
                  {doc.status === 'ready' && <CheckCircle size={16} />}
                  {doc.status === 'error' && <AlertCircle size={16} />}
                  {doc.status === 'processing' && <Loader size={16} />}
                </DocumentIcon>
                <DocumentInfo>
                  <DocumentName>{doc.name}</DocumentName>
                  <DocumentMeta>
                    {formatFileSize(doc.size)} • {doc.uploadedAt.toLocaleDateString()}
                  </DocumentMeta>
                </DocumentInfo>
              </DocumentItem>
            ))
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

      {/* Add CSS animation for spinning */}
      <style>
        {`
          @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
          }
        `}
      </style>
    </PanelContainer>
  );
}; 