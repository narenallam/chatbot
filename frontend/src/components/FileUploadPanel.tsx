import React, { useCallback, useState } from 'react';
import styled from 'styled-components';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, CheckCircle, AlertCircle, Loader, RotateCcw, FileImage, FileSpreadsheet, Presentation } from 'lucide-react';
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

interface FileUploadPanelProps {
  onFileUpload: (files: File[]) => Promise<any>;
  onRegisterCallback?: (callback: (file: File, result: any) => void) => void;
  uploadedDocuments?: UploadedDocument[];
  onRefreshDocuments?: () => Promise<void>;
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

const EnhancedDropzoneHint = styled.div`
  color: #00ffff;
  font-size: 0.7rem;
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
  color: #888888;
  font-size: 0.9rem;
  font-weight: 600;
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
  align-items: flex-start;
  gap: 12px;
  transition: all 0.2s ease;
  cursor: pointer;
  
  &:hover {
    background: rgba(255, 255, 255, 0.06);
    border-color: #444;
  }
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

const DocumentHash = styled.div`
  color: #666;
  font-size: 0.65rem;
  font-family: monospace;
  margin-top: 2px;
  opacity: 0.8;
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

export const FileUploadPanel: React.FC<FileUploadPanelProps> = ({ 
  onFileUpload,
  uploadedDocuments = [],
  onRefreshDocuments
}) => {
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [selectedDocument, setSelectedDocument] = useState<UploadedDocument | null>(null);

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
    onDrop,
    accept: {
      // Enhanced document service supported file types
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'application/vnd.openxmlformats-officedocument.presentationml.presentation': ['.pptx'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'image/png': ['.png'],
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/heic': ['.heic']
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
      default: return 'Document';
    }
  };

  const handleDocumentClick = (document: UploadedDocument) => {
    setSelectedDocument(document);
  };

  const handleCloseDetails = () => {
    setSelectedDocument(null);
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
            Supports PDF, DOCX, PPTX, XLSX, PNG, JPG, HEIC • Max 500MB
          </DropzoneHint>
          <EnhancedDropzoneHint>
            Enhanced OCR processing for images • Duplicate detection enabled
          </EnhancedDropzoneHint>
        </DropzoneContent>
      </DropzoneContainer>

      {/* Documents Section */}
      <DocumentsSection>
        <DocumentsHeader>
          <DocumentsTitle>
            <FileText size={16} />
            Documents ({uploadedDocuments.length})
          </DocumentsTitle>
          <RefreshButton onClick={handleRefresh} disabled={!onRefreshDocuments || isRefreshing}>
            {isRefreshing ? <Loader size={12} /> : <RotateCcw size={12} />}
            Refresh
          </RefreshButton>
        </DocumentsHeader>
        
        <DocumentsList>
          {uploadedDocuments.length > 0 ? (
            uploadedDocuments.map((doc) => (
              <DocumentItem key={doc.id} onClick={() => handleDocumentClick(doc)}>
                <DocumentIcon $status={doc.status}>
                  {doc.status === 'ready' && <CheckCircle size={20} />}
                  {doc.status === 'error' && <AlertCircle size={20} />}
                  {doc.status === 'processing' && <Loader size={20} />}
                </DocumentIcon>
                <DocumentInfo>
                  <DocumentName>
                    <FileTypeIcon>
                      {getFileTypeIcon(doc.name)}
                    </FileTypeIcon>
                    {doc.name || 'Unknown Document'}
                  </DocumentName>
                  <DocumentMeta>
                    {formatFileSize(doc.size || 0)} • {getFileTypeName(doc.name)} • {formatUploadDate(doc.uploadedAt)}
                    {doc.newFileName && (
                      <span> • Stored as: {doc.newFileName}</span>
                    )}
                  </DocumentMeta>
                  {doc.fileHash && (
                    <DocumentHash>
                      Hash: {doc.fileHash.substring(0, 16)}...
                    </DocumentHash>
                  )}
                  {doc.status === 'error' && doc.errorMessage && (
                    <DocumentError>
                      Error: {doc.errorMessage}
                    </DocumentError>
                  )}
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
    </PanelContainer>
  );
}; 