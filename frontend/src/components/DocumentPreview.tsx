import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { X, ExternalLink, Download, FileText } from 'lucide-react';

interface DocumentPreviewProps {
  documentId: string;
  filename: string;
  isOpen: boolean;
  onClose: () => void;
}

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';

const Overlay = styled.div<{ $isOpen: boolean }>`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.8);
  z-index: 1000;
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: ${props => props.$isOpen ? 1 : 0};
  visibility: ${props => props.$isOpen ? 'visible' : 'hidden'};
  transition: all 0.3s ease;
  padding: 20px;
`;

const PreviewContainer = styled.div<{ $isOpen: boolean }>`
  background: #1e1e1e;
  border-radius: 12px;
  width: 90vw;
  height: 90vh;
  max-width: 1200px;
  max-height: 800px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  border: 1px solid #333;
  transform: ${props => props.$isOpen ? 'scale(1) translateY(0)' : 'scale(0.95) translateY(20px)'};
  transition: all 0.3s ease;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
`;

const Header = styled.div`
  display: flex;
  align-items: center;
  justify-content: between;
  padding: 16px 20px;
  border-bottom: 1px solid #333;
  background: #252525;
`;

const Title = styled.div`
  flex: 1;
  display: flex;
  align-items: center;
  gap: 8px;
`;

const TitleText = styled.h3`
  color: #fff;
  font-size: 1rem;
  font-weight: 600;
  margin: 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 400px;
`;

const Actions = styled.div`
  display: flex;
  align-items: center;
  gap: 12px;
`;

const ActionButton = styled.button`
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid #444;
  color: #ccc;
  padding: 8px 12px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 0.9rem;
  display: flex;
  align-items: center;
  gap: 6px;
  transition: all 0.2s ease;

  &:hover {
    background: rgba(255, 255, 255, 0.15);
    border-color: #555;
    color: #fff;
  }
`;

const CloseButton = styled.button`
  background: none;
  border: none;
  color: #ccc;
  cursor: pointer;
  padding: 8px;
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s ease;

  &:hover {
    background: rgba(255, 255, 255, 0.1);
    color: #fff;
  }
`;

const PreviewContent = styled.div`
  flex: 1;
  display: flex;
  overflow: hidden;
`;

const PDFViewer = styled.iframe`
  width: 100%;
  height: 100%;
  border: none;
  background: #fff;
`;

const ImageViewer = styled.img`
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  background: #fff;
  border-radius: 4px;
`;



const PreviewFrame = styled.div`
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #f5f5f5;
  overflow: hidden;
`;



const LoadingContainer = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: #888;
  gap: 16px;
`;

const LoadingText = styled.div`
  font-size: 1rem;
  font-weight: 500;
`;

const LoadingSpinner = styled.div`
  width: 40px;
  height: 40px;
  border: 3px solid #333;
  border-top: 3px solid #00ffff;
  border-radius: 50%;
  animation: spin 1s linear infinite;

  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

const ErrorContainer = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: #ff6b6b;
  gap: 16px;
  padding: 40px;
`;

const ErrorText = styled.div`
  font-size: 1rem;
  font-weight: 500;
  text-align: center;
`;

const RetryButton = styled.button`
  background: #ff6b6b;
  border: none;
  color: white;
  padding: 10px 20px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 0.9rem;
  font-weight: 500;
  transition: all 0.2s ease;

  &:hover {
    background: #ff5252;
  }
`;

export const DocumentPreview: React.FC<DocumentPreviewProps> = ({
  documentId,
  filename,
  isOpen,
  onClose
}) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [originalUrl, setOriginalUrl] = useState<string | null>(null);
  const [documentInfo, setDocumentInfo] = useState<any>(null);
  const [isImage, setIsImage] = useState(false);

  useEffect(() => {
    if (isOpen && documentId) {
      loadPreview();
    }
  }, [isOpen, documentId]);

  const loadPreview = async () => {
    setLoading(true);
    setError(null);
    setPreviewUrl(null);
    setOriginalUrl(null);
    setDocumentInfo(null);
    setIsImage(false);

    try {
      console.log('Loading preview for document:', documentId);
      
      // Check if document exists and get info
      const infoResponse = await fetch(`${BACKEND_URL}/api/documents/${documentId}/info`);
      console.log('Info response status:', infoResponse.status);
      
      if (!infoResponse.ok) {
        throw new Error('Document not found');
      }

      const docInfo = await infoResponse.json();
      console.log('Document info:', docInfo);
      setDocumentInfo(docInfo);

      // Check if this is an image type based on content type
      const contentType = docInfo.metadata?.content_type || docInfo.contentType || '';
      const imageTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/bmp', 'image/webp', 'image/tiff'];
      const isImageFile = imageTypes.some(type => contentType.toLowerCase().includes(type.toLowerCase()));
      
      setIsImage(isImageFile);
      
      if (isImageFile) {
        // For images, set original URL for native display
        const originalImageUrl = `${BACKEND_URL}/api/documents/original/${documentId}`;
        setOriginalUrl(originalImageUrl);
        console.log('Original image URL:', originalImageUrl);
      }
      
      // Set the preview URL (PDF conversion for all files)
      const pdfUrl = `${BACKEND_URL}/api/documents/preview/${documentId}`;
      console.log('PDF Preview URL:', pdfUrl);
      
      setPreviewUrl(pdfUrl);
      setLoading(false);
    } catch (err) {
      console.error('Preview load error:', err);
      setError(err instanceof Error ? err.message : 'Failed to load document');
      setLoading(false);
    }
  };

  const handleDownload = async () => {
    try {
      // Use dedicated download endpoints that force download
      const downloadUrl = isImage 
        ? `${BACKEND_URL}/api/documents/download/${documentId}` // Original file for images
        : `${BACKEND_URL}/api/documents/download-pdf/${documentId}`; // PDF for other files
      
      console.log('Downloading from:', downloadUrl);
      
      // Use fetch to download and create blob
      const response = await fetch(downloadUrl);
      if (!response.ok) {
        throw new Error('Download failed');
      }
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      
      // Create a temporary link to download the blob
      const link = document.createElement('a');
      link.href = url;
      link.download = isImage ? filename : `${filename.replace(/\.[^/.]+$/, "")}.pdf`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      // Clean up the blob URL
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Download error:', error);
      setError('Failed to download file');
    }
  };

  const handleOpenExternal = () => {
    const openUrl = isImage && originalUrl ? originalUrl : previewUrl;
    if (openUrl) {
      window.open(openUrl, '_blank');
    }
  };

  const handleOverlayClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key === 'Escape') {
      onClose();
    }
  };

  useEffect(() => {
    if (isOpen) {
      document.addEventListener('keydown', handleKeyDown);
      return () => document.removeEventListener('keydown', handleKeyDown);
    }
  }, [isOpen]);

  return (
    <Overlay $isOpen={isOpen} onClick={handleOverlayClick}>
      <PreviewContainer $isOpen={isOpen}>
        <Header>
          <Title>
            <FileText size={20} color="#00ffff" />
            <TitleText title={filename}>{filename}</TitleText>
          </Title>
          <Actions>
            {(previewUrl || originalUrl) && (
              <>
                <ActionButton 
                  onClick={handleDownload} 
                  title={isImage ? "Download original image" : "Download PDF"}
                >
                  <Download size={16} />
                  Download
                </ActionButton>
                <ActionButton 
                  onClick={handleOpenExternal} 
                  title={isImage ? "Open original image" : "Open PDF in new tab"}
                >
                  <ExternalLink size={16} />
                  Open
                </ActionButton>
              </>
            )}
            <CloseButton onClick={onClose} title="Close preview">
              <X size={20} />
            </CloseButton>
          </Actions>
        </Header>

        <PreviewContent>
          {loading && (
            <LoadingContainer>
              <LoadingSpinner />
              <LoadingText>Loading document preview...</LoadingText>
            </LoadingContainer>
          )}

          {error && (
            <ErrorContainer>
              <FileText size={48} color="#ff6b6b" />
              <ErrorText>
                Failed to load document preview
                <br />
                {error}
              </ErrorText>
              <RetryButton onClick={loadPreview}>
                Try Again
              </RetryButton>
            </ErrorContainer>
          )}

          {!loading && !error && (
            <PreviewFrame>
              {isImage && originalUrl ? (
                <ImageViewer 
                  src={originalUrl} 
                  alt={filename}
                  onLoad={() => console.log('Image loaded successfully')}
                  onError={() => {
                    console.log('Image load failed, falling back to PDF');
                    setIsImage(false); // Fallback to PDF view
                  }}
                />
              ) : previewUrl ? (
                <PDFViewer 
                  src={`${previewUrl}#view=FitH`}
                  title={filename}
                  onLoad={() => console.log('PDF loaded successfully')}
                  onError={() => setError('Failed to load document preview')}
                />
              ) : null}
            </PreviewFrame>
          )}
        </PreviewContent>
      </PreviewContainer>
    </Overlay>
  );
}; 