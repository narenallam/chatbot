import React from 'react';
import styled from 'styled-components';
import { FileText, ExternalLink } from 'lucide-react';

interface DocumentSource {
  document_id: string;
  filename: string;
  chunk_text: string;
  similarity_score: number;
}

interface DocumentSourcesProps {
  sources: DocumentSource[];
  onDocumentClick: (documentId: string, filename: string) => void;
}

const SourcesContainer = styled.div`
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
`;

const SourcesHeader = styled.div`
  color: #888;
  font-size: 0.8rem;
  font-weight: 600;
  margin-bottom: 8px;
  display: flex;
  align-items: center;
  gap: 6px;
`;

const SourcesList = styled.div`
  display: flex;
  flex-direction: column;
  gap: 6px;
`;

const SourceItem = styled.button`
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 17px;
  padding: 8px 10px;
  transition: all 0.2s ease;
  width: 100%;
  text-align: left;
  cursor: pointer;
  display: block;
  outline: none;
  border-width: 1px;
  border-style: solid;
  
  &:hover, &:focus {
    background: rgba(255, 255, 255, 0.06);
    border-color: rgba(0, 255, 255, 0.3);
  }
`;

const SourceHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 4px;
`;

const SourceFilename = styled.button`
  background: none;
  border: none;
  color: #00ffff;
  font-size: 0.8rem;
  font-weight: 500;
  cursor: pointer;
  padding: 0;
  display: flex;
  align-items: center;
  gap: 4px;
  text-align: left;
  transition: all 0.2s ease;
  
  &:hover {
    color: #ffffff;
    text-decoration: underline;
  }
`;

const SourceScore = styled.span`
  color: #666;
  font-size: 0.7rem;
  font-family: "Noto Sans Mono", monospace !important;
  font-weight: 300 !important;
`;

const SourcePreview = styled.div`
  color: #aaa;
  font-size: 0.75rem;
  line-height: 1.3;
  overflow: hidden;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  margin-top: 4px;
`;

const NoSources = styled.div`
  color: #666;
  font-size: 0.8rem;
  font-style: italic;
  text-align: center;
  padding: 8px;
`;

export const DocumentSources: React.FC<DocumentSourcesProps> = ({
  sources,
  onDocumentClick
}) => {
  if (!sources || sources.length === 0) {
    return null;
  }

  const handleDocumentClick = (documentId: string, filename: string) => {
    onDocumentClick(documentId, filename);
  };

  // Remove duplicates based on document_id
  const uniqueSources = sources.reduce((acc, current) => {
    const existing = acc.find(item => item.document_id === current.document_id);
    if (!existing) {
      acc.push(current);
    }
    return acc;
  }, [] as DocumentSource[]);

  return (
    <SourcesContainer>
      <SourcesHeader>
        <FileText size={14} />
        Referenced Documents ({uniqueSources.length})
      </SourcesHeader>
      
      <SourcesList>
        {uniqueSources.map((source, index) => (
          <SourceItem
            key={`${source.document_id}-${index}`}
            onClick={() => handleDocumentClick(source.document_id, source.filename)}
            title={`Click to preview ${source.filename}`}
          >
            <SourceHeader>
              <span style={{ display: 'flex', alignItems: 'center', gap: 4, color: '#00ffff', fontWeight: 300 }}>
                <FileText size={12} />
                {source.filename}
                <ExternalLink size={10} />
              </span>
              <SourceScore>
                {(source.similarity_score * 100).toFixed(1)}%
              </SourceScore>
            </SourceHeader>
            <SourcePreview title={source.chunk_text}>
              {source.chunk_text}
            </SourcePreview>
          </SourceItem>
        ))}
      </SourcesList>
    </SourcesContainer>
  );
}; 