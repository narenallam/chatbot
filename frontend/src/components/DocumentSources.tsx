import React from 'react';
import styled from 'styled-components';
import { FileText, ExternalLink, Globe, Clock, Star } from 'lucide-react';

interface DocumentSource {
  document_id?: string | null;
  filename: string;
  chunk_text: string;
  similarity_score: number;
  source_type?: string;
  url?: string;
  provider?: string;
  is_recent?: boolean;
  authority_score?: number;
}

interface DocumentSourcesProps {
  sources: DocumentSource[];
  onDocumentClick: (documentId: string, filename: string) => void;
}

const WebSourceItem = styled.a`
  background: rgba(0, 255, 255, 0.05);
  border: 1px solid rgba(0, 255, 255, 0.15);
  border-radius: 17px;
  padding: 8px 10px;
  transition: all 0.2s ease;
  width: 100%;
  text-align: left;
  cursor: pointer;
  display: block;
  outline: none;
  text-decoration: none;
  color: inherit;
  
  &:hover, &:focus {
    background: rgba(0, 255, 255, 0.1);
    border-color: rgba(0, 255, 255, 0.4);
    text-decoration: none;
    color: inherit;
  }
`;

const ProviderBadge = styled.span`
  background: rgba(0, 255, 255, 0.2);
  color: #00ffff;
  padding: 2px 6px;
  border-radius: 8px;
  font-size: 0.65rem;
  font-weight: 500;
  margin-left: 6px;
`;

const RecentBadge = styled.span`
  background: rgba(255, 165, 0, 0.2);
  color: #ffa500;
  padding: 2px 6px;
  border-radius: 8px;
  font-size: 0.65rem;
  font-weight: 500;
  margin-left: 4px;
  display: flex;
  align-items: center;
  gap: 2px;
`;

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

const NoSourcesMessage = styled.div`
  padding: 12px 16px;
  color: #888;
  font-style: italic;
  font-size: 0.9em;
  text-align: center;
  border: 1px dashed #333;
  border-radius: 4px;
  margin-top: 8px;
`;

const SourceSectionHeader = styled.div`
  font-weight: 500;
  color: #00ffff;
  font-size: 0.9em;
  margin: 12px 0 8px 0;
  padding-bottom: 4px;
  border-bottom: 1px solid #333;
`;

export const DocumentSources: React.FC<DocumentSourcesProps> = ({
  sources,
  onDocumentClick
}) => {
  const handleDocumentClick = (documentId: string, filename: string) => {
    if (documentId) {
      onDocumentClick(documentId, filename);
    }
  };

  // Separate web and document sources
  const documentSources = sources?.filter(s => s.source_type === 'document' && s.document_id) || [];
  const webSources = sources?.filter(s => s.source_type === 'web_search') || [];
  
  // Remove duplicates for documents based on document_id
  const uniqueDocSources = documentSources.reduce((acc, current) => {
    const existing = acc.find(item => item.document_id === current.document_id);
    if (!existing) {
      acc.push(current);
    }
    return acc;
  }, [] as DocumentSource[]);

  // Remove duplicates for web sources based on URL
  const uniqueWebSources = webSources.reduce((acc, current) => {
    const existing = acc.find(item => item.url === current.url);
    if (!existing) {
      acc.push(current);
    }
    return acc;
  }, [] as DocumentSource[]);

  const totalSources = uniqueDocSources.length + uniqueWebSources.length;

  // Don't render anything if no sources
  if (totalSources === 0) {
    return null;
  }

  return (
    <SourcesContainer>
      <SourcesHeader>
        <FileText size={14} />
        Referenced Sources ({totalSources})
      </SourcesHeader>
      
      <SourcesList>
        {/* Document Sources */}
        {uniqueDocSources.length > 0 && (
          <>
            <SourceSectionHeader>📄 Document Sources ({uniqueDocSources.length})</SourceSectionHeader>
            {uniqueDocSources.map((source, index) => (
              <SourceItem
                key={`doc-${source.document_id}-${index}`}
                onClick={() => handleDocumentClick(source.document_id!, source.filename)}
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
          </>
        )}
        
        {/* Web Sources */}
        {uniqueWebSources.length > 0 && (
          <>
            <SourceSectionHeader>🌐 Web Sources ({uniqueWebSources.length})</SourceSectionHeader>
            {uniqueWebSources.map((source, index) => (
              <WebSourceItem
                key={`web-${source.url}-${index}`}
                href={source.url}
                target="_blank"
                rel="noopener noreferrer"
                title={`Open ${source.filename} in new tab`}
              >
                <SourceHeader>
                  <span style={{ display: 'flex', alignItems: 'center', gap: 4, color: '#00ffff', fontWeight: 300, flexWrap: 'wrap' }}>
                    <Globe size={12} />
                    {source.filename}
                    <ExternalLink size={10} />
                    {source.provider && <ProviderBadge>{source.provider}</ProviderBadge>}
                    {source.is_recent && (
                      <RecentBadge>
                        <Clock size={10} />
                        Recent
                      </RecentBadge>
                    )}
                  </span>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    {source.authority_score && source.authority_score > 0.7 && (
                      <Star size={10} style={{ color: '#ffa500' }} />
                    )}
                    <SourceScore>
                      {(source.similarity_score * 100).toFixed(1)}%
                    </SourceScore>
                  </div>
                </SourceHeader>
                <SourcePreview title={source.chunk_text}>
                  {source.chunk_text}
                </SourcePreview>
              </WebSourceItem>
            ))}
          </>
        )}
      </SourcesList>
    </SourcesContainer>
  );
}; 