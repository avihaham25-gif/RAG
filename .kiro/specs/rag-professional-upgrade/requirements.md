# Requirements Document

## Introduction

This document defines the requirements for upgrading the existing RAG (Retrieval-Augmented Generation) system to professional grade. The current system supports Hebrew document querying via a Streamlit UI and CLI interfaces, using ChromaDB for vector storage and Ollama-hosted LLMs for generation. The upgrade focuses on three core pillars: answer accuracy (Faithfulness), commitment to Hebrew language output, and graceful handling of missing information. The architectural improvements include stronger document parsing, hybrid search, reranking, and chunking optimization — all built on top of the existing codebase.

## Glossary

- **RAG_System**: The complete Retrieval-Augmented Generation pipeline including document ingestion, indexing, retrieval, and generation components
- **Indexing_Pipeline**: The subsystem responsible for loading documents, splitting them into chunks, generating embeddings, and storing them in the vector database (currently `rag_script.py`)
- **Retrieval_Pipeline**: The subsystem responsible for accepting a user query, searching the vector store, and returning relevant document chunks
- **Generation_Pipeline**: The subsystem responsible for taking retrieved context and a user query, then producing a final answer via the LLM
- **Reranker**: A cross-encoder model that rescores retrieved chunks by jointly evaluating query-chunk relevance, improving precision over initial retrieval
- **Hybrid_Search**: A retrieval strategy combining dense vector similarity search with sparse keyword-based search (BM25) to improve recall
- **Chunking_Engine**: The component responsible for splitting documents into semantically coherent segments suitable for embedding and retrieval
- **Parser**: The component responsible for extracting structured text content from source documents in various formats (DOCX, PDF, TXT)
- **Faithfulness_Score**: A metric measuring the proportion of claims in the generated answer that are directly supported by the retrieved context
- **Confidence_Score**: A numeric indicator (0.0–1.0) reflecting how well the retrieved context supports answering the user query
- **Hebrew_Response**: A generated answer written entirely in Hebrew, including transliteration of technical terms where necessary

## Requirements

### Requirement 1: Multi-Format Document Parsing

**User Story:** As a knowledge base administrator, I want the system to ingest documents in multiple formats, so that I can index all organizational knowledge regardless of file type.

#### Acceptance Criteria

1. WHEN a document with a `.docx` extension is provided, THE Parser SHALL extract its text content including headings, paragraphs, and table cell text, producing one or more text segments where each segment contains the textual content in reading order.
2. WHEN a document with a `.pdf` extension is provided, THE Parser SHALL extract its text content separating paragraphs by double newline characters (`\n\n`) so that distinct paragraph blocks in the source PDF appear as distinct text blocks in the output.
3. WHEN a document with a `.txt` extension is provided, THE Parser SHALL read its full content using UTF-8 encoding and produce it as a single text segment.
4. IF a document fails to load due to corruption or unsupported encoding, THEN THE Parser SHALL print a message to standard output containing the file path and a description of the error, skip the failed document, and continue processing remaining documents without interrupting the batch.
5. THE Parser SHALL preserve the source file path in the metadata of each extracted text segment, where a text segment is a unit of text produced by the document loader prior to any chunking or splitting.
6. IF a file has an extension other than `.docx`, `.pdf`, or `.txt`, THEN THE Parser SHALL skip that file without producing an error or halting the batch.
7. WHEN a document is loaded successfully, THE Parser SHALL produce at least one non-empty text segment (containing 1 or more characters of visible text) for that document.

### Requirement 2: Semantic Chunking Optimization

**User Story:** As a search quality engineer, I want documents to be split into semantically coherent chunks, so that retrieved results contain complete thoughts rather than fragmented text.

#### Acceptance Criteria

1. THE Chunking_Engine SHALL split documents into segments by first attempting splits at section boundaries (lines matching a heading pattern such as numbered headings, markdown-style headings, or lines of fewer than 80 characters followed by an empty line), then at paragraph boundaries (sequences of two or more consecutive newline characters), and only then applying character-length limits within a paragraph.
2. THE Chunking_Engine SHALL produce chunks with a configurable maximum size (default: 1500 characters, minimum allowed: 200 characters, maximum allowed: 10000 characters) and a configurable overlap size (default: 200 characters), where the overlap value must be less than 50% of the configured maximum chunk size.
3. WHEN a document contains explicit section headers, THE Chunking_Engine SHALL split at the section boundary unless doing so would produce a chunk exceeding the configured maximum size, in which case it SHALL split at the next paragraph boundary within that section.
4. THE Chunking_Engine SHALL attach metadata to each chunk containing: source file path, zero-based chunk index, and section title (set to the nearest preceding section header text, or an empty string if no section header precedes the chunk).
5. WHEN a chunk is produced, THE Chunking_Engine SHALL ensure it contains at least 100 characters of content, excluding overlap characters.
6. IF the remaining content at the end of a document is fewer than 100 characters, THEN THE Chunking_Engine SHALL merge that remaining content into the preceding chunk rather than emitting it as a separate chunk.

### Requirement 3: Unified Embedding Model

**User Story:** As a system maintainer, I want a single consistent embedding model used across all components, so that query embeddings align with document embeddings for accurate retrieval.

#### Acceptance Criteria

1. THE RAG_System SHALL use the `paraphrase-multilingual-MiniLM-L12-v2` embedding model for both document indexing and query encoding.
2. THE RAG_System SHALL define the embedding model name in a single shared configuration point, and all components (indexing, retrieval, and database inspection) SHALL reference that single configuration point rather than hardcoding the model name independently.
3. IF the embedding model fails to load, THEN THE RAG_System SHALL display an error message indicating the model name that failed and the reason for the failure, and SHALL halt execution without falling back to an alternative model.
4. WHEN a retrieval query is executed against an existing index, THE RAG_System SHALL verify that the embedding model used for querying matches the model that was used to generate the stored index, and IF a mismatch is detected, THEN THE RAG_System SHALL report an error indicating the expected and actual model names and refuse to return results.

### Requirement 4: Hybrid Search Retrieval

**User Story:** As a user querying in Hebrew, I want the system to combine semantic and keyword search, so that exact term matches and conceptual similarity both contribute to finding relevant documents.

#### Acceptance Criteria

1. WHEN a user submits a query, THE Retrieval_Pipeline SHALL perform both a dense vector similarity search and a sparse BM25 keyword search, each retrieving up to 50 candidate chunks independently before fusion.
2. THE Retrieval_Pipeline SHALL merge results from both search methods using Reciprocal Rank Fusion (RRF) with a configurable k parameter (default: 60, valid range: 1 to 1000) that controls rank smoothing.
3. THE Retrieval_Pipeline SHALL return a configurable number of candidate chunks (default: 20, valid range: 1 to 100) from the combined ranked list before reranking.
4. WHEN the BM25 index is queried, THE Retrieval_Pipeline SHALL tokenize the Hebrew query by splitting on whitespace and Unicode punctuation characters (including maqaf ־, geresh ׳, and gershayim ״) while preserving niqqud (diacritics) as part of the token.
5. WHEN new documents are ingested into the vector store, THE Retrieval_Pipeline SHALL rebuild or update the BM25 index to reflect the same document set before serving subsequent queries.
6. IF either the dense vector search or the BM25 search returns zero results for a query, THEN THE Retrieval_Pipeline SHALL return the results from the other search method alone, ranked by that method's score, without failing the query.
7. IF both the dense vector search and the BM25 search return zero results, THEN THE Retrieval_Pipeline SHALL return an empty result set and indicate to the caller that no matching documents were found.

### Requirement 5: Reranking of Retrieved Results

**User Story:** As a user, I want retrieved results to be reranked by relevance before they reach the LLM, so that the generation model receives the most pertinent context.

#### Acceptance Criteria

1. WHEN the Retrieval_Pipeline returns candidate chunks (up to a maximum of 50 candidates), THE Reranker SHALL score each chunk against the original user query using a cross-encoder model and produce a relevance score between 0.0 and 1.0 for each chunk.
2. WHEN the Reranker completes scoring, THE Reranker SHALL sort chunks by descending relevance score and pass the top-k chunks (configurable, minimum 1, maximum 20, default: 5) to the Generation_Pipeline.
3. IF the Retrieval_Pipeline returns fewer chunks than the configured top-k value, THEN THE Reranker SHALL pass all available scored chunks to the Generation_Pipeline without padding or error.
4. WHEN the Reranker scores a chunk, THE Reranker SHALL attach the numeric relevance score to that chunk's metadata before passing it to the Generation_Pipeline.
5. IF the Reranker model fails to initialize or does not return scores within 10 seconds, THEN THE Retrieval_Pipeline SHALL fall back to returning the top-k chunks ordered by the original retrieval ranking without reranking.

### Requirement 6: Hebrew Language Commitment

**User Story:** As a Hebrew-speaking user, I want all system responses to be in Hebrew regardless of the source document language, so that I can consume answers naturally without language switching.

#### Acceptance Criteria

1. THE Generation_Pipeline SHALL produce all responses exclusively in Hebrew, with the sole exception of proper nouns (names of people, organizations, products, or standards) that have no widely-accepted Hebrew equivalent.
2. WHEN the retrieved context contains English technical terms, THE Generation_Pipeline SHALL translate those terms into their accepted Hebrew equivalent; IF no accepted Hebrew equivalent exists, THEN THE Generation_Pipeline SHALL transliterate the term into Hebrew characters and present it in parentheses alongside a Hebrew description.
3. THE Generation_Pipeline SHALL format responses in right-to-left (RTL) text direction.
4. WHEN the user query is in any language (including Hebrew, English, or mixed), THE Generation_Pipeline SHALL respond entirely in Hebrew.
5. THE RAG_System SHALL include a Hebrew language enforcement instruction as both a system-level and a user-level prompt directive to the LLM.
6. IF a response generated by the Generation_Pipeline contains more than 2 consecutive Latin-alphabet words that are not proper nouns, THEN THE Generation_Pipeline SHALL treat this as a language enforcement failure and re-prompt the LLM with a strengthened Hebrew-only instruction.

### Requirement 7: Faithfulness and Accuracy

**User Story:** As a knowledge worker, I want the system to answer only based on the retrieved documents, so that I can trust the responses are grounded in the actual source material.

#### Acceptance Criteria

1. THE Generation_Pipeline SHALL generate answers using only information that is explicitly stated or directly paraphrasable from the retrieved context chunks, without introducing external knowledge or unsupported inferences.
2. WHEN generating an answer, THE Generation_Pipeline SHALL include a citation referencing the source document name for each factual assertion (any statement that attributes a fact, figure, date, or named entity to the source material).
3. IF the retrieved context chunks contain no information relevant to the user's question, THEN THE Generation_Pipeline SHALL respond with a predefined insufficient-information message (e.g., "המידע אינו קיים במסמכים") instead of generating an answer.
4. THE RAG_System SHALL compute a Faithfulness_Score on a scale of 0.0 to 1.0 for each generated response by decomposing the response into individual claims and verifying each claim is supported by the retrieved context.
5. WHEN the computed Faithfulness_Score falls below 0.7, THE RAG_System SHALL display the response to the user accompanied by a visible low-confidence warning indicator adjacent to the answer, informing the user that the response may not be fully supported by the source documents.

### Requirement 8: Missing Information Handling

**User Story:** As a user, I want the system to clearly tell me when it cannot find relevant information, so that I am not misled by fabricated answers.

#### Acceptance Criteria

1. WHEN the Retrieval_Pipeline returns chunks whose maximum relevance score (on a 0.0–1.0 scale) is below a configurable threshold (default: 0.3), THE RAG_System SHALL classify the query as having insufficient context.
2. WHEN a query is classified as having insufficient context, THE Generation_Pipeline SHALL respond with the predefined Hebrew message "המידע אינו קיים במסמכים" without forwarding the query to the LLM for answer generation.
3. IF the Retrieval_Pipeline returns zero chunks for a query, THEN THE RAG_System SHALL classify the query as having insufficient context.
4. WHEN context is classified as insufficient, THE RAG_System SHALL append a suggestion in Hebrew to the response indicating that rephrasing the query or adding relevant documents may improve results.
5. THE RAG_System SHALL compute a Confidence_Score (0.0–1.0) for each response by averaging the reranker relevance scores of the top-k retrieved chunks (where k matches the retrieval count configured in the Retrieval_Pipeline).
6. THE RAG_System SHALL display the Confidence_Score rounded to two decimal places alongside each response in the user interface.

### Requirement 9: Structured Response Format

**User Story:** As a user, I want responses to follow a consistent structure, so that I can quickly find the answer and its source.

#### Acceptance Criteria

1. THE Generation_Pipeline SHALL format each response as exactly two sections: a Hebrew title of no more than 10 words on the first line, followed by a single paragraph containing the answer on a new line.
2. THE Generation_Pipeline SHALL append source citations on a separate line after the answer paragraph, listing each contributing source by its document file name.
3. WHEN multiple source documents contribute to the answer, THE Generation_Pipeline SHALL list all source document file names that were used to compose the answer, separated by commas.
4. THE Generation_Pipeline SHALL NOT include introductory preambles before the answer content, including phrases that reference the retrieval process or the documents as a source (e.g., "based on the documents", "according to the sources", "from the retrieved information").
5. IF the Generation_Pipeline produces a response that does not conform to the required structure (title line, answer paragraph, citation line), THEN THE Generation_Pipeline SHALL reformat the response to match the required structure before presenting it to the user.

### Requirement 10: Indexing Pipeline Robustness

**User Story:** As a system administrator, I want the indexing process to handle large document sets reliably, so that the knowledge base stays up to date without manual intervention.

#### Acceptance Criteria

1. WHEN the Indexing_Pipeline completes processing of each document, THE Indexing_Pipeline SHALL output a progress message to stdout indicating the current document number out of total documents and the cumulative chunk count generated so far.
2. IF a single document fails during indexing, THEN THE Indexing_Pipeline SHALL skip that document, output an error message to stdout containing the file name and error description, and continue processing the remaining documents without loss of previously indexed data.
3. IF all documents in a batch fail during indexing, THEN THE Indexing_Pipeline SHALL complete without crashing, output a summary indicating zero documents were successfully indexed, and still write the manifest file recording the failures.
4. WHEN indexing completes, THE Indexing_Pipeline SHALL write a JSON manifest file to the database directory recording: the list of successfully indexed file names with their individual chunk counts, the list of failed file names with their error descriptions, total chunks generated, and an ISO 8601 UTC completion timestamp.
5. THE Indexing_Pipeline SHALL complete indexing of up to 100 documents of up to 50 pages each within 10 minutes on a machine with at least 4 CPU cores and 8 GB RAM.
