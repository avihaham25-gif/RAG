# Implementation Plan: RAG Professional Upgrade

## Overview

This implementation plan follows a Progressive Disclosure approach, breaking the RAG system upgrade into atomic, independently implementable tasks organized in layers. Each layer builds on the previous one: Foundation → Ingestion → Retrieval → Generation → Integration → Testing. All code is in Python, using the existing ecosystem (ChromaDB, LangChain, Ollama, Streamlit).

## Tasks

- [ ] 1. Foundation Layer — Configuration and Shared Abstractions
  - [ ] 1.1 Create project package structure and config module
    - Create `config.py` with `RAGConfig` dataclass containing all parameters (embedding model, chunk sizes, retrieval counts, thresholds, paths, LLM settings)
    - Implement validation logic for parameter ranges (max_chunk_size 200-10000, overlap < 50% of max, rrf_k 1-1000, etc.)
    - Create `parsers/__init__.py`, `chunking/__init__.py`, `indexing/__init__.py`, `retrieval/__init__.py`, `generation/__init__.py`
    - _Requirements: 3.2, 2.2, 4.2, 4.3, 5.2, 8.1, 8.5_

  - [ ] 1.2 Define core data models and base parser interface
    - Create `parsers/base.py` with `TextSegment` dataclass and `BaseParser` ABC (with `parse` and `supports` abstract methods)
    - Create shared data types: `Chunk` (content, metadata with source/chunk_index/section_title), `ScoredChunk`, `GenerationResult`, `IndexingManifest`
    - _Requirements: 1.5, 2.4_

- [ ] 2. Ingestion Layer — Parsers
  - [ ] 2.1 Implement TXT parser
    - Create `parsers/txt_parser.py` implementing `BaseParser`
    - Read file with UTF-8 encoding, produce single `TextSegment`
    - Handle encoding errors gracefully (print message, skip)
    - _Requirements: 1.3, 1.4, 1.7_

  - [ ] 2.2 Implement DOCX parser
    - Create `parsers/docx_parser.py` implementing `BaseParser` using `python-docx`
    - Extract headings, paragraphs, and table cell text in reading order
    - Handle corrupt files gracefully (print message, skip)
    - _Requirements: 1.1, 1.4, 1.5, 1.7_

  - [ ] 2.3 Implement PDF parser
    - Create `parsers/pdf_parser.py` implementing `BaseParser` using `pypdf`
    - Extract text with paragraph separation via double newlines
    - Handle corrupt/encrypted files gracefully (print message, skip)
    - _Requirements: 1.2, 1.4, 1.5, 1.7_

  - [ ] 2.4 Implement parser registry and batch processing
    - Create a parser dispatcher that routes files by extension to the correct parser
    - Implement batch processing: iterate files, skip unsupported extensions silently, catch per-file errors, continue processing
    - Ensure metadata `source` field is set on every TextSegment
    - _Requirements: 1.4, 1.5, 1.6_

  - [ ]* 2.5 Write property tests for parsers
    - **Property 1: TXT Parser Round-Trip** — write to .txt file and parse back, verify content equality
    - **Property 2: Batch Processing Resilience** — mix valid/invalid files, verify all valid files parsed
    - **Property 3: Metadata Completeness Invariant** — verify source, chunk_index, section_title on all chunks
    - **Property 4: Unsupported Extension Filtering** — non-supported extensions produce zero segments
    - **Property 5: Non-Empty Extraction Guarantee** — successfully loaded docs produce non-empty segments
    - **Validates: Requirements 1.3, 1.4, 1.5, 1.6, 1.7**

- [ ] 3. Ingestion Layer — Chunking
  - [ ] 3.1 Implement semantic chunking engine
    - Create `chunking/semantic_chunker.py` with `SemanticChunker` class
    - Implement section detection (numbered headings, markdown headings, short lines followed by blank)
    - Implement split priority: section boundaries → paragraph boundaries → character limits
    - Enforce max_chunk_size, overlap, and min_chunk_size constraints
    - Implement trailing fragment merge (< min_size chars merged into preceding chunk)
    - Attach metadata: source, chunk_index (zero-based), section_title
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

  - [ ]* 3.2 Write property tests for chunking engine
    - **Property 6: Maximum Chunk Size Invariant** — all chunks ≤ max_chunk_size for any valid config
    - **Property 7: Chunking Structural Boundary Priority** — sections split before paragraphs, paragraphs before chars
    - **Property 8: Minimum Chunk Content Size** — all chunks ≥ min_chunk_size unless doc is shorter
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.5, 2.6**

- [ ] 4. Ingestion Layer — Embedding and BM25 Index
  - [ ] 4.1 Implement unified embedding generator
    - Create `indexing/embedder.py` wrapping HuggingFace embeddings with `paraphrase-multilingual-MiniLM-L12-v2`
    - Read model name from `RAGConfig` (single source of truth)
    - Implement model mismatch detection: store model name in index metadata, verify on query
    - Handle model load failures with descriptive error and halt
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [ ] 4.2 Implement BM25 index builder and searcher
    - Create `indexing/bm25_index.py` with `BM25Index` class using `rank-bm25` (BM25Okapi)
    - Implement Hebrew tokenizer: split on whitespace and Unicode punctuation (maqaf ־, geresh ׳, gershayim ״) while preserving niqqud
    - Implement `build()` to construct index from chunks and persist to disk (pickle)
    - Implement `search()` to query the index and return scored chunks
    - Implement `load()` to restore index from disk
    - _Requirements: 4.4, 4.5_

  - [ ]* 4.3 Write property tests for BM25 tokenizer
    - **Property 10: Hebrew Tokenization Preserves Niqqud** — niqqud stays attached to tokens, splits on correct punctuation
    - **Validates: Requirements 4.4**

  - [ ] 4.4 Implement ingestion pipeline orchestrator
    - Create `indexing/pipeline.py` orchestrating: parse → chunk → embed → store in ChromaDB + build BM25 index
    - Implement per-document progress output to stdout (doc N/total, cumulative chunks)
    - Write JSON manifest on completion (successful files, failed files, total chunks, ISO 8601 timestamp)
    - Write sync marker file (`last_updated.txt`)
    - Handle per-document failures without losing prior indexed data
    - Handle all-failures gracefully (zero successes manifest still written)
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

  - [ ]* 4.5 Write property tests for indexing manifest
    - **Property 20: Indexing Manifest Accuracy** — verify manifest structure, chunk counts, timestamps
    - **Property 21: Embedding Model Mismatch Detection** — verify mismatch detection with expected/actual names
    - **Validates: Requirements 10.4, 3.4**

- [ ] 5. Checkpoint — Foundation and Ingestion Complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 6. Retrieval Layer — Hybrid Search and RRF Fusion
  - [ ] 6.1 Implement dense vector search module
    - Create `retrieval/dense_search.py` wrapping ChromaDB similarity search
    - Accept query string, return up to k=50 scored chunks
    - Use embedder from config for query encoding
    - _Requirements: 4.1_

  - [ ] 6.2 Implement sparse BM25 search module
    - Create `retrieval/sparse_search.py` wrapping `BM25Index.search()`
    - Accept query string, return up to k=50 scored chunks
    - Handle missing BM25 index gracefully (return empty list, log warning)
    - _Requirements: 4.1, 4.6_

  - [ ] 6.3 Implement RRF fusion
    - Create `retrieval/rrf_fusion.py` with `RRFFusion` class
    - Implement RRF formula: score(d) = sum(1/(k + rank_i(d))) for each ranker
    - Handle single-method fallback when one method returns empty results
    - Handle both-empty case (return empty set)
    - Return top_n fused candidates sorted by descending fused score
    - _Requirements: 4.2, 4.3, 4.6, 4.7_

  - [ ]* 6.4 Write property tests for RRF fusion
    - **Property 9: RRF Fusion Score Correctness** — verify formula computation and output ordering
    - **Property 11: Single-Method Fallback on Empty Results** — verify graceful single-source fallback
    - **Validates: Requirements 4.2, 4.6**

- [ ] 7. Retrieval Layer — Cross-Encoder Reranking
  - [ ] 7.1 Implement cross-encoder reranker
    - Create `retrieval/reranker.py` with `CrossEncoderReranker` class using `cross-encoder/ms-marco-MiniLM-L6-v2`
    - Score each candidate chunk against user query, produce relevance score in [0.0, 1.0]
    - Sort by descending score, return top-k (configurable, default 5)
    - Attach relevance_score to chunk metadata
    - Implement timeout (10s) with fallback to original ranking
    - Implement initialization failure fallback to original ranking
    - Handle fewer candidates than top_k gracefully
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [ ]* 7.2 Write property tests for reranker
    - **Property 12: Reranker Output Ordering and Score Attachment** — verify descending sort, top-k limit, score range
    - **Validates: Requirements 5.1, 5.2, 5.4**

- [ ] 8. Checkpoint — Retrieval Layer Complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 9. Generation Layer — Confidence and Faithfulness Scoring
  - [ ] 9.1 Implement confidence scorer
    - Compute confidence as arithmetic mean of reranker relevance scores, rounded to 2 decimal places
    - Classify query as insufficient context when max score < threshold (0.3) or zero chunks returned
    - Return predefined Hebrew message "המידע אינו קיים במסמכים" + rephrase suggestion for insufficient context
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

  - [ ]* 9.2 Write property tests for confidence scoring
    - **Property 13: Confidence Score Computation** — verify arithmetic mean and rounding
    - **Property 14: Insufficient Context Classification** — verify threshold logic and zero-chunk handling
    - **Validates: Requirements 8.1, 8.2, 8.3, 8.5, 8.6**

  - [ ] 9.3 Implement faithfulness scorer
    - Create `generation/faithfulness.py` with `FaithfulnessScorer` class
    - Decompose response into individual claims
    - Verify each claim against context chunks
    - Compute score as M/N (supported/total claims), handle N=0 → 1.0
    - Set `low_confidence_warning` flag when score < 0.7
    - _Requirements: 7.1, 7.4, 7.5_

  - [ ]* 9.4 Write property tests for faithfulness scoring
    - **Property 15: Faithfulness Score Range and Proportionality** — verify M/N computation and [0,1] range
    - **Property 16: Low-Confidence Warning Threshold** — verify warning flag logic at 0.7 boundary
    - **Validates: Requirements 7.4, 7.5**

- [ ] 10. Generation Layer — Hebrew Prompts, Validation, and Formatting
  - [ ] 10.1 Implement Hebrew prompt builder
    - Create `generation/prompt_builder.py` with `HebrewPromptBuilder` class
    - Build prompt with system-level and user-level Hebrew enforcement directives
    - Include context chunks and user query in prompt template
    - Instruct LLM to respond only from context, include citations, translate/transliterate technical terms
    - _Requirements: 6.1, 6.2, 6.4, 6.5, 7.1, 7.2_

  - [ ] 10.2 Implement Hebrew response validator
    - Create `generation/response_validator.py` with `ResponseValidator` class
    - Detect sequences of >2 consecutive Latin-alphabet words (excluding proper nouns)
    - Return validation result (is_valid, cleaned_response)
    - _Requirements: 6.6_

  - [ ]* 10.3 Write property test for Latin word sequence detection
    - **Property 19: Latin Word Sequence Detection** — verify detection of >2 consecutive Latin words
    - **Validates: Requirements 6.6**

  - [ ] 10.4 Implement structured response formatter
    - Create `generation/formatter.py` with `ResponseFormatter` class
    - Enforce format: Line 1 = Hebrew title (≤10 words), then answer paragraph, then source citations line
    - Strip introductory preambles ("based on the documents", "according to the sources", etc.)
    - Handle malformed LLM output by reformatting automatically
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

  - [ ]* 10.5 Write property tests for response formatting
    - **Property 17: Response Structure Format** — verify three-part structure (title, answer, citations)
    - **Property 18: No Introductory Preambles** — verify banned phrases are stripped
    - **Validates: Requirements 9.1, 9.2, 9.3, 9.4**

- [ ] 11. Checkpoint — Generation Layer Complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 12. Integration Layer — Refactor Existing Interfaces
  - [ ] 12.1 Refactor app_ui.py (Streamlit UI) to use new modules
    - Replace inline embedding/ChromaDB logic with calls to retrieval and generation modules
    - Wire: query → dense_search + sparse_search → RRF fusion → reranker → confidence check → prompt builder → LLM → validator → formatter
    - Display confidence score (2 decimal places) alongside response
    - Display low-confidence warning when faithfulness < 0.7
    - Display RTL Hebrew response with citations
    - Handle insufficient context with predefined message
    - _Requirements: 6.3, 8.6, 7.5, 8.2, 8.4_

  - [ ] 12.2 Refactor chat_rag.py (CLI interface) to use new modules
    - Replace inline logic with calls to new retrieval and generation modules
    - Use consistent embedding model from RAGConfig
    - Display confidence score and faithfulness warning in CLI output
    - Handle insufficient context with predefined message
    - _Requirements: 3.2, 8.6, 7.5, 8.2_

  - [ ] 12.3 Refactor check_db.py to use shared config
    - Replace hardcoded `all-MiniLM-L6-v2` with model from RAGConfig
    - Use shared ChromaDB initialization from config
    - _Requirements: 3.2_

  - [ ] 12.4 Replace rag_script.py with new indexing pipeline
    - Create entry point that uses `indexing/pipeline.py` as the primary indexing command
    - Support same document directory input as existing script
    - Ensure BM25 index is built alongside vector store
    - _Requirements: 4.5, 10.1, 10.4_

- [ ] 13. Checkpoint — Integration Complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 14. Testing Layer — Unit Tests and Integration Tests
  - [ ]* 14.1 Write unit tests for DOCX parser edge cases
    - Test tables, nested headings, empty documents, large files
    - _Requirements: 1.1_

  - [ ]* 14.2 Write unit tests for PDF parser edge cases
    - Test multi-page documents, paragraph separation, empty pages
    - _Requirements: 1.2_

  - [ ]* 14.3 Write unit tests for reranker fallback behavior
    - Mock timeout scenarios and initialization failures
    - Verify fallback to original ranking
    - _Requirements: 5.5_

  - [ ]* 14.4 Write unit tests for Hebrew response validator
    - Test proper noun exceptions, mixed text, edge cases
    - _Requirements: 6.6_

  - [ ]* 14.5 Write integration test for full ingestion pipeline
    - Parse → chunk → embed → store flow with test documents
    - Verify manifest, BM25 index, and vector store populated
    - _Requirements: 10.1, 10.2, 10.4_

  - [ ]* 14.6 Write integration test for full retrieval pipeline
    - Query → dual search → fuse → rerank flow
    - Verify scored chunks returned in correct order
    - _Requirements: 4.1, 4.2, 5.1, 5.2_

  - [ ]* 14.7 Write integration test for end-to-end query flow
    - User question → structured Hebrew response with citations and scores
    - Test insufficient context path
    - Test low-confidence warning path
    - _Requirements: 7.3, 8.2, 8.4, 9.1_

- [ ] 15. Final Checkpoint — All Tests Pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at layer boundaries
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and edge cases
- The implementation language is Python (as specified in the design document)
- All modules share configuration through the `RAGConfig` dataclass
- Existing interfaces (app_ui.py, chat_rag.py, check_db.py) are preserved but refactored to use new modules

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2"] },
    { "id": 2, "tasks": ["2.1", "2.2", "2.3", "3.1"] },
    { "id": 3, "tasks": ["2.4", "2.5", "3.2"] },
    { "id": 4, "tasks": ["4.1", "4.2"] },
    { "id": 5, "tasks": ["4.3", "4.4"] },
    { "id": 6, "tasks": ["4.5", "6.1", "6.2"] },
    { "id": 7, "tasks": ["6.3"] },
    { "id": 8, "tasks": ["6.4", "7.1"] },
    { "id": 9, "tasks": ["7.2", "9.1"] },
    { "id": 10, "tasks": ["9.2", "9.3", "10.1", "10.2"] },
    { "id": 11, "tasks": ["9.4", "10.3", "10.4"] },
    { "id": 12, "tasks": ["10.5", "12.1", "12.2", "12.3", "12.4"] },
    { "id": 13, "tasks": ["14.1", "14.2", "14.3", "14.4"] },
    { "id": 14, "tasks": ["14.5", "14.6", "14.7"] }
  ]
}
```
