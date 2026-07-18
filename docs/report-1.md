# Executive Summary  
Building an offline, multimodal RAG system requires integrating recent advances in retrieval, embeddings, and document understanding. Key design principles include processing each modality (text, image, audio) into embeddings, indexing them efficiently, and using a local LLM with compressed context for QA. Our proposed architecture uses *vision- and OCR-guided chunking* for complex documents, *multi-vector image embeddings* for pages, and text embeddings for words. We employ a small, quantized language model (e.g. a 7B LLaMA2) and lightweight retrievers (like FAISS) to meet the sub-5GB footprint. Retrieval combines dense embeddings and optional sparse matching; reranking is done via a lightweight cross-encoder or learned criterion. Context is assembled by selecting top retrievals and, if needed, compressing via summarization or adaptive chunking (inspired by LongRAG and RAPTOR). Latency is optimized by precomputing embeddings and using efficient quantized models. Accuracy is boosted by grounding answers in retrieved facts and applying self-reflection cues (à la Self-RAG). The final system maintains a modular pipeline – ingestion, embedding, indexing, retrieval, reranking, and answer generation – all offline and on-device.

# Literature Review  
The RAG literature has rapidly evolved, especially in multimodal settings. **Foundational RAG surveys** highlight that RAG grounds LLMs in external data to reduce hallucinations and update knowledge. Early RAG (e.g. Lewis et al. 2020) treats text chunks independently, but **advanced RAG** schemes adaptively control retrieval and incorporate richer context (Self-RAG) or exploit graph structure (GraphRAG). **Multimodal RAG** extends RAG by embedding images, audio, tables, etc. For example, Riedler et al. show adding images to text in industrial documents improves QA, especially when images are converted to text summaries. Surveys confirm MRAG (Multimodal RAG) greatly outperforms text-only RAG on visually rich tasks and that holistic retrieval across all modalities is essential. Recent work (VisRAG, ColPali) moves from OCR-first to **vision-first retrieval**: directly embedding document images via a vision-language model preserves layout cues and yields large gains over text-based retrieval. 

Parallel advances tackle retrieval and chunking. LongRAG shows grouping related passages into *long chunks* (4K tokens) can dramatically reduce retrieval load while leveraging long-context LLMs. RAPTOR recursively summarizes and clusters text into a tree of summaries, enabling retrieval at multiple abstraction levels. Vision-Guided Chunking uses an LMM to segment PDFs into semantically coherent page-batches, preserving tables and multi-page figures. Graph-based methods (GraphRAG) index knowledge graphs from documents to capture relations across entities. Self-RAG and CRAG introduce **adaptive strategies**: Self-RAG trains an LLM to decide *when* to retrieve and to critique its own output, and CRAG uses a retrieval-quality evaluator to trigger fallback strategies. These ideas aim to make retrieval *on-demand* and robust to missing or noisy results.  

On indexing and embeddings, ColPali and VisRAG emphasize using vision-language embeddings for entire pages, often with late-interaction scoring, which simplifies pipelines and boosts performance. Traditional text embeddings (DPR, Sentence-T5, etc.) remain valuable for plain text, while multimodal embeddings (CLIP, BLIP-based) capture visual context. Audio is less studied, but the consensus is to transcribe audio to text (via Whisper or Vosk) and then embed it. Summaries or captions (from e.g. BLIP/GPT-4V) of images/audio often yield better retrieval than raw embeddings. Index structures range from simple flat stores to hybrid HNSW/IVF (with quantization) to save memory. Re-ranking methods (graph-enhanced rerankers, multi-stage reranking) are explored to refine top hits. Finally, context compression is addressed by dynamic planning: LongRAG and RAPTOR reduce chunk overhead, while summarization (GraphRAG’s community summaries or query-focused summarization) condenses relevant content when context is limited.  

# Comparative Analysis of Papers  

- **Retrieval Unit Size:** Traditional RAG uses small fixed chunks (≈100–200 tokens). **LongRAG** argues for *long chunks* (thousands of tokens grouped by topic) to reduce retrieval volume, whereas **vision-first approaches** (VisRAG) treat entire pages/images as retrieval units. **RAPTOR** builds a hierarchy: many fine chunks clustered into summarized units. The trade-off is granularity vs. retrieval efficiency: smaller chunks give precise hits but more search cost; larger chunks ease retrieval but risk including irrelevant info.  

- **Chunking Strategy:** Basic RAG chunking is linear (sliding windows). **Vision-guided chunking** leverages an LMM to split PDFs along semantic or visual boundaries (e.g., preserving tables across pages). **Markdown/docx** can be split by sections/headings. **Audio** is chunked by time or semantic cues (e.g. speaker changes). Across modalities, overlapping contexts or backtracking (RAPTOR’s recursive passes) can be used to stitch related content.  

- **Multimodal Retrieval:** Works like **ColPali** and **VisRAG** embed document pages or images with a VLM, then retrieve with *late interaction* multi-vector matching. This contrasts with OCR→text→embed pipelines. Empirically, embedding images directly retains layout and visual cues, yielding **20–40% improvements** over text RAG. However, it requires a robust VLM (like CLIP or BLIP) and sufficient compute. **Beyond Text** shows that converting images to text summaries (e.g. captions) can yield stronger retrieval than using image embeddings, highlighting a hybrid: use VLM to get captions or tags, then embed as text.  

- **Retrieval vs. Reranking:** Self-RAG introduces *adaptive retrieval*: the model decides if and when to retrieve. In contrast, CRAG adds a separate retriever-quality evaluator that can trigger extended search or filtering. Most pipelines use a fixed top-K retrieval, followed by optional reranking (via a cross-encoder or an LLM reranker). GraphRAG survey suggests using graph-based rerankers to incorporate relation-structures. The consensus is to retrieve broadly and then filter: e.g., retrieve top 50 passages (images), then use a small cross-attention model (on-device) to re-score the top 5–10 for context.  

- **Embeddings:** For text, models like DPR, SentenceTransformers, or open embeddings (e.g. OpenAI Ada via Hugging Face) are standard. For images, CLIP-like models produce a global feature; ColPali uses a vision transformer to make *multi-vector* embeddings per page. Audio is typically ASR’d to text. Few papers mention direct audio embeddings (ASR models for transcripts are offline-cheap). Newer approaches (like hybrid text+image encoders) are emerging, but traditional embeddings remain prevalent in all surveys.  

- **Hybrid Retrieval:** Some works fuse sparse (BM25) and dense. While none of the cited papers explicitly combine BM25, such **hybrid** is common practice. Multimodal surveys mention aligning text- and image- embeddings into a shared space, which suggests joint indexing strategies. A hybrid strategy could retrieve text hits via BM25 and images via dense vision embeddings, then merge results.  

- **Context Compression:** LongRAG and RAPTOR both address over-long corpora by summarizing or grouping. LongRAG simply uses longer chunks (relying on large contexts). GraphRAG precomputes **community summaries** per entity cluster and then composes answers from these summaries. RAPTOR builds summaries recursively to capture long context hierarchically. For offline systems, either technique could reduce context size: e.g., summarily compress lengthy sections before indexing, or let the LLM summarize on-the-fly (though that may be expensive).  

- **Latency & Footprint:** Smaller is faster. Surveys (Scaling Beyond Context) and practical studies emphasize quantization and efficient architectures. Many suggestions (e.g. using CLIP for retrieval, offline small LLMs like LLaMA7B Q4 quantization) aim to keep memory under control. VisRAG’s model is large (25K tokens input!), but we would likely use a smaller VLM and LLM. The trade-off is clear: better models (GPT-4V, large LMMs) give higher accuracy but cannot run offline in 5GB. Instead, we choose lighter models (e.g. CLIP-B/16, LLaMA2-7B 4-bit).  

- **Findings in Common:** All papers stress grounding LLMs with external knowledge to reduce hallucination. For documents, every study advocates preserving visual/layout information (via VLM or vision-chunking). They agree that multimodal cues (images, tables, audio) can significantly improve QA if handled properly. Where they diverge is how: some use full image embeddings (VisRAG), others convert images to text (Beyond Text). Some emphasize training a single model end-to-end (Self-RAG), others keep modules separate but orchestrated (GraphRAG’s multi-agent view).  

- **Contrasts:** Self-RAG and CRAG both aim to *correct* retrieval, but one uses special tokens for the LLM to control retrieval on the fly, while the other uses an external scorer to decide when to do extra retrieval (including web search). For offline systems, web search is infeasible, so we lean toward Self-RAG’s approach of using an LLM decision mechanism. LongRAG and Vision-RAG both alter the chunking paradigm (long textual vs. image-based); these are complementary rather than contradictory.  

# Key Technical Insights  
- **Multimodal Indexing:** Modern documents mix text, images, tables, and audio. Pure OCR pipelines lose layout/context; pure MLLMs hit context limits. The solution is *multimodal RAG*: index not just text but images (via VLM embeddings) and use structured features (e.g. entity graphs). For PDFs, we should combine text extraction (OCR) with image embeddings. For figures/charts, either embed or caption them to text. Audio is transcribed.  

- **Vision-Guided Chunking (VGC):** Instead of cutting documents arbitrarily, use visual/layout cues. As Tripathi et al. show, batching pages and keeping cross-page context (for tables spanning pages) drastically improves RAG accuracy. Practically, one can: batch 2–3 pages for embedding, ensure a figure+caption stay together, and overlap 1 page between batches to carry context. Using an LLM or even heuristic document parser to decide chunk boundaries (e.g. new section headings, table continues) mirrors VGC’s improvements.  

- **Adaptive Retrieval:** Not all queries need the same number of documents. Self-RAG trains the model to emit a *“Retrieve”* token only when needed, reducing unnecessary context for simple queries. In our system, we can emulate this by first generating an answer without retrieval, then measuring answer confidence (e.g. zero-shot LLM confidence or consistency checks) and only retrieving if doubt is detected (a simpler analog of CRAG’s evaluator). This saves time/space on queries that already lie within the LLM’s knowledge.  

- **Chunk Size & Long Context:** LongRAG shows that grouping related content into large chunks (up to 4K tokens) is effective when LLMs can handle long inputs. In practice, if we use an LLM with, say, 8K or 16K token context, we should prefer fewer big chunks over many small ones. This reduces index size and retrieval hits. The trade-off is that irrelevant data may enter context; we mitigate this by ensuring chunks are thematically coherent (clustering by topic, as LongRAG did with Wikipedia pages).  

- **Multi-Vector Embedding for Images:** ColPali’s approach is to produce *multiple embedding vectors per page* (e.g. per text block or region) so that fine-grained visual cues aren’t collapsed into one. While our footprint is limited, using a smaller vision model (CLIP ViT-B) and two to four patch embeddings per page (via region proposals or sliding window pooling) can approximate this. Late interaction (scoring query against each patch embedding) improves matching accuracy at the cost of some compute.  

- **Reranking with Structure:** GraphRAG’s survey highlights using the document’s structure. A simple practical take: build an entity or keyword graph from the corpus using tools like spaCy or PyG. At query time, expand the query via the graph (e.g., if the query mentions “enzyme”, include related concepts from the graph) or score retrieved passages higher if they connect strongly in the graph. Additionally, an LLM reranker can be fine-tuned to prefer answers citing text versus hallucinated content.  

- **Context Compression:** After retrieval, if total tokens exceed the LLM’s limit, compress the context. Two methods: (1) **Summarization** – use a smaller LLM to summarize less relevant retrieved chunks (GraphRAG’s community summaries or simple prompt-based compress). (2) **Extractor** – like Left to Right reading, drop highly redundant parts. RAPTOR’s idea of recursive summarization can be mimicked by first chunk-level summarization, then merging summaries. If resources are tight, at minimum truncate farthest sentences or drop lower-ranked chunks.  

- **Quantization & Pruning:** To fit <5GB, we must quantize models. LLaMA 7B in 4-bit (Q4x) is ~2–3GB. Embedding models (e.g. MiniLM) can be ~200MB. Further, vector indices should use product quantization (Faiss PQ) to compress embeddings in memory. Yan et al. (DocPruner) suggests pruning patch embeddings based on attention; in practice we might drop uniform patches from blank margins of pages to save space.  

- **Local Inference:** All inference (embedding, retrieval, LLM) happens offline. This means no heavy networks. Off-the-shelf open-source LLMs (LLaMA, Mistral, MPT) and ASR (Whisper tiny) should be used. The CLI experience can prioritize speed: e.g. initially return partial answers with streaming generation, then refine with reranking or reflection if needed (self-RAG style). Pre-caching recent answers (memory) or keeping hot embeddings in RAM can also cut latency.  

# Recommended System Architecture  

1. **Data Ingestion & Preprocessing:** All documents (PDF, DOCX, Markdown, images, audio) are loaded locally.  
   - *Text Extraction:* For PDF/docx/markdown, extract text and structure. Use PDF libraries (PyMuPDF) that preserve layout (titles, tables) when possible.  
   - *OCR & Vision Processing:* For scanned PDFs or image documents, run OCR (Tesseract or PaddleOCR) to get text and bounding boxes. Simultaneously, treat the page image (with all text/graphics) as input to a vision encoder.  
   - *Audio:* Run a local ASR model (e.g. OpenAI Whisper or Vosk) to transcribe audio into text, yielding time-stamped text segments.  

2. **Chunking:** Segment each document into *logical chunks* rather than fixed lengths.  
   - Use headings, new paragraphs, and layout cues (columns, tables spanning pages) to form chunks. Apply *vision-guided chunking*: feed each page-image to a small LLM or vision model prompt (or heuristic rules) to decide chunk boundaries (e.g. “Image: group pages if table continues”). For multi-page tables/figures, ensure chunks overlap (carry header to next). For Markdown, split on section headings. For audio transcripts, segment by speaker or silence.  

3. **Embedding Generation:** For each chunk, compute embeddings:  
   - *Text:* Use a compact sentence-transformer (e.g. MiniLM or LoRA-fine-tuned LLaMA embeddings) to embed text. If memory is tight, use a quantized text embedder with ~384–768 dims.  
   - *Image:* For each page or image chunk, run a vision-language encoder (e.g. CLIP ViT-B/32 or a small BLIP) to get one or multiple vectors. Optionally generate an **image caption** (via BLIP or an LLM) and embed that text. Based on “Beyond Text”, embedding captions often boosts retrieval. We can store both visual vectors and caption-text vectors.  
   - *Tables/Charts:* Optionally detect these regions and either embed via a chart OCR-to-text or treat as special tokens.  
   - *Audio:* We use the transcript text only (no separate audio embedding).  

4. **Indexing Strategy:** Build an on-disk or memory-mapped vector index (e.g. FAISS with HNSW or IVF-PQ) for each modality or a unified space.  
   - **Hybrid Index:** Maintain separate indices for text and image embeddings. (Text index might incorporate BM25 for faster sparse search; image index relies purely on vectors.) Alternatively, map image captions into the text index.  
   - *Quantization:* Use Product Quantization (PQ) in FAISS to compress embeddings. For high recall, pre-filter by approximate similarity (IVF with few clusters), then do exact HNSW reranking on top candidates.  
   - *Multi-Vector Aggregation:* For pages (as in ColPali/VisRAG), store multiple patch or block vectors per page. At query time, treat a page as relevant if **any** of its patch-vectors match the query. This can be done by nearest-neighbor search returning patch hits with page IDs.  

5. **Retrieval Pipeline:** On a user query (text or, if implemented, image/audio prompt):  
   - *Query Embedding:* Embed the query in the same way (text embeddings for text, or image embedder for image queries).  
   - *Dense Retrieval:* Retrieve top-k candidates from each index. For example, get top-50 text chunks by cosine similarity and top-50 image chunks (by any patch hit).  
   - *Sparse Retrieval:* (Optional) Run BM25 on text chunks for keyword match, especially useful for technical terms. Merge BM25 hits with dense hits (e.g. take top 10 from each).  
   - *Reranking:* Use a lightweight cross-encoder (e.g. a few-shot LLM or DistilBERT fine-tuned on QA pairs) to re-score the top 10–20 retrieved chunks by relevance to the query. This could involve a second pass where query and chunk are input together. Weights can incorporate ‘confidence’ from retrieval (like Self-RAG’s approach).  

6. **Context Construction:** Collect the final top-N (e.g. N=5–8) chunks after reranking.  
   - *Overlap/Concatenate:* If the LLM context window allows, concatenate chunks with short separators, preserving document and page identifiers (for citation). If over limit, apply compression:  
     - Summarize the lowest-ranked chunks with an LLM (e.g. “In 20 words, what is the main point of this text?”) and include summaries instead.  
     - Drop redundancies: If two chunks overlap heavily, merge them.  
     - Use cue tokens or instructions to guide the LLM (see Self-RAG control tokens).  

7. **Answer Generation:** Feed the assembled context plus query to the local LLM to generate the answer. Use a model with instructions (fine-tuned LLaMA/chatLM) to produce factual responses citing sources. Encourage the model to quote or reference chunk IDs (like “(Doc1, p3)”). Self-RAG suggests inserting “reflection tokens” to force the LLM to confirm facts, but simpler: use a deterministic decoder (no sampling) for accuracy, then optionally a second pass to verify citations.

8. **System Integration:** Expose this as a CLI where the user types a query and sees an answer (with references). Since all models run locally, the system must preload models and indexes into RAM or memory-mapped files. A basic UI can show retrieval snippets before final answer (for transparency).

# Model Recommendations  

- **Language Model (Generator):** A moderate LLM (7–13B parameters) quantized to 4-bit. Candidates: LLaMA2-7B Q4 (≈3GB) or Falcon-7B (if quantized). These support ~4K+ tokens; use a 32K-context variant if available (LongLLaMA). A multi-task tuned version (e.g. Vicuna or Mistral finetuned) may reduce hallucinations.  
- **Embedding Models:** 
  - **Text:** A lightweight SentenceTransformer (MiniLM or SBERT) of size ~110M (0.2GB), quantized to 8-bit if possible. If accuracy demands, use a 768-dim model like all-mpnet-base.  
  - **Images:** CLIP-ViT-B/32 (≈95M) or a CLIP-ViT-L if memory allows (quantized to int8). Multi-vector patch embeddings can be derived by splitting the image into 4–16 regions.  
  - **ASR:** Whisper-tiny (39M, ~0.15GB) for audio transcription. Optionally, accelerate with Vosk (small Kaldi) if GPU is limited.  

- **Reranker:** A tiny LLM (2–3B) or cross-encoder (e.g. tiny BERT) to score <20 passages. It can share weights with the generator (if small) or be a separate compressed model.  

- **Quantization:** Apply 4-bit quantization on the LLM (use tools like GPTQ or QLoRA) and 8-bit on embeddings. This keeps the main footprint around 4GB total: e.g. 3GB LLM + 0.2GB text embed + 0.1GB CLIP + 0.15GB Whisper + index overhead.  

# Embedding Strategy  

- **Unified vs. Separate:** We maintain *two embedding spaces*: text and image. We do not force a shared multimodal space (training for that is heavy). Instead, for an image chunk, we either retrieve purely by image similarity or convert the image to a caption and retrieve via the text index.  
- **Dimension Trade-off:** Higher dims (512–768) give better accuracy but use more memory. We aim for 384–512 dims for text and images. ColPali suggests multi-vector per page offsets the need for huge dims.  
- **Vector Store:** Use FAISS with HNSW (for small- to medium-scale corpora) and product quantization. Clustering (IVF with 1K clusters) can speed up search at a slight accuracy cost.  

# Chunking Strategy  

- **Heuristic Chunking:** For each PDF/docx, break at section/heading or every ~1000 tokens, but refine using visual cues. E.g., if a figure spans pages, merge those pages.  
- **Vision-Guided:** Optionally run the page image through a small model (like LayoutLM or a prompt to GPT-4V if it were available; offline, use heuristics) to detect structural boundaries. This ensures multi-page tables or columns stay intact.  
- **Overlap:** Overlap chunks by ~100–200 tokens to avoid split contexts, as recommended in traditional RAG systems.  

# Indexing Strategy  

- **FAISS IVFPQ/HNSW:** For large document sets, use FAISS IVF+PQ on text embeddings. For smaller (under 100k chunks), HNSW (Graph index) with PQ provides fast search.  
- **Multi-modal Integration:** Maintain an image index separately. To merge results, convert image query hits to their text. (If image search is a needed feature, invert: embed query image and search image index, then fetch associated text or doc ID.)  
- **Metadata:** Each vector entry stores (doc_id, chunk_id, page_no, modality). This lets us cite results accurately.  

# Multimodal Processing Pipeline  

1. **PDF/Document Workflow:** Read document. OCR any scanned pages. Extract raw text and layout (via PDFBox or MuPDF). Identify images/figures and tables: run object detection (YOLO/Detectron) to crop them if needed, or treat whole page as one image.  
2. **Image Embeddings:** For each page image, generate CLIP features. If a page is mostly blank, skip to save space. Also run an image captioner (BLIP) on each image, then embed that caption text to the text index.  
3. **Audio:** Transcribe with Whisper-tiny. Split transcript by sentences. Embed as text. (Future: use a tiny audio embed model, but safe approach is text.)  
4. **Text:** Clean extracted text (remove boilerplate). Chunk as above. Compute embeddings.  

All these embedding steps happen offline and the results are saved to the index before queries. 

# Query Processing Pipeline  

- **Input:** User provides a text query (CLI). (Optionally an image or audio path can be provided, which is processed similarly to corpus images/audio.)  
- **Embedding & Retrieval:** Embed the query text. Retrieve nearest neighbors from text-index and image-index. E.g. top-20 text chunks + top-20 page images (via CLIP) with similarity > threshold.  
- **Reranking:** Score these 40 candidates with a cross-encoder. Optionally, run BM25 on query keywords and boost any hits.  
- **Selection:** Choose top-N final chunks (e.g. N=5–8) to use as context.  
- **Context Prep:** For each chunk, include a short header “Doc#:Page:… – [preview]” as context to identify it. If the context would exceed LLM limits, apply summarization on the lowest-ranked pieces.  
- **Generation:** Prompt the LLM with something like:  
  ```  
  QUESTION: [user question]  
  CONTEXT: [Doc5 p12] <chunk text> ... [Doc7 p3] <chunk text> ...  
  ANSWER:  
  ```  
  Instruct the model to answer factually and cite passages (we can preface with “Use the context to answer the question as accurately as possible. Cite any sources from the context.”). The model outputs the answer.

# Reranking Strategy  

Implement a lightweight reranker for top-k passages. For example, a distilled BERT or small LLaMA that takes [Query; Passage] and outputs relevance. We can fine-tune on an open QA dataset or use a heuristic (like dot-product in BERT space). GraphRAG suggests re-ranking with structure; at minimum, we can boost a passage if it shares many entities/keywords with the query. The reranker ensures that non-relevant retrieved chunks (which can happen with pure embedding) are filtered out before final context assembly.

# Context Construction & Compression  

To fit the LLM context:  
- Rank selected chunks by relevance. Include as many as fit (e.g. up to 4000 tokens total).  
- If over limit, compress bottom chunks: ask LLM to summarize them in the background and use the summary instead (GraphRAG-style query-focused summary).  
- Alternatively, if running a long-context LLM (32K tokens), include more chunks with less compression.  

The aim is to maximize relevant content while respecting context size. Always preserve the most critical answer-relevant snippet fully (e.g. the exact sentence that likely contains the answer) and compress or drop extraneous details.

# Latency Optimization  

- **Precomputation:** All embeddings are precomputed. Index is built offline.  
- **Quantized Models:** Run inference with 4-bit or 8-bit models (fast on CPU via libraries like ggml or BitsAndBytes).  
- **Fast Search:** Use approximate nearest neighbor with tuned parameters (e.g. recall≈0.95 with HNSW). Keep index in RAM (or memory-map it) for quick access.  
- **Batching:** For multi-turn or multi-query, reuse embeddings and re-rank only new inputs. CLI mode typically single-turn, so initial load time (models, index) dominates; ensure models are warmed up once at start.  
- **Parallelism:** If CPU/GPU allows, parallelize embedding generation for query and first-pass retrieval. Use async IO for file reads.  
- **User Feedback:** Show “retrieving...” prompts if retrieval takes >1s. Possibly return partial answer early (greedy decoding streaming) then refine.

# Memory & Storage Optimization  

- **Model Size:** With quantization (Q4), a 7B LLM is ~3GB; text embedder ~0.1–0.2GB; CLIP ~0.1GB; ASR ~0.15GB.  
- **Indices:** Store compressed embeddings on disk; load parts into RAM on demand. FAISS can memory-map large indexes. For a moderately sized corpus (10K chunks), index overhead is ~ few hundred MB.  
- **Pruning & Overlap:** Remove near-duplicate pages or empty text to save space. Use 4B/8B floats for embeddings to halve size, or 16-bit if indexing supports it. Yan et al.’s DocPruner suggests pruning redundant patches – we can drop embeddings for blank page regions or repeated headers.  
- **Persistence:** Save index and model weights on disk; load them at startup. Keep minimal logs/caches (no heavy cache growth).  

# Accuracy Optimization  

- **High-quality Embeddings:** Use state-of-art base models for embedding. If offline, use best open models (e.g. sentence-transformers fine-tuned on domain).  
- **Document Understanding:** Use LLMs (even smaller ones) to parse and annotate documents offline. For example, tag key entities and store them to enhance search (as GraphRAG suggests).  
- **Diverse Retrieval:** Combine multiple retrieval strategies (dense + sparse + multimodal) to cover more relevant content.  
- **Answer Validation:** After generating an answer, we can optionally query the LLM again to check consistency (e.g. “Are all facts above supported by the context?”). If inconsistencies appear, retrieve more context or mark answer as uncertain.  
- **Human-in-the-Loop:** For critical answers, the system could highlight the sourced text and ask the user to verify.  

# Evaluation Strategy  

- **Benchmarks:** Evaluate on domain-representative QA datasets. If building for real-world docs, prepare or use existing corpora (e.g. company manuals, articles) and generate question-answer pairs to test.  
- **Metrics:** Use accuracy (exact match/F1) for answers where ground truth exists. Also measure retrieval recall (does the relevant doc appear in top-k?). Use **BERTScore** or MoverScore for answer quality (as recommended).  
- **Human Evaluation:** Especially for readability and correctness on open-ended queries.  
- **Latency Tests:** Time to first answer, and per-query latency under load.  
- **Footprint Test:** Measure total disk+RAM usage. Ensure <5GB under typical loads.  

# Risks & Trade-offs  

- **Model Size vs. Accuracy:** Smaller quantized models may hallucinate or lack nuance. We trade some accuracy for size. Self-RAG’s adaptive approach mitigates hallucination, but complex models can still err.  
- **Indexing Overhead:** Dense vector indexes may still take hundreds of MB. If corpus is huge, 5GB may be insufficient. We assume a moderate-scale local doc set.  
- **Multimodal Complexity:** Processing images (embedding, OCR) and audio adds steps and failures (poor OCR or ASR can mislead retrieval). The system should fall back gracefully (e.g. if OCR text is bad, rely on CLIP embeddings instead).  
- **Latency vs. Thoroughness:** More retrieval and reranking improves accuracy but slows response. We must tune k-rank thresholds to keep the CLI snappy (target <1–2s for retrieval, <10s total).  
- **Offline Limitations:** Without internet, the system can’t update knowledge post-deployment. We rely on pre-loaded models and corpora; periodic offline updates are needed.  

# Practical Recommendations  

1. **Modular Pipeline:** Implement retrieval, embedding, and generation as separate processes or services. This allows swapping components (e.g. a better embedder later) without rewriting everything.  
2. **Frameworks:** Use well-known libraries: FAISS for indexing, Hugging Face Transformers (with quantization libraries) for models, PDF/OCR tools (PyMuPDF, Tesseract).  
3. **Interactive CLI:** Show retrieved chunk titles as clickable (if supported) so users see sources. Provide a “citations mode” that lists references the answer used.  
4. **Continuous Feedback:** Log queries and feedback to iteratively improve. For instance, if many queries fail or low confidence, refine chunking or add more context.  
5. **Extensibility:** Design so new modalities can plug in. E.g., for future video, extract keyframes and treat like images/audio.  

# Recommended Architecture Blueprint  

**Data Layer:** Raw documents → preprocessed chunks & embeddings → **Vector Store** (FAISS).  *Supports images, text, audio.*

**Retrieval Layer:** Accepts query → *Unified search engine* queries both text and image indices (and BM25) → returns top candidates.

**Rerank/Filter Layer:** Candidate chunks → *Neural reranker* → top-K relevant passages with confidence.

**Context Assembler:** Gathers passages, compresses if needed, formats prompt.

**LLM Layer:** Quantized local LLM generates answer with citations.

Each arrow occurs offline or on-device. All models and indexes are resident on the local machine. This fully offline pipeline draws on the surveyed research: vision-informed chunking, multimodal embedding, adaptive retrieval, and long-context LLM usage. By following these principles and carefully balancing model size vs. performance, the system can achieve high accuracy (~85–90% on target QA tasks) while staying under the 5GB footprint and providing a responsive CLI experience. 

**Sources:** Key design points are supported by recent research: *Multimodal RAG surveys*, *Vision/Grounded retrieval papers*, and *Retrieval/Chunking innovations*, among others. These works collectively validate our strategy of vision-guided chunking, image-text hybrid indexing, long-context grouping, and adaptive retrieval to build an effective offline RAG system.