"""
IITM BS RAG Pipeline — Stage 6: Uploader
==========================================
INPUT:  output/chunks/all_chunks_embedded.json
OUTPUT: Qdrant collection "iitm_bs"

Provider: Google Gemini (text-embedding-004, dim=768)
"""

import json
import hashlib
import logging
import google.generativeai as genai
from pathlib import Path
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    PayloadSchemaType,
)

from config import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    GEMINI_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    VECTOR_WEIGHT,
    BM25_WEIGHT,
    LOG_LEVEL,
    LOG_FORMAT,
    ALL_CHUNKS_FILE,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("uploader")

EMBEDDED_FILE = ALL_CHUNKS_FILE.parent / "all_chunks_embedded.json"
BATCH_SIZE    = 100


def chunk_id_to_point_id(chunk_id: str) -> int:
    full_hash = hashlib.md5(chunk_id.encode()).hexdigest()
    return int(full_hash[:8], 16)


def build_payload(chunk: dict) -> dict:
    chunk_type = chunk.get("chunk_type", "text")

    payload = {
        "chunk_id":       chunk.get("chunk_id", ""),
        "chunk_type":     chunk_type,
        "content":        chunk.get("content", ""),
        "embed_text":     chunk.get("embed_text", ""),
        "heading":        chunk.get("heading", ""),
        "heading_level":  chunk.get("heading_level", 0),
        "doc_title":      chunk.get("doc_title", ""),
        "section":        chunk.get("section", ""),
        "breadcrumb":     chunk.get("breadcrumb", ""),
        "source_url":     chunk.get("source_url", ""),
        "parent_doc":     chunk.get("parent_doc", ""),
        "references":     chunk.get("references", []),
        "hyde_questions": chunk.get("hyde_questions", []),
        "created_at":     chunk.get("created_at", ""),
        "version":        chunk.get("version", "1"),
        "token_count":    chunk.get("token_count", 0),
    }

    if chunk_type == "image":
        payload["image_file"]    = chunk.get("image_file", "")
        payload["image_content"] = chunk.get("image_content", "")
        payload["image_type"]    = chunk.get("image_type", "")
        payload["scan_method"]   = chunk.get("scan_method", "")

    if chunk_type == "reference_link":
        payload["link_url"]         = chunk.get("link_url", "")
        payload["link_text"]        = chunk.get("link_text", "")
        payload["what_it_contains"] = chunk.get("what_it_contains", "")
        payload["when_to_refer"]    = chunk.get("when_to_refer", "")
        payload["category"]         = chunk.get("category", "")
        payload["access"]           = chunk.get("access", "public")
        payload["found_in_doc"]     = chunk.get("found_in_doc", "")

    if chunk_type == "restricted_doc":
        payload["link_url"]     = chunk.get("link_url", "")
        payload["skip_reason"]  = chunk.get("skip_reason", "")
        payload["note"]         = chunk.get("note", "")
        payload["access"]       = chunk.get("access", "restricted")
        payload["found_in_doc"] = chunk.get("found_in_doc", "")

    return payload


_qdrant_instance = None

def get_qdrant_client(timeout: int = 10) -> QdrantClient:
    """
    Tries connecting to configured QDRANT_URL.
    If Qdrant server is offline/unreachable, seamlessly falls back to an embedded
    local disk database in ./app/qdrant_db so uploads work without needing Docker!
    """
    global _qdrant_instance
    if _qdrant_instance is not None:
        return _qdrant_instance

    if QDRANT_URL and not QDRANT_URL.startswith("http://localhost"):
        _qdrant_instance = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=timeout)
        return _qdrant_instance

    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=3)
        client.get_collections()
        _qdrant_instance = client
        return _qdrant_instance
    except Exception as e:
        db_path = Path(__file__).parent / "qdrant_db"
        logger.info(f"Qdrant server not active at {QDRANT_URL}. Falling back to embedded local storage at {db_path}")
        _qdrant_instance = QdrantClient(path=str(db_path))
        return _qdrant_instance


def ensure_collection_exists(client: QdrantClient):
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config={
                "dense": VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            },
        )
        logger.info(f"Created collection: {QDRANT_COLLECTION}")
        index_fields = [
            ("chunk_type",    PayloadSchemaType.KEYWORD),
            ("doc_title",     PayloadSchemaType.KEYWORD),
            ("parent_doc",    PayloadSchemaType.KEYWORD),
            ("section",       PayloadSchemaType.KEYWORD),
            ("access",        PayloadSchemaType.KEYWORD),
            ("heading_level", PayloadSchemaType.INTEGER),
        ]
        for field, schema in index_fields:
            try:
                client.create_payload_index(
                    collection_name=QDRANT_COLLECTION,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception as e:
                logger.warning(f"Index {field} skipped: {e}")


def upload_chunks_batch(chunks: list[dict]):
    """
    Upsert a batch of chunks into Qdrant collection safely.
    Used for live document and URL ingestion.
    """
    qdrant = get_qdrant_client()
    ensure_collection_exists(qdrant)

    points = []
    for chunk in chunks:
        embedding = chunk.get("embedding", [])
        chunk_id = chunk.get("chunk_id", "")
        if not embedding or not chunk_id:
            continue

        point_id = chunk_id_to_point_id(chunk_id)
        content_text = chunk.get("embed_text", chunk.get("content", ""))
        sparse_indices, sparse_values = build_sparse_vector(content_text)
        payload = build_payload(chunk)

        points.append(PointStruct(
            id=point_id,
            vector={
                "dense": embedding,
                "sparse": {
                    "indices": sparse_indices,
                    "values": sparse_values,
                }
            },
            payload=payload,
        ))

    if points:
        for i in range(0, len(points), BATCH_SIZE):
            batch = points[i:i + BATCH_SIZE]
            qdrant.upsert(collection_name=QDRANT_COLLECTION, points=batch)

    return len(points)


def delete_document_chunks(doc_title: str) -> int:
    """
    Delete all points in Qdrant matching a doc_title.
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    try:
        qdrant = get_qdrant_client()
        if not qdrant.collection_exists(QDRANT_COLLECTION):
            return True
        qdrant.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=Filter(
                must=[FieldCondition(
                    key="doc_title",
                    match=MatchValue(value=doc_title)
                )]
            )
        )
        return True
    except Exception as e:
        logger.warning(f"Could not delete '{doc_title}': {e}")
        return False


def list_indexed_documents() -> list[dict]:
    """
    Retrieve all unique ingested document titles, URLs, and chunk counts from Qdrant.
    """
    docs_map = {}
    try:
        qdrant = get_qdrant_client()
        if not qdrant.collection_exists(QDRANT_COLLECTION):
            return []
        offset = None
        while True:
            scroll_res = qdrant.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=250,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points, next_offset = scroll_res
            for p in points:
                payload = p.payload or {}
                doc_title = payload.get("doc_title", "Untitled Document")
                source_url = payload.get("source_url", "")
                created_at = payload.get("created_at", "")
                if doc_title not in docs_map:
                    docs_map[doc_title] = {
                        "doc_title": doc_title,
                        "source_url": source_url,
                        "chunk_count": 0,
                        "created_at": created_at,
                    }
                docs_map[doc_title]["chunk_count"] += 1

            if not next_offset:
                break
            offset = next_offset
    except Exception as e:
        logger.warning(f"Error listing documents: {e}")
        return []
    return list(docs_map.values())



    return list(docs_map.values())


def setup_collection(client: QdrantClient):
    existing = [c.name for c in client.get_collections().collections]
    if QDRANT_COLLECTION in existing:
        client.delete_collection(QDRANT_COLLECTION)
        print(f"  🗑  Deleted existing collection: {QDRANT_COLLECTION}")

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config={
            "dense": VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False)
            )
        },
    )
    print(f"  ✅ Created collection: {QDRANT_COLLECTION}")
    print(f"     Dense:  {EMBEDDING_DIM}d cosine (Gemini text-embedding-004)")
    print(f"     Sparse: BM25 (keyword)")
    print(f"     Weights: vector={VECTOR_WEIGHT}, bm25={BM25_WEIGHT}")



def build_sparse_vector(text: str) -> tuple[list[int], list[float]]:
    import re
    from collections import Counter

    words = re.findall(r'\b[a-z]{2,}\b', text.lower())
    if not words:
        return [0], [0.0]

    tf    = Counter(words)
    total = len(words)

    index_map = {}
    for word, count in tf.items():
        word_idx = int(hashlib.md5(word.encode()).hexdigest()[:6], 16) % 100000
        tf_score = count / total
        if word_idx in index_map:
            index_map[word_idx] += tf_score
        else:
            index_map[word_idx] = tf_score

    return list(index_map.keys()), [float(v) for v in index_map.values()]


def run():
    print("\n" + "═" * 65)
    print("  IITM BS RAG Pipeline — Stage 6: Uploader")
    print("  Embedding provider: Google Gemini (text-embedding-004)")
    print("═" * 65)

    if not EMBEDDED_FILE.exists():
        print(f"\n  ❌ {EMBEDDED_FILE} not found")
        print(f"     Run python embedder.py first")
        return

    print(f"\n  Loading embedded chunks...")
    with open(EMBEDDED_FILE) as f:
        chunks = json.load(f)
    print(f"  Loaded {len(chunks)} chunks")

    missing_emb = sum(1 for c in chunks if "embedding" not in c)
    if missing_emb:
        print(f"  ❌ {missing_emb} chunks missing embeddings — run embedder.py first")
        return

    sample_emb = next(c for c in chunks if "embedding" in c)
    actual_dim = len(sample_emb["embedding"])
    if actual_dim != EMBEDDING_DIM:
        print(f"  ❌ Dimension mismatch: embeddings are {actual_dim}d but config says {EMBEDDING_DIM}d")
        print(f"     Update EMBEDDING_DIM = {actual_dim} in config.py")
        return

    print(f"  ✅ All chunks have embeddings (dim={actual_dim})")

    print(f"\n  Connecting to Qdrant at {QDRANT_URL}...")
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
    try:
        qdrant.get_collections()
        print(f"  ✅ Connected to Qdrant")
    except Exception as e:
        print(f"  ❌ Cannot connect: {e}")
        return

    print(f"\n  Setting up collection...")
    setup_collection(qdrant)

    print(f"\n  Building {len(chunks)} points...")

    points        = []
    id_map        = {}
    id_collisions = 0

    for chunk in chunks:
        embedding = chunk.get("embedding", [])
        chunk_id  = chunk.get("chunk_id", "")
        point_id  = chunk_id_to_point_id(chunk_id)

        if point_id in id_map:
            logger.warning(f"ID collision: {chunk_id} and {id_map[point_id]} → {point_id}")
            id_collisions += 1
        id_map[point_id] = chunk_id

        content_text = chunk.get("embed_text", chunk.get("content", ""))
        sparse_indices, sparse_values = build_sparse_vector(content_text)
        payload = build_payload(chunk)

        points.append(PointStruct(
            id     = point_id,
            vector = {
                "dense": embedding,
                "sparse": {
                    "indices": sparse_indices,
                    "values":  sparse_values,
                }
            },
            payload = payload,
        ))

    print(f"\n  Uploading to Qdrant in batches of {BATCH_SIZE}...")
    for i in tqdm(range(0, len(points), BATCH_SIZE), desc="  Uploading"):
        batch = points[i:i + BATCH_SIZE]
        qdrant.upsert(collection_name=QDRANT_COLLECTION, points=batch)

    print(f"\n  Creating payload indexes...")
    index_fields = [
        ("chunk_type",    PayloadSchemaType.KEYWORD),
        ("doc_title",     PayloadSchemaType.KEYWORD),
        ("parent_doc",    PayloadSchemaType.KEYWORD),
        ("section",       PayloadSchemaType.KEYWORD),
        ("access",        PayloadSchemaType.KEYWORD),
        ("heading_level", PayloadSchemaType.INTEGER),
    ]
    for field, schema in index_fields:
        try:
            qdrant.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name=field,
                field_schema=schema,
            )
        except Exception as e:
            logger.warning(f"Index {field} skipped: {e}")

    info  = qdrant.get_collection(QDRANT_COLLECTION)
    count = info.points_count
    print(f"\n  ✅ Upload complete!")
    print(f"     Collection: {QDRANT_COLLECTION}")
    print(f"     Points:     {count}")

    # ── Test searches using Gemini ────────────────────────────────
    print(f"\n  Running test searches via Gemini...")
    genai.configure(api_key=GEMINI_API_KEY)

    test_queries = [
        "eligibility criteria for foundation level admission",
        "what is the fee for diploma programme",
        "OPPE exam rules and camera setup",
    ]

    for query in test_queries:
        print(f"\n  Query: '{query}'")
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=query,
            task_type="retrieval_query",
        )
        vec = result["embedding"]

        results = qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=vec,
            using="dense",
            limit=3,
            with_payload=True,
        ).points

        print(f"  Top 3 results:")
        for r in results:
            print(f"    [{r.score:.3f}] {r.payload.get('chunk_type',''):15s} | {r.payload.get('breadcrumb','')[:50]}")

    type_counts = {}
    for chunk in chunks:
        t = chunk.get("chunk_type", "text")
        type_counts[t] = type_counts.get(t, 0) + 1

    print(f"\n  {'═' * 40}")
    print(f"  Collection:     {QDRANT_COLLECTION}")
    print(f"  Total points:   {count}")
    print(f"  Embedding dim:  {actual_dim} (Gemini text-embedding-004)")
    for t, c in sorted(type_counts.items()):
        print(f"    {t:20s}: {c}")
    print(f"\n  Hybrid search: vector({VECTOR_WEIGHT}) + BM25({BM25_WEIGHT})")
    print(f"\n  Next step: uvicorn main:app --port 8000")
    print("═" * 65)


if __name__ == "__main__":
    run()