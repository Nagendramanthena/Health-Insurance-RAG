import os
import json
import hashlib
import time
from typing import Optional, Dict, List, Tuple
from loguru import logger
import redis

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import (
    REDIS_URL,
    SEMANTIC_CACHE_THRESHOLD,
    SEMANTIC_CACHE_COLLECTION,
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
)

class SemanticCache:
    """
    Cognitive Semantic Cache for LLM responses.
    
    1. Query Normalization:
       - Uses a lightweight ChatOpenAI call (gpt-4o-mini) to translate conversational, 
         first-person user queries into standard, formal third-person policy search statements
         before vector search. This bridges the semantic gap (e.g. "asthma inhalers" -> "chronic pre-existing conditions and maintenance medications").
         
    2. Redis Mode:
       - Persists query text, embedding vector of normalized query, and response JSON in Redis.
       - Caches vectors and queries in Python memory on startup for sub-millisecond similarity search.
       - Uses plain cosine similarity in Python.
       - Does NOT require Redis Stack/RediSearch, making it 100% compatible with standard Redis (e.g. local redis-server on HF, Upstash, AWS).
       
    3. Local Fallback Mode:
       - Used if Redis is unavailable.
       - Stores vectors and response JSON in a local ChromaDB collection (`semantic_cache`).
       - Matches using ChromaDB's native vector similarity search on normalized query vectors.
    """
    def __init__(self):
        # Initialize Embeddings
        logger.info(f"Initializing SemanticCache embeddings with model {EMBEDDING_MODEL}...")
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        
        # Initialize query normalizer LLM
        logger.info("Initializing SemanticCache normalizer LLM (gpt-4o-mini)...")
        self.normalizer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        
        # Redis configuration
        self.redis_url = REDIS_URL
        self.redis_client: Optional[redis.Redis] = None
        self.redis_available = False
        
        # In-memory index of vectors for fast search in Redis Mode
        # Format: {cache_id: {"query": query_text, "vector": list[float], "plan_tier": plan_tier}}
        self._cache_memory: Dict[str, Dict] = {}
        
        # Local Fallback Store (Chroma)
        self.vector_store: Optional[Chroma] = None
        
        # Try to connect to Redis
        try:
            logger.info(f"Connecting to Redis at {self.redis_url}...")
            self.redis_client = redis.Redis.from_url(
                self.redis_url, 
                socket_timeout=1.5,
                socket_connect_timeout=1.5,
                decode_responses=True  # Decode hash values as string
            )
            self.redis_client.ping()
            self.redis_available = True
            logger.info("Successfully connected to Redis. Running in Redis Cache Mode.")
            
            # Load existing cache items into memory
            self._load_cache_into_memory()
            
        except Exception as e:
            logger.warning(f"Redis not available: {e}. Falling back to ChromaDB Local Cache Mode.")
            self.redis_available = False
            self._init_chroma_store()

    def _init_chroma_store(self):
        """Initialize Chroma collection for fallback caching."""
        try:
            self.vector_store = Chroma(
                persist_directory=CHROMA_PERSIST_DIR,
                embedding_function=self.embeddings,
                collection_name=SEMANTIC_CACHE_COLLECTION
            )
            logger.info(f"Initialized fallback ChromaDB cache collection: '{SEMANTIC_CACHE_COLLECTION}'")
        except Exception as e:
            logger.error(f"Failed to initialize fallback ChromaDB: {e}")

    def _load_cache_into_memory(self):
        """Pre-load all cached vectors from Redis into Python memory for fast scanning."""
        if not self.redis_client:
            return
            
        start_time = time.time()
        try:
            # Retrieve all cache IDs
            cache_ids = self.redis_client.smembers("cache:ids")
            if not cache_ids:
                logger.info("Redis cache is empty. In-memory index initialized empty.")
                return
                
            logger.info(f"Pre-loading {len(cache_ids)} cached queries from Redis...")
            
            # Fetch the query and vector hashes in pipelined execution
            pipeline = self.redis_client.pipeline()
            for cid in cache_ids:
                pipeline.hmget(f"cache:data:{cid}", ["query", "vector", "plan_tier"])
            
            results = pipeline.execute()
            
            loaded_count = 0
            for cid, (query, vector_str, plan_tier) in zip(cache_ids, results):
                if query and vector_str:
                    try:
                        vector = json.loads(vector_str)
                        self._cache_memory[cid] = {
                            "query": query,
                            "vector": vector,
                            "plan_tier": plan_tier or "Unknown"
                        }
                        loaded_count += 1
                    except Exception as ve:
                        logger.error(f"Error parsing vector for key {cid}: {ve}")
            
            logger.info(f"Loaded {loaded_count} cache keys into Python memory in {time.time() - start_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Error pre-loading cache from Redis: {e}")

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        OpenAI text-embedding-3-small vectors are already normalized (L2 norm = 1.0).
        Thus, cosine similarity is exactly the dot product.
        Capped between -1.0 and 1.0 to prevent float precision overflows.
        """
        if len(vec_a) != len(vec_b):
            return 0.0
        val = sum(a * b for a, b in zip(vec_a, vec_b))
        return min(1.0, max(-1.0, val))

    def normalize_query(self, query: str) -> str:
        """
        Normalize conversational, user-specific queries into formal, third-person health insurance search queries.
        e.g., "asthma inhaler wait" -> "waiting periods for pre-existing conditions and maintenance medications"
        """
        if not query or len(query.strip()) < 4:
            return query
            
        try:
            system_prompt = (
                "You are a health insurance query normalizer. Convert the user's conversational, first-person query "
                "into a standard, formal, third-person search query about health insurance policies, deductibles, "
                "coverage rules, or providers. Strip away personal details (like names, specific diagnosis dates, "
                "pronouns, conversational filler). Keep plan names and general terms. Keep it concise.\n\n"
                "Examples:\n"
                "- 'What is the policy regarding waiting periods for chronic, pre-existing conditions before the plan starts paying for specialist visits and maintenance medications?' "
                "-> 'Waiting periods for pre-existing conditions, specialist visits, and maintenance medications'\n"
                "- 'I was diagnosed with asthma last year and just signed up for this insurance. Do I have to wait a certain number of months before you guys will cover my inhalers and pulmonologist appointments, or am I good to go right now?' "
                "-> 'Waiting periods for pre-existing conditions, specialist visits, and maintenance medications'\n"
                "- 'Who are the skin doctors in Aurora?' -> 'In-network dermatologists in Aurora'\n"
                "- 'How much do I pay for Metformin on Silver?' -> 'Copay for Metformin on Silver plan'"
            )
            
            response = self.normalizer_llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=query)
            ])
            
            normalized = response.content.strip()
            logger.info(f"Normalized query: '{query[:50]}...' -> '{normalized}'")
            return normalized
        except Exception as e:
            logger.warning(f"Failed to normalize query: {e}. Using original query.")
            return query

    def _extract_entities(self, text: str) -> set:
        """Extract key plan tiers, drugs, specialties, and locations from query to prevent false semantic matches."""
        key_entities = {
            # Tiers
            "bronze", "silver", "gold",
            # Drugs
            "metformin", "lisinopril", "amlodipine", "atorvastatin", "lipitor", "omeprazole", 
            "levothyroxine", "sertraline", "escitalopram", "fluoxetine", "alprazolam", 
            "lorazepam", "metoprolol", "carvedilol", "furosemide", "hydrochlorothiazide", 
            "tirzepatide", "etanercept", "spironolactone",
            # Specialties
            "pulmonology", "oncology", "pediatrics", "ophthalmology", "cardiology", 
            "urology", "hematology", "rheumatology", "nephrology", "dermatology", "dermatologist",
            # Cities
            "chicago", "rockford", "miami", "joliet", "naperville", "brooklyn", 
            "queens", "oakland", "springfield"
        }
        text_lower = text.lower()
        found = set()
        for ent in key_entities:
            if ent in text_lower:
                found.add(ent)
        # Handle some common typos/synonyms
        if "silbr" in text_lower:
            found.add("silver")
        if "dermatologist" in found:
            found.add("dermatology")
        return found

    def check(self, query: str, plan_tier: str = "Unknown", normalized_query: Optional[str] = None) -> Optional[dict]:
        """
        Check the semantic cache for a match.
        
        Returns:
          dict containing the response data and hit metadata if found, else None.
        """
        if not query or len(query.strip()) < 4:
            return None
            
        start_time = time.time()
        try:
            # 1. Normalize query to standard terminology
            norm_query = normalized_query or self.normalize_query(query)
            
            # 2. Embed the normalized query
            query_vector = self.embeddings.embed_query(norm_query)
            
            if self.redis_available and self.redis_client:
                # Redis Mode: Scan memory cache for similarity
                if not self._cache_memory:
                    return None
                    
                best_id = None
                best_score = -1.0
                
                # Compute similarities in Python memory, filtering by plan_tier
                for cid, item in self._cache_memory.items():
                    # Filter by plan_tier case-insensitively
                    if item.get("plan_tier", "Unknown").lower() != plan_tier.lower():
                        continue
                        
                    # Entity safeguard check: ensure the key entities (tiers, drugs, specialties, locations)
                    # mentioned in the user's query match the cached item's query.
                    user_entities = self._extract_entities(norm_query)
                    cached_entities = self._extract_entities(item["query"])
                    if user_entities != cached_entities:
                        continue
                        
                    sim = self._cosine_similarity(query_vector, item["vector"])
                    if sim > best_score:
                        best_score = sim
                        best_id = cid
                
                # Check threshold
                if best_id and best_score >= SEMANTIC_CACHE_THRESHOLD:
                    # Cache Hit! Retrieve full response from Redis
                    cached_json = self.redis_client.hget(f"cache:data:{best_id}", "response")
                    if cached_json:
                        response = json.loads(cached_json)
                        # Inject cache metadata
                        response["cached"] = True
                        response["cache_similarity"] = round(best_score * 100, 1)
                        response["matched_query"] = self._cache_memory[best_id]["query"]
                        
                        logger.info(f"⚡ Redis Semantic Cache HIT (Plan: {plan_tier}, Similarity: {response['cache_similarity']}%) in {time.time() - start_time:.3f}s")
                        return response
                        
            else:
                # Fallback ChromaDB Mode
                if not self.vector_store:
                    self._init_chroma_store()
                if not self.vector_store:
                    return None
                    
                # Search ChromaDB with plan_tier filter using the normalized query
                results = self.vector_store.similarity_search_with_score(
                    norm_query, 
                    k=1, 
                    filter={"plan_tier": plan_tier}
                )
                if results:
                    doc, distance = results[0]
                    # Convert distance to similarity score
                    similarity = min(1.0, max(-1.0, 1.0 - (distance / 2.0)))
                    
                    if similarity >= SEMANTIC_CACHE_THRESHOLD:
                        # Entity safeguard check
                        user_entities = self._extract_entities(norm_query)
                        cached_query = doc.metadata.get("original_query", doc.page_content)
                        cached_entities = self._extract_entities(cached_query)
                        
                        if user_entities == cached_entities:
                            response_json = doc.metadata.get("response_json")
                            if response_json:
                                response = json.loads(response_json)
                                response["cached"] = True
                                response["cache_similarity"] = round(similarity * 100, 1)
                                response["matched_query"] = doc.metadata.get("original_query", doc.page_content)
                                
                                logger.info(f"⚡ Local ChromaDB Cache HIT (Plan: {plan_tier}, Similarity: {response['cache_similarity']}%) in {time.time() - start_time:.3f}s")
                                return response
                            
        except Exception as e:
            logger.error(f"Error checking semantic cache: {e}")
            
        return None

    def store(self, query: str, response: dict, plan_tier: str = "Unknown", normalized_query: Optional[str] = None) -> None:
        """
        Store query response in the semantic cache.
        """
        # Skip caching error responses or empty results
        if not query or not response or "error" in response:
            return
            
        try:
            # 1. Normalize query to standard terminology
            norm_query = normalized_query or self.normalize_query(query)
            
            # Generate deterministic cache ID based on normalized query
            cache_id = hashlib.sha256(norm_query.encode("utf-8")).hexdigest()
            query_vector = self.embeddings.embed_query(norm_query)
            
            # Remove any existing cache flags from the saved dictionary
            clean_response = response.copy()
            clean_response.pop("cached", None)
            clean_response.pop("cache_similarity", None)
            clean_response.pop("matched_query", None)
            
            response_json = json.dumps(clean_response)
            
            if self.redis_available and self.redis_client:
                # Store in Redis
                pipeline = self.redis_client.pipeline()
                # Hash contents
                pipeline.hset(f"cache:data:{cache_id}", mapping={
                    "query": query,  # original user query for UI
                    "vector": json.dumps(query_vector),
                    "response": response_json,
                    "plan_tier": plan_tier,
                    "timestamp": time.time()
                })
                # Add to ID set
                pipeline.sadd("cache:ids", cache_id)
                # Set TTL on the hash (e.g. 7 days = 604800 seconds)
                pipeline.expire(f"cache:data:{cache_id}", 604800)
                pipeline.execute()
                
                # Update Python memory index (storing normalized vector)
                self._cache_memory[cache_id] = {
                    "query": query,
                    "vector": query_vector,
                    "plan_tier": plan_tier
                }
                logger.info(f"💾 Query cached in Redis under ID cache:data:{cache_id[:8]}... (Plan: {plan_tier})")
                
            else:
                # Store in fallback ChromaDB
                if not self.vector_store:
                    self._init_chroma_store()
                if not self.vector_store:
                    return
                    
                # To prevent bloating, check if this query ID is already in Chroma
                existing = self.vector_store.get(ids=[cache_id])
                if existing and existing["ids"]:
                    # Update metadata
                    self.vector_store.update_document(
                        document_id=cache_id,
                        document=Document(
                            page_content=norm_query,
                            metadata={
                                "cache_id": cache_id, 
                                "original_query": query, 
                                "response_json": response_json, 
                                "plan_tier": plan_tier, 
                                "timestamp": time.time()
                            }
                        )
                    )
                    logger.info(f"💾 Updated query cache in ChromaDB under ID {cache_id[:8]}... (Plan: {plan_tier})")
                else:
                    # Add new
                    self.vector_store.add_texts(
                        texts=[norm_query],
                        metadatas=[{
                            "cache_id": cache_id, 
                            "original_query": query, 
                            "response_json": response_json, 
                            "plan_tier": plan_tier, 
                            "timestamp": time.time()
                        }],
                        ids=[cache_id]
                    )
                    logger.info(f"💾 Created query cache in ChromaDB under ID {cache_id[:8]}... (Plan: {plan_tier})")
                    
        except Exception as e:
            logger.error(f"Error storing query in semantic cache: {e}")

    def clear(self) -> None:
        """Clear all entries in the cache."""
        try:
            if self.redis_available and self.redis_client:
                cache_ids = self.redis_client.smembers("cache:ids")
                if cache_ids:
                    pipeline = self.redis_client.pipeline()
                    for cid in cache_ids:
                        pipeline.delete(f"cache:data:{cid}")
                    pipeline.delete("cache:ids")
                    pipeline.execute()
                self._cache_memory.clear()
                logger.info("Cleared Redis semantic cache.")
            else:
                if self.vector_store:
                    # Recreate collection to wipe it
                    self.vector_store.delete_collection()
                    self._init_chroma_store()
                    logger.info("Cleared local ChromaDB semantic cache.")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

# Global Cache Manager Singleton
cache_manager = SemanticCache()
