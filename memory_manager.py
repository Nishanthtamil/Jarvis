import json
import logging
import weaviate
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

# Import specific v4 classes for configuration and querying
from weaviate.classes.config import Property, DataType
from weaviate.classes.query import MetadataQuery

logger = logging.getLogger(__name__)

class MemoryManager:
    """
    Manages persistent memory storage for Jarvis system using Weaviate v4.
    Handles both research memory and personal memory.
    """

    def __init__(self, weaviate_url: str = "http://localhost:8080"):
        """Initialize connection to Weaviate and embedding model."""
        try:
            # v4 connection method - already correct
            self.client = weaviate.connect_to_local()
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self._initialize_schemas()
            logger.info("✅ MemoryManager initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize MemoryManager: {e}")
            raise

    def _initialize_schemas(self):
        """Initialize Weaviate schemas for different memory types using v4 syntax."""
        try:
            # v4: Get all existing collection names
            existing_collections = {c.name for c in self.client.collections.list_all().values()}

            # --- Research Memory Schema Definition ---
            if "ResearchMemory" not in existing_collections:
                self.client.collections.create(
                    name="ResearchMemory",
                    description="Storage for research queries and comprehensive answers",
                    properties=[
                        Property(name="query", data_type=DataType.TEXT, description="Original user question"),
                        Property(name="answer", data_type=DataType.TEXT, description="Comprehensive research answer"),
                        Property(name="sources_used", data_type=DataType.TEXT_ARRAY, description="List of sources used"),
                        Property(name="google_analysis", data_type=DataType.TEXT, description="Google search analysis"),
                        Property(name="bing_analysis", data_type=DataType.TEXT, description="Bing search analysis"),
                        Property(name="reddit_analysis", data_type=DataType.TEXT, description="Reddit community analysis"),
                        Property(name="timestamp", data_type=DataType.DATE, description="When this research was conducted"),
                        Property(name="query_hash", data_type=DataType.TEXT, description="Hash of the query for deduplication"),
                    ]
                )
                logger.info("✅ Created ResearchMemory schema")

            # --- Personal Memory Schema Definition ---
            if "PersonalMemory" not in existing_collections:
                self.client.collections.create(
                    name="PersonalMemory",
                    description="Storage for personal information and context",
                    properties=[
                        Property(name="context_type", data_type=DataType.TEXT, description="Type of personal context"),
                        Property(name="content", data_type=DataType.TEXT, description="The personal information or context"),
                        Property(name="associated_query", data_type=DataType.TEXT, description="Query that led to this info"),
                        Property(name="timestamp", data_type=DataType.DATE, description="When this was learned"),
                        Property(name="relevance_score", data_type=DataType.NUMBER, description="How important this info is"),
                    ]
                )
                logger.info("✅ Created PersonalMemory schema")

        except Exception as e:
            logger.error(f"❌ Error creating schemas: {e}")

    def _generate_query_hash(self, query: str) -> str:
        """Generate a hash for query deduplication."""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()

    def store_research_memory(
        self,
        query: str,
        answer: str,
        google_analysis: str = "",
        bing_analysis: str = "",
        reddit_analysis: str = "",
        sources_used: List[str] = None
    ) -> str:
        """
        Store a complete research session in memory.
        Returns the UUID of the stored memory.
        """
        try:
            # Check if similar query already exists
            existing = self._find_similar_research(query, similarity_threshold=0.85)
            if existing:
                logger.info(f"⚠️ Similar query found, updating existing memory: {existing.uuid}")
                return self._update_research_memory(existing.uuid, answer, google_analysis, bing_analysis, reddit_analysis)

            embedding = self.embedder.encode(query).tolist()
            query_hash = self._generate_query_hash(query)

            memory_data = {
                "query": query,
                "answer": answer,
                "google_analysis": google_analysis,
                "bing_analysis": bing_analysis,
                "reddit_analysis": reddit_analysis,
                "sources_used": sources_used or [],
                "timestamp": datetime.now(),
                "query_hash": query_hash
            }

            # v4: Get collection and insert data
            research_collection = self.client.collections.get("ResearchMemory")
            uuid_obj = research_collection.data.insert(
                properties=memory_data,
                vector=embedding
            )

            logger.info(f"✅ Stored research memory: {uuid_obj}")
            return str(uuid_obj)

        except Exception as e:
            logger.error(f"❌ Error storing research memory: {e}")
            return ""

    def _update_research_memory(
        self,
        uuid: str,
        answer: str,
        google_analysis: str,
        bing_analysis: str,
        reddit_analysis: str
    ) -> str:
        """Update existing research memory with new information."""
        try:
            update_data = {
                "answer": answer,
                "google_analysis": google_analysis,
                "bing_analysis": bing_analysis,
                "reddit_analysis": reddit_analysis,
                "timestamp": datetime.now()
            }

            # v4: Get collection and update data
            research_collection = self.client.collections.get("ResearchMemory")
            research_collection.data.update(
                uuid=uuid,
                properties=update_data
            )

            logger.info(f"✅ Updated research memory: {uuid}")
            return uuid

        except Exception as e:
            logger.error(f"❌ Error updating research memory: {e}")
            return ""

    def store_personal_memory(
        self,
        context_type: str,
        content: str,
        associated_query: str = "",
        relevance_score: float = 1.0
    ) -> str:
        """
        Store personal context/information.
        Returns the UUID of the stored memory.
        """
        try:
            embedding = self.embedder.encode(content).tolist()
            memory_data = {
                "context_type": context_type,
                "content": content,
                "associated_query": associated_query,
                "timestamp": datetime.now(),
                "relevance_score": relevance_score
            }

            # v4: Get collection and insert data
            personal_collection = self.client.collections.get("PersonalMemory")
            uuid_obj = personal_collection.data.insert(
                properties=memory_data,
                vector=embedding
            )

            logger.info(f"✅ Stored personal memory: {uuid_obj}")
            return str(uuid_obj)

        except Exception as e:
            logger.error(f"❌ Error storing personal memory: {e}")
            return ""

    def retrieve_research_context(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant research memories for a given query.
        Returns list of relevant past research sessions.
        """
        try:
            query_embedding = self.embedder.encode(query).tolist()

            # v4: Get collection and perform query
            research_collection = self.client.collections.get("ResearchMemory")
            response = research_collection.query.near_vector(
                near_vector=query_embedding,
                limit=limit,
                return_metadata=MetadataQuery(certainty=True) # Certainty is now in metadata
            )

            # Process and filter results from the new response object
            relevant_memories = []
            for item in response.objects:
                if item.metadata.certainty > 0.7:
                    memory_dict = item.properties
                    memory_dict['_additional'] = {'certainty': item.metadata.certainty, 'id': item.uuid}
                    relevant_memories.append(memory_dict)

            logger.info(f"📚 Retrieved {len(relevant_memories)} relevant research memories")
            return relevant_memories

        except Exception as e:
            logger.error(f"❌ Error retrieving research context: {e}")
            return []

    def retrieve_personal_context(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant personal context for a given query.
        Returns list of relevant personal information.
        """
        try:
            query_embedding = self.embedder.encode(query).tolist()

            # v4: Get collection and perform query
            personal_collection = self.client.collections.get("PersonalMemory")
            response = personal_collection.query.near_vector(
                near_vector=query_embedding,
                limit=limit,
                return_metadata=MetadataQuery(certainty=True)
            )

            relevant_memories = []
            for item in response.objects:
                if item.metadata.certainty > 0.6:
                    memory_dict = item.properties
                    memory_dict['_additional'] = {'certainty': item.metadata.certainty, 'id': item.uuid}
                    relevant_memories.append(memory_dict)

            logger.info(f"👤 Retrieved {len(relevant_memories)} relevant personal memories")
            return relevant_memories

        except Exception as e:
            logger.error(f"❌ Error retrieving personal context: {e}")
            return []

    def _find_similar_research(self, query: str, similarity_threshold: float = 0.8) -> Optional[Any]:
        """Find if a similar research query already exists, returns the v4 Object."""
        try:
            query_embedding = self.embedder.encode(query).tolist()
            
            research_collection = self.client.collections.get("ResearchMemory")
            response = research_collection.query.near_vector(
                near_vector=query_embedding,
                limit=1,
                return_metadata=MetadataQuery(certainty=True)
            )

            if response.objects and response.objects[0].metadata.certainty > similarity_threshold:
                return response.objects[0]
            return None

        except Exception as e:
            logger.error(f"❌ Error finding similar research: {e}")
            return None

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memories using v4 aggregation."""
        try:
            # v4: Much simpler aggregation
            research_collection = self.client.collections.get("ResearchMemory")
            personal_collection = self.client.collections.get("PersonalMemory")
            
            research_agg = research_collection.aggregate.over_all(total_count=True)
            personal_agg = personal_collection.aggregate.over_all(total_count=True)

            research_total = research_agg.total_count
            personal_total = personal_agg.total_count   

            return {
                "research_memories": research_total,
                "personal_memories": personal_total,
                "total_memories": research_total + personal_total
            }

        except Exception as e:
            logger.error(f"❌ Error getting memory stats: {e}")
            return {"research_memories": 0, "personal_memories": 0, "total_memories": 0}

    def clear_memories(self, memory_type: str = "all") -> bool:
        """Clear stored memories. Use with caution!"""
        try:
            if memory_type in ["all", "research"]:
                # v4: Use client.collections.delete()
                self.client.collections.delete("ResearchMemory")
                logger.info("🗑️ Cleared research memories")

            if memory_type in ["all", "personal"]:
                self.client.collections.delete("PersonalMemory")
                logger.info("🗑️ Cleared personal memories")

            # Recreate schemas
            self._initialize_schemas()
            return True

        except Exception as e:
            logger.error(f"❌ Error clearing memories: {e}")
            return False
            
    def close(self):
        """Close the Weaviate client connection."""
        try:
            self.client.close()
            logger.info("🔌 Weaviate connection closed.")
        except Exception as e:
            logger.error(f"❌ Error closing Weaviate connection: {e}")