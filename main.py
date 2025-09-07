import asyncio
import logging
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from structured_response import JarvisResponse
from memory_manager import MemoryManager
from tool_registry import create_default_registry
from agent_controller import AgentController
from structured_response import ResponseBuilder, ResponseType
from web_operations import serp_search, reddit_search_api

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress verbose logs from external libraries
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)

class JarvisSystem:
    """
    Main Jarvis system that orchestrates all components.
    """
    
    def __init__(self):
        """Initialize all system components."""
        load_dotenv()
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.getenv("gemini_api_key")
        )
        
        # Initialize memory manager
        logger.info("🧠 Initializing memory system...")
        self.memory_manager = MemoryManager()
        
        # Initialize tool registry
        logger.info("🛠️ Initializing tool registry...")
        self.tool_registry = create_default_registry(
            serp_search_func=serp_search,
            reddit_search_func=reddit_search_api,
            memory_manager=self.memory_manager
        )
        
        # Initialize agent controller
        logger.info("🤖 Initializing agent controller...")
        self.agent_controller = AgentController(
            tool_registry=self.tool_registry,
            memory_manager=self.memory_manager,
            llm_client=self.llm
        )
        
        logger.info("✅ Jarvis system initialized successfully!")
    
    async def process_query(self, query: str, output_format: str = "console") -> str:
        """
        Process a user query and return formatted response.
        
        Args:
            query: User's question
            output_format: 'console', 'simple', or 'json'
            
        Returns:
            Formatted response string
        """
        try:
            # Process the query through the agent controller
            result = await self.agent_controller.process_query(query)
            
            if not result["success"]:
                return self._format_error_response(result, output_format)
            
            # Build structured response
            response = self._build_structured_response(result)
            
            # Format based on requested output
            if output_format == "console":
                return response.to_console_format()
            elif output_format == "simple":
                return response.to_simple_format()
            elif output_format == "json":
                return response.json(indent=2)
            else:
                return response.to_console_format()
                
        except Exception as e:
            logger.error(f"❌ Error processing query: {e}")
            return f"❌ System error: {str(e)}"
    
    def _build_structured_response(self, result: dict) -> 'JarvisResponse':
        """Build a structured response from agent controller result."""
        
        
        builder = ResponseBuilder()
        response_data = result["response"]
        
        # Set basic information
        builder.set_query(response_data.get("query", ""))
        
        # Determine response type
        intent = response_data.get("intent", "research")
        if intent == "memory_retrieval":
            builder.set_response_type(ResponseType.MEMORY_RECALL)
        else:
            builder.set_response_type(ResponseType.RESEARCH)
        
        # Set main answer
        answer_text = response_data.get("answer", "No answer generated")
        summary = self._extract_summary(answer_text)
        
        builder.set_answer(
            summary=summary,
            detailed_answer=answer_text,
            confidence=0.8,  # Can be enhanced with actual confidence calculation
            completeness=0.8
        )
        
        # Add source information
        sources = response_data.get("sources", {})
        for source_name, available in sources.items():
            builder.add_source(source_name, available)
        
        # Set memory context
        memory_context = response_data.get("memory_context", {})
        builder.set_memory_context(
            research_items=memory_context.get("research_items", 0),
            personal_items=memory_context.get("personal_items", 0)
        )
        
        # Set execution metadata
        execution_meta = response_data.get("execution_metadata", {})
        builder.set_execution_metadata(
            processing_time=result.get("execution_time", 0.0),
            tools_used=execution_meta.get("tools_used", []),
            warnings=result.get("warnings", []),
            memory_updated=result.get("memory_updated", False)
        )
        
        # Add any errors
        for error in result.get("errors", []):
            builder.add_error(error)
        
        return builder.build()
    
    def _extract_summary(self, answer_text: str, max_length: int = 150) -> str:
        """Extract a summary from the full answer text."""
        if not answer_text:
            return "No summary available"
        
        # Simple extraction: first sentence or first N characters
        sentences = answer_text.split('. ')
        if sentences and len(sentences[0]) <= max_length:
            return sentences[0] + '.'
        
        # Fallback to character limit
        if len(answer_text) <= max_length:
            return answer_text
        
        return answer_text[:max_length].rsplit(' ', 1)[0] + '...'
    
    def _format_error_response(self, result: dict, output_format: str) -> str:
        """Format an error response."""
        error_msg = f"❌ Processing failed: {', '.join(result.get('errors', ['Unknown error']))}"
        
        if output_format == "json":
            return f'{{"error": "{error_msg}", "success": false}}'
        else:
            return error_msg
    
    def get_system_status(self) -> dict:
        """Get system status information."""
        try:
            controller_status = self.agent_controller.get_status()
            return {
                "system": "operational",
                "components": {
                    "memory_manager": "connected",
                    "tool_registry": f"{len(self.tool_registry.tools)} tools loaded",
                    "agent_controller": "ready",
                    "llm": "connected"
                },
                "details": controller_status
            }
        except Exception as e:
            return {
                "system": "error",
                "error": str(e)
            }
    
    async def run_interactive_mode(self):
        """Run the system in interactive console mode."""
        print("🤖 Jarvis Research Assistant (Utility Track)")
        print("=" * 50)
        
        # Show system status
        status = self.get_system_status()
        if status["system"] == "operational":
            print("✅ System Status: All components operational")
            print(f"🛠️  Tools: {status['details']['tools']['total_tools']} available")
            print(f"🧠 Memory: {status['details']['memory']['total_memories']} memories stored")
        else:
            print(f"❌ System Status: {status.get('error', 'Unknown error')}")
            return
        
        print("\nType 'exit' to quit, 'status' for system info, 'memory' for memory stats")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n🔍 Ask me anything: ").strip()
                
                if user_input.lower() == 'exit':
                    print("\n👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'status':
                    status = self.get_system_status()
                    print(f"\n📊 System Status: {status['system']}")
                    if status["system"] == "operational":
                        details = status['details']
                        print(f"   Tools: {details['tools']['total_tools']}")
                        print(f"   Memory: {details['memory']['total_memories']} items")
                        print(f"   Agent: {details['agent_status']}")
                    continue
                
                elif user_input.lower() == 'memory':
                    stats = self.memory_manager.get_memory_stats()
                    print(f"\n🧠 Memory Statistics:")
                    print(f"   Research memories: {stats['research_memories']}")
                    print(f"   Personal memories: {stats['personal_memories']}")
                    print(f"   Total: {stats['total_memories']}")
                    continue
                
                elif not user_input:
                    continue
                
                print(f"\n🔄 Processing query...")
                
                # Process the query
                response = await self.process_query(user_input, "console")
                print(f"\n{response}")
                
            except KeyboardInterrupt:
                print(f"\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")

async def main():
    """Main entry point."""
    jarvis = None

    try:
        # Check for Weaviate connection
        print("🔌 Checking Weaviate connection...")
        print("   Make sure Weaviate is running: docker-compose up -d")
        print("   Weaviate should be available at http://localhost:8080")
        
        # Initialize system
        jarvis = JarvisSystem()
        
        # Run interactive mode
        await jarvis.run_interactive_mode()
        
    except Exception as e:
        logger.error(f"❌ Failed to start Jarvis system: {e}")
        print(f"\n❌ Startup failed: {str(e)}")
        print("\n🔧 Troubleshooting:")
        print("   1. Ensure Weaviate is running: docker-compose up -d")
        print("   2. Check your .env file has gemini_api_key and BRIGHTDATA_API_KEY")
        print("   3. Verify network connectivity")
    
    finally:
        if jarvis and jarvis.memory_manager:
            print("\n closing connections...")
            jarvis.memory_manager.close()

if __name__ == "__main__":
    asyncio.run(main())