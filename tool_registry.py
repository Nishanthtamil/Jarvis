import logging
from typing import Dict, Any, Callable, List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class ToolResult:
    """Standardized result from tool execution."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class BaseTool(ABC):
    """Base class for all tools in the registry."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get tool information for LLM reasoning."""
        return {
            "name": self.name,
            "description": self.description
        }

class GoogleSearchTool(BaseTool):
    """Google search tool implementation."""
    
    def __init__(self, search_function: Callable):
        super().__init__(
            name="google_search",
            description="Search Google for factual information, official sources, and authoritative content"
        )
        self.search_function = search_function
    
    async def execute(self, query: str, **kwargs) -> ToolResult:
        """Execute Google search."""
        try:
            logger.info(f"🔍 Google searching: {query}")
            results = self.search_function(query, engine="google")
            
            if results:
                return ToolResult(
                    success=True,
                    data=results,
                    metadata={"source": "google", "query": query}
                )
            else:
                return ToolResult(
                    success=False,
                    error="No results returned from Google search"
                )
                
        except Exception as e:
            logger.error(f"❌ Google search error: {e}")
            return ToolResult(
                success=False,
                error=f"Google search failed: {str(e)}"
            )

class BingSearchTool(BaseTool):
    """Bing search tool implementation."""
    
    def __init__(self, search_function: Callable):
        super().__init__(
            name="bing_search",
            description="Search Bing for complementary information, Microsoft ecosystem content, and technical documentation"
        )
        self.search_function = search_function
    
    async def execute(self, query: str, **kwargs) -> ToolResult:
        """Execute Bing search."""
        try:
            logger.info(f"🔍 Bing searching: {query}")
            results = self.search_function(query, engine="bing")
            
            if results:
                return ToolResult(
                    success=True,
                    data=results,
                    metadata={"source": "bing", "query": query}
                )
            else:
                return ToolResult(
                    success=False,
                    error="No results returned from Bing search"
                )
                
        except Exception as e:
            logger.error(f"❌ Bing search error: {e}")
            return ToolResult(
                success=False,
                error=f"Bing search failed: {str(e)}"
            )

class RedditSearchTool(BaseTool):
    """Reddit search tool implementation."""
    
    def __init__(self, search_function: Callable):
        super().__init__(
            name="reddit_search",
            description="Search Reddit for community insights, user experiences, and social discussions"
        )
        self.search_function = search_function
    
    async def execute(self, query: str, **kwargs) -> ToolResult:
        """Execute Reddit search."""
        try:
            logger.info(f"🔍 Reddit searching: {query}")
            results = self.search_function(query)
            
            if results:
                return ToolResult(
                    success=True,
                    data=results,
                    metadata={"source": "reddit", "query": query}
                )
            else:
                return ToolResult(
                    success=False,
                    error="No results returned from Reddit search"
                )
                
        except Exception as e:
            logger.error(f"❌ Reddit search error: {e}")
            return ToolResult(
                success=False,
                error=f"Reddit search failed: {str(e)}"
            )

class MemoryRetrievalTool(BaseTool):
    """Memory retrieval tool implementation."""
    
    def __init__(self, memory_manager):
        super().__init__(
            name="memory_retrieve",
            description="Retrieve relevant information from past research sessions and personal context"
        )
        self.memory_manager = memory_manager
    
    async def execute(self, query: str, memory_type: str = "both", **kwargs) -> ToolResult:
        """Execute memory retrieval."""
        try:
            logger.info(f"🧠 Memory retrieving: {query} (type: {memory_type})")
            
            results = {"research_context": [], "personal_context": []}
            
            if memory_type in ["both", "research"]:
                results["research_context"] = self.memory_manager.retrieve_research_context(query)
            
            if memory_type in ["both", "personal"]:
                results["personal_context"] = self.memory_manager.retrieve_personal_context(query)
            
            return ToolResult(
                success=True,
                data=results,
                metadata={
                    "source": "memory",
                    "query": query,
                    "memory_type": memory_type,
                    "research_items": len(results["research_context"]),
                    "personal_items": len(results["personal_context"])
                }
            )
                
        except Exception as e:
            logger.error(f"❌ Memory retrieval error: {e}")
            return ToolResult(
                success=False,
                error=f"Memory retrieval failed: {str(e)}"
            )

class ToolRegistry:
    """
    Registry for managing all available tools in the Jarvis system.
    Provides plug-and-play tool management.
    """
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        logger.info("🛠️ ToolRegistry initialized")
    
    def register_tool(self, tool: BaseTool) -> bool:
        """Register a new tool in the registry."""
        try:
            self.tools[tool.name] = tool
            logger.info(f"✅ Registered tool: {tool.name}")
            return True
        except Exception as e:
            logger.error(f"❌ Error registering tool {tool.name}: {e}")
            return False
    
    def unregister_tool(self, tool_name: str) -> bool:
        """Remove a tool from the registry."""
        try:
            if tool_name in self.tools:
                del self.tools[tool_name]
                logger.info(f"🗑️ Unregistered tool: {tool_name}")
                return True
            else:
                logger.warning(f"⚠️ Tool not found for unregistration: {tool_name}")
                return False
        except Exception as e:
            logger.error(f"❌ Error unregistering tool {tool_name}: {e}")
            return False
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self.tools.get(tool_name)
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools with their information."""
        return [tool.get_info() for tool in self.tools.values()]
    
    def get_tools_for_llm(self) -> str:
        """Get formatted tool descriptions for LLM reasoning."""
        if not self.tools:
            return "No tools available."
        
        tool_descriptions = []
        for tool in self.tools.values():
            tool_descriptions.append(f"- {tool.name}: {tool.description}")
        
        return "Available tools:\n" + "\n".join(tool_descriptions)
    
    async def execute_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool by name with given parameters."""
        tool = self.get_tool(tool_name)
        
        if not tool:
            return ToolResult(
                success=False,
                error=f"Tool '{tool_name}' not found in registry"
            )
        
        try:
            return await tool.execute(**kwargs)
        except Exception as e:
            logger.error(f"❌ Error executing tool {tool_name}: {e}")
            return ToolResult(
                success=False,
                error=f"Tool execution failed: {str(e)}"
            )
    
    def tool_exists(self, tool_name: str) -> bool:
        """Check if a tool exists in the registry."""
        return tool_name in self.tools
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get statistics about the tool registry."""
        return {
            "total_tools": len(self.tools),
            "tool_names": list(self.tools.keys()),
            "tools_by_category": self._categorize_tools()
        }
    
    def _categorize_tools(self) -> Dict[str, List[str]]:
        """Categorize tools by their type."""
        categories = {
            "search": [],
            "memory": [],
            "other": []
        }
        
        for tool_name, tool in self.tools.items():
            if "search" in tool_name.lower():
                categories["search"].append(tool_name)
            elif "memory" in tool_name.lower():
                categories["memory"].append(tool_name)
            else:
                categories["other"].append(tool_name)
        
        return categories

def create_default_registry(serp_search_func, reddit_search_func, memory_manager) -> ToolRegistry:
    """
    Create a tool registry with default tools for the Jarvis system.
    
    Args:
        serp_search_func: Function for Google/Bing search
        reddit_search_func: Function for Reddit search
        memory_manager: Memory manager instance
    
    Returns:
        Configured ToolRegistry instance
    """
    registry = ToolRegistry()
    
    # Register search tools
    google_tool = GoogleSearchTool(serp_search_func)
    bing_tool = BingSearchTool(serp_search_func)
    reddit_tool = RedditSearchTool(reddit_search_func)
    memory_tool = MemoryRetrievalTool(memory_manager)
    
    registry.register_tool(google_tool)
    registry.register_tool(bing_tool)
    registry.register_tool(reddit_tool)
    registry.register_tool(memory_tool)
    
    logger.info("🎯 Default tool registry created with 4 tools")
    return registry