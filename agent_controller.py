import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class AgentState(Enum):
    """States that the agent can be in during execution."""
    IDLE = "idle"
    UNDERSTANDING = "understanding"
    MEMORY_CHECK = "memory_check"
    TOOL_SELECTION = "tool_selection"
    TOOL_EXECUTION = "tool_execution"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    RESPONSE_FORMATTING = "response_formatting"
    MEMORY_STORAGE = "memory_storage"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class ExecutionContext:
    """Context object that holds all execution state."""
    query: str
    state: AgentState = AgentState.IDLE
    start_time: datetime = field(default_factory=datetime.now)
    
    # Tool results
    tool_results: Dict[str, Any] = field(default_factory=dict)
    
    # Memory context
    research_context: List[Dict[str, Any]] = field(default_factory=list)
    personal_context: List[Dict[str, Any]] = field(default_factory=list)
    
    # Analysis results
    analyses: Dict[str, str] = field(default_factory=dict)
    
    # Final response
    structured_response: Dict[str, Any] = field(default_factory=dict)
    final_answer: str = ""
    
    # Error handling
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Metadata
    execution_time: float = 0.0
    tools_used: List[str] = field(default_factory=list)
    memory_updated: bool = False

class AgentController:
    """
    Main controller for the Jarvis agent system.
    Manages the complete execution flow from input to response.
    """
    
    def __init__(self, tool_registry, memory_manager, llm_client):
        """
        Initialize the agent controller.
        
        Args:
            tool_registry: ToolRegistry instance
            memory_manager: MemoryManager instance
            llm_client: LLM client for analysis
        """
        self.tool_registry = tool_registry
        self.memory_manager = memory_manager
        self.llm = llm_client
        
        # Execution pipeline
        self.pipeline_steps = [
            self._understand_query,
            self._check_memory,
            self._select_tools,
            self._execute_tools,
            self._analyze_results,
            self._synthesize_response,
            self._format_response,
            self._store_memory
        ]
        
        logger.info("🤖 AgentController initialized")
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Main entry point for processing a user query.
        
        Args:
            query: User's question/request
            
        Returns:
            Dictionary containing the structured response
        """
        context = ExecutionContext(query=query)
        
        try:
            logger.info(f"🚀 Starting query processing: {query[:100]}...")
            
            # Execute pipeline
            for step in self.pipeline_steps:
                try:
                    await step(context)
                    if context.state == AgentState.ERROR:
                        break
                except Exception as e:
                    logger.error(f"❌ Pipeline step failed: {step.__name__}: {e}")
                    context.errors.append(f"Step {step.__name__} failed: {str(e)}")
                    context.state = AgentState.ERROR
                    break
            
            # Calculate execution time
            context.execution_time = (datetime.now() - context.start_time).total_seconds()
            
            # Prepare final response
            response = self._prepare_final_response(context)
            
            logger.info(f"✅ Query processed successfully in {context.execution_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"❌ Critical error in query processing: {e}")
            return self._prepare_error_response(context, str(e))
    
    async def _understand_query(self, context: ExecutionContext):
        """Step 1: Understand and parse the user query."""
        context.state = AgentState.UNDERSTANDING
        logger.info("🧠 Understanding query...")
        
        # Basic query analysis
        query_lower = context.query.lower()
        
        # Set query metadata
        context.structured_response["query"] = context.query
        context.structured_response["timestamp"] = context.start_time.isoformat()
        
        # Simple intent detection (can be enhanced with ML later)
        if any(word in query_lower for word in ["remember", "recall", "what did", "you told me"]):
            context.structured_response["intent"] = "memory_retrieval"
        else:
            context.structured_response["intent"] = "research"
        
        logger.info(f"📝 Query understood - Intent: {context.structured_response['intent']}")
    
    async def _check_memory(self, context: ExecutionContext):
        """Step 2: Check memory for relevant context."""
        context.state = AgentState.MEMORY_CHECK
        logger.info("🧠 Checking memory for context...")
        
        try:
            # Execute memory retrieval tool
            memory_result = await self.tool_registry.execute_tool(
                "memory_retrieve",
                query=context.query
            )
            
            if memory_result.success:
                context.research_context = memory_result.data.get("research_context", [])
                context.personal_context = memory_result.data.get("personal_context", [])
                
                logger.info(f"📚 Found {len(context.research_context)} research memories, {len(context.personal_context)} personal memories")
            else:
                logger.warning(f"⚠️ Memory retrieval failed: {memory_result.error}")
                context.warnings.append(f"Memory retrieval failed: {memory_result.error}")
                
        except Exception as e:
            logger.error(f"❌ Memory check error: {e}")
            context.warnings.append(f"Memory check error: {str(e)}")
    
    async def _select_tools(self, context: ExecutionContext):
        """Step 3: Select which tools to use for this query."""
        context.state = AgentState.TOOL_SELECTION
        logger.info("🛠️ Selecting tools...")
        
        # Default tool selection logic (can be enhanced with LLM reasoning)
        selected_tools = []
        
        # Always use search tools for research queries
        if context.structured_response["intent"] == "research":
            selected_tools.extend(["google_search", "bing_search", "reddit_search"])
        
        # For memory-focused queries, prioritize memory but still do some research
        elif context.structured_response["intent"] == "memory_retrieval":
            if not context.research_context and not context.personal_context:
                # No relevant memory found, do fresh research
                selected_tools.extend(["google_search", "bing_search"])
        
        context.tools_used = selected_tools
        logger.info(f"🎯 Selected tools: {', '.join(selected_tools)}")
    
    async def _execute_tools(self, context: ExecutionContext):
        """Step 4: Execute selected tools concurrently."""
        context.state = AgentState.TOOL_EXECUTION
        logger.info("⚡ Executing tools...")
        
        if not context.tools_used:
            logger.info("ℹ️ No tools to execute, using memory context only")
            return
        
        # Execute tools concurrently
        tasks = []
        for tool_name in context.tools_used:
            task = self.tool_registry.execute_tool(tool_name, query=context.query)
            tasks.append((tool_name, task))
        
        # Wait for all tools to complete
        for tool_name, task in tasks:
            try:
                result = await task
                context.tool_results[tool_name] = result
                
                if result.success:
                    logger.info(f"✅ {tool_name} completed successfully")
                else:
                    logger.warning(f"⚠️ {tool_name} failed: {result.error}")
                    context.warnings.append(f"{tool_name} failed: {result.error}")
                    
            except Exception as e:
                logger.error(f"❌ {tool_name} execution error: {e}")
                context.errors.append(f"{tool_name} execution error: {str(e)}")
    
    async def _analyze_results(self, context: ExecutionContext):
        """Step 5: Analyze results from each source."""
        context.state = AgentState.ANALYSIS
        logger.info("🔍 Analyzing results...")
        
        # Import analysis functions
        from prompts import (
            get_google_analysis_messages,
            get_bing_analysis_messages, 
            get_reddit_analysis_messages
        )
        
        # Analyze Google results
        if "google_search" in context.tool_results:
            google_result = context.tool_results["google_search"]
            if google_result.success:
                try:
                    messages = get_google_analysis_messages(context.query, str(google_result.data))
                    reply = self.llm.invoke(messages)
                    context.analyses["google"] = reply.content
                    logger.info("✅ Google analysis completed")
                except Exception as e:
                    logger.error(f"❌ Google analysis failed: {e}")
                    context.warnings.append(f"Google analysis failed: {str(e)}")
        
        # Analyze Bing results
        if "bing_search" in context.tool_results:
            bing_result = context.tool_results["bing_search"]
            if bing_result.success:
                try:
                    messages = get_bing_analysis_messages(context.query, str(bing_result.data))
                    reply = self.llm.invoke(messages)
                    context.analyses["bing"] = reply.content
                    logger.info("✅ Bing analysis completed")
                except Exception as e:
                    logger.error(f"❌ Bing analysis failed: {e}")
                    context.warnings.append(f"Bing analysis failed: {str(e)}")
        
        # Analyze Reddit results
        if "reddit_search" in context.tool_results:
            reddit_result = context.tool_results["reddit_search"]
            if reddit_result.success:
                try:
                    messages = get_reddit_analysis_messages(
                        context.query, 
                        str(reddit_result.data), 
                        []  # Post data would be retrieved separately
                    )
                    reply = self.llm.invoke(messages)
                    context.analyses["reddit"] = reply.content
                    logger.info("✅ Reddit analysis completed")
                except Exception as e:
                    logger.error(f"❌ Reddit analysis failed: {e}")
                    context.warnings.append(f"Reddit analysis failed: {str(e)}")
        
        logger.info(f"📊 Completed {len(context.analyses)} analyses")
    
    async def _synthesize_response(self, context: ExecutionContext):
        """Step 6: Synthesize all information into a coherent response."""
        context.state = AgentState.SYNTHESIS
        logger.info("🔗 Synthesizing response...")
        
        try:
            from prompts import get_synthesis_messages
            
            # Prepare synthesis input
            google_analysis = context.analyses.get("google", "")
            bing_analysis = context.analyses.get("bing", "")
            reddit_analysis = context.analyses.get("reddit", "")
            
            # Include memory context if available
            memory_context = ""
            if context.research_context or context.personal_context:
                memory_parts = []
                
                if context.research_context:
                    memory_parts.append("Previous research:")
                    for memory in context.research_context[:2]:  # Limit to most relevant
                        memory_parts.append(f"- Q: {memory.get('query', '')}")
                        memory_parts.append(f"  A: {memory.get('answer', '')[:200]}...")
                
                if context.personal_context:
                    memory_parts.append("\nPersonal context:")
                    for memory in context.personal_context[:3]:  # Limit to most relevant
                        memory_parts.append(f"- {memory.get('content', '')}")
                
                memory_context = "\n".join(memory_parts)
            
            # Create enhanced synthesis prompt that includes memory
            messages = get_synthesis_messages(
                context.query,
                google_analysis + "\n\nMemory Context:\n" + memory_context if memory_context else google_analysis,
                bing_analysis,
                reddit_analysis
            )
            
            reply = self.llm.invoke(messages)
            context.final_answer = reply.content
            
            logger.info("✅ Response synthesis completed")
            
        except Exception as e:
            logger.error(f"❌ Synthesis failed: {e}")
            context.errors.append(f"Synthesis failed: {str(e)}")
            
            # Fallback synthesis
            context.final_answer = self._create_fallback_response(context)
    
    async def _format_response(self, context: ExecutionContext):
        """Step 7: Format the response into structured output."""
        context.state = AgentState.RESPONSE_FORMATTING
        logger.info("📋 Formatting response...")
        
        # Create structured response
        context.structured_response.update({
            "answer": context.final_answer,
            "sources": {
                "google": bool(context.analyses.get("google")),
                "bing": bool(context.analyses.get("bing")), 
                "reddit": bool(context.analyses.get("reddit")),
                "memory": bool(context.research_context or context.personal_context)
            },
            "memory_context": {
                "research_items": len(context.research_context),
                "personal_items": len(context.personal_context)
            },
            "execution_metadata": {
                "tools_used": context.tools_used,
                "warnings": context.warnings,
                "processing_time": (datetime.now() - context.start_time).total_seconds()
            }
        })
        
        logger.info("📝 Response formatted successfully")
    
    async def _store_memory(self, context: ExecutionContext):
        """Step 8: Store the interaction in memory for future reference."""
        context.state = AgentState.MEMORY_STORAGE
        logger.info("💾 Storing memory...")
        
        try:
            # Store research memory
            uuid = self.memory_manager.store_research_memory(
                query=context.query,
                answer=context.final_answer,
                google_analysis=context.analyses.get("google", ""),
                bing_analysis=context.analyses.get("bing", ""),
                reddit_analysis=context.analyses.get("reddit", ""),
                sources_used=context.tools_used
            )
            
            if uuid:
                context.memory_updated = True
                logger.info(f"✅ Memory stored with UUID: {uuid}")
            else:
                logger.warning("⚠️ Failed to store memory")
                context.warnings.append("Failed to store memory")
                
        except Exception as e:
            logger.error(f"❌ Memory storage failed: {e}")
            context.warnings.append(f"Memory storage failed: {str(e)}")
        
        # Mark as completed
        context.state = AgentState.COMPLETED
    
    def _create_fallback_response(self, context: ExecutionContext) -> str:
        """Create a fallback response when synthesis fails."""
        parts = []
        
        if context.analyses.get("google"):
            parts.append(f"From Google: {context.analyses['google'][:300]}...")
        
        if context.analyses.get("bing"):
            parts.append(f"From Bing: {context.analyses['bing'][:300]}...")
        
        if context.analyses.get("reddit"):
            parts.append(f"From Reddit: {context.analyses['reddit'][:300]}...")
        
        if not parts and (context.research_context or context.personal_context):
            parts.append("Based on previous research and personal context...")
            if context.research_context:
                parts.append(f"Previous research found: {context.research_context[0].get('answer', '')[:300]}...")
        
        return "\n\n".join(parts) if parts else "I apologize, but I encountered issues processing your request. Please try again."
    
    def _prepare_final_response(self, context: ExecutionContext) -> Dict[str, Any]:
        """Prepare the final response dictionary."""
        return {
            "success": context.state == AgentState.COMPLETED,
            "response": context.structured_response,
            "execution_time": context.execution_time,
            "memory_updated": context.memory_updated,
            "errors": context.errors,
            "warnings": context.warnings,
            "state": context.state.value
        }
    
    def _prepare_error_response(self, context: ExecutionContext, error: str) -> Dict[str, Any]:
        """Prepare an error response."""
        return {
            "success": False,
            "response": {
                "query": context.query,
                "answer": "I apologize, but I encountered an error while processing your request. Please try again.",
                "error": error
            },
            "execution_time": (datetime.now() - context.start_time).total_seconds(),
            "memory_updated": False,
            "errors": context.errors + [error],
            "warnings": context.warnings,
            "state": AgentState.ERROR.value
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the agent controller."""
        tool_stats = self.tool_registry.get_registry_stats()
        memory_stats = self.memory_manager.get_memory_stats()
        
        return {
            "agent_status": "ready",
            "tools": tool_stats,
            "memory": memory_stats,
            "pipeline_steps": len(self.pipeline_steps)
        }