from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum

class ResponseType(str, Enum):
    """Types of responses the system can generate."""
    RESEARCH = "research"
    MEMORY_RECALL = "memory_recall"
    ERROR = "error"
    PARTIAL = "partial"

class SourceInfo(BaseModel):
    """Information about a source used in the response."""
    name: str = Field(..., description="Source name (google, bing, reddit, memory)")
    available: bool = Field(..., description="Whether this source was successfully used")
    items_found: Optional[int] = Field(None, description="Number of items found from this source")
    confidence: Optional[float] = Field(None, description="Confidence score for this source", ge=0.0, le=1.0)

class MemoryContext(BaseModel):
    """Context about memory usage in the response."""
    research_items: int = Field(0, description="Number of research memories used")
    personal_items: int = Field(0, description="Number of personal memories used")
    total_items: int = Field(0, description="Total memory items considered")
    
    @validator('total_items', always=True)
    def calculate_total(cls, v, values):
        return values.get('research_items', 0) + values.get('personal_items', 0)

class ExecutionMetadata(BaseModel):
    """Metadata about the execution process."""
    processing_time: float = Field(..., description="Total processing time in seconds")
    tools_used: List[str] = Field(default_factory=list, description="List of tools that were executed")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal issues during execution")
    memory_updated: bool = Field(False, description="Whether new memory was stored")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the response was generated")

class ConflictingInfo(BaseModel):
    """Information about conflicting data from different sources."""
    topic: str = Field(..., description="What the conflict is about")
    sources: Dict[str, str] = Field(..., description="Different viewpoints by source")
    confidence: Optional[float] = Field(None, description="Confidence in resolving the conflict")

class StructuredAnswer(BaseModel):
    """Main structured answer format."""
    # Core content
    summary: str = Field(..., description="Brief summary of the answer")
    detailed_answer: str = Field(..., description="Comprehensive answer to the query")
    
    # Source breakdown
    facts: Optional[str] = Field(None, description="Factual information from authoritative sources")
    community_insights: Optional[str] = Field(None, description="Insights from community discussions")
    expert_opinions: Optional[str] = Field(None, description="Expert opinions and analysis")
    
    # Quality indicators
    confidence_level: float = Field(..., description="Overall confidence in the answer", ge=0.0, le=1.0)
    completeness: float = Field(..., description="How complete the answer is", ge=0.0, le=1.0)
    
    # Conflicts and limitations
    conflicting_information: List[ConflictingInfo] = Field(
        default_factory=list, 
        description="Information that conflicts between sources"
    )
    limitations: List[str] = Field(
        default_factory=list, 
        description="Known limitations or gaps in the answer"
    )
    
    # Follow-up suggestions
    related_questions: List[str] = Field(
        default_factory=list, 
        description="Suggested follow-up questions"
    )

class JarvisResponse(BaseModel):
    """Complete response format for Jarvis system."""
    # Request information
    query: str = Field(..., description="Original user query")
    response_type: ResponseType = Field(..., description="Type of response generated")
    
    # Main answer
    answer: StructuredAnswer = Field(..., description="Structured answer content")
    
    # Source information
    sources: Dict[str, SourceInfo] = Field(
        default_factory=dict, 
        description="Information about sources used"
    )
    
    # Memory context
    memory_context: MemoryContext = Field(
        default_factory=MemoryContext, 
        description="Information about memory usage"
    )
    
    # Execution information
    execution: ExecutionMetadata = Field(..., description="Execution metadata")
    
    # Success indicators
    success: bool = Field(..., description="Whether the response was generated successfully")
    errors: List[str] = Field(default_factory=list, description="Any errors that occurred")
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }
    
    def to_console_format(self) -> str:
        """Convert to a human-readable console format."""
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append(f"🤖 Jarvis Response ({self.response_type.value})")
        lines.append("=" * 80)
        
        # Query
        lines.append(f"\n📝 Query: {self.query}")
        
        # Main answer
        lines.append(f"\n📋 Summary: {self.answer.summary}")
        lines.append(f"\n📖 Detailed Answer:\n{self.answer.detailed_answer}")
        
        # Source breakdown
        if self.answer.facts:
            lines.append(f"\n📊 Facts:\n{self.answer.facts}")
        
        if self.answer.community_insights:
            lines.append(f"\n👥 Community Insights:\n{self.answer.community_insights}")
        
        if self.answer.expert_opinions:
            lines.append(f"\n🎓 Expert Opinions:\n{self.answer.expert_opinions}")
        
        # Quality indicators
        lines.append(f"\n🎯 Confidence: {self.answer.confidence_level:.1%}")
        lines.append(f"📈 Completeness: {self.answer.completeness:.1%}")
        
        # Conflicts
        if self.answer.conflicting_information:
            lines.append("\n⚠️ Conflicting Information:")
            for conflict in self.answer.conflicting_information:
                lines.append(f"  • {conflict.topic}")
                for source, viewpoint in conflict.sources.items():
                    lines.append(f"    - {source}: {viewpoint}")
        
        # Limitations
        if self.answer.limitations:
            lines.append("\n🚧 Limitations:")
            for limitation in self.answer.limitations:
                lines.append(f"  • {limitation}")
        
        # Sources used
        lines.append(f"\n🔍 Sources Used:")
        for source_name, source_info in self.sources.items():
            status = "✅" if source_info.available else "❌"
            lines.append(f"  {status} {source_name.title()}")
            if source_info.items_found:
                lines.append(f"     ({source_info.items_found} items)")
        
        # Memory context
        if self.memory_context.total_items > 0:
            lines.append(f"\n🧠 Memory Used:")
            lines.append(f"  • Research memories: {self.memory_context.research_items}")
            lines.append(f"  • Personal memories: {self.memory_context.personal_items}")
        
        # Follow-up suggestions
        if self.answer.related_questions:
            lines.append(f"\n💡 Related Questions:")
            for question in self.answer.related_questions:
                lines.append(f"  • {question}")
        
        # Execution info
        lines.append(f"\n⚡ Execution:")
        lines.append(f"  • Processing time: {self.execution.processing_time:.2f}s")
        lines.append(f"  • Tools used: {', '.join(self.execution.tools_used)}")
        if self.execution.memory_updated:
            lines.append(f"  • Memory updated: ✅")
        
        # Warnings and errors
        if self.execution.warnings:
            lines.append(f"\n⚠️ Warnings:")
            for warning in self.execution.warnings:
                lines.append(f"  • {warning}")
        
        if self.errors:
            lines.append(f"\n❌ Errors:")
            for error in self.errors:
                lines.append(f"  • {error}")
        
        lines.append("\n" + "=" * 80)
        
        return "\n".join(lines)
    
    def to_simple_format(self) -> str:
        """Convert to a simple format for basic usage."""
        return f"Query: {self.query}\n\nAnswer: {self.answer.detailed_answer}"

class ResponseBuilder:
    """Helper class to build structured responses."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset the builder for a new response."""
        self._query = ""
        self._response_type = ResponseType.RESEARCH
        self._answer_data = {}
        self._sources = {}
        self._memory_context = MemoryContext()
        self._execution_data = {}
        self._success = True
        self._errors = []
    
    def set_query(self, query: str) -> 'ResponseBuilder':
        """Set the original query."""
        self._query = query
        return self
    
    def set_response_type(self, response_type: ResponseType) -> 'ResponseBuilder':
        """Set the response type."""
        self._response_type = response_type
        return self
    
    def set_answer(
        self, 
        summary: str, 
        detailed_answer: str, 
        confidence: float = 0.8,
        completeness: float = 0.8
    ) -> 'ResponseBuilder':
        """Set the main answer content."""
        self._answer_data.update({
            "summary": summary,
            "detailed_answer": detailed_answer,
            "confidence_level": confidence,
            "completeness": completeness
        })
        return self
    
    def add_source(
        self, 
        name: str, 
        available: bool, 
        items_found: Optional[int] = None,
        confidence: Optional[float] = None
    ) -> 'ResponseBuilder':
        """Add source information."""
        self._sources[name] = SourceInfo(
            name=name,
            available=available,
            items_found=items_found,
            confidence=confidence
        )
        return self
    
    def set_memory_context(
        self, 
        research_items: int = 0, 
        personal_items: int = 0
    ) -> 'ResponseBuilder':
        """Set memory context."""
        self._memory_context = MemoryContext(
            research_items=research_items,
            personal_items=personal_items
        )
        return self
    
    def set_execution_metadata(
        self,
        processing_time: float,
        tools_used: List[str],
        warnings: List[str] = None,
        memory_updated: bool = False
    ) -> 'ResponseBuilder':
        """Set execution metadata."""
        self._execution_data = {
            "processing_time": processing_time,
            "tools_used": tools_used,
            "warnings": warnings or [],
            "memory_updated": memory_updated
        }
        return self
    
    def add_error(self, error: str) -> 'ResponseBuilder':
        """Add an error."""
        self._errors.append(error)
        self._success = False
        return self
    
    def add_conflicting_info(
        self, 
        topic: str, 
        sources: Dict[str, str], 
        confidence: Optional[float] = None
    ) -> 'ResponseBuilder':
        """Add conflicting information."""
        if "conflicting_information" not in self._answer_data:
            self._answer_data["conflicting_information"] = []
        
        self._answer_data["conflicting_information"].append(
            ConflictingInfo(topic=topic, sources=sources, confidence=confidence)
        )
        return self
    
    def add_limitation(self, limitation: str) -> 'ResponseBuilder':
        """Add a limitation to the response."""
        if "limitations" not in self._answer_data:
            self._answer_data["limitations"] = []
        
        self._answer_data["limitations"].append(limitation)
        return self
    
    def build(self) -> JarvisResponse:
        """Build the final structured response."""
        # Ensure required fields have defaults
        if not self._answer_data.get("summary"):
            self._answer_data["summary"] = "No summary available"
        
        if not self._answer_data.get("detailed_answer"):
            self._answer_data["detailed_answer"] = "No detailed answer available"
        
        if "confidence_level" not in self._answer_data:
            self._answer_data["confidence_level"] = 0.5
        
        if "completeness" not in self._answer_data:
            self._answer_data["completeness"] = 0.5
        
        # Build structured answer
        structured_answer = StructuredAnswer(**self._answer_data)
        
        # Build execution metadata
        execution = ExecutionMetadata(**self._execution_data)
        
        return JarvisResponse(
            query=self._query,
            response_type=self._response_type,
            answer=structured_answer,
            sources=self._sources,
            memory_context=self._memory_context,
            execution=execution,
            success=self._success,
            errors=self._errors
        )