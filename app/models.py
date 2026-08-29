"""
API Request and Response Models
Pydantic models for input validation and output serialization.
"""

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    message: str = Field(
        ...,
        description="The user's message to the agent",
        min_length=1,
        max_length=10000,
    )

    thread_id: str = Field(description="Conversation thread ID", default="default")


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""

    response: str
    thread_id: str
    model_used: str
    cached: bool
    processing_time: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HealthResponse(BaseModel):
    """Response model for health endpoint."""

    status: str = Field(default="healthy")
    environment: str = Field(default="development")
    version: str = Field(default="1.0.0")
    checks: dict = {}


class MetricsResponse(BaseModel):
    """Response model for metrics endpoint."""

    total_requests: int = Field(default=0)
    total_errors: int = Field(default=0)
    error_rate: str = Field(default="")
    avg_latency_ms: float = Field(default=0.0)
    cache_hit_rate: int = Field(default=0)
    total_input_tokens: int = Field(default=0)
    total_output_tokens: int = Field(default=0)


class ErrorResponse(BaseModel):
    """Response model for error endpoint."""

    error: str
    details: str | None = None
    request_id: str | None = None
