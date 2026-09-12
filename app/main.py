import time
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from langsmith import traceable
from langsmith.utils import tracing_is_enabled
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.agent import ProductionRAGAgent
from app.cache import ResponseCache
from app.config import get_settings
from app.models import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
)
from app.monitoring import MetricsCollector, get_logger
from app.security import SecurityPipeline

load_dotenv()

logger = get_logger()

limiter = Limiter(key_func=get_remote_address)

# Global app state (initialized in lifespan)
security: Optional[SecurityPipeline] = None
cache: Optional[ResponseCache] = None
metrics: Optional[MetricsCollector] = None
agent: Optional[ProductionRAGAgent] = None


class Timer:
    def __init__(self):
        self.start_time = None
        self.elapsed_ms = 0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = int((time.time() - self.start_time) * 1000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initializes the application state before serving requests.
    """
    global security, cache, metrics, agent

    settings = get_settings()

    logger.info(
        "Initializing application...",
        extra={
            "extra_data": {
                "app_env": settings.app_env,
                "primary_model": settings.primary_model,
                "tracing_is_enabled": tracing_is_enabled(),
            }
        },
    )

    security = SecurityPipeline()
    cache = ResponseCache(ttl_seconds=settings.cache_ttl_seconds)
    metrics = MetricsCollector()
    agent = ProductionRAGAgent()

    logger.info("Application initialized.")

    yield

    if metrics is not None:
        logger.info(
            "Application shutdown.", extra={"extra_data": metrics.get_summary()}
        )


app = FastAPI(
    title="Production LangGraph RAG Application",
    description="A production-ready RAG application using LangGraph and FastAPI.",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(get_settings().rate_limit)
@traceable(name="chat_endpoint")
async def chat_endpoint(request: Request, body: ChatRequest):
    """
    Main Chat endpoint

    Handles incoming chat requests and returns a response.
    Flow:
        - Security check (injection + PI masking)
        - Cache lookup
        - LangGraph agent invocation (if cache miss)
        - Output validation
        - Cache store
        - Return response
    """
    if security is None or cache is None or agent is None or metrics is None:
        raise HTTPException(status_code=503, detail="Application not initialized")

    with Timer() as timer:
        security_notes = []

        # Step 1: Security check
        is_allowed, cleaned_message, notes = security.check_input(body.message)
        security_notes.extend(notes)

        if not is_allowed:
            logger.warning(
                "Request blocked by security check",
                extra={
                    "extra_data": {
                        "security_notes": security_notes,
                        "reason": notes,
                        "thread_id": body.thread_id,
                    }
                },
            )
            metrics.record_request(latency_ms=0, error=True)
            raise HTTPException(
                status_code=400, detail="Your message was blocked by security check"
            )

        # Step 2: Cache lookup
        cached_response = cache.get(cleaned_message)

        if cached_response is not None:
            metrics.record_request(latency_ms=0, cache_hit=True)
            logger.info(
                "Cache hit",
                extra={
                    "extra_data": {
                        "message": cleaned_message,
                        "thread_id": body.thread_id,
                    }
                },
            )
            return ChatResponse(
                response=cached_response,
                thread_id=body.thread_id,
                model_used="cached",
                cached=True,
                processing_time_ms=timer.elapsed_ms,
                security_notes=security_notes,
            )

        # Step 3: RAG processing
        try:
            result = agent.invoke(cleaned_message)
        except Exception as e:
            logger.error(
                f"Error during agent invocation: {str(e)}",
                extra={
                    "extra_data": {
                        "message": cleaned_message,
                        "thread_id": body.thread_id,
                        "error": str(e),
                    }
                },
            )
            metrics.record_request(latency_ms=0, error=True)
            raise HTTPException(status_code=500, detail=str(e))

        response_text = result["response"]
        model_used = result["model_used"]

        # Step 4: Output Validation

        validated_response, output_warnings = security.check_output(response_text)
        security_notes.extend(output_warnings)

        # Step 5: Cache Store
        cache.set(cleaned_message, validated_response)

    # Step 6: Return Response
    input_tokens = int(len(cleaned_message.split()) * 1.3)
    output_tokens = int(len(validated_response.split()) * 1.3)

    metrics.record_request(
        latency_ms=timer.elapsed_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        error=False,
        cache_hit=False,
    )

    return ChatResponse(
        response=validated_response,
        thread_id=body.thread_id,
        model_used=model_used,
        cached=False,
        processing_time_ms=timer.elapsed_ms,
        security_notes=security_notes,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    settings = get_settings()

    checks = {
        "agent": agent is not None,
        "security": security is not None,
        "cache": cache is not None,
    }

    all_healthy = all(checks.values())

    return HealthResponse(
        status="healthy" if all_healthy else "unhealthy",
        environment=settings.app_env,
        checks=checks,
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Metrics endpoint for monitoring dashboards"""
    if metrics is None:
        raise HTTPException(status_code=503, detail="Metrics not initialized")
    summary = metrics.get_summary()
    return MetricsResponse(**summary)


@app.get("/cache/stats")
async def get_cache_stats():
    """Cache statistics endpoint"""
    if cache is None:
        raise HTTPException(status_code=503, detail="Cache not initialized")
    return cache.stats
