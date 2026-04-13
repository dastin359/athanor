"""
Arize Phoenix / OpenTelemetry integration for ARC solver observability.

This module provides optional Phoenix instrumentation alongside the existing
custom observability system. Both systems can run in parallel:
- Custom logs: Detailed markdown logs in gpt5_logs/
- Phoenix: Real-time traces viewable in Phoenix UI at http://127.0.0.1:6006

To enable Phoenix:
1. Install dependencies:
   - pip install arize-phoenix
   - pip install openinference-instrumentation-openai
   - pip install openinference-instrumentation-google-genai google-genai
2. Start Phoenix: python -m phoenix.server.main serve
3. Set ENABLE_PHOENIX=true in .env or environment
"""

import asyncio
import os
import warnings
from typing import Optional, Any, Dict
from contextlib import contextmanager
import logging
import json

# Suppress OpenTelemetry attribute validation warnings for complex types (like Part objects)
# These appear when Phoenix instrumentation tries to log multimodal content
warnings.filterwarnings("ignore", message="Invalid type.*in attribute.*value sequence")

# Suppress OpenTelemetry SDK span attribute validation warnings
# These are printed directly when trying to set non-primitive span attributes
logging.getLogger("opentelemetry.sdk.trace").setLevel(logging.ERROR)
logging.getLogger("opentelemetry.attributes").setLevel(logging.ERROR)

# Optional imports - gracefully degrade if not installed
# IMPORTANT: Phoenix tracing capability should not depend on any specific model-provider instrumentor.
# We keep provider instrumentors optional and gate only on core OpenTelemetry + Phoenix exporter deps.
try:
    from openinference.semconv.resource import ResourceAttributes
except ImportError:
    ResourceAttributes = None  # type: ignore

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk import trace as trace_sdk
    from opentelemetry.sdk.resources import Resource
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False
    print("⚠️  Phoenix OpenTelemetry libraries not installed. Run: pip install arize-phoenix")

try:
    from openinference.instrumentation.openai import OpenAIInstrumentor
except ImportError:
    OpenAIInstrumentor = None  # type: ignore

try:
    from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
except ImportError:
    GoogleGenAIInstrumentor = None  # type: ignore

try:
    from openinference.instrumentation.anthropic import AnthropicInstrumentor
except ImportError:
    AnthropicInstrumentor = None  # type: ignore


class PhoenixObservability:
    """
    Optional Phoenix/OpenTelemetry instrumentation for the ARC solver.
    
    Usage:
        phoenix = PhoenixObservability()
        if phoenix.enabled:
            phoenix.instrument()
            
            with phoenix.span("planner_phase") as span:
                span.set_attribute("puzzle_id", puzzle_id)
                # ... agent logic ...
    """
    
    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:6006/v1/traces",
        project_name: Optional[str] = None,
    ):
        """
        Initialize Phoenix observability.
        
        Args:
            endpoint: Phoenix OTLP HTTP endpoint (default: localhost:6006)
        """
        self.endpoint = endpoint
        self.project_name = project_name or os.getenv("PHOENIX_PROJECT_NAME", "ARC")
        self.quiet = os.getenv("PHOENIX_QUIET", "false").lower() in ("1", "true", "yes", "on")
        self.enabled = self._check_enabled()
        self.tracer_provider: Optional[Any] = None
        self.tracer: Optional[Any] = None
        
        if self.enabled:
            self._setup()
    
    def _check_enabled(self) -> bool:
        """Check if Phoenix should be enabled."""
        if not PHOENIX_AVAILABLE:
            return False

        # Check environment variables (prefer CLI flag propagated as PHOENIX_ENABLED)
        raw_value = os.getenv("ENABLE_PHOENIX")
        if raw_value is None:
            raw_value = os.getenv("PHOENIX_ENABLED", "false")

        enable_phoenix = str(raw_value).lower()
        return enable_phoenix in ("true", "1", "yes", "on")
    
    def _setup(self):
        """Set up OpenTelemetry tracer and Phoenix exporter."""
        if not PHOENIX_AVAILABLE:
            return
        
        try:
            # Create tracer provider with project-level resource attributes so Phoenix
            # groups all traces under the desired project name (e.g., "ARC").
            resource_attributes = {"service.name": "arc_solver"}
            if ResourceAttributes is not None and hasattr(ResourceAttributes, "PROJECT_NAME"):
                resource_attributes[ResourceAttributes.PROJECT_NAME] = self.project_name
            else:
                # Fallback key if semantic conventions are unavailable
                resource_attributes["phoenix.project_name"] = self.project_name

            resource = None
            try:
                if 'Resource' in globals():
                    resource = Resource(attributes=resource_attributes)
            except Exception:
                resource = None

            # Configure span limits to prevent message truncation
            # Our API calls include large system prompts (15K+ chars) and multimodal content
            # Default OpenTelemetry limits: 12KB attribute value, 128 array items
            # Increase limits to capture full messages
            from opentelemetry.sdk.trace import SpanLimits
            span_limits = SpanLimits(
                max_attribute_length=200000,  # 200KB per attribute (enough for full prompts)
                max_attributes=4096,           # Needed for OpenInference-style flattened multimodal messages
                max_events=256,                # More events per span
                max_links=256,                 # More links per span
            )
            
            self.tracer_provider = trace_sdk.TracerProvider(
                resource=resource if resource else None,
                span_limits=span_limits
            )
            
            # Add Phoenix OTLP exporter
            # Use BatchSpanProcessor for better performance and reliability
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            self.span_processor = BatchSpanProcessor(
                OTLPSpanExporter(endpoint=self.endpoint),
                max_export_batch_size=512,
                max_queue_size=2048,
                schedule_delay_millis=2000  # Export every 2 seconds
            )
            self.tracer_provider.add_span_processor(self.span_processor)
            
            # Set as global tracer provider
            trace.set_tracer_provider(self.tracer_provider)
            
            # Get tracer for our app
            self.tracer = self.tracer_provider.get_tracer("arc_solver")
            
            if not self.quiet:
                print(f"✅ Phoenix observability enabled: {self.endpoint}")
                print(f"   Phoenix project: {self.project_name}")
                print(f"   View traces at: http://127.0.0.1:6006")
            
        except Exception as e:
            print(f"⚠️  Phoenix setup failed: {e}")
            self.enabled = False
    
    def instrument_openai(self):
        """
        Instrument OpenAI SDK to auto-capture all API calls.
        
        This should be called once at app startup, BEFORE creating OpenAI client.
        """
        if not self.enabled or not PHOENIX_AVAILABLE:
            return
        
        if OpenAIInstrumentor is None:
            if not self.quiet:
                print("⚠️  OpenAI instrumentation not installed. Run: pip install openinference-instrumentation-openai")
            return

        try:
            # CRITICAL: Must instrument BEFORE creating OpenAI client
            OpenAIInstrumentor().instrument(tracer_provider=self.tracer_provider)
            if not self.quiet:
                print("✅ OpenAI SDK instrumented for Phoenix")
                print(f"   Traces will appear at: http://127.0.0.1:6006")
        except Exception as e:
            print(f"⚠️  OpenAI instrumentation failed: {e}")

    def instrument_google_genai(self):
        """Instrument google-genai SDK to auto-capture GenAI API calls."""
        if not self.enabled or not PHOENIX_AVAILABLE:
            return

        if GoogleGenAIInstrumentor is None:
            if not self.quiet:
                print("⚠️  Google GenAI instrumentation not installed. Run: pip install openinference-instrumentation-google-genai google-genai")
            return

        try:
            # The current OpenInference google-genai instrumentation can be noisy when requests
            # include multimodal parts or tool/function parts. Suppress the known spammy loggers.
            # Suppress both the extractor and the parent instrumentation logger
            logging.getLogger("openinference.instrumentation.google_genai").setLevel(logging.CRITICAL)
            logging.getLogger("openinference.instrumentation.google_genai._request_attributes_extractor").setLevel(logging.CRITICAL)
            logging.getLogger("openinference.instrumentation").setLevel(logging.ERROR)

            GoogleGenAIInstrumentor().instrument(tracer_provider=self.tracer_provider)
            if not self.quiet:
                print("✅ Google GenAI SDK instrumented for Phoenix")
        except Exception as e:
            print(f"⚠️  Google GenAI instrumentation failed: {e}")

    def instrument_anthropic(self):
        """Instrument Anthropic SDK to auto-capture Claude API calls."""
        if not self.enabled or not PHOENIX_AVAILABLE:
            return

        if AnthropicInstrumentor is None:
            if not self.quiet:
                print("⚠️  Anthropic instrumentation not installed. Run: pip install openinference-instrumentation-anthropic")
            return

        try:
            AnthropicInstrumentor().instrument(tracer_provider=self.tracer_provider)
            if not self.quiet:
                print("✅ Anthropic SDK instrumented for Phoenix")
        except Exception as e:
            print(f"⚠️  Anthropic instrumentation failed: {e}")
    
    @contextmanager
    def span(self, name: str, attributes: Optional[Dict[str, Any]] = None, force_flush: bool = False):
        """
        Create a span for tracing a code block.
        
        Args:
            name: Span name
            attributes: Optional dict of attributes to attach
            force_flush: If True, force export spans immediately after this span ends
        
        Example:
            with phoenix.span("planner_analysis", {"puzzle_id": "abc123"}, force_flush=True):
                # ... planning logic ...
                # Span will be exported immediately when this block exits
        """
        if not self.enabled or not self.tracer:
            # No-op context manager if Phoenix is disabled
            yield None
            return
        
        with self.tracer.start_as_current_span(name) as span:
            # Add attributes
            if attributes:
                for key, value in attributes.items():
                    # Convert to string for safety
                    span.set_attribute(str(key), str(value))
            
            yield span
            
            # Force flush if requested (exports span immediately for real-time viewing)
            if force_flush and hasattr(self, 'span_processor'):
                try:
                    self.span_processor.force_flush(timeout_millis=1000)
                except Exception as e:
                    # Don't fail if flush fails
                    pass
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """
        Add an event to the current span.
        
        Args:
            name: Event name
            attributes: Optional event attributes
        """
        if not self.enabled:
            return
        
        current_span = trace.get_current_span()
        if current_span:
            current_span.add_event(name, attributes=attributes or {})
    
    def set_attribute(self, key: str, value: Any):
        """
        Set an attribute on the current span.
        
        Args:
            key: Attribute key
            value: Attribute value
        """
        if not self.enabled:
            return
        
        current_span = trace.get_current_span()
        if current_span:
            current_span.set_attribute(str(key), str(value))

    def set_large_attribute(self, key: str, value: Any, chunk_size: int = 190000, max_parts: int = 8):
        if not self.enabled:
            return

        try:
            if isinstance(value, str):
                text = value
            else:
                text = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            text = str(value)

        self.set_attribute(f"{key}.length", len(text))

        if len(text) <= chunk_size:
            self.set_attribute(key, text)
            self.set_attribute(f"{key}.parts", 1)
            return

        parts = (len(text) + chunk_size - 1) // chunk_size
        if parts <= max_parts:
            for i in range(parts):
                start = i * chunk_size
                self.set_attribute(f"{key}.part_{i}", text[start:start + chunk_size])
            self.set_attribute(f"{key}.parts", parts)
            return

        head = text[:chunk_size]
        tail = text[-chunk_size:]
        self.set_attribute(f"{key}.part_0", head)
        self.set_attribute(f"{key}.part_last", tail)
        self.set_attribute(f"{key}.parts", parts)
        self.set_attribute(f"{key}.truncated", True)
    
    def record_exception(self, exception: Exception):
        """
        Record an exception in the current span.
        
        Args:
            exception: Exception to record
        """
        if not self.enabled:
            return
        
        current_span = trace.get_current_span()
        if current_span:
            current_span.record_exception(exception)
    
    def shutdown(self):
        """
        Shutdown Phoenix and flush all remaining traces.
        Call this at app exit to ensure all traces are exported.
        """
        if not self.enabled or not self.span_processor:
            return
        
        try:
            # Force flush all pending spans
            self.span_processor.force_flush(timeout_millis=5000)
            # Shutdown span processor
            self.span_processor.shutdown()
            if not self.quiet:
                print("✅ Phoenix traces flushed and shutdown complete")
        except Exception as e:
            if not self.quiet:
                print(f"⚠️  Phoenix shutdown warning: {e}")


# Global instance (singleton pattern)
_phoenix_instance: Optional[PhoenixObservability] = None


def get_phoenix() -> PhoenixObservability:
    """
    Get the global Phoenix observability instance.
    
    Returns:
        PhoenixObservability instance (may be disabled)
    """
    global _phoenix_instance
    if _phoenix_instance is None:
        _phoenix_instance = PhoenixObservability()
    return _phoenix_instance


def initialize_phoenix(
    instrument_openai: bool = True,
    instrument_google_genai: bool = False,
    instrument_anthropic: bool = False
) -> PhoenixObservability:
    """
    Initialize Phoenix observability at app startup.
    
    Args:
        instrument_openai: If True, instrument OpenAI SDK
        instrument_google_genai: If True, instrument Google GenAI SDK
        instrument_anthropic: If True, instrument Anthropic SDK
    
    Returns:
        PhoenixObservability instance
    
    Example:
        # In run_solver.py main():
        phoenix = initialize_phoenix()
    """
    phoenix = get_phoenix()
    
    if phoenix.enabled and instrument_openai:
        phoenix.instrument_openai()

    if phoenix.enabled and instrument_google_genai:
        phoenix.instrument_google_genai()

    if phoenix.enabled and instrument_anthropic:
        phoenix.instrument_anthropic()
    
    return phoenix


# Decorator for easy function tracing
def trace_function(name: Optional[str] = None, **attributes):
    """
    Decorator to trace a function execution.
    
    Args:
        name: Optional span name (defaults to function name)
        **attributes: Additional attributes to attach
    
    Example:
        @trace_function(agent="planner")
        async def planning_phase():
            # ... planning logic ...
    """
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            phoenix = get_phoenix()
            span_name = name or func.__name__
            
            with phoenix.span(span_name, attributes):
                return func(*args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            phoenix = get_phoenix()
            span_name = name or func.__name__
            
            with phoenix.span(span_name, attributes):
                return await func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    
    return decorator


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Phoenix Observability Test")
    print("=" * 60)
    print()
    
    # Test 1: Check availability
    print(f"Phoenix libraries available: {PHOENIX_AVAILABLE}")
    print(f"ENABLE_PHOENIX env var: {os.getenv('ENABLE_PHOENIX', 'not set')}")
    print()
    
    # Test 2: Initialize
    phoenix = initialize_phoenix(instrument_openai=False)
    print(f"Phoenix enabled: {phoenix.enabled}")
    print()
    
    if phoenix.enabled:
        # Test 3: Create spans
        print("Creating test spans...")
        
        with phoenix.span("test_solver_run", {"puzzle_id": "test_001"}):
            phoenix.set_attribute("phase", "planning")
            
            with phoenix.span("planner_analysis"):
                phoenix.add_event("hypothesis_generated", {"count": 3})
                print("  - Created planner_analysis span")
            
            with phoenix.span("coder_implementation"):
                phoenix.add_event("code_executed", {"lines": 42})
                print("  - Created coder_implementation span")
        
        print()
        print("✅ Test spans created. Check Phoenix UI at http://127.0.0.1:6006")
    else:
        print("⚠️  Phoenix not enabled. Set ENABLE_PHOENIX=true to test.")
    
    print()
    print("=" * 60)
