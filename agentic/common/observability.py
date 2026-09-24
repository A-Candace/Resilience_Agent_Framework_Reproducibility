import logging
from contextlib import ExitStack, contextmanager

logger = logging.getLogger(__name__)


def configure_observability() -> None:
    """Configure Phoenix/OpenTelemetry when dependencies and collector are available."""
    try:
        from phoenix.otel import register
        from agentic.common.settings import get_settings
        cfg = get_settings()
        register(project_name=cfg.otel_service_name, endpoint=cfg.phoenix_collector_endpoint)
    except Exception as exc:  # observability must never prevent the app from starting
        logger.warning("Phoenix tracing not configured: %s", exc)


@contextmanager
def trace_span(name: str, **attributes):
    """Yield an OpenTelemetry span, or None if tracing isn't available.

    Only span *creation* is guarded — an unconfigured collector degrades to a
    no-op. Exceptions raised by the caller's code inside the `with` block are
    never caught here; they propagate normally (after the span records them).
    """
    stack = ExitStack()
    try:
        from opentelemetry import trace
        span = stack.enter_context(trace.get_tracer(__name__).start_as_current_span(name))
        for key, value in attributes.items():
            span.set_attribute(key, str(value))
    except Exception as exc:  # tracing must never break the code it wraps
        stack.close()
        logger.warning("Tracing span %r not created: %s", name, exc)
        yield None
        return

    with stack:
        yield span
