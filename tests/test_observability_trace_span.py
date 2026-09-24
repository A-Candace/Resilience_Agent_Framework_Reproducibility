"""trace_span() must degrade gracefully when tracing isn't configured, and must
never swallow or mangle exceptions raised by the code it wraps."""
import logging

import pytest
from opentelemetry import trace as otel_trace

from agentic.common.observability import trace_span


class _RecordingSpan:
    def __init__(self):
        self.attributes = {}
        self.exited_with = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.exited_with = exc_type
        return False

    def set_attribute(self, key, value):
        self.attributes[key] = value


class _RecordingTracer:
    def __init__(self, span):
        self.span = span
        self.started = []

    def start_as_current_span(self, name):
        self.started.append(name)
        return self.span


def _patch_tracer(monkeypatch, tracer):
    monkeypatch.setattr(otel_trace, "get_tracer", lambda *a, **kw: tracer)


def test_yields_none_and_does_not_crash_when_span_creation_fails(monkeypatch, caplog):
    def exploding_get_tracer(*args, **kwargs):
        raise RuntimeError("phoenix collector unreachable")

    monkeypatch.setattr(otel_trace, "get_tracer", exploding_get_tracer)

    body_ran = False
    with caplog.at_level(logging.WARNING, logger="agentic.common.observability"):
        with trace_span("agent.run", session_id="s1") as span:
            assert span is None
            body_ran = True

    assert body_ran
    assert any("not created" in record.message for record in caplog.records)


def test_caller_exception_propagates_unchanged(monkeypatch):
    recording_span = _RecordingSpan()
    _patch_tracer(monkeypatch, _RecordingTracer(recording_span))

    sentinel = ValueError("the caller's actual error")

    with pytest.raises(ValueError) as excinfo:
        with trace_span("agent.run", session_id="s2"):
            raise sentinel

    # the exact exception object, not a RuntimeError about a generator
    assert excinfo.value is sentinel
    # the span was still closed, and saw the exception
    assert recording_span.exited_with is ValueError


def test_caller_exception_propagates_even_when_tracing_is_unavailable(monkeypatch):
    def exploding_get_tracer(*args, **kwargs):
        raise RuntimeError("phoenix collector unreachable")

    monkeypatch.setattr(otel_trace, "get_tracer", exploding_get_tracer)

    sentinel = KeyError("missing thing")

    with pytest.raises(KeyError) as excinfo:
        with trace_span("agent.run", session_id="s3") as span:
            assert span is None
            raise sentinel

    assert excinfo.value is sentinel


def test_happy_path_yields_span_and_sets_attributes(monkeypatch):
    recording_span = _RecordingSpan()
    tracer = _RecordingTracer(recording_span)
    _patch_tracer(monkeypatch, tracer)

    with trace_span("agent.run", session_id="s4", user="alice") as span:
        assert span is recording_span

    assert tracer.started == ["agent.run"]
    assert recording_span.attributes == {"session_id": "s4", "user": "alice"}
    assert recording_span.exited_with is None
