import os
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

def init_tracing(service_name: str):
    if trace.get_tracer_provider().__class__.__name__ == "TracerProvider":
        # already set (avoid double-init in reloads)
        return

    resource = Resource.create({
        "service.name": service_name,
        "service.namespace": "sentinelx",
        "deployment.environment": os.getenv("SENTINELX_ENV", "local"),
    })

    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://jaeger:4318")
    exporter = OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces")

    provider.add_span_processor(BatchSpanProcessor(exporter))
