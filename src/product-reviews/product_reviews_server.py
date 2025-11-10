#!/usr/bin/python

# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0


# Python
import os
from concurrent import futures
import random

# Pip
import grpc
from opentelemetry import trace, metrics
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
    OTLPLogExporter,
)
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource

# Local
import logging
import demo_pb2
import demo_pb2_grpc
from grpc_health.v1 import health_pb2
from grpc_health.v1 import health_pb2_grpc
from database import (
    fetch_product_reviews_from_db,
    fetch_avg_product_review_score_from_db
)

from openfeature import api
from openfeature.contrib.provider.flagd import FlagdProvider

from metrics import (
    init_metrics
)

# Import both AI assistant implementations
import openai_assistant
import langchain_agent

llm_host = None
llm_port = None
llm_mock_url = None
llm_base_url = None
llm_api_key = None
llm_model = None

# Configuration flag to choose implementation
USE_LANGCHAIN_AGENT = os.environ.get('USE_LANGCHAIN_AGENT', 'false').lower() == 'true'

class ProductReviewService(demo_pb2_grpc.ProductReviewServiceServicer):
    def GetProductReviews(self, request, context):
        logger.info(f"Receive GetProductReviews for product id:{request.product_id}")
        product_reviews = get_product_reviews(request.product_id)

        return product_reviews

    def GetAverageProductReviewScore(self, request, context):
        logger.info(f"Receive GetAverageProductReviewScore for product id:{request.product_id}")
        product_reviews = get_average_product_review_score(request.product_id)

        return product_reviews

    def AskProductAIAssistant(self, request, context):
        logger.info(f"Receive AskProductAIAssistant for product id:{request.product_id}, question: {request.question}")
        ai_assistant_response = get_ai_assistant_response(request.product_id, request.question)

        return ai_assistant_response

    def Check(self, request, context):
        return health_pb2.HealthCheckResponse(
            status=health_pb2.HealthCheckResponse.SERVING)

    def Watch(self, request, context):
        return health_pb2.HealthCheckResponse(
            status=health_pb2.HealthCheckResponse.UNIMPLEMENTED)

def get_product_reviews(request_product_id):

    with tracer.start_as_current_span("get_product_reviews") as span:

        span.set_attribute("app.product.id", request_product_id)

        product_reviews = demo_pb2.GetProductReviewsResponse()
        records = fetch_product_reviews_from_db(request_product_id)

        for row in records:
            logger.info(f"  username: {row[0]}, description: {row[1]}, score: {str(row[2])}")
            product_reviews.product_reviews.add(
                    username=row[0],
                    description=row[1],
                    score=str(row[2])
            )

        span.set_attribute("app.product_reviews.count", len(product_reviews.product_reviews))

        # Collect metrics for this service
        product_review_svc_metrics["app_product_review_counter"].add(len(product_reviews.product_reviews), {'product.id': request_product_id})

        return product_reviews

def get_average_product_review_score(request_product_id):

    with tracer.start_as_current_span("get_average_product_review_score") as span:

        span.set_attribute("app.product.id", request_product_id)

        product_review_score = demo_pb2.GetAverageProductReviewScoreResponse()
        avg_score = fetch_avg_product_review_score_from_db(request_product_id)
        product_review_score.average_score = avg_score

        span.set_attribute("app.product_reviews.average_score", avg_score)

        return product_review_score

def get_ai_assistant_response(request_product_id, question):
    """
    Get AI assistant response using the configured implementation.
    Routes to either OpenAI SDK or LangChain agent based on USE_LANGCHAIN_AGENT env var.
    """

    with tracer.start_as_current_span("get_ai_assistant_response") as span:

        ai_assistant_response = demo_pb2.AskProductAIAssistantResponse()

        span.set_attribute("app.product.id", request_product_id)
        span.set_attribute("app.product.question", question)

        # Check feature flags
        llm_rate_limit_error = check_feature_flag("llmRateLimitError")
        llm_inaccurate_response = check_feature_flag("llmInaccurateResponse")

        logger.info(f"llmRateLimitError feature flag: {llm_rate_limit_error}")
        logger.info(f"llmInaccurateResponse feature flag: {llm_inaccurate_response}")
        logger.info(f"Using {'LangChain' if USE_LANGCHAIN_AGENT else 'OpenAI SDK'} implementation")

        span.set_attribute("app.use_langchain", USE_LANGCHAIN_AGENT)

        # Determine if we should trigger rate limit
        check_rate_limit = False
        if llm_rate_limit_error:
            random_number = random.random()
            logger.info(f"Generated a random number: {str(random_number)}")
            check_rate_limit = random_number < 0.5

        # Choose implementation
        if USE_LANGCHAIN_AGENT:
            result = langchain_agent.get_ai_assistant_response_langchain(
                product_id=request_product_id,
                question=question,
                base_url=llm_base_url,
                api_key=llm_api_key,
                model=llm_model,
                tracer=tracer,
                check_rate_limit=check_rate_limit,
                check_inaccurate=llm_inaccurate_response,
                mock_url=llm_mock_url
            )
        else:
            result = openai_assistant.get_ai_assistant_response_openai(
                product_id=request_product_id,
                question=question,
                base_url=llm_base_url,
                api_key=llm_api_key,
                model=llm_model,
                tracer=tracer,
                check_rate_limit=check_rate_limit,
                check_inaccurate=llm_inaccurate_response,
                mock_url=llm_mock_url
            )

        ai_assistant_response.response = result["response"]

        # Collect metrics for this service
        product_review_svc_metrics["app_ai_assistant_counter"].add(
            1,
            {
                'product.id': request_product_id,
                'implementation': 'langchain' if USE_LANGCHAIN_AGENT else 'openai_sdk'
            }
        )

        return ai_assistant_response

def must_map_env(key: str):
    value = os.environ.get(key)
    if value is None:
        raise Exception(f'{key} environment variable must be set')
    return value

def check_feature_flag(flag_name: str):
    # Initialize OpenFeature
    client = api.get_client()
    return client.get_boolean_value(flag_name, False)

if __name__ == "__main__":
    service_name = must_map_env('OTEL_SERVICE_NAME')

    api.set_provider(FlagdProvider(host=os.environ.get('FLAGD_HOST', 'flagd'), port=os.environ.get('FLAGD_PORT', 8013)))

    # Initialize Traces and Metrics
    tracer = trace.get_tracer_provider().get_tracer(service_name)
    meter = metrics.get_meter_provider().get_meter(service_name)

    product_review_svc_metrics = init_metrics(meter)

    # Initialize Logs
    logger_provider = LoggerProvider(
        resource=Resource.create(
            {
                'service.name': service_name,
            }
        ),
    )
    set_logger_provider(logger_provider)
    log_exporter = OTLPLogExporter(insecure=True)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
    handler = LoggingHandler(level=logging.NOTSET, logger_provider=logger_provider)

    # Attach OTLP handler to logger
    logger = logging.getLogger('main')
    logger.addHandler(handler)

    # Create gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    # Add class to gRPC server
    service = ProductReviewService()
    demo_pb2_grpc.add_ProductReviewServiceServicer_to_server(service, server)
    health_pb2_grpc.add_HealthServicer_to_server(service, server)

    llm_host = must_map_env('LLM_HOST')
    llm_port = must_map_env('LLM_PORT')
    llm_mock_url = f"http://{llm_host}:{llm_port}/v1"
    llm_base_url = must_map_env('LLM_BASE_URL')
    llm_api_key = must_map_env('OPENAI_API_KEY')
    llm_model = must_map_env('LLM_MODEL')

    catalog_addr = must_map_env('PRODUCT_CATALOG_ADDR')
    pc_channel = grpc.insecure_channel(catalog_addr)
    product_catalog_stub = demo_pb2_grpc.ProductCatalogServiceStub(pc_channel)

    # Initialize both implementations with dependencies
    openai_assistant.init_openai_assistant(product_catalog_stub)
    langchain_agent.init_langchain_agent(product_catalog_stub)

    logger.info(f"AI Assistant Implementation: {'LangChain Agent' if USE_LANGCHAIN_AGENT else 'OpenAI SDK'}")

    # Start server
    port = must_map_env('PRODUCT_REVIEWS_PORT')
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    logger.info(f'Product reviews service started, listening on port {port}')
    server.wait_for_termination()
