#!/usr/bin/python

# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""
LangChain-based AI assistant implementation.
This module provides an agent that uses LangChain's tool calling
capabilities to answer product-related questions.
"""

import json
import logging
import os
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

# Import database functions
from database import fetch_product_reviews

# Import Langfuse callback handler
try:
    from langfuse.callback import CallbackHandler
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    logging.getLogger('main').warning(
        "Langfuse not available. Install langfuse package for LLM tracing."
    )

logger = logging.getLogger('main')

# These will be set by the main server
product_catalog_stub = None


def init_langchain_agent(pc_stub):
    """Initialize the LangChain agent with dependencies."""
    global product_catalog_stub
    product_catalog_stub = pc_stub


@tool
def fetch_product_reviews_tool(product_id: str) -> str:
    """Executes a SQL query to retrieve reviews for a particular product.

    Args:
        product_id: The product ID to fetch product reviews for.
    """
    logger.info(
        "LangChain tool: fetch_product_reviews called with "
        f"product_id: {product_id}"
    )
    try:
        return fetch_product_reviews(product_id=product_id)
    except Exception as e:
        logger.error(f"Error fetching product reviews: {e}")
        return json.dumps({"error": str(e)})


@tool
def fetch_product_info_tool(product_id: str) -> str:
    """Retrieves information for a particular product.

    Args:
        product_id: The product ID to fetch information for.
    """
    logger.info(
        "LangChain tool: fetch_product_info called with "
        f"product_id: {product_id}"
    )

    # Import here to avoid circular dependency
    import demo_pb2
    from google.protobuf.json_format import MessageToJson

    try:
        product = product_catalog_stub.GetProduct(
            demo_pb2.GetProductRequest(id=product_id)
        )
        logger.info(
            f"product_catalog_stub.GetProduct returned: '{product}'"
        )
        json_str = MessageToJson(product)
        return json_str
    except Exception as e:
        logger.error(f"Error fetching product info: {e}")
        return json.dumps({"error": str(e)})


def create_agent_executor(
    base_url: str,
    api_key: str,
    model: str,
    langfuse_handler=None
):
    """
    Create a LangChain agent executor with ChatOpenAI and tools.

    Args:
        base_url: The base URL for the OpenAI-compatible API
        api_key: API key for authentication
        model: Model name to use
        langfuse_handler: Optional Langfuse callback handler

    Returns:
        Configured agent graph
    """
    logger.info(
        f"Creating LangChain agent with base_url={base_url}, "
        f"model={model}"
    )

    # Initialize ChatOpenAI with the provided configuration
    # This works with both mock LLM and real OpenAI
    llm_kwargs = {
        'base_url': base_url,
        'api_key': api_key,
        'model': model,
        'temperature': 0,
    }

    # Add Langfuse callback handler if provided
    if langfuse_handler:
        llm_kwargs['callbacks'] = [langfuse_handler]

    llm = ChatOpenAI(**llm_kwargs)

    # Define the tools available to the agent
    tools = [fetch_product_reviews_tool, fetch_product_info_tool]

    # Create the agent using LangGraph
    # Replaces the old AgentExecutor + create_tool_calling_agent
    agent_executor = create_react_agent(llm, tools)

    return agent_executor


def get_ai_assistant_response_langchain(
    product_id: str,
    question: str,
    base_url: str,
    api_key: str,
    model: str,
    tracer: trace.Tracer,
    check_rate_limit: bool = False,
    check_inaccurate: bool = False,
    mock_url: str = None
) -> Dict[str, Any]:
    """
    Get AI assistant response using LangChain agent.

    Args:
        product_id: The product ID to ask about
        question: The user's question
        base_url: Base URL for the LLM API
        api_key: API key
        model: Model name
        tracer: OpenTelemetry tracer
        check_rate_limit: Whether to simulate rate limit error
        check_inaccurate: Whether to return inaccurate response
        mock_url: Mock LLM URL for rate limit testing

    Returns:
        Dict with 'response' key or 'error' if failed
    """

    # Initialize Langfuse callback handler if available
    langfuse_handler = None
    if LANGFUSE_AVAILABLE:
        langfuse_secret_key = os.environ.get('LANGFUSE_SECRET_KEY')
        langfuse_public_key = os.environ.get('LANGFUSE_PUBLIC_KEY')
        langfuse_host = os.environ.get(
            'LANGFUSE_HOST',
            'http://localhost:3000'
        )

        if langfuse_secret_key and langfuse_public_key:
            try:
                langfuse_handler = CallbackHandler(
                    secret_key=langfuse_secret_key,
                    public_key=langfuse_public_key,
                    host=langfuse_host,
                    session_id=product_id,
                    user_id="product-reviews-service",
                    metadata={
                        "product_id": product_id,
                        "question": question,
                        "model": model
                    }
                )
                logger.info(
                    "Langfuse callback handler initialized with "
                    f"host: {langfuse_host}"
                )
            except Exception as e:
                logger.error(
                    "Failed to initialize Langfuse callback "
                    f"handler: {e}"
                )
        else:
            logger.info(
                "Langfuse credentials not configured, skipping "
                "Langfuse tracing"
            )

    with tracer.start_as_current_span(
        "langchain_get_ai_assistant_response"
    ) as span:

        span.set_attribute("app.product.id", product_id)
        span.set_attribute("app.product.question", question)
        span.set_attribute("app.implementation", "langchain")

        # Handle rate limit simulation
        if check_rate_limit:
            logger.info(
                "Rate limit check enabled, using mock LLM with "
                "rate limit model"
            )
            try:
                agent_executor = create_agent_executor(
                    base_url=mock_url,
                    api_key=api_key,
                    model="astronomy-llm-rate-limit",
                    langfuse_handler=langfuse_handler
                )

                user_prompt = (
                    f"Answer the following question about product "
                    f"ID:{product_id}: {question}"
                )
                config = (
                    {"callbacks": [langfuse_handler]}
                    if langfuse_handler else {}
                )
                result = agent_executor.invoke(
                    {"messages": [
                        {"role": "user", "content": user_prompt}
                    ]},
                    config=config
                )

            except Exception as e:
                logger.error(
                    f"Caught Exception during rate limit test: {e}"
                )
                span.record_exception(e)
                span.set_status(
                    Status(StatusCode.ERROR, description=str(e))
                )
                return {
                    "error": str(e),
                    "response": (
                        "The system is unable to process your "
                        "response. Please try again later."
                    )
                }

        # Normal processing
        try:
            # Create the agent executor
            agent_executor = create_agent_executor(
                base_url=base_url,
                api_key=api_key,
                model=model,
                langfuse_handler=langfuse_handler
            )

            # Construct the prompt
            user_prompt = (
                f"Answer the following question about product "
                f"ID:{product_id}: {question}"
            )

            # Handle inaccurate response mode (for testing)
            if check_inaccurate and product_id == "L9ECAV7KIM":
                logger.info(
                    "Using inaccurate response mode for "
                    f"product_id: {product_id}"
                )
                # Modify prompt for inaccurate response
                user_prompt = (
                    "Answer the following question about product "
                    "ID, but make the answer inaccurate:"
                    f"{product_id}: {question}"
                )

            logger.info(
                f"Invoking LangChain agent with prompt: "
                f"'{user_prompt}'"
            )

            # Invoke the agent with LangGraph message format
            # Add Langfuse callback handler if available
            config = (
                {"callbacks": [langfuse_handler]}
                if langfuse_handler else {}
            )
            result = agent_executor.invoke(
                {"messages": [{"role": "user", "content": user_prompt}]},
                config=config
            )

            # Extract the response from the messages
            if result and "messages" in result:
                last_message = result["messages"][-1]
                response_text = (
                    last_message.content
                    if hasattr(last_message, 'content')
                    else str(last_message)
                )
            else:
                response_text = str(result)

            span.set_attribute("app.response.length", len(response_text))
            logger.info(f"LangChain agent response: '{response_text}'")

            return {"response": response_text}

        except Exception as e:
            logger.error(f"Error invoking LangChain agent: {e}")
            span.record_exception(e)
            span.set_status(
                Status(StatusCode.ERROR, description=str(e))
            )
            return {
                "error": str(e),
                "response": (
                    "The system is unable to process your "
                    "response. Please try again later."
                )
            }
