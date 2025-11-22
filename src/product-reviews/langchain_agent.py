#!/usr/bin/python

# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""
LangChain-based AI assistant implementation (simplified version).
This module provides an agent that uses LangChain's tool calling
capabilities to answer product-related questions.
"""

import json
import logging
from typing import Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from opentelemetry import trace

# Import database functions
from database import fetch_product_reviews

logger = logging.getLogger('main')

# These will be set by the main server
product_catalog_stub = None


def init_langchain_agent(pc_stub):
    """Initialize the LangChain agent with dependencies."""
    global product_catalog_stub
    product_catalog_stub = pc_stub


@tool
def fetch_product_reviews_tool(product_id: str) -> str:
    """Retrieve reviews for a particular product.

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
    model: str
):
    """
    Create a LangChain agent executor with ChatOpenAI and tools.

    Args:
        base_url: The base URL for the OpenAI-compatible API
        api_key: API key for authentication
        model: Model name to use

    Returns:
        Configured agent graph
    """
    logger.info(
        f"Creating LangChain agent with base_url={base_url}, "
        f"model={model}"
    )

    # Initialize ChatOpenAI with the provided configuration
    # This works with both mock LLM and real OpenAI
    llm = ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=0,
    )

    # Define the tools available to the agent
    tools = [fetch_product_reviews_tool, fetch_product_info_tool]

    # Create the agent using LangGraph
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

    # Raise error if unsupported features are requested
    if check_rate_limit or check_inaccurate:
        raise ValueError(
            "check_rate_limit and check_inaccurate are not supported in this simplified version"
        )

    try:
        # Create the agent executor
        agent_executor = create_agent_executor(
            base_url=base_url,
            api_key=api_key,
            model=model
        )

        # Construct the prompt
        user_prompt = (
            f"Answer the following question about product "
            f"ID:{product_id}: {question}"
        )

        logger.info(
            f"Invoking LangChain agent with prompt: "
            f"'{user_prompt}'"
        )

        # Invoke the agent with LangGraph message format
        result = agent_executor.invoke(
            {"messages": [{"role": "user", "content": user_prompt}]}
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

        logger.info(f"LangChain agent response: '{response_text}'")

        return {"response": response_text}

    except json.JSONDecodeError as e:
        error_msg = (
            f"JSON parsing error: {e}. This usually means the LLM API "
            "returned an error response instead of valid JSON. "
            "Check your API key and endpoint configuration."
        )
        logger.error(error_msg)
        return {
            "error": error_msg,
            "response": (
                "The system is unable to process your "
                "response. Please try again later."
            )
        }
    except Exception as e:
        logger.error(f"Error invoking LangChain agent: {e}")
        return {
            "error": str(e),
            "response": (
                "The system is unable to process your "
                "response. Please try again later."
            )
        }
