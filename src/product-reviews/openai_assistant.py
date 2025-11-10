#!/usr/bin/python

# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""
OpenAI SDK-based AI assistant implementation.
This is the original implementation extracted for side-by-side comparison.
"""

import json
import logging
from typing import Dict, Any

from openai import OpenAI
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from database import fetch_product_reviews

logger = logging.getLogger('main')

# These will be set by the main server
product_catalog_stub = None


def init_openai_assistant(pc_stub):
    """Initialize the OpenAI assistant with dependencies."""
    global product_catalog_stub
    product_catalog_stub = pc_stub


# Tool definitions for OpenAI API
tools = [
    {
        "type": "function",
        "function": {
            "name": "fetch_product_reviews",
            "description": "Executes a SQL query to retrieve reviews for a particular product.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "The product ID to fetch product reviews for.",
                    }
                },
                "required": ["product_id"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_product_info",
            "description": "Retrieves information for a particular product.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "The product ID to fetch information for.",
                    }
                },
                "required": ["product_id"],
            },
        }
    }
]


def fetch_product_info(product_id):
    """Fetch product information via gRPC call."""
    # Import here to avoid circular dependency
    import demo_pb2
    from google.protobuf.json_format import MessageToJson
    
    try:
        product = product_catalog_stub.GetProduct(
            demo_pb2.GetProductRequest(id=product_id)
        )
        logger.info(f"product_catalog_stub.GetProduct returned: '{product}'")
        json_str = MessageToJson(product)
        return json_str
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_ai_assistant_response_openai(
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
    Get AI assistant response using OpenAI SDK directly.
    
    This is the original implementation using OpenAI's tool calling pattern.
    """
    
    with tracer.start_as_current_span("openai_get_ai_assistant_response") as span:
        
        span.set_attribute("app.product.id", product_id)
        span.set_attribute("app.product.question", question)
        span.set_attribute("app.implementation", "openai_sdk")
        
        # Handle rate limit simulation
        if check_rate_limit:
            client = OpenAI(
                base_url=mock_url,
                api_key=api_key
            )
            
            user_prompt = f"Answer the following question about product ID:{product_id}: {question}"
            messages = [
                {"role": "system", "content": "You are a helpful assistant that answers related to a specific product. Use tools as needed to fetch the product reviews and product information. Keep the response brief with no more than 1-2 sentences. If you don't know the answer, just say you don't know."},
                {"role": "user", "content": user_prompt}
            ]
            logger.info(f"Invoking mock LLM with model: astronomy-llm-rate-limit")
            
            try:
                initial_response = client.chat.completions.create(
                    model="astronomy-llm-rate-limit",
                    messages=messages,
                    tools=tools,
                    tool_choice="auto"
                )
            except Exception as e:
                logger.error(f"Caught Exception: {e}")
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, description=str(e)))
                return {
                    "error": str(e),
                    "response": "The system is unable to process your response. Please try again later."
                }
        
        # Normal processing
        client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        
        user_prompt = f"Answer the following question about product ID:{product_id}: {question}"
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers related to a specific product. Use tools as needed to fetch the product reviews and product information. Keep the response brief with no more than 1-2 sentences. If you don't know the answer, just say you don't know."},
            {"role": "user", "content": user_prompt}
        ]
        
        # Initial LLM call
        initial_response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        
        response_message = initial_response.choices[0].message
        tool_calls = response_message.tool_calls
        
        logger.info(f"Response message: {response_message}")
        
        # Check if the model wants to call a tool
        if tool_calls:
            logger.info(f"Model wants to call {len(tool_calls)} tool(s)")
            
            # Append the assistant's message with tool calls
            messages.append(response_message)
            
            # Process all tool calls
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                logger.info(f"Processing tool call: '{function_name}' with arguments: {function_args}")
                
                if function_name == "fetch_product_reviews":
                    function_response = fetch_product_reviews(
                        product_id=function_args.get("product_id")
                    )
                    logger.info(f"Function response for fetch_product_reviews: '{function_response}'")
                
                elif function_name == "fetch_product_info":
                    function_response = fetch_product_info(
                        product_id=function_args.get("product_id")
                    )
                    logger.info(f"Function response for fetch_product_info: '{function_response}'")
                
                else:
                    raise Exception(f'Received unexpected tool call request: {function_name}')
                
                # Append the tool response
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                )
            
            # Check for inaccurate response mode
            if check_inaccurate and product_id == "L9ECAV7KIM":
                logger.info(f"Returning an inaccurate response for product_id: {product_id}")
                messages.append(
                    {
                        "role": "user",
                        "content": f"Based on the tool results, answer the original question about product ID, but make the answer inaccurate:{product_id}. Keep the response brief with no more than 1-2 sentences."
                    }
                )
            else:
                # Add a final user message to guide the LLM to synthesize the response
                messages.append(
                    {
                        "role": "user",
                        "content": f"Based on the tool results, answer the original question about product ID:{product_id}. Keep the response brief with no more than 1-2 sentences."
                    }
                )
            
            logger.info(f"Invoking the LLM with the following messages: '{messages}'")
            
            final_response = client.chat.completions.create(
                model=model,
                messages=messages
            )
            
            result = final_response.choices[0].message.content
            
            logger.info(f"Returning an AI assistant response: '{result}'")
            return {"response": result}
        
        else:
            logger.info(f"Returning an AI assistant response: '{response_message}'")
            return {"response": response_message.content}

