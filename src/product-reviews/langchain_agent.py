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

# Import Langfuse callback handler and client
try:
    from langfuse.callback import CallbackHandler
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    logging.getLogger('main').warning(
        "Langfuse not available. Install langfuse package for LLM tracing."
    )

logger = logging.getLogger('main')

# These will be set by the main server
product_catalog_stub = None
tracer = None


def init_langchain_agent(pc_stub, service_tracer=None):
    """Initialize the LangChain agent with dependencies."""
    global product_catalog_stub, tracer
    product_catalog_stub = pc_stub
    if service_tracer:
        tracer = service_tracer
    else:
        # Fallback: get tracer from global provider using service name
        import os
        service_name = os.environ.get('OTEL_SERVICE_NAME', 'product-reviews-service')
        tracer = trace.get_tracer_provider().get_tracer(service_name)


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
    agent_executor = create_react_agent(llm, tools)

    return agent_executor


def calculate_llm_judge_score(
    trace_id: str,
    product_id: str,
    question: str,
    response: str,
    langfuse_client: Langfuse,
    base_url: str,
    api_key: str,
    model: str
) -> Dict[str, Any]:
    """
    Calculate a score using LLM as a judge and create it in Langfuse 2.x.
    
    Args:
        trace_id: The Langfuse trace ID to attach the score to
        product_id: Product ID for context
        question: Original user question
        response: The AI assistant's response to evaluate
        langfuse_client: Langfuse client instance
        base_url: Base URL for the judge LLM
        api_key: API key for the judge LLM
        model: Model name for the judge (must not be empty)
        
    Returns:
        Dict with score and reasoning
    """
    # Validate model parameter
    if not model or not model.strip():
        logger.error(
            f"Judge model is empty or None, cannot calculate score. "
            f"Model value: '{model}'"
        )
        return {
            "overall_score": 0.0,
            "reasoning": "Judge model not configured",
            "error": "Model parameter is required"
        }
    logger.info(
        f"Calculating LLM judge score for trace {trace_id} "
        f"using model: '{model}'"
    )
    
    # Create evaluation prompt for the judge
    judge_prompt = f"""You are an expert evaluator assessing the quality of an AI assistant's response.

Context:
- Product ID: {product_id}
- User Question: {question}
- AI Response: {response}

Evaluate the response on the following criteria (each on a scale of 0-1):
1. Relevance: Does the response directly address the user's question?
2. Accuracy: Is the information provided factually correct?
3. Completeness: Does the response provide sufficient information?
4. Clarity: Is the response clear and well-structured?

Provide your evaluation as a JSON object with:
- overall_score: A single score from 0-1 representing overall quality
- relevance_score: Score for relevance (0-1)
- accuracy_score: Score for accuracy (0-1)
- completeness_score: Score for completeness (0-1)
- clarity_score: Score for clarity (0-1)
- reasoning: A brief explanation of your evaluation

Respond ONLY with valid JSON, no additional text."""

    try:
        # Use the judge LLM to evaluate
        # Ensure model is not None or empty
        if not model:
            raise ValueError(
                f"Model parameter is required but got: '{model}'"
            )
        
        logger.info(
            f"Initializing judge LLM with model='{model}', "
            f"base_url='{base_url}'"
        )
        
        # Create a child span for the judge LLM call to ensure proper parent
        # linkage. Use the global tracer provider to ensure we're using the
        # same tracer that the LangChain instrumentation uses.
        judge_response = None
        try:
            # Get tracer from global provider to match what
            # instrumentation uses
            tracer_name = (
                __name__ or "product-reviews-service"
            )
            global_tracer = (
                trace.get_tracer_provider().get_tracer(tracer_name)
            )

            # Create the ChatOpenAI instance outside the span to avoid
            # context issues but invoke it inside the span so the
            # instrumentation picks up the parent
            judge_llm = ChatOpenAI(
                base_url=base_url,
                api_key=api_key,
                model=model,
                temperature=0
            )

            # Use start_as_current_span to ensure proper parent context
            # inheritance. The LangChain instrumentation should read the
            # current span from the context
            with global_tracer.start_as_current_span(
                "llm_judge_evaluation"
            ) as judge_span:
                judge_span.set_attribute("app.judge.model", model)
                judge_span.set_attribute(
                    "app.judge.product_id", product_id
                )

                judge_response = judge_llm.invoke(judge_prompt)
        except Exception as invoke_error:
            logger.error(
                f"Error invoking judge LLM: {invoke_error}. "
                f"Model: {model}, Base URL: {base_url}"
            )
            return {
                "overall_score": 0.0,
                "reasoning": (
                    f"Judge LLM invocation failed: {str(invoke_error)}"
                ),
                "error": str(invoke_error)
            }

        if judge_response is None:
            logger.error("Judge LLM response is None")
            return {
                "overall_score": 0.0,
                "reasoning": "Judge LLM returned None",
                "error": "No response from judge LLM"
            }

        judge_text = (
            judge_response.content
            if hasattr(judge_response, 'content')
            else str(judge_response)
        )
        
        # Check if response looks like an error message
        if not judge_text or not judge_text.strip():
            logger.error("Judge LLM returned empty response")
            return {
                "overall_score": 0.0,
                "reasoning": "Judge LLM returned empty response",
                "error": "Empty response from judge LLM"
            }
        
        # Check for common error patterns
        error_patterns = [
            "internal server error",
            "internal error",
            "error:",
            "exception:",
            "failed",
            "unable to"
        ]
        judge_text_lower = judge_text.lower()
        if any(pattern in judge_text_lower for pattern in error_patterns):
            logger.error(
                f"Judge LLM returned error response: {judge_text[:200]}"
            )
            return {
                "overall_score": 0.0,
                "reasoning": f"Judge LLM error: {judge_text[:100]}",
                "error": "Judge LLM returned an error response"
            }
        
        # Parse the JSON response
        try:
            # Extract JSON from response (in case there's extra text)
            import re
            json_match = re.search(r'\{.*\}', judge_text, re.DOTALL)
            if json_match:
                judge_text = json_match.group(0)
            else:
                # No JSON found in response
                logger.error(
                    f"No JSON found in judge response: {judge_text[:200]}"
                )
                return {
                    "overall_score": 0.0,
                    "reasoning": "Judge LLM did not return valid JSON",
                    "error": "No JSON found in response"
                }
            
            evaluation = json.loads(judge_text)
            overall_score = evaluation.get('overall_score', 0.0)
            reasoning = evaluation.get('reasoning', 'No reasoning provided')
            
            # Create score in Langfuse 2.x using the client
            langfuse_client.score(
                trace_id=trace_id,
                name="llm_judge_score",
                value=overall_score,
                comment=reasoning,
                metadata={
                    "relevance_score": evaluation.get(
                        'relevance_score', 0.0
                    ),
                    "accuracy_score": evaluation.get(
                        'accuracy_score', 0.0
                    ),
                    "completeness_score": evaluation.get(
                        'completeness_score', 0.0
                    ),
                    "clarity_score": evaluation.get(
                        'clarity_score', 0.0
                    ),
                    "product_id": product_id,
                    "question": question
                }
            )
            
            logger.info(
                f"Created LLM judge score {overall_score} "
                f"for trace {trace_id}"
            )
            
            return {
                "overall_score": overall_score,
                "reasoning": reasoning,
                "detailed_scores": {
                    "relevance": evaluation.get('relevance_score', 0.0),
                    "accuracy": evaluation.get('accuracy_score', 0.0),
                    "completeness": evaluation.get('completeness_score', 0.0),
                    "clarity": evaluation.get('clarity_score', 0.0)
                }
            }
            
        except json.JSONDecodeError as e:
            error_msg = (
                f"Failed to parse judge LLM response as JSON: {e}. "
                f"Response was: {judge_text[:500]}"
            )
            logger.error(error_msg)
            return {
                "overall_score": 0.0,
                "reasoning": "Failed to parse judge evaluation as JSON",
                "error": f"JSON decode error: {str(e)}"
            }
            
    except Exception as e:
        logger.error(f"Error calculating LLM judge score: {e}")
        return {
            "overall_score": 0.0,
            "reasoning": f"Error during evaluation: {str(e)}",
            "error": str(e)
        }


def get_ai_assistant_response_langchain(
    product_id: str,
    question: str,
    base_url: str,
    api_key: str,
    model: str,
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
        check_rate_limit: Whether to simulate rate limit error
        check_inaccurate: Whether to return inaccurate response
        mock_url: Mock LLM URL for rate limit testing

    Returns:
        Dict with 'response' key or 'error' if failed
    """

    # Initialize Langfuse callback handler and client if available
    langfuse_handler = None
    langfuse_client = None
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
                
                # Initialize Langfuse client for score creation
                # (version 2.x compatible)
                langfuse_client = Langfuse(
                    secret_key=langfuse_secret_key,
                    public_key=langfuse_public_key,
                    host=langfuse_host
                )
                
                logger.info(
                    "Langfuse callback handler and client initialized with "
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

    # Keep top-level business span with custom attributes
    # LangChain auto instrumentation will create child spans automatically
    with tracer.start_as_current_span(
        "langchain_get_ai_assistant_response"
    ) as span:

        span.set_attribute("app.product.id", product_id)
        span.set_attribute("app.product.question", question)
        span.set_attribute("app.implementation", "langchain")
        span.set_attribute("app.model", model)

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
                # Auto instrumentation will trace this invoke call
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
            # LangChain auto instrumentation will automatically trace:
            # - Agent execution steps
            # - LLM calls
            # - Tool invocations
            # - Chain operations
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

            # Calculate LLM judge score if enabled
            if langfuse_client and langfuse_handler:
                try:
                    # Flush handler to ensure trace is created
                    langfuse_handler.flush()
                    
                    # Create a trace explicitly for scoring
                    # (Langfuse 2.x compatible)
                    # This trace will be linked to the session_id, allowing
                    # us to associate scores with the agent's execution
                    langfuse_trace = langfuse_client.trace(
                        name="ai_assistant_response",
                        session_id=product_id,
                        user_id="product-reviews-service",
                        metadata={
                            "product_id": product_id,
                            "question": question,
                            "model": model
                        }
                    )
                    trace_id = langfuse_trace.id
                    
                    # Check if LLM judge scoring is enabled
                    enable_llm_judge = (
                        os.environ.get(
                            'ENABLE_LLM_JUDGE_SCORING', 'false'
                        ).lower() == 'true'
                    )
                    # Use LLM_JUDGE_MODEL if set, otherwise use the same model
                    judge_model_env = os.environ.get('LLM_JUDGE_MODEL', '')
                    judge_model = (
                        judge_model_env.strip()
                        if judge_model_env and judge_model_env.strip()
                        else model
                    )
                    logger.info(
                        f"Using judge model: '{judge_model}' "
                        f"(LLM_JUDGE_MODEL='{judge_model_env}', "
                        f"main model='{model}')"
                    )
                    
                    if enable_llm_judge:
                        score_result = calculate_llm_judge_score(
                            trace_id=trace_id,
                            product_id=product_id,
                            question=question,
                            response=response_text,
                            langfuse_client=langfuse_client,
                            base_url=base_url,
                            api_key=api_key,
                            model=judge_model
                        )
                        logger.info(
                            f"LLM judge score: "
                            f"{score_result.get('overall_score', 'N/A')}"
                        )
                        span.set_attribute(
                            "app.llm_judge.score",
                            score_result.get('overall_score', 0.0)
                        )
                except Exception as e:
                    logger.error(f"Error calculating LLM judge score: {e}")
                    # Don't fail the request if scoring fails

            return {"response": response_text}

        except json.JSONDecodeError as e:
            error_msg = (
                f"JSON parsing error: {e}. This usually means the LLM API "
                "returned an error response instead of valid JSON. "
                "Check your API key and endpoint configuration."
            )
            logger.error(error_msg)
            span.record_exception(e)
            span.set_status(
                Status(StatusCode.ERROR, description=error_msg)
            )
            return {
                "error": error_msg,
                "response": (
                    "The system is unable to process your "
                    "response. Please try again later."
                )
            }
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
