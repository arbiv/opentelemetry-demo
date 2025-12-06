#!/usr/bin/python

# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""
LLM Judge scoring module.
This module provides functions to evaluate AI assistant responses
using an LLM as a judge.
"""

import json
import logging
import re
from typing import Dict, Any

from langchain_openai import ChatOpenAI

try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    logging.getLogger('main').warning(
        "Langfuse not available. Install langfuse package for LLM scoring."
    )

logger = logging.getLogger('main')


def validate_judge_model(model: str) -> Dict[str, Any]:
    """
    Validate that the judge model parameter is provided.
    
    Args:
        model: Model name to validate
        
    Returns:
        Dict with error info if invalid, None if valid
    """
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
    return None


def create_judge_prompt(product_id: str, question: str, response: str) -> str:
    """
    Create the evaluation prompt for the judge LLM.
    
    Args:
        product_id: Product ID for context
        question: Original user question
        response: The AI assistant's response to evaluate
        
    Returns:
        Formatted prompt string
    """
    return f"""You are an expert evaluator assessing the quality of an AI assistant's response.

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


def invoke_judge_llm(
    prompt: str,
    base_url: str,
    api_key: str,
    model: str
) -> str:
    """
    Invoke the judge LLM with the evaluation prompt.
    
    Args:
        prompt: The evaluation prompt
        base_url: Base URL for the judge LLM
        api_key: API key for the judge LLM
        model: Model name for the judge
        
    Returns:
        Response text from the LLM
        
    Raises:
        Exception: If LLM invocation fails
    """
    logger.info(
        f"Initializing judge LLM with model='{model}', "
        f"base_url='{base_url}'"
    )
    
    judge_llm = ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=0
    )
    
    judge_response = judge_llm.invoke(prompt)
    
    if judge_response is None:
        raise ValueError("Judge LLM returned None")
    
    judge_text = (
        judge_response.content
        if hasattr(judge_response, 'content')
        else str(judge_response)
    )
    
    if not judge_text or not judge_text.strip():
        raise ValueError("Judge LLM returned empty response")
    
    return judge_text


def validate_judge_response(judge_text: str) -> Dict[str, Any]:
    """
    Validate the judge LLM response for errors.
    
    Args:
        judge_text: The response text from the judge LLM
        
    Returns:
        Dict with error info if invalid, None if valid
    """
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
    return None


def extract_json_from_response(judge_text: str) -> str:
    """
    Extract JSON from the judge response, handling cases where
    there might be extra text.
    
    Args:
        judge_text: The response text from the judge LLM
        
    Returns:
        Extracted JSON string
        
    Raises:
        ValueError: If no JSON found in response
    """
    json_match = re.search(r'\{.*\}', judge_text, re.DOTALL)
    if json_match:
        return json_match.group(0)
    else:
        logger.error(
            f"No JSON found in judge response: {judge_text[:200]}"
        )
        raise ValueError("No JSON found in response")


def parse_judge_response(judge_text: str) -> Dict[str, Any]:
    """
    Parse the judge LLM response as JSON.
    
    Args:
        judge_text: The response text from the judge LLM
        
    Returns:
        Parsed evaluation dict
        
    Raises:
        json.JSONDecodeError: If response is not valid JSON
        ValueError: If no JSON found in response
    """
    # Extract JSON from response (in case there's extra text)
    json_text = extract_json_from_response(judge_text)
    evaluation = json.loads(json_text)
    return evaluation


def create_langfuse_score(
    langfuse_client: Langfuse,
    trace_id: str,
    evaluation: Dict[str, Any],
    product_id: str,
    question: str
) -> None:
    """
    Create a score in Langfuse 2.x.
    
    Args:
        langfuse_client: Langfuse client instance
        trace_id: The Langfuse trace ID to attach the score to
        evaluation: Parsed evaluation dict from judge
        product_id: Product ID for context
        question: Original user question
    """
    overall_score = evaluation.get('overall_score', 0.0)
    reasoning = evaluation.get('reasoning', 'No reasoning provided')
    
    langfuse_client.score(
        trace_id=trace_id,
        name="llm_judge_score",
        value=overall_score,
        comment=reasoning,
        metadata={
            "relevance_score": evaluation.get('relevance_score', 0.0),
            "accuracy_score": evaluation.get('accuracy_score', 0.0),
            "completeness_score": evaluation.get('completeness_score', 0.0),
            "clarity_score": evaluation.get('clarity_score', 0.0),
            "product_id": product_id,
            "question": question
        }
    )
    
    logger.info(
        f"Created LLM judge score {overall_score} "
        f"for trace {trace_id}"
    )


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
    validation_error = validate_judge_model(model)
    if validation_error:
        return validation_error
    
    logger.info(
        f"Calculating LLM judge score for trace {trace_id} "
        f"using model: '{model}'"
    )
    
    try:
        # Create evaluation prompt
        judge_prompt = create_judge_prompt(product_id, question, response)
        
        # Invoke judge LLM
        try:
            judge_text = invoke_judge_llm(judge_prompt, base_url, api_key, model)
        except Exception as invoke_error:
            logger.error(
                f"Error invoking judge LLM: {invoke_error}. "
                f"Model: {model}, Base URL: {base_url}"
            )
            return {
                "overall_score": 0.0,
                "reasoning": f"Judge LLM invocation failed: {str(invoke_error)}",
                "error": str(invoke_error)
            }
        
        # Validate response
        validation_error = validate_judge_response(judge_text)
        if validation_error:
            return validation_error
        
        # Parse response
        try:
            evaluation = parse_judge_response(judge_text)
        except ValueError as e:
            logger.error(f"Failed to extract JSON from judge response: {e}")
            return {
                "overall_score": 0.0,
                "reasoning": "Judge LLM did not return valid JSON",
                "error": str(e)
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
        
        # Create score in Langfuse
        overall_score = evaluation.get('overall_score', 0.0)
        reasoning = evaluation.get('reasoning', 'No reasoning provided')
        
        create_langfuse_score(
            langfuse_client,
            trace_id,
            evaluation,
            product_id,
            question
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
        
    except Exception as e:
        logger.error(f"Error calculating LLM judge score: {e}")
        return {
            "overall_score": 0.0,
            "reasoning": f"Error during evaluation: {str(e)}",
            "error": str(e)
        }

