"""Bedrock client for LLM and embedding operations."""
import json
import time
from typing import Dict, List, Optional
import boto3


class BedrockClient:
    """Client for Amazon Bedrock LLM and embedding operations."""
    
    def __init__(self, region: str = "us-east-1"):
        self.bedrock_runtime = boto3.client("bedrock-runtime", region_name=region)
        self.region = region
        
    def embed_text(self, text: str, model_id: str = "amazon.titan-embed-text-v2:0") -> List[float]:
        """Generate embedding for text.
        
        Args:
            text: Text to embed
            model_id: Embedding model ID
            
        Returns:
            List of floats representing the embedding vector
        """
        start_time = time.time()
        
        body = json.dumps({
            "inputText": text,
            "dimensions": 1024,
            "normalize": True
        })
        
        response = self.bedrock_runtime.invoke_model(
            modelId=model_id,
            body=body,
            contentType="application/json",
            accept="application/json"
        )
        
        result = json.loads(response["body"].read())
        latency = time.time() - start_time
        
        return result["embedding"], latency
    
    def generate_text(
        self,
        messages: List[Dict[str, str]],
        model_id: str = "us.anthropic.claude-sonnet-4-20250514-v1:0",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> tuple[str, Dict[str, any]]:
        """Generate text using Bedrock LLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model_id: LLM model ID
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: Optional system prompt
            
        Returns:
            Tuple of (generated_text, metrics_dict)
        """
        start_time = time.time()
        
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages
        }
        
        if system_prompt:
            body["system"] = system_prompt
        
        response = self.bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json"
        )
        
        result = json.loads(response["body"].read())
        latency = time.time() - start_time
        
        # Extract text from response
        text = result["content"][0]["text"]
        
        # Collect metrics
        metrics = {
            "latency_ms": latency * 1000,
            "input_tokens": result["usage"]["input_tokens"],
            "output_tokens": result["usage"]["output_tokens"],
            "total_tokens": result["usage"]["input_tokens"] + result["usage"]["output_tokens"]
        }
        
        return text, metrics
    
    def calculate_cost(self, input_tokens: int, output_tokens: int, model_id: str) -> float:
        """Calculate cost based on token usage.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model_id: Model ID for pricing lookup
            
        Returns:
            Cost in USD
        """
        # Pricing per 1M tokens (as of 2026)
        pricing = {
            "us.anthropic.claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
            "global.anthropic.claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
            "us.anthropic.claude-sonnet-4-20250514-v1:0": {"input": 3.0, "output": 15.0},
            "anthropic.claude-sonnet-4-20250514-v1:0": {"input": 3.0, "output": 15.0},
            "us.anthropic.claude-sonnet-5": {"input": 3.0, "output": 15.0},
            "anthropic.claude-3-5-sonnet-20241022-v2:0": {"input": 3.0, "output": 15.0},
            "amazon.titan-embed-text-v2:0": {"input": 0.0001, "output": 0.0}  # Per embedding
        }
        
        if model_id not in pricing:
            # Default fallback pricing
            input_price = 3.0
            output_price = 15.0
        else:
            input_price = pricing[model_id]["input"]
            output_price = pricing[model_id]["output"]
        
        cost = (input_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price
        return cost
