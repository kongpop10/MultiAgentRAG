import requests
import json
import time
from typing import Dict, List, Any, Optional, Union
from config import *
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Model configurations with their specific attributes
MODELS = {
    'llama-3.1-70b-versatile': {
        'id': 'llama-3.1-70b-versatile',
        'max_tokens': 8192,
        'purpose': 'reasoning'
    },
    'mixtral-8x7b-32768': {
        'id': 'mixtral-8x7b-32768',
        'max_tokens': 32768,
        'purpose': 'knowledge'
    },
    'gemma-7b-it': {
        'id': 'gemma-7b-it',
        'max_tokens': 8192,
        'purpose': 'synthesis'
    }
}

API_KEY = os.getenv('GROQ_API_KEY')
API_URL = 'https://api.groq.com/openai/v1/chat/completions'

class ReasoningOrchestrator:
    def __init__(self, rag_system):
        self.rag = rag_system
        self.step_agents = {}

    def plan_reasoning_steps(self, context: str) -> List[str]:
        planner = self.rag.create_agent(
            'llama-3.1-70b-versatile',
            REASONING_PLANNER_PROMPT
        )
        plan_response = planner(context)
        
        try:
            # Extract steps from the response
            steps = []
            for line in plan_response.split('\n'):
                if line.strip().startswith(('Step', '1.', '2.', '3.', '4.', '5.')):
                    steps.append(line.split('.', 1)[1].strip())
            return steps if steps else ['Analyze', 'Evaluate', 'Conclude']
        except:
            return ['Analyze', 'Evaluate', 'Conclude']

    def execute_reasoning_step(self, step_number: int, step_description: str, context: str, previous_steps: Dict[str, str] = None) -> str:
        step_prompt = f"""Previous steps results: {json.dumps(previous_steps) if previous_steps else 'None'}
        
        Current step ({step_number}): {step_description}
        
        Context: {context}
        
        Execute this specific reasoning step and provide your analysis."""
        
        step_agent = self.rag.create_agent(
            'llama-3.1-70b-versatile',
            f"{REASONING_STEP_PROMPT}\nYou are specifically responsible for: {step_description}"
        )
        return step_agent(step_prompt)

    def orchestrate_reasoning(self, context: str) -> Dict[str, str]:
        # Plan the reasoning steps
        steps = self.plan_reasoning_steps(context)
        print(f"\nPlanned Reasoning Steps: {steps}")
        
        # Execute each step sequentially
        results = {}
        for i, step in enumerate(steps, 1):
            print(f"\nExecuting Reasoning Step {i}: {step}")
            results[f"Step {i}"] = self.execute_reasoning_step(i, step, context, results)
            time.sleep(1)  # Prevent rate limiting
        
        return results

class RAGSystem:
    def __init__(self):
        self.context_window = {}
        for model_id, config in MODELS.items():
            self.context_window[model_id] = config['max_tokens']
        self.reasoning_orchestrator = ReasoningOrchestrator(self)

    def handle_rate_limit(self, response: requests.Response) -> bool:
        if response.status_code == 429:
            try:
                wait_time = float(response.json().get('error', {}).get('message', '').split('in ')[1].split('s.')[0])
                print(f"\nRate limit reached. Waiting {wait_time:.2f} seconds...")
                time.sleep(wait_time + 1)
                return True
            except:
                time.sleep(30)
                return True
        return False

    def create_agent(self, model_id: str, system_prompt: str):
        def get_response(user_prompt: str, max_retries: int = 3) -> str:
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        API_URL,
                        headers={
                            'Authorization': f'Bearer {API_KEY}',
                            'Content-Type': 'application/json'
                        },
                        json={
                            'model': model_id,
                            'messages': [
                                {'role': 'system', 'content': system_prompt},
                                {'role': 'user', 'content': user_prompt}
                            ],
                            'max_tokens': min(2000, self.context_window[model_id] // 2),
                            'temperature': 0.7
                        }
                    )
                    
                    if response.status_code == 200:
                        return response.json()['choices'][0]['message']['content']
                    elif self.handle_rate_limit(response):
                        continue
                    else:
                        return f"Error: {response.status_code} - {response.text}"
                except Exception as e:
                    if attempt == max_retries - 1:
                        return f"Error: {str(e)}"
                    time.sleep(2 ** attempt)
            return "Maximum retries reached"
        return get_response

    def process_query(self, user_input: str) -> Dict[str, Any]:
        # 1. Knowledge Retrieval using Mixtral
        knowledge_agent = self.create_agent(
            'mixtral-8x7b-32768',
            KNOWLEDGE_RETRIEVAL_PROMPT
        )
        knowledge_response = knowledge_agent(user_input)
        
        # 2. Multi-step Reasoning using Orchestrator
        reasoning_context = f"""Query: {user_input}
        Retrieved Knowledge: {knowledge_response}"""
        reasoning_results = self.reasoning_orchestrator.orchestrate_reasoning(reasoning_context)
        
        # 3. Final Synthesis using Gemma
        synthesis_agent = self.create_agent(
            'gemma-7b-it',
            SYNTHESIS_PROMPT
        )
        synthesis_prompt = f"""
        Original Query: {user_input}
        
        Retrieved Knowledge:
        {knowledge_response}
        
        Reasoning Steps and Analysis:
        {json.dumps(reasoning_results, indent=2)}
        
        Provide a comprehensive yet concise summary with actionable insights.
        """
        final_response = synthesis_agent(synthesis_prompt)
        
        return {
            'knowledge_retrieval': knowledge_response,
            'reasoning_steps': reasoning_results,
            'final_synthesis': final_response
        }

class MultiAgentChat:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        self.api_url = 'https://api.groq.com/openai/v1/chat/completions'
        self.rag_system = RAGSystem()

    def create(
        self,
        messages: List[Dict[str, str]],
        model: str = "mixtral-8x7b-32768",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        OpenAI-compatible chat completion method.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model identifier (default: mixtral-8x7b-32768)
            temperature: Sampling temperature (default: 0.7)
            max_tokens: Maximum tokens in response (optional)
            stream: Whether to stream responses (not implemented)
            **kwargs: Additional parameters
            
        Returns:
            Dict containing response in OpenAI-compatible format
        """
        try:
            # Extract the user's query from messages
            user_query = ""
            for msg in reversed(messages):
                if msg['role'] == 'user':
                    user_query = msg['content']
                    break
            
            if not user_query:
                raise ValueError("No user message found in the conversation")

            # Process through RAG system
            results = self.rag_system.process_query(user_query)
            
            # Format response in OpenAI-compatible structure
            response = {
                "id": f"rag-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": results['final_synthesis'],
                        "function_call": None,
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": -1,  # Placeholder
                    "completion_tokens": -1,  # Placeholder
                    "total_tokens": -1  # Placeholder
                },
                "system_info": {
                    "knowledge_retrieval": results['knowledge_retrieval'],
                    "reasoning_steps": results['reasoning_steps']
                }
            }
            
            return response

        except Exception as e:
            raise Exception(f"Error in chat completion: {str(e)}")

    def get_models(self) -> Dict[str, Any]:
        """
        Get available models in OpenAI-compatible format.
        """
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "groq",
                    "permission": [],
                    "root": model_id,
                    "parent": None,
                    "context_window": config['max_tokens']
                }
                for model_id, config in MODELS.items()
            ]
        }

# Example usage
def example_usage():
    # Initialize the client
    client = MultiAgentChat()
    
    # Create a chat completion
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Analyze the impact of AI on healthcare"}
    ]
    
    try:
        response = client.create(messages=messages)
        
        # Access the main response
        print("\nFinal Response:")
        print(response['choices'][0]['message']['content'])
        
        # Access additional information
        print("\nKnowledge Retrieval:")
        print(response['system_info']['knowledge_retrieval'])
        
        print("\nReasoning Steps:")
        for step, content in response['system_info']['reasoning_steps'].items():
            print(f"\n{step}:")
            print(content)
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    example_usage()
