import requests
import json
import time
from typing import Dict, List, Any
from config import *
from dotenv import load_dotenv
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
import aiohttp

# Load environment variables
load_dotenv()

# Model configurations
MODELS = {
    'llama-3.1-70b-versatile': {
        'id': 'llama-3.1-70b-versatile',
        'max_tokens': 8192,
        'purpose': 'reasoning'
    },
    'gemma-7b-it': {
        'id': 'gemma-7b-it',
        'max_tokens': 8192,
        'purpose': 'synthesis'
    }
}

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GROQ_API_URL = 'https://api.groq.com/openai/v1/chat/completions'

class GeminiKnowledgeRetriever:
    def __init__(self):
        self.api_key = GOOGLE_API_KEY
        self.api_url = GOOGLE_API_CONFIG['api_url']
        self.model = GOOGLE_API_CONFIG['model']
        self.last_request_time = 0
        self.requests_this_minute = 0
        self.requests_today = 0
        self.minute_start = time.time()
        self.day_start = time.time()
        
        # Load daily request count from persistent storage if available
        self.load_request_count()

    def load_request_count(self):
        try:
            with open('.request_count', 'r') as f:
                data = json.load(f)
                if time.time() - data['day_start'] < 86400:  # If same day
                    self.requests_today = data['count']
                    self.day_start = data['day_start']
        except:
            pass

    def save_request_count(self):
        try:
            with open('.request_count', 'w') as f:
                json.dump({
                    'count': self.requests_today,
                    'day_start': self.day_start
                }, f)
        except:
            pass

    def _check_and_update_rate_limits(self):
        current_time = time.time()
        
        # Check daily limit first
        if self.requests_today >= DOCUMENT_CONFIG['daily_request_limit']:
            remaining_time = 86400 - (current_time - self.day_start)
            raise Exception(f"Daily request limit reached ({DOCUMENT_CONFIG['daily_request_limit']}). "
                          f"Please wait {remaining_time/3600:.1f} hours.")
        
        # Reset minute counters if a minute has passed
        if current_time - self.minute_start >= 60:
            self.requests_this_minute = 0
            self.minute_start = current_time

        # Reset daily counters if a day has passed
        if current_time - self.day_start >= 86400:
            self.requests_today = 0
            self.day_start = current_time
            self.save_request_count()

        # Check rate limits
        if self.requests_this_minute >= GOOGLE_API_CONFIG['rate_limits']['requests_per_minute']:
            wait_time = 60 - (current_time - self.minute_start)
            print(f"\nRate limit reached. Waiting {wait_time:.2f} seconds...")
            time.sleep(wait_time)
            self.requests_this_minute = 0
            self.minute_start = time.time()

        # Ensure minimum delay between requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < GOOGLE_API_CONFIG['rate_limits']['min_delay']:
            time.sleep(GOOGLE_API_CONFIG['rate_limits']['min_delay'] - time_since_last)

    def get_knowledge(self, query: str) -> str:
        self._check_and_update_rate_limits()
        
        try:
            headers = {
                'Content-Type': 'application/json'
            }
            
            data = {
                "contents": [{
                    "parts": [{
                        "text": f"{KNOWLEDGE_RETRIEVAL_PROMPT}\n\nQuery: {query}"
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": GOOGLE_API_CONFIG['max_tokens'],
                    "topP": 0.8,
                    "topK": 40,
                    "candidateCount": 1,
                    "stopSequences": []
                },
                "safetySettings": [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_NONE"
                    }
                ]
            }
            
            url = f"{self.api_url}?key={self.api_key}"
            
            self.last_request_time = time.time()
            self.requests_this_minute += 1
            self.requests_today += 1
            self.save_request_count()  # Save updated count
            
            response = requests.post(
                url,
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                return response.json()['candidates'][0]['content']['parts'][0]['text']
            else:
                print(f"Gemini API Error: {response.status_code} - {response.text}")
                return f"Error: {response.status_code}"
                
        except Exception as e:
            print(f"Error in Gemini knowledge retrieval: {str(e)}")
            return f"Error: {str(e)}"

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

class EnhancedGeminiRetriever:
    def __init__(self):
        self.api_key = GOOGLE_API_KEY
        self.api_url = GOOGLE_API_CONFIG['api_url']
        max_workers = getattr(DOCUMENT_CONFIG, 'max_parallel_requests', 3)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
    async def process_chunk_async(self, chunk: str, query: str) -> str:
        try:
            headers = {
                'Content-Type': 'application/json'
            }
            
            data = {
                "contents": [{
                    "parts": [{
                        "text": f"{KNOWLEDGE_RETRIEVAL_PROMPT}\n\nContext: {chunk}\nQuery: {query}"
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": GOOGLE_API_CONFIG['max_tokens'],
                    "topP": 0.8,
                    "topK": 40
                },
                "safetySettings": GOOGLE_API_CONFIG['safety_settings']
            }
            
            url = f"{self.api_url}?key={self.api_key}"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['candidates'][0]['content']['parts'][0]['text']
                    else:
                        print(f"Chunk processing error: {response.status}")
                        return ""
        except Exception as e:
            print(f"Chunk processing error: {str(e)}")
            return ""

    def get_knowledge(self, query: str) -> str:
        try:
            headers = {
                'Content-Type': 'application/json'
            }
            
            data = {
                "contents": [{
                    "parts": [{
                        "text": f"{KNOWLEDGE_RETRIEVAL_PROMPT}\n\nQuery: {query}"
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": GOOGLE_API_CONFIG['max_tokens'],
                    "topP": 0.8,
                    "topK": 40
                },
                "safetySettings": GOOGLE_API_CONFIG['safety_settings']
            }
            
            url = f"{self.api_url}?key={self.api_key}"
            
            response = requests.post(
                url,
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                return response.json()['candidates'][0]['content']['parts'][0]['text']
            else:
                print(f"Gemini API Error: {response.status_code} - {response.text}")
                return f"Error: {response.status_code}"
                
        except Exception as e:
            print(f"Error in Gemini knowledge retrieval: {str(e)}")
            return f"Error: {str(e)}"

class HybridKnowledgeRetriever:
    def __init__(self):
        self.gemini_retriever = EnhancedGeminiRetriever()
        self.mixtral_retriever = MixtralKnowledgeRetriever()
        
    async def process_chunks_parallel(self, chunks: List[str], query: str) -> List[str]:
        tasks = []
        for chunk in chunks:
            try:
                task = self.gemini_retriever.process_chunk_async(chunk, query)
                tasks.append(task)
            except Exception as e:
                print(f"Gemini chunk processing failed, falling back to Mixtral: {str(e)}")
                # Fallback to Mixtral for failed chunks
                tasks.append(self.mixtral_retriever.get_knowledge(query, chunk))
        return await asyncio.gather(*tasks)
        
    def get_knowledge(self, query: str, document: str = None) -> str:
        try:
            if document:
                estimated_tokens = len(document.split()) * 1.3
                
                # Use Mixtral for very large documents
                if estimated_tokens > DOCUMENT_CONFIG['use_mixtral_threshold']:
                    print("Document exceeds Gemini threshold, using Mixtral...")
                    return self.mixtral_retriever.get_knowledge(query, document)
                
                # Enhanced chunking for medium documents
                if estimated_tokens > DOCUMENT_CONFIG['small_doc_limit']:
                    chunks = self.chunk_document_with_overlap(
                        document,
                        DOCUMENT_CONFIG['chunk_size'],
                        DOCUMENT_CONFIG['overlap']
                    )
                    
                    # Process chunks with fallback handling
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        responses = loop.run_until_complete(
                            self.process_chunks_parallel(chunks, query)
                        )
                    except Exception as e:
                        print(f"Parallel processing failed, falling back to Mixtral: {str(e)}")
                        loop.close()
                        return self.mixtral_retriever.get_knowledge(query, document)
                    loop.close()
                    
                    # Combine and synthesize responses
                    combined_response = "\n".join(filter(None, responses))
                    synthesis_query = f"Synthesize these findings:\n{combined_response}"
                    try:
                        return self.gemini_retriever.get_knowledge(synthesis_query)
                    except Exception as e:
                        print(f"Synthesis failed, falling back to Mixtral: {str(e)}")
                        return self.mixtral_retriever.get_knowledge(synthesis_query)
            
            # Try Gemini first for simple queries or small documents
            try:
                return self.gemini_retriever.get_knowledge(query)
            except Exception as e:
                print(f"Gemini processing failed, falling back to Mixtral: {str(e)}")
                return self.mixtral_retriever.get_knowledge(query, document)
                
        except Exception as e:
            print(f"Error in knowledge retrieval, using fallback: {str(e)}")
            return self.mixtral_retriever.get_knowledge(query, document)
    
    def chunk_document_with_overlap(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        words = text.split()
        chunks = []
        start = 0
        
        while start < len(words):
            end = start + chunk_size
            chunk = words[start:end]
            chunks.append(' '.join(chunk))
            start = end - overlap
            
        return chunks

class MixtralKnowledgeRetriever:
    def __init__(self):
        self.api_key = GROQ_API_KEY
        self.model = GROQ_API_CONFIG['model']
        self.api_url = GROQ_API_CONFIG['api_url']
        
    async def process_chunk_async(self, chunk: str, query: str) -> str:
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'model': self.model,
                'messages': [
                    {'role': 'system', 'content': KNOWLEDGE_RETRIEVAL_PROMPT},
                    {'role': 'user', 'content': f"Context: {chunk}\nQuery: {query}"}
                ],
                'max_tokens': GROQ_API_CONFIG['max_tokens'] // 2,
                'temperature': GROQ_API_CONFIG['temperature']
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result['choices'][0]['message']['content']
                    else:
                        print(f"Chunk processing error: {response.status}")
                        return ""
        except Exception as e:
            print(f"Chunk processing error: {str(e)}")
            return ""

    def get_knowledge(self, query: str, document: str = None) -> str:
        try:
            if document:
                # Split into chunks if document is large
                if len(document.split()) > DOCUMENT_CONFIG['model_thresholds']['medium']:
                    chunks = self.chunk_document(
                        document,
                        GROQ_API_CONFIG['chunk_settings']['size'],
                        GROQ_API_CONFIG['chunk_settings']['overlap']
                    )
                    
                    # Process chunks in parallel
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    responses = loop.run_until_complete(
                        self.process_chunks_parallel(chunks, query)
                    )
                    loop.close()
                    
                    # Combine responses
                    combined_response = "\n".join(filter(None, responses))
                    return self.synthesize_responses(combined_response)
            
            # Direct processing for small documents or simple queries
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            content = query if not document else f"Document: {document}\n\nQuery: {query}"
            
            data = {
                'model': self.model,
                'messages': [
                    {'role': 'system', 'content': KNOWLEDGE_RETRIEVAL_PROMPT},
                    {'role': 'user', 'content': content}
                ],
                'max_tokens': GROQ_API_CONFIG['max_tokens'] // 2,
                'temperature': GROQ_API_CONFIG['temperature']
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                raise Exception(f"API Error: {response.status_code}")
                
        except Exception as e:
            print(f"Error in Mixtral processing: {str(e)}")
            # Fallback to Gemini for errors
            return self.fallback_to_gemini(query, document)

    def fallback_to_gemini(self, query: str, document: str = None) -> str:
        try:
            gemini = GeminiKnowledgeRetriever()
            return gemini.get_knowledge(query)
        except Exception as e:
            return f"Both primary and fallback models failed: {str(e)}"

    async def process_chunks_parallel(self, chunks: List[str], query: str) -> List[str]:
        tasks = [self.process_chunk_async(chunk, query) for chunk in chunks]
        return await asyncio.gather(*tasks)

    def chunk_document(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        words = text.split()
        chunks = []
        start = 0
        
        while start < len(words):
            end = start + chunk_size
            chunk = words[start:end]
            chunks.append(' '.join(chunk))
            start = end - overlap
            
        return chunks

    def synthesize_responses(self, combined_response: str) -> str:
        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'model': self.model,
                'messages': [
                    {'role': 'system', 'content': SYNTHESIS_PROMPT},
                    {'role': 'user', 'content': f"Synthesize these findings:\n{combined_response}"}
                ],
                'max_tokens': GROQ_API_CONFIG['max_tokens'] // 2,
                'temperature': 0.7
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                raise Exception(f"Synthesis API Error: {response.status_code}")
        except Exception as e:
            print(f"Synthesis error: {str(e)}")
            return combined_response

class RAGSystem:
    def __init__(self):
        self.context_window = {}
        for model_id, config in MODELS.items():
            self.context_window[model_id] = config['max_tokens']
        self.reasoning_orchestrator = ReasoningOrchestrator(self)
        self.knowledge_retriever = HybridKnowledgeRetriever()

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
                        GROQ_API_URL,
                        headers={
                            'Authorization': f'Bearer {GROQ_API_KEY}',
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
        # 1. Knowledge Retrieval using Gemini
        knowledge_response = self.knowledge_retriever.get_knowledge(user_input)
        
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

def main():
    rag_system = RAGSystem()
    print("Enhanced RAG System with Multi-Step Reasoning - Ready!")
    print("Using models:", ", ".join(MODELS.keys()))
    
    while True:
        user_input = input("\nEnter your query (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        
        print("\nProcessing your query across multiple models...")
        results = rag_system.process_query(user_input)
        
        print("\n=== Knowledge Retrieval ===")
        print(results['knowledge_retrieval'])
        
        print("\n=== Reasoning Steps ===")
        for step, result in results['reasoning_steps'].items():
            print(f"\n{step}:")
            print(result)
        
        print("\n=== Final Synthesis ===")
        print(results['final_synthesis'])
        
        time.sleep(2)

if __name__ == "__main__":
    main()
