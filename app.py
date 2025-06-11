# app.py
from fastapi import FastAPI, HTTPException, Depends, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from dotenv import load_dotenv
import os
from mail_service import send_email
from groq import Groq
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
import tiktoken
from collections import deque
import time
import httpx
from redis.asyncio import Redis
import hashlib
import logging
from fastapi_limiter.depends import RateLimiter
from fastapi_limiter import FastAPILimiter
import json
from starlette.middleware.base import BaseHTTPMiddleware
import asyncio
from redis.exceptions import RedisError

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Premier Malta Chat API",
    description="API for the Premier Malta chatbot",
    version="1.0.0"
)

class RealIPMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        original_ip = request.client.host
        logger.info(f"Original client IP: {original_ip}")
        
        # Log all headers for debugging
        logger.info("=== Request Headers ===")
        for header, value in request.headers.items():
            logger.info(f"{header}: {value}")
        logger.info("=====================")
        
        # Try to get IP from X-Forwarded-For header first
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # X-Forwarded-For can contain multiple IPs, take the first one
            new_ip = forwarded.split(",")[0].strip()
            logger.info(f"Found IP from X-Forwarded-For: {new_ip}")
            request.client.host = new_ip
        # If no X-Forwarded-For, try X-Real-IP
        elif request.headers.get("X-Real-IP"):
            new_ip = request.headers.get("X-Real-IP")
            logger.info(f"Found IP from X-Real-IP: {new_ip}")
            request.client.host = new_ip
        else:
            logger.info(f"No proxy headers found, using original IP: {original_ip}")
            
        return await call_next(request)

# Add the middleware to your FastAPI app (before CORS middleware)
app.add_middleware(RealIPMiddleware)

# Add CORS middleware with more specific settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # For local development
        "https://2024premiermalta.com",  # Your main domain
        "https://*.2024premiermalta.com",  # Any subdomains
        # Add your WordPress site domain here
        "https://your-wordpress-site.com",  # Replace with your actual WordPress domain
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],  # Allow all headers
    expose_headers=["Content-Length", "X-Request-ID"],
    max_age=3600,  # Cache preflight requests for 1 hour
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize Redis
# Note: Ensure UPSTASH_REDIS_REST_URL starts with rediss:// or redis://
redis_url = os.getenv("UPSTASH_REDIS_REST_URL")
redis_password = os.getenv("UPSTASH_REDIS_REST_TOKEN")

if not redis_url or not redis_password:
    logger.error("UPSTASH_REDIS_REST_URL or UPSTASH_REDIS_REST_TOKEN not found in environment variables.")
    raise ValueError("Redis configuration is missing")

try:
    redis_client = Redis.from_url(
        redis_url,
        password=redis_password,
        encoding="utf-8",
        decode_responses=True,
        socket_timeout=5.0,
        socket_connect_timeout=5.0
    )
    logger.info("✅ Redis client initialized.")
except Exception as e:
    logger.error(f"❌ Failed to initialize Redis client: {e}")
    raise

async def get_redis() -> Redis:
    return redis_client

# Initialize clients and stores
openai_api_key = os.getenv("OPENAI_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

if not openai_api_key:
    logger.error("OPENAI_API_KEY not found in environment variables.")
    # Handle missing API key - the OpenAIEmbeddings initialization will fail
    # This will be caught below.

if not qdrant_url or not qdrant_api_key:
    logger.error("QDRANT_URL or QDRANT_API_KEY not found in environment variables.")
    # Handle missing API keys - the QdrantClient initialization will fail
    # This will be caught below.

try:
    embedder = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=openai_api_key, # Use the potentially None variable
    )
    logger.info("✅ OpenAI Embeddings initialized.")
except Exception as e:
    logger.error(f"❌ Failed to initialize OpenAI Embeddings: {e}")
    raise # Re-raise to prevent the app from starting

try:
    qdrant_client = QdrantClient(
        url=qdrant_url, # Use the potentially None variable
        api_key=qdrant_api_key, # Use the potentially None variable
        timeout=10.0,
        prefer_grpc=True,
    )
    logger.info("✅ Qdrant client initialized.")
    # Optional: Perform a quick test connection
    # try:
    #     qdrant_client.get_collection(collection_name="premier_malta")
    #     logger.info("✅ Qdrant connection successful.")
    # except Exception as e:
    #      logger.error(f"❌ Qdrant connection failed or collection not found: {e}")
except Exception as e:
    logger.error(f"❌ Failed to initialize Qdrant client: {e}")
    raise # Re-raise to prevent the app from starting

# Note: vectorstore depends on qdrant_client and embedder, so it must be initialized after them
vectorstore = QdrantVectorStore(
    client=qdrant_client,
    collection_name="premier_malta", # Ensure this matches the collection name in Embdediing.py
    embedding=embedder,
)
logger.info(f"✅ Qdrant VectorStore initialized for collection: {vectorstore.collection_name}.")

# Constants
MAX_CONVERSATION_HISTORY = 0 # Consider increasing this for limited conversational memory
conversation_history = deque(maxlen=MAX_CONVERSATION_HISTORY)
CACHE_TTL = 300

class AskRequest(BaseModel):
    prompt: str
    tags: Optional[List[str]] = None

class AskResponse(BaseModel):
    response: str

class EmailRequest(BaseModel):
    name: str
    email: Optional[str] = None
    user_issue: str

class EmailResponse(BaseModel):
    status: str
    message: Optional[str] = None

SECTION_URL_MAP = {
    "2024 PREMIER MALTA": "https://2024premiermalta.com",
    "Agenda": "http://2024premiermalta.com/itinerary",
    "Activities": "http://2024premiermalta.com/activities",
    "Hotel information": "http://2024premiermalta.com/hotel-information",
    "Pre/Post Extensions": "http://2024premiermalta.com/pre-post-extensions",
    "2024 U.S. Award Trip Policy": "http://2024premiermalta.com/wp-content/uploads/2025/01/2024-North-American-Award-Trip-Policy_updated-10.1.2024.pdf",
    "Travel Tips": "http://2024premiermalta.com/travel-tips",
    "SUGGESTED ATTIRE": "http://2024premiermalta.com/suggested-attire",
    "Restaurants": "http://2024premiermalta.com/restaurants",
    "Full Day Gozo": "http://2024premiermalta.com/activity/full-day-gozo",
    "Tour of Malta's Capital: Valetta": "http://2024premiermalta.com/activity/valetta",
    "Spa Treatments at The Beauty Clinic": "http://2024premiermalta.com/activity/spa-treatments-at-the-beauty-clinic",
    "Half Day Ta' Betta Wine Estate": "http://2024premiermalta.com/activity/half-day-ta-betta-wine-estate",
    "Golf at Royal Malta Golf Club": "http://2024premiermalta.com/activity/golf-at-royal-malta-golf-club",
    "Trekking & Fishing Village (Southern Malta)": "http://2024premiermalta.com/activity/trekking-fishing-village-southern-malta",
    "Half Day Mosta & Mdina": "http://2024premiermalta.com/activity/half-day-mosta-mdina"
}

def format_doc(doc):
    # max_length = 500 # You can remove or comment out this line
    content = doc.page_content # Use the full content
    section_hierarchy = doc.metadata.get("section_hierarchy", "Unknown Section")
    source_url = doc.metadata.get("source_url", "No URL available")
    return (
        f"From {section_hierarchy}:\n"
        f"{content}\n" # Use the full content here
        f"Source: {source_url}\n"
        "---"
    )

def retrieve_relevant_docs(query, top_k=40, score_threshold=0.6):
    """
    Retrieves relevant documents from the vector store.
    Adjust top_k and score_threshold based on performance testing:
    - Increase score_threshold if retrieving too many irrelevant documents.
    - Decrease score_threshold if missing relevant documents.
    - Increase top_k if more context is needed by the LLM (after filtering/deduplication).
    - Decrease top_k for efficiency if good responses are generated with fewer documents.
    """
    try:
        logger.info(f"🔍 Searching Qdrant for: '{query}' with top_k={top_k}, score_threshold={score_threshold}")

        # Search the vector store, fetch more initially than the final top_k
        kb_docs = vectorstore.similarity_search_with_relevance_scores(query, k=top_k * 2)

        if not kb_docs:
            logger.warning("❌ No documents found in initial search.")
            return []

        logger.info(f"🧠 Retrieved {len(kb_docs)} raw docs from Qdrant.")

        # Filter by score threshold
        filtered_docs = [(doc, score) for doc, score in kb_docs if score >= score_threshold]

        if not filtered_docs:
            logger.warning(f"⚠️ No documents passed the score threshold ({score_threshold}).")
            return []

        logger.info(f"✅ {len(filtered_docs)} documents passed score threshold.")

        # Deduplicate using SHA-256 hash of full content
        seen_hashes = set()
        unique_docs = []

        for doc, score in filtered_docs:
            content = doc.page_content.strip()
            if len(content) < 20:
                logger.debug(f"Skipping document with short content length ({len(content)} chars): {content[:50]}...")
                continue  # skip short content

            # Using the full content hash for more robust deduplication
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                doc.metadata["relevance_score"] = score  # optionally attach score
                unique_docs.append(doc)

            if len(unique_docs) >= top_k:
                logger.debug(f"Reached desired top_k ({top_k}) unique documents, stopping deduplication.")
                break

        if not unique_docs:
            logger.warning("📦 No unique relevant documents found after deduplication.")
            return []

        logger.info(f"📦 Returning {len(unique_docs)} unique, relevant documents.")
        return unique_docs

    except Exception as e:
        logger.error(f"⚠️ Retrieval error: {e}", exc_info=True) # Log exception details
        return []

async def groq_async_completion(messages, model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.3):
    """
    Sends messages to the Groq API for chat completion with retries and better timeout handling.
    """
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        logger.error("GROQ_API_KEY not found in environment variables.")
        raise HTTPException(status_code=500, detail="LLM API key not configured.")

    max_retries = 3
    base_timeout = 30.0  # 30 seconds base timeout
    retry_delay = 1.0    # 1 second delay between retries

    for attempt in range(max_retries):
        try:
            logger.info(f"🤖 Attempt {attempt + 1}/{max_retries} - Sending messages to Groq API (Model: {model}, Temp: {temperature})")
            
            async with httpx.AsyncClient(timeout=base_timeout) as client:
                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {groq_api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": model,
                        "messages": messages,
                        "temperature": temperature
                    },
                )
                response.raise_for_status()
                logger.info("✅ Groq API call successful.")
                return response.json()

        except httpx.TimeoutException as e:
            logger.warning(f"⚠️ Timeout on attempt {attempt + 1}/{max_retries}: {str(e)}")
            if attempt == max_retries - 1:
                raise HTTPException(
                    status_code=503,
                    detail="LLM API request timed out after multiple attempts. Please try again."
                )
            await asyncio.sleep(retry_delay * (attempt + 1))  # Exponential backoff

        except httpx.HTTPStatusError as e:
            logger.error(f"❌ Groq API HTTP error occurred: {e.response.status_code} - {e.response.text}", exc_info=True)
            raise HTTPException(status_code=e.response.status_code, detail=f"LLM API error: {e.response.text}")

        except httpx.RequestError as e:
            logger.error(f"❌ Groq API request error occurred: {e}", exc_info=True)
            if attempt == max_retries - 1:
                raise HTTPException(status_code=503, detail=f"LLM API request failed: {e}")
            await asyncio.sleep(retry_delay * (attempt + 1))

        except Exception as e:
            logger.error(f"❌ An unexpected error occurred during Groq API call: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="An unexpected LLM error occurred.")

@app.post("/ask", response_model=AskResponse)
async def ask_bot(request: AskRequest, req: Request, redis: Redis = Depends(get_redis)):
    """
    Handles user questions with rate limiting and chat history.
    """
    # Skip rate limiting for OPTIONS requests
    if req.method == "OPTIONS":
        return {"response": "OK"}

    # Get user's IP for rate limiting
    ip = req.client.host
    ip_key = f"rate:ip:{ip}"
    
    try:
        # Check rate limit
        ip_count = await redis.incr(ip_key)
        if ip_count == 1:
            await redis.expire(ip_key, 86400)  # 24 hours
        if ip_count > 15:  # Max 15 requests per day
            raise HTTPException(429, "Daily quota exceeded. Please try again tomorrow.")

        # Store user message in chat history
        hist_key = f"chat:{ip}"
        await redis.lpush(hist_key, json.dumps({
            "role": "user",
            "msg": request.prompt,
            "timestamp": int(time.time())
        }))
        await redis.expire(hist_key, 604800)  # 7 days

        start_time = time.time()
        user_prompt = request.prompt.strip()
        logger.info(f"Received /ask request for prompt: '{user_prompt}'")

        cache_key = f"pm:ask:{hashlib.sha256(user_prompt.encode('utf-8')).hexdigest()}" # Use hash for cache key

        try:
            cached_response = await redis.get(cache_key)
            if cached_response:
                logger.info("✅ Cache hit for prompt.")
                return AskResponse(response=cached_response)

            logger.info("🔄 Cache miss for prompt.")
            relevant_docs = retrieve_relevant_docs(user_prompt)

            if not relevant_docs:
                fallback_msg = "I couldn't find any information about that in the 2024 Premier Malta website. Please ask something related to the website content."
                logger.warning("No relevant documents found, returning fallback message.")
                # Cache the fallback message as well
                try:
                    await redis.set(cache_key, fallback_msg, ex=CACHE_TTL)
                    logger.info("✅ Cached fallback message.")
                except Exception as cache_e:
                    logger.error(f"❌ Failed to cache fallback message: {cache_e}", exc_info=True)
                return AskResponse(response=fallback_msg)

            context = "\n\n".join(format_doc(doc) for doc in relevant_docs)

            # --- Debugging: Write context to a file ---
            debug_file_path = "debug_context.txt"
            try:
                with open(debug_file_path, "w", encoding="utf-8") as f:
                    f.write(context)
                logger.debug(f"Context written to {debug_file_path} for debugging.")
            except Exception as e:
                logger.error(f"Failed to write context to debug file {debug_file_path}: {e}", exc_info=True)
            # --- End Debugging Code ---


            # Refined System Message
            messages = [
                 {
                    "role": "system",
                    "content": """You are a helpful, ethical, and respectful customer support assistant for the 2024 Premier Malta website.

                    ---

                    ###  STRICT RULE: ONLY Use Provided Information
                    - **You CANNOT use general knowledge.**
                    - **You MUST respond ONLY using information found in the "Context from 2024 Premier Malta website" provided below and potentially from the "Previous Answer".**
                    - If the user's question is NOT about the 2024 Premier Malta website content AT ALL, respond *only* with:
                      > "I can only answer questions about the 2024 Premier Malta website. Please ask something related to the website content."
                    - If the user's question *is* about the website content, but you cannot find the answer in the provided context, respond *only* with:
                      > "I couldn't find any information about that in the 2024 Premier Malta website. Please ask something related to the website content."

                    ---

                    ### ✅ Response Formatting Rules:
                    - Use **clean Markdown**: headings, bullets, bold/italics for clarity.
                    - Be user-friendly, neutral, and professional.
                    - **ALWAYS include the source URL** for any information you provide, using this format: `[Source: Page Name](https://url)`. Example: `[Source: Suggested Attire](http://2024premiermalta.com/suggested-attire)`. Use the `section_hierarchy` or infer a good Page Name if `section_hierarchy` is "Unknown Section".
                    - Use `[Name](https://url)` format for all links — never raw URLs.

                    ### 🔒 Core Guidelines:
                    - ✋ Never mention backend, tech, or internal systems (AI, system, database, code, server, vector store, embeddings, etc.).
                    - 🛑 No troubleshooting unless explicitly mentioned in the info.
                    - 🧱 Read-only mode — no suggestions for changes or edits to the website or trip details.
                    - ❌ Do not respond to out-of-scope topics (e.g., login issues, development problems, personal account details).

                    ---
                    ### 🧷 Ethics:
                    - Only validated, user-facing info from the website.
                    - Strict confidentiality.
                    - Stay on-topic and professional.
                    """
                }
            ]

            if conversation_history: # Check if history is enabled (maxlen > 0)
                 logger.debug(f"Adding {len(conversation_history)} previous turns to messages.")
                 for prev_prompt, prev_response in conversation_history:
                     messages.extend([
                         {"role": "user", "content": f"Previous Question: {prev_prompt}"},
                         {"role": "assistant", "content": f"Previous Answer: {prev_response}"}
                     ])
            else:
                logger.debug("Conversation history is disabled.")


            messages.append({
                "role": "user",
                "content": f"Context from 2024 Premier Malta website:\n{context}\nSITEMAP: {SECTION_URL_MAP}\n\nCurrent Question: {user_prompt}"
            })

            completion = await groq_async_completion(messages)
            response = completion["choices"][0]["message"]["content"]

            try:
                enc = tiktoken.get_encoding("cl100k_base")
                logger.info(f"Tokens - Input: {len(enc.encode(user_prompt))}, Output: {len(enc.encode(response))}")
            except Exception as token_e:
                logger.warning(f"Could not encode tokens: {token_e}")


            if MAX_CONVERSATION_HISTORY > 0: # Only append if history is enabled
                conversation_history.append((user_prompt, response))

            try:
                await redis.set(cache_key, response, ex=CACHE_TTL)
                logger.info("✅ Cached response.")
            except Exception as cache_e:
                 logger.error(f"❌ Failed to cache response: {cache_e}", exc_info=True)

            # Store bot response in chat history
            await redis.lpush(hist_key, json.dumps({
                "role": "bot",
                "msg": response,
                "timestamp": int(time.time())
            }))

            end_time = time.time()
            logger.info(f"Request processed successfully in {end_time - start_time:.2f} seconds.")

            return AskResponse(response=response)

        except HTTPException as http_e:
            logger.error(f"❌ HTTPException occurred: {http_e.detail}", exc_info=True)
            raise http_e # Re-raise FastAPI HTTPExceptions
        except Exception as e:
            logger.error(f"❌ An unexpected critical error occurred during /ask request: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="An internal server error occurred.")

    except RedisError as e:
        logger.error(f"Redis error during rate limiting: {e}")
        # If Redis fails, allow the request but log the error
        logger.warning("Rate limiting temporarily disabled due to Redis error")
        # Continue with the request

@app.post("/send-support-email", response_model=EmailResponse)
async def send_support_email(request: EmailRequest, req: Request, redis: Redis = Depends(get_redis)):
    """
    Sends a support email with chat history as an attachment.
    """
    try:
        if not request.email:
            raise HTTPException(400, "Email is required")

        # Get chat history
        hist_key = f"chat:{req.client.host}"
        chat_history = await redis.lrange(hist_key, 0, -1)
        chat_history = [json.loads(msg) for msg in chat_history][::-1]  # Reverse to get chronological order
        
        # Send email with chat history
        success = send_email(
            request.name,
            request.email,
            request.user_issue,
            chat_history=chat_history
        )
        
        if success:
            return EmailResponse(
                status="success",
                message="Email sent successfully with chat history"
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to send email")
            
    except Exception as e:
        logger.error(f"Failed to send support email: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to send email: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint. Checks basic service availability.
    Could be expanded to check connections to Redis, Qdrant, etc.
    """
    # Basic health check - assumes the app is running if this endpoint is reachable
    # For a more robust check, you could ping external services here
    health_status = {"status": "healthy"}
    # Example of checking Redis connectivity in health check (requires async ping)
    # try:
    #     await redis_client.ping()
    #     health_status["redis"] = "connected"
    # except Exception:
    #     health_status["redis"] = "disconnected"
    # Log health check access (optional)
    # logger.debug("Health check endpoint accessed.")
    return health_status

@app.on_event("startup")
async def startup_event():
    """Initialize FastAPI Limiter with Redis."""
    try:
        # Test Redis connection first
        await redis_client.ping()
        logger.info("✅ Redis connection test successful.")
        
        # Initialize FastAPI Limiter with the Redis client
        await FastAPILimiter.init(
            redis_client,
            prefix="fastapi-limiter"
        )
        logger.info("✅ FastAPI Limiter initialized.")
    except Exception as e:
        logger.error(f"❌ Failed to initialize FastAPI Limiter: {e}")
        # Don't re-raise to allow the app to start even if rate limiting fails
        logger.warning("⚠️ Rate limiting will be disabled due to initialization failure.")

if __name__ == "__main__":
    import uvicorn
    # When running with reload=True, uvicorn manages restarts on code changes.
    # workers=4 is generally for production where reload is False.
    # We will use reload=True for development convenience.
    logger.info("Starting Uvicorn server...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
