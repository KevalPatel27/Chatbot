# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
from mail_service import send_email
from groq import Groq
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
import datetime
import uuid
import tiktoken
import json
from collections import deque
from functools import lru_cache
import time

load_dotenv()
app = Flask(__name__)
CORS(app)

# Initialize clients and stores
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embedder = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

# Initialize Qdrant client with optimizations
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=10.0,  # Set timeout
    prefer_grpc=True,  # Use gRPC for better performance
)

vectorstore = QdrantVectorStore(
    client=qdrant_client,
    collection_name="premier_malta",
    embedding=embedder
)

# Constants
MAX_CONVERSATION_HISTORY = 5
conversation_history = deque(maxlen=MAX_CONVERSATION_HISTORY)

CACHE_TTL = 300  # 5 minutes

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
    """Format document with hierarchical metadata for context."""
    section_hierarchy = doc.metadata.get("section_hierarchy", "Unknown Section")
    source_url = doc.metadata.get("source_url", "No URL available")
    return (
        f"From {section_hierarchy}:\n"
        f"{doc.page_content}\n"
        f"Source: {source_url}\n"
        "---"
    )

def retrieve_relevant_docs(query, top_k=10, score_threshold=0.2):
    """Retrieve both knowledge base docs and past conversations."""
    try:
        # 1. Retrieve from knowledge base with optimized parameters
        print(f"\n🔍 Searching for: {query}")
        kb_docs = vectorstore.similarity_search_with_score(
            query, 
            k=15,  # Reduced from 30 to 15
            score_threshold=score_threshold  # Add score threshold at query level
        )
        
        if not kb_docs:
            return None
            
        # 2. Quick filter for content length
        filtered_docs = [
            doc for doc, score in kb_docs 
            if len(doc.page_content) > 20
        ]
        
        # 3. Deduplicate using a more efficient method
        seen_content = set()
        unique_docs = []
        for doc in filtered_docs:
            content_hash = hash(doc.page_content[:50])  # Reduced from 100 to 50
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
                if len(unique_docs) >= top_k:
                    break
        
        return unique_docs
    except Exception as e:
        print(f"⚠️ Retrieval error: {e}")
        return None

def store_conversation(prompt, response, sources=None, input_tokens=None, output_tokens=None):
    """Store conversation with optimized batching."""
    try:
        metadata = {
            "type": "conversation",
            "timestamp": datetime.datetime.now().isoformat(),
            "sources": "|".join(s for s in (sources or []) if s and isinstance(s, str)) or "No sources",
            "token_data": json.dumps({
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens if input_tokens and output_tokens else None
            })
        }

        # Batch the texts and metadata
        texts = [prompt, response]
        metadatas = [metadata, metadata]
        ids = [int(uuid.uuid4().int % (2**63)), int(uuid.uuid4().int % (2**63))]
        
        # Single batch operation
        vectorstore.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids
        )
    except Exception as e:
        print(f"❌ Error storing conversation: {e}")

@lru_cache(maxsize=100)
def get_cached_docs(query, timestamp):
    """Cache document retrieval results."""
    return retrieve_relevant_docs(query)

@app.route("/ask", methods=["POST"])
def ask_bot():
    data = request.json or {}
    user_prompt = data.get("prompt", "").strip()
    
    # Use cached results if available
    current_time = int(time.time() / CACHE_TTL)
    relevant_docs = get_cached_docs(user_prompt, current_time)
    
    if not relevant_docs:
        return jsonify({
            "response": "I apologize, but I couldn't find any specific information about that in the 2024 Premier Malta website. Could you please try rephrasing your question or ask about something else related to the website content?"
        })

    # 2. Format context
    for doc in relevant_docs:
        if "section_hierarchy" not in doc.metadata:
            print(f"⚠️ Missing 'section_hierarchy' in doc: {doc.metadata}")
    context = "\n\n".join(format_doc(doc) for doc in relevant_docs)
    print(f"Context length: {len(context)} chars")

    # 3. Prepare messages
    print("\n--- Message Preparation ---")
    messages = [
        {
            "role": "system",
            "content": """You are a helpful, ethical, and respectful customer support assistant for the 2024 Premier Malta website. You can ONLY respond using information from the provided context and past interactions.

            ---

            ### ✅ Response Rules:
            - Use **clean Markdown**: headings, bullets, bold/italics for clarity.
            - Be user-friendly, neutral, and professional.
            - **Always include the source URL** of the verified information in this format:  
              `[Source: Page Name](https://url)`  
              Example: `[Source: 2024 Premier Malta](https://2024premiermalta.com/page)`

            ### 🔗 Links:
            - Use `[Name](https://url)` format — never raw URLs.
            - Include links for:
            - Docs: `[Policy PDF](...)`
            - Pages: `[Account Settings](...)`
            - Downloads: `[Form](...)`
            - If no link exists:  
            > "I don't have a direct link, but here's the info: [...]"

            ---

            ### 🔒 Core Guidelines:
            - **ONLY respond to questions about 2024 Premier Malta website content**
            - ❌ If the question is not about the website content, respond with:
              > "I can only answer questions about the 2024 Premier Malta website. Please ask something related to the website content."
            - ❌ No assumptions, guessing, or general knowledge
            - ✋ Never mention backend, tech, or internal systems
            - 🛑 No troubleshooting unless explicitly mentioned in the info
            - 🧱 Read-only mode — no suggestions for changes or edits
            - ❌ Never use words like: *AI, system, database, code, server, etc.*
            - ❌ Out-of-scope = no response (e.g. login, dev issues)

            If nothing relevant is found:  
            > "I couldn't find any information about that in the 2024 Premier Malta website. Please ask something related to the website content."

            If partial info found:  
            > "Here's what I found: [...] For more, see: [Resource Name](URL)"

            ---

            **End every response with (bolded)**:  
            **Not found this helpful, please click the button below to let us know!**

            ---

            ### 🧷 Ethics:
            - Only validated, user-facing info from the website
            - Strict confidentiality
            - Stay on-topic and professional
            """
        }
    ]

    # Add conversation history to messages
    if conversation_history:
        print("Adding conversation history...")
        for prev_prompt, prev_response in conversation_history:
            messages.extend([
                {"role": "user", "content": f"Previous Question: {prev_prompt}"},
                {"role": "assistant", "content": f"Previous Answer: {prev_response}"}
            ])
        print(f"Added {len(conversation_history)} previous exchanges")

    # Add current context and query with explicit website reference
    messages.append({
        "role": "user",
        "content": f"""Context from 2024 Premier Malta website:
        {context}
        SITEMAP for url:-{SECTION_URL_MAP}

        Current Question about 2024 Premier Malta if question is not related to the website content respond with:
        > "I can only answer questions about the 2024 Premier Malta website. Please ask something related to the website content."
        {user_prompt}
        """
    })

    try:
        # 4. Call LLM
        completion = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=messages,
            temperature=0.1,
        )
        
        response = completion.choices[0].message.content
        
        # Calculate tokens for only user prompt and response
        enc = tiktoken.get_encoding("cl100k_base")
        input_tokens = len(enc.encode(user_prompt))
        output_tokens = len(enc.encode(response))
        print(f"Tokens - Input (prompt only): {input_tokens}, Output: {output_tokens}")
        
        # Update conversation history
        conversation_history.append((user_prompt, response))
        print(f"Updated conversation history (size: {len(conversation_history)})")
        
        # 5. Store conversation
        store_conversation(
            prompt=user_prompt,
            response=response,
            sources=[doc.metadata.get("source_url") for doc in relevant_docs],
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
        
        return jsonify({"response": response})
         
    except Exception as e:
        print(f"⚠️ Critical error: {str(e)}")
        return jsonify({"error": "Service unavailable"}), 503

@app.route('/send-support-email', methods=['POST'])
def send_support_email():
    data = request.get_json() or {}
    try:
        send_email(data.get('name'), data.get('email'), data.get('user_issue'))
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
