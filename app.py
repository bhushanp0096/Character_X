#!/usr/bin/env python3
"""
PersonaSynth - Multi-source Character Synthesizer
Advanced chatbot backend with knowledge graph integration
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from langchain_google_genai import ChatGoogleGenerativeAI
import sqlite3
import json
import re
import os
import requests
import spacy
import networkx as nx
from datetime import datetime, timedelta
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
from io import BytesIO
import threading
import time
from collections import defaultdict, Counter
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY_HERE")  # Replace with actual API key
DATABASE_PATH = "personasynth.db"
UPLOAD_FOLDER = "uploads"
MAX_CONTEXT_LENGTH = 8000
CONFIDENCE_THRESHOLD = 0.7

# Initialize Google Gemini AI with LangChain
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.7,  # More creative for emotional responses
        google_api_key=GOOGLE_API_KEY,
        max_tokens=2048,
        timeout=30
    )
    logger.info("✅ Google Gemini 2.0 Flash initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize Gemini: {e}")
    llm = None

# Load spaCy model for NER
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

# Knowledge Graph using NetworkX
class PersonaKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.entity_embeddings = {}
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.fact_vectors = None
        
    def add_fact(self, subject, predicate, object_val, confidence=1.0, source="unknown"):
        """Add a fact to the knowledge graph with provenance"""
        fact_id = f"{subject}_{predicate}_{object_val}_{len(self.graph.nodes)}"
        
        # Add nodes
        self.graph.add_node(subject, type="entity")
        self.graph.add_node(object_val, type="entity")
        
        # Add edge with metadata
        self.graph.add_edge(subject, object_val, 
                          predicate=predicate,
                          confidence=confidence,
                          source=source,
                          timestamp=datetime.now().isoformat(),
                          fact_id=fact_id)
        
        return fact_id
    
    def get_facts_about(self, entity):
        """Get all facts about a specific entity"""
        facts = []
        
        # Outgoing edges (entity as subject)
        for neighbor in self.graph.successors(entity):
            edge_data = self.graph[entity][neighbor]
            facts.append({
                'subject': entity,
                'predicate': edge_data['predicate'],
                'object': neighbor,
                'confidence': edge_data['confidence'],
                'source': edge_data['source']
            })
        
        # Incoming edges (entity as object)
        for predecessor in self.graph.predecessors(entity):
            edge_data = self.graph[predecessor][entity]
            facts.append({
                'subject': predecessor,
                'predicate': edge_data['predicate'],
                'object': entity,
                'confidence': edge_data['confidence'],
                'source': edge_data['source']
            })
        
        return facts
    
    def find_related_entities(self, entity, max_distance=2):
        """Find entities related to the given entity within max_distance"""
        if entity not in self.graph:
            return []
        
        related = []
        visited = set()
        queue = [(entity, 0)]
        
        while queue:
            current, distance = queue.pop(0)
            if current in visited or distance > max_distance:
                continue
                
            visited.add(current)
            if distance > 0:  # Don't include the entity itself
                related.append(current)
            
            # Add neighbors
            for neighbor in self.graph.neighbors(current):
                if neighbor not in visited:
                    queue.append((neighbor, distance + 1))
        
        return related
    
    def export_persona_json(self):
        """Export the persona as a structured JSON"""
        persona_data = {
            'entities': {},
            'relationships': [],
            'metadata': {
                'created': datetime.now().isoformat(),
                'total_facts': self.graph.number_of_edges(),
                'total_entities': self.graph.number_of_nodes()
            }
        }
        
        # Export entities and their attributes
        for node in self.graph.nodes(data=True):
            entity_name = node[0]
            facts = self.get_facts_about(entity_name)
            persona_data['entities'][entity_name] = {
                'facts': facts,
                'related_count': len(self.find_related_entities(entity_name))
            }
        
        # Export relationships
        for edge in self.graph.edges(data=True):
            persona_data['relationships'].append({
                'subject': edge[0],
                'object': edge[1],
                'predicate': edge[2]['predicate'],
                'confidence': edge[2]['confidence'],
                'source': edge[2]['source']
            })
        
        return persona_data

# Global knowledge graph instance
persona_kg = PersonaKnowledgeGraph()

class PersonaSynth:
    def __init__(self):
        self.init_database()
        self.conversation_history = defaultdict(list)
        self.persona_cache = {}
        
    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Sources table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL,
            content TEXT,
            url TEXT,
            confidence REAL DEFAULT 1.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            persona_name TEXT
        )
        ''')
        
        # Conversations table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            persona_name TEXT NOT NULL,
            user_message TEXT,
            bot_response TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            context_facts TEXT
        )
        ''')
        
        # Facts table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            persona_name TEXT NOT NULL,
            subject TEXT,
            predicate TEXT,
            object TEXT,
            confidence REAL,
            source TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def extract_entities_and_facts(self, text, source="unknown"):
        """Extract entities and relationships from text using spaCy"""
        if not nlp:
            return []
        
        doc = nlp(text)
        facts = []
        
        # Extract named entities
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Simple pattern-based fact extraction
        patterns = [
            (r"(.+) is (.+)", "is"),
            (r"(.+) has (.+)", "has"),
            (r"(.+) lives in (.+)", "lives_in"),
            (r"(.+) works as (.+)", "works_as"),
            (r"(.+) likes (.+)", "likes"),
            (r"(.+) was born in (.+)", "born_in"),
            (r"(.+) married (.+)", "married_to"),
            (r"(.+) knows (.+)", "knows"),
        ]
        
        for pattern, predicate in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                subject = match.group(1).strip()
                obj = match.group(2).strip()
                
                # Clean up extracted text
                subject = re.sub(r'^(the|a|an)\s+', '', subject, flags=re.IGNORECASE)
                obj = re.sub(r'^(the|a|an)\s+', '', obj, flags=re.IGNORECASE)
                
                if len(subject) > 2 and len(obj) > 2:
                    facts.append({
                        'subject': subject,
                        'predicate': predicate,
                        'object': obj,
                        'confidence': 0.8,
                        'source': source
                    })
        
        return facts
    
    def add_source(self, persona_name, source_type, content, url=None, confidence=1.0):
        """Add a new source and extract facts from it"""
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Store source
        cursor.execute('''
        INSERT INTO sources (type, content, url, confidence, persona_name)
        VALUES (?, ?, ?, ?, ?)
        ''', (source_type, content, url, confidence, persona_name))
        
        source_id = cursor.lastrowid
        
        # Extract facts from content
        facts = self.extract_entities_and_facts(content, f"{source_type}_{source_id}")
        
        # Store facts in database and knowledge graph
        for fact in facts:
            cursor.execute('''
            INSERT INTO facts (persona_name, subject, predicate, object, confidence, source)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (persona_name, fact['subject'], fact['predicate'], 
                  fact['object'], fact['confidence'], fact['source']))
            
            # Add to knowledge graph
            persona_kg.add_fact(
                fact['subject'], fact['predicate'], fact['object'],
                fact['confidence'], fact['source']
            )
        
        conn.commit()
        conn.close()
        
        return len(facts)
    
    def get_persona_context(self, persona_name, user_message):
        """Get relevant context for generating response"""
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Get all facts for persona
        cursor.execute('''
        SELECT subject, predicate, object, confidence, source
        FROM facts WHERE persona_name = ?
        ORDER BY confidence DESC
        ''', (persona_name,))
        
        facts = cursor.fetchall()
        
        # Get recent conversation history
        cursor.execute('''
        SELECT user_message, bot_response
        FROM conversations 
        WHERE persona_name = ?
        ORDER BY timestamp DESC
        LIMIT 5
        ''', (persona_name,))
        
        history = cursor.fetchall()
        conn.close()
        
        # Build context
        context = {
            'facts': [
                {
                    'subject': f[0], 'predicate': f[1], 'object': f[2],
                    'confidence': f[3], 'source': f[4]
                }
                for f in facts[:20]  # Limit context size
            ],
            'history': [
                {'user': h[0], 'bot': h[1]}
                for h in history
            ]
        }
        
        return context
    
    def generate_response(self, persona_name, user_message):
        """Generate response using Gemini with persona context"""
        if not llm:
            return "Sorry, the AI model is not available. Please check your API key configuration."
            
        context = self.get_persona_context(persona_name, user_message)
        
        # Build prompt with persona context
        prompt = f"""You are roleplaying as {persona_name}. Use the following information about this character to respond authentically:

FACTS ABOUT {persona_name.upper()}:
"""
        
        # Add facts to prompt
        for fact in context['facts'][:15]:  # Limit to prevent token overflow
            confidence_indicator = "✓" if fact['confidence'] > 0.8 else "~"
            prompt += f"{confidence_indicator} {fact['subject']} {fact['predicate'].replace('_', ' ')} {fact['object']}\n"
        
        prompt += f"""
RECENT CONVERSATION HISTORY:
"""
        
        # Add conversation history
        for conv in context['history'][:3]:
            prompt += f"User: {conv['user']}\n{persona_name}: {conv['bot']}\n\n"
        
        prompt += f"""
INSTRUCTIONS:
- Stay in character as {persona_name}
- Use the facts above to inform your personality and knowledge
- Be consistent with established facts
- If you don't know something, respond as the character would
- Keep responses conversational and engaging
- Don't mention that you're using provided facts
- Show personality traits and speaking patterns consistent with the character
- Draw from your knowledge base but don't sound robotic

Current message to respond to: "{user_message}"

Response as {persona_name}:"""

        try:
            # Use LangChain ChatGoogleGenerativeAI
            response = llm.invoke(prompt)
            bot_response = response.content if hasattr(response, 'content') else str(response)
            
            # Store conversation
            self.store_conversation(persona_name, user_message, bot_response, context)
            
            # Extract new facts from the conversation
            self.update_persona_from_conversation(persona_name, user_message, bot_response)
            
            return bot_response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I'm having trouble responding right now as {persona_name}. Could you try again?"
    
    def store_conversation(self, persona_name, user_message, bot_response, context):
        """Store conversation in database"""
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO conversations (persona_name, user_message, bot_response, context_facts)
        VALUES (?, ?, ?, ?)
        ''', (persona_name, user_message, bot_response, json.dumps(context['facts'][:10])))
        
        conn.commit()
        conn.close()
    
    def update_persona_from_conversation(self, persona_name, user_message, bot_response):
        """Extract new facts from conversation and update persona"""
        conversation_text = f"User: {user_message}\n{persona_name}: {bot_response}"
        
        # Extract facts from the conversation
        facts = self.extract_entities_and_facts(conversation_text, "conversation")
        
        # Filter facts that seem to be about the persona
        persona_facts = []
        for fact in facts:
            if (persona_name.lower() in fact['subject'].lower() or 
                persona_name.lower() in fact['object'].lower()):
                persona_facts.append(fact)
        
        # Add persona facts to database with lower confidence
        if persona_facts:
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            
            for fact in persona_facts:
                cursor.execute('''
                INSERT INTO facts (persona_name, subject, predicate, object, confidence, source)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (persona_name, fact['subject'], fact['predicate'], 
                      fact['object'], 0.6, fact['source']))  # Lower confidence for conversation-derived facts
                
                # Add to knowledge graph
                persona_kg.add_fact(
                    fact['subject'], fact['predicate'], fact['object'],
                    0.6, fact['source']
                )
            
            conn.commit()
            conn.close()
    
    def process_pdf(self, persona_name, pdf_data):
        """Extract text from PDF and add as source"""
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_data))
            text = ""
            
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            facts_added = self.add_source(persona_name, "pdf", text)
            return text[:500], facts_added  # Return preview and fact count
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return None, 0
    
    def scrape_web_content(self, url):
        """Simple web scraping (basic implementation)"""
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Very basic text extraction (would need BeautifulSoup for better parsing)
            text = response.text
            # Remove HTML tags roughly
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text[:5000]  # Limit text length
            
        except Exception as e:
            logger.error(f"Error scraping web content: {e}")
            return None
    
    def get_persona_stats(self, persona_name):
        """Get statistics about a persona"""
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Count facts
        cursor.execute('SELECT COUNT(*) FROM facts WHERE persona_name = ?', (persona_name,))
        fact_count = cursor.fetchone()[0]
        
        # Count sources
        cursor.execute('SELECT COUNT(*) FROM sources WHERE persona_name = ?', (persona_name,))
        source_count = cursor.fetchone()[0]
        
        # Count conversations
        cursor.execute('SELECT COUNT(*) FROM conversations WHERE persona_name = ?', (persona_name,))
        conversation_count = cursor.fetchone()[0]
        
        # Get source types
        cursor.execute('SELECT type, COUNT(*) FROM sources WHERE persona_name = ? GROUP BY type', (persona_name,))
        source_types = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            'fact_count': fact_count,
            'source_count': source_count,
            'conversation_count': conversation_count,
            'source_types': source_types
        }

# Initialize PersonaSynth
persona_synth = PersonaSynth()

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# API Routes
@app.route('/')
def serve_frontend():
    """Serve the frontend HTML file"""
    return send_from_directory('.', 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.json
        persona_name = data.get('persona_name', 'Default')
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Generate response
        response = persona_synth.generate_response(persona_name, user_message)
        
        return jsonify({
            'response': response,
            'persona_name': persona_name,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/add_source', methods=['POST'])
def add_source():
    """Add a new source for persona building"""
    try:
        data = request.json
        persona_name = data.get('persona_name', 'Default')
        source_type = data.get('type')  # 'text', 'url', 'bio'
        content = data.get('content', '')
        url = data.get('url')
        
        if source_type == 'url' and url:
            # Scrape web content
            scraped_content = persona_synth.scrape_web_content(url)
            if scraped_content:
                facts_added = persona_synth.add_source(persona_name, 'web', scraped_content, url)
                return jsonify({
                    'success': True,
                    'facts_added': facts_added,
                    'content_preview': scraped_content[:200]
                })
            else:
                return jsonify({'error': 'Failed to scrape URL'}), 400
        
        elif content:
            facts_added = persona_synth.add_source(persona_name, source_type, content)
            return jsonify({
                'success': True,
                'facts_added': facts_added
            })
        
        else:
            return jsonify({'error': 'Content or URL is required'}), 400
            
    except Exception as e:
        logger.error(f"Error adding source: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    """Upload and process PDF file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        persona_name = request.form.get('persona_name', 'Default')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and file.filename.lower().endswith('.pdf'):
            pdf_data = file.read()
            content_preview, facts_added = persona_synth.process_pdf(persona_name, pdf_data)
            
            if content_preview:
                return jsonify({
                    'success': True,
                    'facts_added': facts_added,
                    'content_preview': content_preview
                })
            else:
                return jsonify({'error': 'Failed to process PDF'}), 400
        
        else:
            return jsonify({'error': 'Only PDF files are allowed'}), 400
            
    except Exception as e:
        logger.error(f"Error uploading PDF: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/persona_stats/<persona_name>')
def persona_stats(persona_name):
    """Get statistics about a persona"""
    try:
        stats = persona_synth.get_persona_stats(persona_name)
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting persona stats: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/export_persona/<persona_name>')
def export_persona(persona_name):
    """Export persona as JSON"""
    try:
        # Get persona data from database
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM facts WHERE persona_name = ?', (persona_name,))
        facts = cursor.fetchall()
        
        cursor.execute('SELECT * FROM sources WHERE persona_name = ?', (persona_name,))
        sources = cursor.fetchall()
        
        conn.close()
        
        # Export knowledge graph
        kg_export = persona_kg.export_persona_json()
        
        persona_export = {
            'persona_name': persona_name,
            'exported_at': datetime.now().isoformat(),
            'database_facts': [
                {
                    'id': f[0], 'subject': f[2], 'predicate': f[3],
                    'object': f[4], 'confidence': f[5], 'source': f[6]
                }
                for f in facts
            ],
            'sources': [
                {
                    'id': s[0], 'type': s[1], 'content': s[2][:200],
                    'url': s[3], 'confidence': s[4]
                }
                for s in sources
            ],
            'knowledge_graph': kg_export,
            'stats': persona_synth.get_persona_stats(persona_name)
        }
        
        return jsonify(persona_export)
        
    except Exception as e:
        logger.error(f"Error exporting persona: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/list_personas')
def list_personas():
    """List all available personas"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT DISTINCT persona_name FROM facts')
        personas = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        
        return jsonify({'personas': personas})
        
    except Exception as e:
        logger.error(f"Error listing personas: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'kg_nodes': persona_kg.graph.number_of_nodes(),
        'kg_edges': persona_kg.graph.number_of_edges()
    })

if __name__ == '__main__':
    print("🤖 PersonaSynth Server Starting...")
    print("📊 Multi-source Character Synthesizer")
    print("🌐 Frontend: http://localhost:5000")
    print("⚡ API ready for character synthesis!")
    
    # Start the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)