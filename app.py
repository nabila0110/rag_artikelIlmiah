from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import logging
from pathlib import Path
import sys
# tambah project root ke path
sys.path.insert(0, str(Path(__file__).parent))

from config import config
from utils.retrieval import RetrievalSystem
from utils.generation import GenerationSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s-%(name)s-%(levelname)s-%(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

env = 'development'
app.config.from_object(config[env])

CORS(app)

retrieval_system = None
generation_system = None

def get_retrieval_system():
    global retrieval_system
    if retrieval_system is None:
        logger.info("Initializing retrieval system...")
        retrieval_system = RetrievalSystem(
            chunks_file=app.config['CHUNKS_FILE'],
            faiss_index_file=app.config['FAISS_INDEX_FILE'],
            model_path=app.config['EMBEDDING_MODEL_PATH']
        )
    return retrieval_system

def get_generation_system():
    global generation_system
    if generation_system is None:
        logger.info("Initializing generation system...")
        generation_system = GenerationSystem(
            model_name=app.config['LLM_MODEL'],
            temperature=app.config['LLM_TEMPERATURE'],
            max_tokens=app.config['LLM_MAX_TOKENS']
        )
    return generation_system

#ROUTES
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search():
    try:
        data=request.get_json()

        if not data or 'query' not in data:
            return jsonify({
                'error': 'Query parameter required'
            }), 400
        
        query = data['query'].strip()
        top_k = min(data.get('top_k', app.config['DEFAULT_TOP_K'])), app.config['MAX_TOP_K']
        generate_answer = data.get('generate_answer', False)

        if not query:
            return jsonify({
                'error': 'Query cannot be empty'
            }), 400
        
        logger.info(f"Search request: query='{query}', top_k={top_k}, generate={generate_answer}")

        retrieval = get_retrieval_system()
        results = retrieval.search(query, top_k=top_k)

        response = {
            'query': query,
            'num_results': len(results),
            'results':results
        }

        #generate answer if requested
        if generate_answer:
            generation = get_generation_system()
            generation_result = generation.generate_answer(
                query=query,
                retrieved_chunks=results,
                max_context_chunks=app.config['MAX_CONTEXT_CHUNKS']
            )
            #cited reference
            response['answer'] = generation_result['answer']
            response['context_chunks_used']=generation_result['context_chunks_used']

            #additional reference
            response['cited_references']=results[:app.config['MAX_CONTEXT_CHUNKS']]
            response['additional_references']=results[app.config['MAX_CONTEXT_CHUNKS']:]
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in search: {e}")
        return jsonify({
            'error': str(e)
        }), 500
    
@app.route('/api/stats')
def stats():
    try:
        retrieval = get_retrieval_system()
        stats = retrieval.get_statistics()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error geting stats: {e}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/health') #cek
def health():
    try:
        retrieval = get_retrieval_system()
        generation = get_generatioon_system()

        return jsonify({
            'status': 'healthy',
            'retrieval': 'ok',
            'generation': 'ok'
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500
    
#EROR HANDLERS
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not Found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    return jsonify({})

#MAIN
if __name__=='__main__':
    logger.info("Starting Flask application...")
    logger.info(f"Environment: {env}")
    logger.info(f"Debug mode: {app.config['DEBUG']}")

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=app.config['DEBUG']
    )