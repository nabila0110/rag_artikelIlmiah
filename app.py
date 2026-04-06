from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import logging
from pathlib import Path
import sys

# Tambah project root ke path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from utils.retrieval import RetrievalSystem
from utils.generation import GenerationSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s-%(name)s-%(levelname)s-%(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)

CORS(app)

retrieval_system = None
generation_system = None
systems_initialized = False

def initialize_systems():
    global retrieval_system, generation_system, systems_initialized

    if systems_initialized:
        return

    logger.info("="*60)
    logger.info("Pre-loading retrieval system...")
    logger.info("="*60)
    retrieval_system = RetrievalSystem(
        chunks_file=app.config['CHUNKS_FILE'],
        faiss_index_file=app.config['FAISS_INDEX_FILE'],
        model_path=app.config['EMBEDDING_MODEL_PATH']
    )
    logger.info("Retrieval system loaded successfully")

    logger.info("Pre-loading generation system...")
    generation_system = GenerationSystem(
        model_name=app.config['LLM_MODEL'],
        temperature=app.config['LLM_TEMPERATURE'],
        max_tokens=app.config['LLM_MAX_TOKENS']
    )
    logger.info("Generation system loaded successfully")

    systems_initialized = True
    logger.info("="*60)
    logger.info("All systems ready!")
    logger.info("="*60)

def ensure_systems_initialized():
    if not systems_initialized:
        raise RuntimeError("Systems not initialized. Jalankan aplikasi melalui run.py agar startup initialization berjalan.")

#ROUTES
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search():
    try:
        ensure_systems_initialized()
        data=request.get_json()

        if not data or 'query' not in data:
            return jsonify({
                'error': 'Query parameter required'
            }), 400
        
        query = data['query'].strip()
        top_k = min(data.get('top_k', app.config['DEFAULT_TOP_K']), app.config['MAX_TOP_K'])
        generate_answer = data.get('generate_answer', False)

        if not query:
            return jsonify({
                'error': 'Query cannot be empty'
            }), 400
        
        logger.info(f"Search request: query='{query}', top_k={top_k}, generate={generate_answer}")

        results = retrieval_system.search(query, top_k=top_k)

        response = {
            'query': query,
            'num_results': len(results),
            'results':results
        }

        #generate answer if requested
        if generate_answer:
            generation_result = generation_system.generate_answer(
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
        ensure_systems_initialized()
        stats = retrieval_system.get_statistics()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error geting stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ready')
def ready():
    try:
        retrieval_ready = retrieval_system is not None
        generation_ready = generation_system is not None
        return jsonify({
            'ready': systems_initialized and retrieval_ready and generation_ready,
            'retrieval_ready': retrieval_ready,
            'generation_ready': generation_ready
        }), 200
    except Exception as e:
        logger.error(f"Error checking readiness: {e}")
        return jsonify({'ready': False, 'error': str(e)}), 500
    
# @app.route('/health') #cek
# def health():
#     try:
#         retrieval = get_retrieval_system()
#         generation = get_generation_system()

#         return jsonify({
#             'status': 'healthy',
#             'retrieval': 'ok',
#             'generation': 'ok'
#         })
#     except Exception as e:
#         return jsonify({
#             'status': 'unhealthy',
#             'error': str(e)
#         }), 500
    
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
    logger.info(f"Debug mode: {app.config['DEBUG']}")

    initialize_systems()

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=app.config['DEBUG'],
        use_reloader=False
    )