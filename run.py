import sys
from pathlib import Path

# Tambah project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app import app, logger, initialize_systems
from config import Config

def main():
    # Load config
    app.config.from_object(Config)

    # Inisialisasi semua komponen berat saat startup
    initialize_systems()
    
    print("\n" + "="*60)
    print("FLASK RAG APPLICATION")
    print("="*60)
    print(f"Debug mode: {app.config['DEBUG']}")
    print("="*60)
    print("   URL: http://localhost:5000")
    print("\n" + "="*60 + "\n")
    
    # Run app
    try:
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=app.config['DEBUG'],
            use_reloader=False
        )
    except KeyboardInterrupt:
        print("\n\nApplication stopped by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise


if __name__ == '__main__':
    main()