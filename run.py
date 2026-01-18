import sys
import os
from pathlib import Path

#Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app import app, logger
from config import config

def main():
    #Get environment from command line
    env = sys.argv[1] if len(sys.argv) > 1 else 'development'
    
    if env not in config:
        print(f"Invalid environment: {env}")
        print(f"Available: {list(config.keys())}")
        sys.exit(1)
    
    # Load config
    app.config.from_object(config[env])
    
    print("\n" + "="*60)
    print("FLASK RAG APPLICATION")
    print("="*60)
    print(f"Environment: {env}")
    print(f"Debug mode: {app.config['DEBUG']}")
    print("="*60)
    print("   URL: http://localhost:5000")
    print("\n" + "="*60 + "\n")
    
    # Run app
    try:
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=app.config['DEBUG']
        )
    except KeyboardInterrupt:
        print("\n\nApplication stopped by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise


if __name__ == '__main__':
    main()