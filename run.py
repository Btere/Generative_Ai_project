#!/usr/bin/env python3
"""
Main entry point for the Stran Connect Virtual Try-On System.

This script starts the Flask web application with proper configuration
and error handling. It includes comprehensive typing annotations.

Usage:
    python run.py [--host HOST] [--port PORT] [--debug]

Author: Stran Connect Team
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, NoReturn
import logging

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from web.app import VirtualTryOnApp
except ImportError as e:
    print(f"Error importing application modules: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Stran Connect Virtual Try-On System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                          # Run on default host and port
  python run.py --host 0.0.0.0 --port 8080  # Run on all interfaces, port 8080
  python run.py --debug                  # Run in debug mode
        """
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Host address to bind to (default: 127.0.0.1)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port number to listen on (default: 5000)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> bool:
    """
    Validate command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        True if arguments are valid, False otherwise
    """
    # Validate port range
    if not (1 <= args.port <= 65535):
        logger.error(f"Invalid port number: {args.port}. Must be between 1 and 65535.")
        return False
    
    # Validate host format (basic check)
    if not args.host:
        logger.error("Host cannot be empty.")
        return False
    
    return True


def main() -> Optional[NoReturn]:
    """
    Main entry point for the application.
    
    Returns:
        Does not return (starts Flask server) or None on error
    """
    try:
        # Parse and validate arguments
        args = parse_arguments()
        
        if not validate_arguments(args):
            sys.exit(1)
        
        # Display startup information
        logger.info("=" * 60)
        logger.info("Stran Connect Virtual Try-On System")
        logger.info("=" * 60)
        logger.info(f"Starting server on {args.host}:{args.port}")
        logger.info(f"Debug mode: {'enabled' if args.debug else 'disabled'}")
        logger.info(f"Access the application at: http://{args.host}:{args.port}")
        logger.info("=" * 60)
        
        # Create and run the application
        app = VirtualTryOnApp()
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug
        )
        
    except KeyboardInterrupt:
        logger.info("\nApplication stopped by user.")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        if args.debug:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()