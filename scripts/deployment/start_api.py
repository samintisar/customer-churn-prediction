#!/usr/bin/env python
"""
Production API Startup Script

Starts the Flask API with Gunicorn for production deployment.

Usage:
    python start_api.py
    
    # Custom settings
    python start_api.py --host 0.0.0.0 --port 8000 --workers 4
"""

import argparse
import os
import sys
from pathlib import Path

def check_requirements():
    """Check if all required files and dependencies exist"""
    print("🔍 Checking requirements...")
    
    # Change to project root directory (two levels up from scripts/deployment)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    os.chdir(project_root)
    
    errors = []
    warnings = []
    
    # Check models
    model_path = Path('models/logistic_regression.pkl')
    fe_path = Path('models/feature_engineer.pkl')
    
    if not model_path.exists():
        errors.append(f"❌ Model not found: {model_path}")
    else:
        print(f"✅ Model found: {model_path}")
    
    if not fe_path.exists():
        errors.append(f"❌ Feature engineer not found: {fe_path}")
    else:
        print(f"✅ Feature engineer found: {fe_path}")
    
    # Check API file
    api_path = Path('app/api.py')
    if not api_path.exists():
        errors.append(f"❌ API file not found: {api_path}")
    else:
        print(f"✅ API file found: {api_path}")
    
    # Check dependencies
    try:
        import flask
        print(f"✅ Flask installed")
    except ImportError:
        errors.append("❌ Flask not installed")
    
    # Check for production server (platform-specific)
    import platform
    if platform.system() == 'Windows':
        try:
            import waitress
            print(f"✅ Waitress installed (Windows production server)")
        except ImportError:
            warnings.append("⚠️  Waitress not installed (required for Windows production)")
    else:
        try:
            import gunicorn
            print(f"✅ Gunicorn installed (Unix production server)")
        except ImportError:
            warnings.append("⚠️  Gunicorn not installed (required for Unix production)")
    
    # Check src modules
    try:
        sys.path.insert(0, str(Path.cwd()))
        from src.feature_engineering import FeatureEngineer
        from src.retention_strategy import classify_risk_tier
        print("✅ Source modules imported successfully")
    except ImportError as e:
        errors.append(f"❌ Error importing source modules: {e}")
    
    if errors:
        print("\n❌ ERRORS FOUND:")
        for error in errors:
            print(f"   {error}")
        return False
    
    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"   {warning}")
    
    print("\n✅ All requirements met!\n")
    return True


def start_development_server(host='127.0.0.1', port=5000):
    """Start Flask development server"""
    print("="*70)
    print("🚀 STARTING DEVELOPMENT SERVER")
    print("="*70)
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"URL:  http://{host}:{port}")
    print("="*70)
    print("\n⚠️  WARNING: Development server - not for production use!")
    print("   For production, use: python start_api.py --production\n")
    
    from app.api import app
    app.run(host=host, port=port, debug=False)


def start_production_server(host='0.0.0.0', port=5000, workers=4):
    """Start production server (Waitress on Windows, Gunicorn on Unix)"""
    import platform
    
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    if platform.system() == 'Windows':
        # Use Waitress for Windows
        print("="*70)
        print("🚀 STARTING PRODUCTION SERVER (WAITRESS)")
        print("="*70)
        print(f"Host:    {host}")
        print(f"Port:    {port}")
        print(f"Threads: {workers}")
        print(f"URL:     http://{host if host != '0.0.0.0' else 'localhost'}:{port}")
        print("="*70)
        print("\n✅ Production-ready server with Waitress (Windows)\n")
        
        try:
            from waitress import serve
            from app.api import app
            
            print(f"Starting Waitress server...")
            print(f"Press Ctrl+C to stop\n")
            
            serve(app, host=host, port=port, threads=workers)
        except KeyboardInterrupt:
            print("\n\n🛑 Server stopped by user")
        except ImportError:
            print("\n❌ Waitress not found!")
            print("   Install with: conda run -n ml-conda pip install waitress")
            sys.exit(1)
    else:
        # Use Gunicorn for Unix/Linux/Mac
        print("="*70)
        print("🚀 STARTING PRODUCTION SERVER (GUNICORN)")
        print("="*70)
        print(f"Host:    {host}")
        print(f"Port:    {port}")
        print(f"Workers: {workers}")
        print(f"URL:     http://{host}:{port}")
        print("="*70)
        print("\n✅ Production-ready server with Gunicorn\n")
        
        # Build gunicorn command
        cmd = [
            'gunicorn',
            'app.api:app',
            f'--bind={host}:{port}',
            f'--workers={workers}',
            '--timeout=120',
            '--access-logfile=logs/access.log',
            '--error-logfile=logs/error.log',
            '--log-level=info',
            '--worker-class=sync',
        ]
        
        print(f"Command: {' '.join(cmd)}\n")
        
        import subprocess
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\n\n🛑 Server stopped by user")
        except FileNotFoundError:
            print("\n❌ Gunicorn not found!")
            print("   Install with: pip install gunicorn")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Start Customer Churn Prediction API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Development server (default)
  python start_api.py
  
  # Production server with Gunicorn
  python start_api.py --production
  
  # Custom host and port
  python start_api.py --host 0.0.0.0 --port 8000
  
  # Production with more workers
  python start_api.py --production --workers 8
        """
    )
    
    parser.add_argument(
        '--production', '-p',
        action='store_true',
        help='Run with Gunicorn (production mode)'
    )
    parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Host to bind (default: 127.0.0.1)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port to bind (default: 5000)'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=4,
        help='Number of worker processes (production only, default: 4)'
    )
    parser.add_argument(
        '--skip-checks',
        action='store_true',
        help='Skip requirement checks'
    )
    
    args = parser.parse_args()
    
    # Check requirements
    if not args.skip_checks:
        if not check_requirements():
            print("\n❌ Please fix the errors above before starting the server.")
            sys.exit(1)
    
    # Start server
    if args.production:
        # Use 0.0.0.0 for production by default
        if args.host == '127.0.0.1':
            args.host = '0.0.0.0'
        start_production_server(args.host, args.port, args.workers)
    else:
        start_development_server(args.host, args.port)


if __name__ == '__main__':
    main()
