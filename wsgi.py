"""
WSGI Entry Point for Production Deployment

This file is used by WSGI servers (Gunicorn, uWSGI) to run the application.

Usage with Gunicorn:
    gunicorn wsgi:app --bind 0.0.0.0:5000 --workers 4
"""

from app.api import app

if __name__ == "__main__":
    app.run()
