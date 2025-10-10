#!/usr/bin/env python3
"""
Simple HTTP server to serve the derivatives dashboard
"""

import http.server
import socketserver
import webbrowser
import os
import sys
from pathlib import Path

def start_dashboard_server(port=8000):
    """Start a simple HTTP server to serve the dashboard"""
    
    # Change to the directory containing the dashboard
    dashboard_dir = Path(__file__).parent
    os.chdir(dashboard_dir)
    
    # Check if dashboard.html exists
    if not Path("dashboard.html").exists():
        print("❌ dashboard.html not found!")
        return
    
    # Create server
    handler = http.server.SimpleHTTPRequestHandler
    
    try:
        with socketserver.TCPServer(("", port), handler) as httpd:
            print(f"🚀 Starting Derivatives Dashboard Server...")
            print(f"📊 Dashboard available at: http://localhost:{port}/dashboard.html")
            print(f"📁 Serving files from: {dashboard_dir}")
            print(f"🛑 Press Ctrl+C to stop the server")
            print("-" * 60)
            
            # Try to open browser automatically
            try:
                webbrowser.open(f"http://localhost:{port}/dashboard.html")
                print("🌐 Dashboard opened in your default browser!")
            except:
                print("⚠️  Could not open browser automatically. Please open manually.")
            
            print("-" * 60)
            
            # Start serving
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except OSError as e:
        if e.errno == 98:  # Address already in use
            print(f"❌ Port {port} is already in use. Try a different port:")
            print(f"   python dashboard_server.py --port {port + 1}")
        else:
            print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start the Derivatives Dashboard Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on (default: 8000)")
    
    args = parser.parse_args()
    
    start_dashboard_server(args.port)
