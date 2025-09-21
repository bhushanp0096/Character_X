#!/usr/bin/env python3
"""
PersonaSynth Run Script
Quick start script with environment checks and setup validation
"""

import os
import sys
import subprocess
from pathlib import Path
import webbrowser
import time
from dotenv import load_dotenv

class PersonaSynthRunner:
    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.env_file = self.project_dir / ".env"
        
    def print_banner(self):
        """Print startup banner"""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                    🤖 PersonaSynth                          ║
║              Multi-source Character Synthesizer              ║
║                                                              ║
║  Starting your intelligent persona chat system...           ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def check_environment(self):
        """Check if environment is properly configured"""
        issues = []
        
        # Check if .env file exists
        if not self.env_file.exists():
            issues.append("❌ .env file not found. Run setup.py first or copy .env.template to .env")
        
        # Load and check environment variables
        load_dotenv(self.env_file)
        
        # Check Google API key
        api_key = os.getenv("GOOGLE_API_KEY", "")
        if not api_key or api_key == "your_google_api_key_here":
            issues.append("❌ Google API key not configured. Edit .env file and add your API key")
        
        # Check if required files exist
        required_files = ["app.py", "index.html"]
        for file in required_files:
            if not (self.project_dir / file).exists():
                issues.append(f"❌ Required file missing: {file}")
        
        return issues
    
    def check_dependencies(self):
        """Check if required Python packages are installed"""
        required_packages = [
            "flask",
            "langchain_google_genai", 
            "spacy",
            "networkx",
            "numpy",
            "scikit-learn",
            "PyPDF2",
            "requests",
            "beautifulsoup4"
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)
        
        return missing_packages
    
    def check_spacy_model(self):
        """Check if spaCy English model is installed"""
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            return True
        except OSError:
            return False
    
    def print_setup_instructions(self, issues, missing_packages):
        """Print setup instructions for found issues"""
        print("\n" + "="*60)
        print("🔧 SETUP REQUIRED")
        print("="*60)
        
        if issues:
            print("\n📋 Configuration Issues:")
            for issue in issues:
                print(f"  {issue}")
        
        if missing_packages:
            print(f"\n📦 Missing Python packages:")
            for package in missing_packages:
                print(f"  - {package}")
            print(f"\n💡 Install missing packages with:")
            print(f"   conda activate personasynth")
            print(f"   pip install {' '.join(missing_packages)}")
        
        if not self.check_spacy_model():
            print(f"\n🔤 spaCy English model missing:")
            print(f"   python -m spacy download en_core_web_sm")
        
        print(f"\n🚀 Quick setup:")
        print(f"   python setup.py")
        print("\n" + "="*60)
    
    def start_application(self):
        """Start the Flask application"""
        try:
            print("🚀 Starting PersonaSynth server...")
            print("📍 Server will be available at: http://localhost:5000")
            print("⚡ Press Ctrl+C to stop the server\n")
            
            # Import and run the Flask app
            from app import app
            
            # Get configuration from environment
            host = os.getenv("FLASK_HOST", "0.0.0.0")
            port = int(os.getenv("FLASK_PORT", "5000"))
            debug = os.getenv("FLASK_DEBUG", "True").lower() == "true"
            
            # Try to open browser automatically
            def open_browser():
                time.sleep(1.5)  # Wait for server to start
                try:
                    webbrowser.open(f"http://localhost:{port}")
                except Exception:
                    pass  # Browser opening is optional
            
            import threading
            browser_thread = threading.Thread(target=open_browser)
            browser_thread.daemon = True
            browser_thread.start()
            
            # Start Flask application
            app.run(host=host, port=port, debug=debug)
            
        except ImportError as e:
            print(f"❌ Failed to import Flask app: {e}")
            print("🔧 Make sure all dependencies are installed")
            return False
        except Exception as e:
            print(f"❌ Failed to start application: {e}")
            return False
    
    def run(self):
        """Main run function"""
        self.print_banner()
        
        # Check environment configuration
        issues = self.check_environment()
        missing_packages = self.check_dependencies()
        
        # If there are setup issues, show instructions and exit
        if issues or missing_packages or not self.check_spacy_model():
            self.print_setup_instructions(issues, missing_packages)
            return False
        
        # Environment is good, start the application
        print("✅ Environment check passed!")
        print("✅ All dependencies found!")
        print("✅ Configuration looks good!")
        print()
        
        return self.start_application()

def main():
    """Main function"""
    try:
        runner = PersonaSynthRunner()
        success = runner.run()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n👋 PersonaSynth stopped by user")
        print("Thanks for using PersonaSynth! 🤖")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()