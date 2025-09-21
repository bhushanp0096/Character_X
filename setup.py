#!/usr/bin/env python3
"""
PersonaSynth Setup Script
Automated setup for the Multi-source Character Synthesizer
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import requests
from typing import Optional

class PersonaSynthSetup:
    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.conda_env_name = "personasynth"
        self.python_version = "3.10"
        
    def print_banner(self):
        """Print setup banner"""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                    🤖 PersonaSynth Setup                     ║
║              Multi-source Character Synthesizer              ║
║                                                              ║
║  This script will set up your PersonaSynth environment      ║
║  with all required dependencies and configurations.         ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def check_conda(self) -> bool:
        """Check if conda is installed"""
        try:
            result = subprocess.run(['conda', '--version'], 
                                  capture_output=True, text=True, check=True)
            print(f"✅ Found conda: {result.stdout.strip()}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Conda not found. Please install Anaconda or Miniconda first.")
            print("   Download from: https://docs.conda.io/en/latest/miniconda.html")
            return False
    
    def create_conda_environment(self) -> bool:
        """Create conda environment from environment.yml"""
        try:
            env_file = self.project_dir / "environment.yml"
            if not env_file.exists():
                print("❌ environment.yml not found!")
                return False
            
            print(f"🔧 Creating conda environment '{self.conda_env_name}'...")
            
            # Remove existing environment if it exists
            try:
                subprocess.run(['conda', 'env', 'remove', '-n', self.conda_env_name, '-y'], 
                             capture_output=True, check=True)
                print(f"🗑️  Removed existing environment '{self.conda_env_name}'")
            except subprocess.CalledProcessError:
                pass  # Environment doesn't exist, continue
            
            # Create new environment
            cmd = ['conda', 'env', 'create', '-f', str(env_file)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Conda environment '{self.conda_env_name}' created successfully!")
                return True
            else:
                print(f"❌ Failed to create conda environment:")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ Error creating conda environment: {e}")
            return False
    
    def install_spacy_model(self) -> bool:
        """Install spaCy English model"""
        try:
            print("📥 Installing spaCy English model...")
            
            # Activate conda environment and install spacy model
            cmd = f"conda activate {self.conda_env_name} && python -m spacy download en_core_web_sm"
            
            if os.name == 'nt':  # Windows
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            else:  # Unix/Linux/macOS
                result = subprocess.run(['bash', '-c', cmd], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ spaCy English model installed successfully!")
                return True
            else:
                print(f"❌ Failed to install spaCy model:")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ Error installing spaCy model: {e}")
            return False
    
    def setup_environment_file(self) -> bool:
        """Set up .env file from template"""
        try:
            env_template = self.project_dir / ".env.template"
            env_file = self.project_dir / ".env"
            
            if env_file.exists():
                print("⚠️  .env file already exists. Skipping creation.")
                return True
            
            if env_template.exists():
                shutil.copy(env_template, env_file)
                print("✅ Created .env file from template")
                print("⚠️  IMPORTANT: Edit .env file and add your Google API key!")
                return True
            else:
                # Create basic .env file
                env_content = """# PersonaSynth Configuration
GOOGLE_API_KEY=your_google_api_key_here
DATABASE_PATH=personasynth.db
UPLOAD_FOLDER=uploads
GEMINI_MODEL=gemini-2.0-flash
TEMPERATURE=0.7
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=True
"""
                with open(env_file, 'w') as f:
                    f.write(env_content)
                print("✅ Created basic .env file")
                print("⚠️  IMPORTANT: Edit .env file and add your Google API key!")
                return True
                
        except Exception as e:
            print(f"❌ Error setting up environment file: {e}")
            return False
    
    def create_directories(self) -> bool:
        """Create necessary directories"""
        try:
            directories = ['uploads', 'logs', 'exports', 'data']
            
            for directory in directories:
                dir_path = self.project_dir / directory
                dir_path.mkdir(exist_ok=True)
                print(f"📁 Created directory: {directory}")
            
            return True
        except Exception as e:
            print(f"❌ Error creating directories: {e}")
            return False
    
    def test_installation(self) -> bool:
        """Test the installation"""
        try:
            print("🧪 Testing installation...")
            
            # Test imports
            test_script = """
import sys
try:
    from flask import Flask
    from langchain_google_genai import ChatGoogleGenerativeAI
    import spacy
    import networkx as nx
    import numpy as np
    import sklearn
    import sqlite3
    print("✅ All core dependencies imported successfully")
    
    # Test spaCy model
    nlp = spacy.load("en_core_web_sm")
    print("✅ spaCy English model loaded successfully")
    
    print("🎉 Installation test passed!")
    sys.exit(0)
    
except Exception as e:
    print(f"❌ Installation test failed: {e}")
    sys.exit(1)
"""
            
            # Write test script to temporary file
            test_file = self.project_dir / "test_installation.py"
            with open(test_file, 'w') as f:
                f.write(test_script)
            
            # Run test in conda environment
            cmd = f"conda activate {self.conda_env_name} && python {test_file}"
            
            if os.name == 'nt':  # Windows
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            else:  # Unix/Linux/macOS
                result = subprocess.run(['bash', '-c', cmd], capture_output=True, text=True)
            
            # Clean up test file
            test_file.unlink(missing_ok=True)
            
            if result.returncode == 0:
                print(result.stdout)
                return True
            else:
                print("❌ Installation test failed:")
                print(result.stderr)
                return False
                
        except Exception as e:
            print(f"❌ Error testing installation: {e}")
            return False
    
    def get_api_key_instructions(self):
        """Display API key setup instructions"""
        instructions = """
╔══════════════════════════════════════════════════════════════╗
║                    🔑 API Key Setup Required                 ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  To use PersonaSynth, you need a Google AI API key:         ║
║                                                              ║
║  1. Go to: https://makersuite.google.com/app/apikey         ║
║  2. Create a new API key                                     ║
║  3. Copy the API key                                         ║
║  4. Edit the .env file in this directory                     ║
║  5. Replace 'your_google_api_key_here' with your actual key ║
║                                                              ║
║  Example:                                                    ║
║  GOOGLE_API_KEY=ABCD    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(instructions)
    
    def print_success_message(self):
        """Print success message with next steps"""
        message = f"""
╔══════════════════════════════════════════════════════════════╗
║                    🎉 Setup Complete!                       ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  PersonaSynth has been set up successfully!                 ║
║                                                              ║
║  Next steps:                                                 ║
║  1. Edit .env file and add your Google API key              ║
║  2. Activate the conda environment:                         ║
║     conda activate {self.conda_env_name:<40} ║
║  3. Run the application:                                     ║
║     python app.py                                            ║
║  4. Open your browser to: http://localhost:5000             ║
║                                                              ║
║  Happy character synthesizing! 🤖                           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(message)
    
    def run_setup(self):
        """Run the complete setup process"""
        self.print_banner()
        
        # Check prerequisites
        if not self.check_conda():
            sys.exit(1)
        
        # Setup steps
        steps = [
            ("Creating conda environment", self.create_conda_environment),
            ("Installing spaCy model", self.install_spacy_model),
            ("Setting up environment file", self.setup_environment_file),
            ("Creating directories", self.create_directories),
            ("Testing installation", self.test_installation),
        ]
        
        for step_name, step_func in steps:
            print(f"\n📋 {step_name}...")
            if not step_func():
                print(f"❌ Setup failed at step: {step_name}")
                sys.exit(1)
        
        # Display final instructions
        self.get_api_key_instructions()
        self.print_success_message()

def main():
    """Main setup function"""
    try:
        setup = PersonaSynthSetup()
        setup.run_setup()
    except KeyboardInterrupt:
        print("\n❌ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()