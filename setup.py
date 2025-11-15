"""
Setup script for Smart Traffic Light System
"""

import os
import subprocess
import sys

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'opencv-python',
        'numpy', 
        'ultralytics',
        'torch',
        'torchvision'
    ]
    
    print("🔍 Checking dependencies...")
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} not found")
            return False
    
    return True

def main():
    print("🚦 Smart Traffic Light System Setup")
    print("=" * 40)
    
    if check_dependencies():
        print("\n✅ All dependencies are installed!")
        print("🎯 You can run the system with:")
        print("   python traffic_light_system.py")
    else:
        print("\n❌ Some dependencies are missing.")
        print("💡 Install them with:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()
