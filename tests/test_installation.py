#!/usr/bin/env python3
"""Test script to verify all installations are working"""

def test_imports():
    """Test all required library imports"""
    try:
        import cv2
        print("✓ OpenCV imported successfully")
        
        import numpy as np
        print("✓ NumPy imported successfully")
        
        import torch
        print(f"✓ PyTorch imported successfully - Version: {torch.__version__}")
        
        import gymnasium as gym
        print(f"✓ Gymnasium imported successfully - Version: {gym.__version__}")
        
        import stable_baselines3
        print(f"✓ Stable-Baselines3 imported successfully")
        
        import matplotlib.pyplot as plt
        print("✓ Matplotlib imported successfully")
        
        import pandas as pd
        print("✓ Pandas imported successfully")
        
        import sklearn
        print("✓ Scikit-learn imported successfully")
        
        print("\n" + "="*50)
        print("ALL IMPORTS SUCCESSFUL!")
        print("="*50)
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_opencv():
    """Test OpenCV functionality"""
    try:
        import cv2
        import numpy as np
        
        # Create test image
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(test_img, (50, 50), 20, (255, 255, 255), -1)
        
        # Test basic operations
        gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        print("✓ OpenCV basic operations working")
        return True
        
    except Exception as e:
        print(f"✗ OpenCV test failed: {e}")
        return False

def test_pytorch():
    """Test PyTorch functionality"""
    try:
        import torch
        import torch.nn as nn
        
        # Test tensor operations
        x = torch.randn(2, 3)
        y = torch.randn(3, 4)
        z = torch.mm(x, y)
        
        # Test neural network
        model = nn.Linear(10, 1)
        input_tensor = torch.randn(1, 10)
        output = model(input_tensor)
        
        print(f"✓ PyTorch operations working - Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
        return True
        
    except Exception as e:
        print(f"✗ PyTorch test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Installation...")
    print("="*50)
    
    success = True
    success &= test_imports()
    success &= test_opencv()  
    success &= test_pytorch()
    
    if success:
        print("\n🎉 ALL TESTS PASSED! Ready for Day 2 implementation.")
    else:
        print("\n❌ Some tests failed. Please check installation.")
