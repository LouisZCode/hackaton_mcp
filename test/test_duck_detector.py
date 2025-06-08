#!/usr/bin/env python3
"""
Test script for the Duck Detector MCP Server
============================================

This script demonstrates the duck detection functionality.
"""

# Import the MCP server (in a real scenario, this would be done via MCP protocol)
import sys
import os

# Add the current directory to path so we can import our MCP server
sys.path.append(os.path.dirname(__file__))

from duck_detector_mcp import detect_duck_in_text, duck_response_prompt, duck_context_prompt

def test_duck_detection():
    """Test the duck detection functionality"""
    
    print("🦆 Duck Detector MCP Server Test")
    print("=" * 40)
    
    # Test cases
    test_texts = [
        "Hello there, I saw a duck today!",
        "The weather is nice today.",
        "I love DUCK hunting in the fall",
        "Look at that rubber duck in the bath",
        "No birds here, just cats and dogs"
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nTest {i}: '{text}'")
        result = detect_duck_in_text(text)
        print(f"Duck detected: {result['duck_detected']}")
        print(f"Message: {result['message']}")
        
        if result['duck_detected']:
            print("🎯 Duck found! Here's what the LLM should do:")
            prompt = duck_context_prompt(text)
            print(f"Prompt: {prompt}")
            print("🦆 Expected response: cuack!")
    
    print("\n" + "=" * 40)
    print("📋 Available MCP Prompts:")
    print("1. duck_response_prompt() - General duck response instruction")
    print("2. duck_context_prompt(text) - Context-aware duck response")
    
    print("\n🔧 MCP Tools:")
    print("1. detect_duck_in_text(text) - Detect duck mentions in text")

if __name__ == "__main__":
    test_duck_detection()
