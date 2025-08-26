#!/usr/bin/env python3
"""
Test script for enhanced speaker name extraction
"""

from utils import extract_names_from_transcript, extract_with_llm, intelligent_split

# Test transcript with clear speaker introductions
test_transcript = """
Hi everyone, I'm John from the marketing team. Let me start by discussing our quarterly results. 
Our campaign performance has been excellent this quarter.

Thanks John. This is Sarah from operations. I wanted to follow up on the supply chain improvements we discussed. 
We've implemented new tracking systems that have reduced delivery times by 15%.

Great points Sarah. My name is Michael and I represent the development team. 
I'd like to share some updates on our latest product features. We've successfully launched the new user interface.

That sounds excellent Michael. Hi, I'm Lisa from customer support. 
Our satisfaction scores have improved significantly since the UI update.
"""

print("Testing enhanced name extraction...")
print("=" * 50)

# Test 1: Pattern-based extraction
print("1. Pattern-based name extraction:")
names = extract_names_from_transcript(test_transcript)
for speaker, content in names.items():
    print(f"   {speaker}: {content[:100]}...")
print()

# Test 2: Intelligent split with name detection
print("2. Intelligent split with name detection:")
speakers = intelligent_split(test_transcript)
for speaker, content in speakers.items():
    print(f"   {speaker}: {content[:100]}...")
print()

# Test 3: Short test for problematic cases
short_transcript = "We discussed the project timeline and budget allocations for next quarter."
print("3. Testing with generic content (should fallback gracefully):")
result = extract_names_from_transcript(short_transcript)
if not result:
    result = intelligent_split(short_transcript)
for speaker, content in result.items():
    print(f"   {speaker}: {content}")

print("\nTest completed! 🎉")
