"""
Test Script for Option 1 (LLM Reader) and Option 2 (Computed Temporal Facts)
=============================================================================

This script tests:
1. Option 1: LLM Reader - Uses LLM to infer answers from retrieved context
2. Option 2: Computed Dates - Crystallization stores computed dates (yesterday → actual date)

Usage: python test_options.py
"""

import json
import os
import sys
from datetime import datetime, timedelta

# Add FCM root folder to path (parent of demo/)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_option1_llm_reader():
    """Test Option 1: LLM Reader for temporal inference"""
    print("\n" + "="*70)
    print("📖 OPTION 1: Testing LLM Reader for Temporal Inference")
    print("="*70)
    
    from fcm.eval.locomo import LLMReader, check_answer_with_llm_reader, normalize_text
    
    # Initialize LLM Reader
    print("\n🔧 Initializing LLM Reader...")
    reader = LLMReader(verbose=True)
    
    # Test cases
    test_cases = [
        {
            "question": "When did Caroline go to the LGBTQ support group?",
            "expected": "7 May 2023",
            "context": "Caroline: I went to a LGBTQ support group yesterday. It was really supportive.",
            "session_date": "1:56 pm on 8 May, 2023"
        },
        {
            "question": "When did Melanie paint a sunrise?",
            "expected": "2022",
            "context": "Melanie: I painted that beautiful lake sunrise last year at my art class.",
            "session_date": "1:56 pm on 8 May, 2023"
        },
        {
            "question": "When is Melanie planning on going camping?",
            "expected": "June 2023",
            "context": "Melanie: I'm planning to go camping next month with my friends.",
            "session_date": "4:04 pm on 25 May, 2023"
        }
    ]
    
    results = []
    for i, tc in enumerate(test_cases, 1):
        print(f"\n{'─'*50}")
        print(f"Test Case {i}:")
        print(f"  Q: {tc['question']}")
        print(f"  Expected: {tc['expected']}")
        print(f"  Session Date: {tc['session_date']}")
        
        is_correct, confidence, llm_answer = check_answer_with_llm_reader(
            retrieved=tc['context'],
            expected=tc['expected'],
            question=tc['question'],
            session_date=tc['session_date'],
            llm_reader=reader
        )
        
        status = "✅ PASS" if is_correct else "❌ FAIL"
        print(f"  LLM Answer: {llm_answer}")
        print(f"  Result: {status} (confidence={confidence:.2f})")
        
        results.append(is_correct)
    
    # Summary
    correct = sum(results)
    total = len(results)
    print(f"\n{'='*50}")
    print(f"Option 1 Results: {correct}/{total} ({correct/total*100:.1f}%)")
    print("="*50)
    
    return correct, total


def test_option2_computed_dates():
    """Test Option 2: Crystallization with computed temporal dates"""
    print("\n" + "="*70)
    print("🔮 OPTION 2: Testing Crystallization with Computed Dates")
    print("="*70)
    
    from fcm import FCMAgentV2, FCMConfigV2
    
    # Create agent
    print("\n🔧 Initializing FCM V2 Agent...")
    config = FCMConfigV2(
        verbose=True,
        crystallize_threshold=2,  # Low threshold for testing
    )
    agent = FCMAgentV2(config=config, user_id="test_option2")
    agent.clear_all_memories()
    
    # Test messages with temporal references
    test_messages = [
        "Caroline: I went to a LGBTQ support group yesterday. It was really supportive.",
        "Caroline: I'm a transgender woman and finding my community.",
        "Melanie: I painted that beautiful lake sunrise last year at my art class.",
    ]
    
    # Session date for the conversation
    session_date = "1:56 pm on 8 May, 2023"
    
    print(f"\n📅 Session Date: {session_date}")
    print(f"\n📝 Ingesting {len(test_messages)} messages...")
    
    for msg in test_messages:
        print(f"  → {msg[:60]}...")
        agent.chat(msg)
    
    # Force crystallization with session_date (Option 2)
    print("\n🔮 Crystallizing with session_date...")
    
    # We need to call crystallize with session_date parameter
    try:
        result = agent.crystallize(force=True, session_date=session_date)
        print(f"   Crystallization result: {result}")
    except Exception as e:
        print(f"   ⚠ Error during crystallization: {e}")
        import traceback
        traceback.print_exc()
    
    # Search for temporal facts
    print("\n🔍 Searching for temporal facts...")
    
    questions = [
        "When did Caroline go to LGBTQ support group?",
        "When did Melanie paint sunrise?",
    ]
    
    results = []
    for q in questions:
        search_result = agent.search(q, strategy="enhanced")
        
        print(f"\n  Q: {q}")
        
        if search_result and search_result.combined_results:
            top = search_result.combined_results[0]
            memory = top.get("memory", "")
            metadata = top.get("metadata", {})
            
            print(f"  Retrieved: {memory[:80]}...")
            print(f"  Metadata: computed_date={metadata.get('computed_date', 'N/A')}")
            
            # Check if computed_date is in the content or metadata
            has_computed_date = (
                "7 may 2023" in memory.lower() or
                "may 2023" in memory.lower() or
                metadata.get('computed_date') is not None
            )
            results.append(has_computed_date)
        else:
            print("  ❌ No results found")
            results.append(False)
    
    # Summary
    correct = sum(results)
    total = len(results)
    print(f"\n{'='*50}")
    print(f"Option 2 Results: {correct}/{total} facts with computed dates")
    print("="*50)
    
    return correct, total


def main():
    """Run all tests"""
    print("\n" + "🧪"*35)
    print("  FCM OPTIONS TESTING: Option 1 + Option 2  ")
    print("🧪"*35)
    
    # Test Option 1
    try:
        opt1_correct, opt1_total = test_option1_llm_reader()
    except Exception as e:
        print(f"\n❌ Option 1 failed with error: {e}")
        opt1_correct, opt1_total = 0, 3
    
    # Test Option 2
    try:
        opt2_correct, opt2_total = test_option2_computed_dates()
    except Exception as e:
        print(f"\n❌ Option 2 failed with error: {e}")
        import traceback
        traceback.print_exc()
        opt2_correct, opt2_total = 0, 2
    
    # Final Summary
    print("\n" + "="*70)
    print("📊 FINAL SUMMARY")
    print("="*70)
    print(f"\n  Option 1 (LLM Reader):      {opt1_correct}/{opt1_total} ({opt1_correct/opt1_total*100:.1f}%)")
    print(f"  Option 2 (Computed Dates):  {opt2_correct}/{opt2_total} ({opt2_correct/opt2_total*100:.1f}%)")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
