"""
Script để kiểm tra cấu trúc LOCOMO dataset
Chạy: python inspect_locomo.py
"""

import json
import os
import sys

# Add FCM root folder to path (parent of demo/)
FCM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, FCM_ROOT)

def inspect_locomo():
    dataset_path = os.path.join(FCM_ROOT, "dataset", "locomo10.json")
    
    # Check file exists
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at {dataset_path}")
        print("   Please download from: https://drive.google.com/drive/folders/1L-cTjTm0ohMsitsHg4dijSPJtqNflwX-")
        return
    
    print("=" * 70)
    print("📊 LOCOMO DATASET STRUCTURE INSPECTION")
    print("=" * 70)
    
    # Load dataset
    with open(dataset_path, "r") as f:
        data = json.load(f)
    
    print(f"\n📈 Basic Info:")
    print(f"   Total conversations: {len(data)}")
    
    if not data:
        print("❌ Dataset is empty!")
        return
    
    # Inspect first conversation
    print(f"\n🔍 First Conversation Structure:")
    first_item = data[0]
    
    print(f"\n   Top-level keys: {list(first_item.keys())}")
    
    # Conversation details
    conversation = first_item.get("conversation", {})
    print(f"\n   Conversation keys: {list(conversation.keys())}")
    print(f"   Speaker A: {conversation.get('speaker_a')}")
    print(f"   Speaker B: {conversation.get('speaker_b')}")
    
    # Count turns
    turns = [k for k in conversation.keys() if k.startswith("turn_")]
    print(f"   Total turns: {len(turns)}")
    
    if turns:
        first_turn_key = turns[0]
        first_turn = conversation[first_turn_key]
        print(f"\n   Sample turn ({first_turn_key}):")
        print(f"     Keys: {list(first_turn.keys())}")
        print(f"     Speaker: {first_turn.get('speaker')}")
        print(f"     Message: {first_turn.get('message', '')[:100]}...")
    
    # Questions
    questions = first_item.get("questions", [])
    print(f"\n   Total questions: {len(questions)}")
    
    if questions:
        first_q = questions[0]
        print(f"\n   Sample question:")
        print(f"     Keys: {list(first_q.keys())}")
        print(f"     Question: {first_q.get('question', '')[:80]}...")
        print(f"     Category: {first_q.get('category')}")
        print(f"     Answer: {first_q.get('answer', '')[:80]}...")
    
    # Statistics
    print(f"\n📊 Dataset Statistics:")
    
    turn_counts = []
    question_counts = []
    categories = {}
    
    for item in data:
        turns = [k for k in item["conversation"].keys() if k.startswith("turn_")]
        turn_counts.append(len(turns))
        
        questions = item.get("questions", [])
        question_counts.append(len(questions))
        
        for q in questions:
            cat = q.get("category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
    
    print(f"   Avg turns per conversation: {sum(turn_counts) / len(turn_counts):.1f}")
    print(f"   Avg questions per conversation: {sum(question_counts) / len(question_counts):.1f}")
    print(f"   Total questions: {sum(question_counts)}")
    
    print(f"\n   Question categories:")
    for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        print(f"     - {cat}: {count}")
    
    print("\n" + "=" * 70)
    print("✅ Inspection Complete!")
    print("=" * 70)

if __name__ == "__main__":
    inspect_locomo()
