"""
Debug LoCoMo - Chạy 1-2 sample để tìm và fix lỗi
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("DEBUG LOCOMO - 1 Sample Test")
print("=" * 60)

# Check API key
api_key = os.getenv("GROQ_API_KEY")
if api_key:
    print(f"✅ GROQ_API_KEY found: {api_key[:20]}...")
else:
    print("❌ GROQ_API_KEY NOT FOUND!")
    sys.exit(1)

# Load 1 sample
print("\n--- Loading Dataset ---")
data = json.load(open('dataset/locomo10.json', encoding='utf-8'))
sample = data[0]

# Parse sample
conv = sample.get('conversation', {})
session_keys = [k for k in conv.keys() if k.startswith("session_") and not k.endswith("_date_time")]
session_keys = sorted(session_keys, key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else 0)

# Get first 30 messages (more context)
messages = []
for session_key in session_keys[:3]:  # First 3 sessions
    session = conv.get(session_key, [])
    if isinstance(session, list):
        for msg in session[:15]:  # First 15 messages per session
            if isinstance(msg, dict) and "text" in msg:
                speaker = msg.get("speaker", "Unknown")
                text = msg.get("text", "")
                messages.append(f"{speaker}: {text}")

# Get first 3 QA pairs
qa_pairs = []
for qa in sample.get("qa", [])[:3]:
    qa_pairs.append({
        "question": qa.get("question", ""),
        "answer": str(qa.get("answer", ""))
    })

print(f"Sample ID: {sample.get('sample_id')}")
print(f"Messages: {len(messages)}")
print(f"QA pairs: {len(qa_pairs)}")

# ==================================================================
# TEST V1
# ==================================================================
print("\n" + "=" * 60)
print("TEST V1")
print("=" * 60)

from fcm import FCMAgent, FCMConfig

config = FCMConfig(verbose=True, crystallize_threshold=3)
agent = FCMAgent(config=config, user_id="debug_v1")
agent.clear_all_memories()

print("\n--- Ingesting Messages ---")
for i, msg in enumerate(messages):
    print(f"[{i+1}/{len(messages)}] {msg[:60]}...")
    agent.chat(msg)

print("\n--- Crystallizing ---")
agent.crystallize(force=True)

print("\n--- Stats ---")
stats = agent.get_stats()
print(f"Liquid: {stats.get('conversation_length', 0)}")
print(f"Crystal: {stats.get('crystal_count', 0)}")
print(f"Solid: {stats.get('solid_count', 0)}")


def check_context_relevance(question: str, retrieved: str) -> tuple[bool, float]:
    """Check if retrieved context is RELEVANT to the question"""
    if not retrieved:
        return False, 0.0
    
    q_lower = question.lower()
    r_lower = retrieved.lower()
    
    import re
    names_in_q = re.findall(r'\b[A-Z][a-z]+\b', question)
    name_matches = sum(1 for name in names_in_q if name.lower() in r_lower)
    
    topic_words = []
    if "lgbtq" in q_lower: topic_words.append("lgbtq")
    if "support group" in q_lower: topic_words.append("support")
    if "paint" in q_lower: topic_words.extend(["paint", "sunrise", "art"])
    if "education" in q_lower: topic_words.extend(["education", "study", "psychology", "counseling"])
    if "identity" in q_lower: topic_words.extend(["transgender", "identity"])
    
    topic_matches = sum(1 for word in topic_words if word in r_lower)
    
    if names_in_q:
        name_score = name_matches / len(names_in_q)
    else:
        name_score = 0.5
    
    if topic_words:
        topic_score = min(topic_matches / max(1, len(topic_words) / 2), 1.0)
    else:
        topic_score = 0.3
    
    relevance = (name_score * 0.4 + topic_score * 0.6)
    is_relevant = relevance >= 0.3
    return is_relevant, relevance


print("\n--- Search Tests (V1) ---")
v1_results = []
for qa in qa_pairs:
    question = qa["question"]
    expected = qa["answer"]
    
    print(f"\nQ: {question}")
    print(f"Expected: {expected}")
    
    result = agent.search(question, strategy="enhanced")
    
    if result and result.get("combined"):
        top = result["combined"][0]
        retrieved = top.get("memory", "")
        source = top.get("metadata", {}).get("fcm_type", "?")
        print(f"Retrieved ({source}): {retrieved[:100]}")
        
        # Check exact match
        exact_match = expected.lower() in retrieved.lower()
        
        # Check context relevance
        is_relevant, relevance_score = check_context_relevance(question, retrieved)
        
        if exact_match:
            print("✅ EXACT MATCH!")
        elif is_relevant:
            print(f"🔶 RELEVANT CONTEXT (score={relevance_score:.2f})")
        else:
            print(f"❌ NOT RELEVANT (score={relevance_score:.2f})")
        
        v1_results.append({
            "question": question,
            "exact_match": exact_match,
            "relevant": is_relevant,
            "relevance_score": relevance_score
        })
    else:
        print("❌ NO RESULTS")
        v1_results.append({
            "question": question,
            "exact_match": False,
            "relevant": False,
            "relevance_score": 0.0
        })

# ==================================================================
# TEST V2
# ==================================================================
print("\n" + "=" * 60)
print("TEST V2")
print("=" * 60)

# Close V1 agent first to release Qdrant lock
del agent
import gc
gc.collect()
import time
time.sleep(1)

from fcm_v2 import FCMAgentV2, FCMConfigV2

config_v2 = FCMConfigV2(
    verbose=True, 
    crystallize_threshold=3,
    enable_active_forgetting=False,
)
agent_v2 = FCMAgentV2(config=config_v2, user_id="debug_v2")
agent_v2.clear_all_memories()

print("\n--- Ingesting Messages ---")
for i, msg in enumerate(messages):
    print(f"[{i+1}/{len(messages)}] {msg[:60]}...")
    agent_v2.chat(msg)

print("\n--- Crystallizing ---")
agent_v2.crystallize(force=True)

print("\n--- Stats ---")
stats_v2 = agent_v2.get_stats()
print(f"Liquid: {stats_v2.get('liquid_count', 0)}")
print(f"Crystal: {stats_v2.get('crystal_count', 0)}")
print(f"Solid: {stats_v2.get('solid_count', 0)}")

print("\n--- Search Tests (V2) ---")
v2_results = []
for qa in qa_pairs:
    question = qa["question"]
    expected = qa["answer"]
    
    print(f"\nQ: {question}")
    print(f"Expected: {expected}")
    
    result = agent_v2.search(question, strategy="enhanced")
    
    if result and result.combined_results:
        top = result.combined_results[0]
        retrieved = top.get("memory", "")
        source = top.get("source_layer", "?")
        score = top.get("enhanced_score", 0)
        print(f"Retrieved ({source}, score={score:.3f}): {retrieved[:100]}")
        
        # Check exact match
        exact_match = expected.lower() in retrieved.lower()
        
        # Check context relevance
        is_relevant, relevance_score = check_context_relevance(question, retrieved)
        
        if exact_match:
            print("✅ EXACT MATCH!")
        elif is_relevant:
            print(f"🔶 RELEVANT CONTEXT (score={relevance_score:.2f})")
        else:
            print(f"❌ NOT RELEVANT (score={relevance_score:.2f})")
        
        v2_results.append({
            "question": question,
            "exact_match": exact_match,
            "relevant": is_relevant,
            "relevance_score": relevance_score
        })
    else:
        print("❌ NO RESULTS")
        v2_results.append({
            "question": question,
            "exact_match": False,
            "relevant": False,
            "relevance_score": 0.0
        })

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

v1_exact = sum(1 for r in v1_results if r["exact_match"])
v1_relevant = sum(1 for r in v1_results if r["relevant"])
v2_exact = sum(1 for r in v2_results if r["exact_match"])
v2_relevant = sum(1 for r in v2_results if r["relevant"])

print(f"\nV1: Exact={v1_exact}/{len(v1_results)}, Relevant={v1_relevant}/{len(v1_results)}")
print(f"V2: Exact={v2_exact}/{len(v2_results)}, Relevant={v2_relevant}/{len(v2_results)}")

print("\n" + "=" * 60)
print("DEBUG COMPLETE")
print("=" * 60)
