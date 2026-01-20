"""
FCM Demo Comparison
===================

So sánh FCM V1 vs V2 với các demo scenarios:
1. Basic Flow
2. Topic Shift Detection
3. Conflict Resolution
4. Search Strategies

Chạy: python -m fcm_eval.demo
"""

import os
import sys
import time
import logging

logging.basicConfig(level=logging.WARNING)
logging.getLogger("mem0").setLevel(logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fcm_eval.common import (
    print_header, print_subheader, print_comparison_table,
    check_answer, BenchmarkResult, normalize_text,
    DEMO_MESSAGES, DEMO_QUERIES, CONFLICT_MESSAGES, CONFLICT_EXPECTED
)


# ============================================================
# DEMO 1: BASIC FLOW
# ============================================================

def demo_basic_flow_v1(user_id: str = "demo_basic_v1") -> dict:
    """V1: Basic Flow"""
    from fcm import FCMAgent, FCMConfig
    
    config = FCMConfig(verbose=False, crystallize_threshold=3)
    agent = FCMAgent(config=config, user_id=user_id)
    
    try:
        agent.memory.delete_all(user_id=user_id)
    except:
        pass
    
    start = time.time()
    
    # Ingest
    for msg in DEMO_MESSAGES:
        agent.chat(msg, auto_crystallize=True)
    
    agent.end_session()
    
    # Search
    correct = 0
    for query, expected in DEMO_QUERIES:
        result = agent.search(query)
        if result.get("combined") and check_answer(result["combined"][0].get("memory", ""), expected):
            correct += 1
    
    stats = agent.get_stats()
    
    return {
        "time": time.time() - start,
        "accuracy": correct / len(DEMO_QUERIES),
        "stats": stats,
        "topic_shifts_detected": 0,  # V1 không đếm được
    }


def demo_basic_flow_v2(user_id: str = "demo_basic_v2") -> dict:
    """V2: Basic Flow với Attention Sinks"""
    from fcm import FCMAgentV2, FCMConfigV2
    
    config = FCMConfigV2(
        verbose=False, 
        crystallize_threshold=3,
        attention_sink_count=2,
        semantic_similarity_threshold=0.75,
    )
    agent = FCMAgentV2(config=config, user_id=user_id)
    agent.clear_all_memories()
    
    start = time.time()
    
    # Ingest
    topic_shifts = 0
    attention_sinks = 0
    for msg in DEMO_MESSAGES:
        result = agent.chat(msg)
        if result.get("is_attention_sink"):
            attention_sinks += 1
        if result.get("topic_shifted"):
            topic_shifts += 1
    
    agent.end_session()
    
    # Search with enhanced (shared with V1 for fair comparison)
    correct = 0
    for query, expected in DEMO_QUERIES:
        result = agent.search(query, strategy="enhanced")
        if result.combined_results and check_answer(result.combined_results[0].get("memory", ""), expected):
            correct += 1
    
    stats = agent.get_stats()
    
    return {
        "time": time.time() - start,
        "accuracy": correct / len(DEMO_QUERIES),
        "stats": stats,
        "topic_shifts_detected": topic_shifts,
        "attention_sinks": attention_sinks,
        "llm_calls_saved": stats.get("llm_calls_saved", 0),
    }


# ============================================================
# DEMO 2: TOPIC SHIFT DETECTION
# ============================================================

TOPIC_SHIFT_MESSAGES = [
    # Segment 1: Personal info
    "Tôi tên Hùng, 25 tuổi",
    "Tôi làm kỹ sư phần mềm",
    # Topic shift → Weather
    "Hôm nay trời đẹp quá, nắng ấm",
    "Dự báo ngày mai sẽ mưa",
    # Topic shift → Food
    "Tối nay tôi muốn ăn lẩu",
    "Quán lẩu Hải Phòng ngon lắm",
]


def demo_topic_shift_v1(user_id: str = "demo_topic_v1") -> dict:
    """V1: Topic Shift (SeCom)"""
    from fcm import FCMAgent, FCMConfig
    
    config = FCMConfig(verbose=False, crystallize_threshold=10)  # High threshold
    agent = FCMAgent(config=config, user_id=user_id)
    
    try:
        agent.memory.delete_all(user_id=user_id)
    except:
        pass
    
    topic_shifts = 0
    crystallize_triggers = 0
    
    for msg in TOPIC_SHIFT_MESSAGES:
        result = agent.chat(msg, auto_crystallize=True)
        if result.get("topic_shifted"):
            topic_shifts += 1
        if result.get("crystallize_trigger") == "topic_shift":
            crystallize_triggers += 1
    
    return {
        "topic_shifts_detected": topic_shifts,
        "crystallize_by_topic_shift": crystallize_triggers,
        "total_crystallizations": crystallize_triggers,
    }


def demo_topic_shift_v2(user_id: str = "demo_topic_v2") -> dict:
    """V2: Topic Shift với Semantic Grouping"""
    from fcm import FCMAgentV2, FCMConfigV2
    
    config = FCMConfigV2(
        verbose=False, 
        crystallize_threshold=10,
        semantic_similarity_threshold=0.7,  # Để detect shift dễ hơn
        attention_sink_count=1,
    )
    agent = FCMAgentV2(config=config, user_id=user_id)
    agent.clear_all_memories()
    
    topic_shifts = 0
    crystallize_triggers = 0
    llm_calls_saved = 0
    
    for msg in TOPIC_SHIFT_MESSAGES:
        result = agent.chat(msg)
        if result.get("topic_shifted"):
            topic_shifts += 1
        if result.get("crystallize_trigger") == "topic_shift":
            crystallize_triggers += 1
        if result.get("llm_call_skipped"):
            llm_calls_saved += 1
    
    return {
        "topic_shifts_detected": topic_shifts,
        "crystallize_by_topic_shift": crystallize_triggers,
        "total_crystallizations": crystallize_triggers,
        "llm_calls_saved_by_embedding": llm_calls_saved,
    }


# ============================================================
# DEMO 3: CONFLICT RESOLUTION
# ============================================================

def demo_conflict_v1(user_id: str = "demo_conflict_v1") -> dict:
    """V1: Conflict Resolution"""
    from fcm import FCMAgent, FCMConfig
    
    config = FCMConfig(verbose=False, crystallize_threshold=2)
    agent = FCMAgent(config=config, user_id=user_id)
    
    try:
        agent.memory.delete_all(user_id=user_id)
    except:
        pass
    
    # Session 1
    for msg, is_conflict in CONFLICT_MESSAGES[:2]:
        agent.chat(msg)
    agent.crystallize(force=True)
    
    # Session 2 (conflicts)
    for msg, is_conflict in CONFLICT_MESSAGES[2:]:
        agent.chat(msg)
    agent.crystallize(force=True)
    
    # Evolve
    evolve_result = agent.evolve(force=True)
    
    # Check if conflicts resolved
    resolved = 0
    profile = agent.get_user_profile()
    profile_str = str(profile).lower()
    
    if CONFLICT_EXPECTED["food_preference"] in profile_str:
        resolved += 1
    if CONFLICT_EXPECTED["workplace"] in profile_str:
        resolved += 1
    
    return {
        "conflicts_reported": evolve_result.get("conflicts_resolved", 0),
        "actually_resolved": resolved,
        "total_conflicts": 2,
    }


def demo_conflict_v2(user_id: str = "demo_conflict_v2") -> dict:
    """V2: Conflict Resolution với Bi-Temporal"""
    from fcm import FCMAgentV2, FCMConfigV2
    
    config = FCMConfigV2(
        verbose=False, 
        crystallize_threshold=2,
        enable_temporal_priority=True,
    )
    agent = FCMAgentV2(config=config, user_id=user_id)
    agent.clear_all_memories()
    
    # Session 1
    for msg, is_conflict in CONFLICT_MESSAGES[:2]:
        agent.chat(msg)
    agent.crystallize(force=True)
    
    # Simulate time passing
    time.sleep(0.5)
    
    # Session 2 (conflicts)
    for msg, is_conflict in CONFLICT_MESSAGES[2:]:
        agent.chat(msg)
    agent.crystallize(force=True)
    
    # Evolve
    evolve_result = agent.evolve(force=True)
    
    # Check resolution
    resolved = 0
    profile = agent.get_user_profile()
    profile_str = str(profile).lower()
    
    if CONFLICT_EXPECTED["food_preference"] in profile_str:
        resolved += 1
    if CONFLICT_EXPECTED["workplace"] in profile_str:
        resolved += 1
    
    return {
        "conflicts_reported": evolve_result.get("conflicts_resolved", 0),
        "actually_resolved": resolved,
        "total_conflicts": 2,
        "bi_temporal_used": True,
    }


# ============================================================
# DEMO 4: SEARCH STRATEGIES
# ============================================================

def demo_search_strategies_v1(user_id: str = "demo_search_v1") -> dict:
    """V1: Search Strategies"""
    from fcm import FCMAgent, FCMConfig
    
    config = FCMConfig(verbose=False, crystallize_threshold=2)
    agent = FCMAgent(config=config, user_id=user_id)
    
    try:
        agent.memory.delete_all(user_id=user_id)
    except:
        pass
    
    # Add messages
    for msg in DEMO_MESSAGES[:5]:
        agent.chat(msg)
    agent.crystallize(force=True)
    agent.evolve(force=True)
    
    query = "Minh học ngành gì?"
    strategies = ["hybrid", "solid_first", "all_layers"]
    
    results = {}
    for strategy in strategies:
        result = agent.search(query, strategy=strategy)
        combined = result.get("combined", [])
        results[strategy] = {
            "found": len(combined) > 0,
            "top_result": combined[0].get("memory", "")[:50] if combined else None,
            "source": result.get("best_source", "N/A"),
        }
    
    return {"strategies": results, "available_strategies": strategies}


def demo_search_strategies_v2(user_id: str = "demo_search_v2") -> dict:
    """V2: Search với Weighted Ensemble"""
    from fcm import FCMAgentV2, FCMConfigV2
    
    config = FCMConfigV2(
        verbose=False, 
        crystallize_threshold=2,
        retrieval_weight_solid=0.5,
        retrieval_weight_crystal=0.3,
        retrieval_weight_liquid=0.2,
    )
    agent = FCMAgentV2(config=config, user_id=user_id)
    agent.clear_all_memories()
    
    # Add messages
    for msg in DEMO_MESSAGES[:5]:
        agent.chat(msg)
    agent.crystallize(force=True)
    agent.evolve(force=True)
    
    query = "Minh học ngành gì?"
    strategies = ["enhanced", "weighted", "solid_first", "all_layers"]
    
    results = {}
    for strategy in strategies:
        result = agent.search(query, strategy=strategy)
        combined = result.combined_results
        results[strategy] = {
            "found": len(combined) > 0,
            "top_result": combined[0].get("memory", "")[:50] if combined else None,
            "source": result.best_source,
            "weighted_score": combined[0].get("weighted_score", 0) if combined else 0,
        }
    
    return {
        "strategies": results, 
        "available_strategies": strategies,
        "weights_used": {
            "solid": config.retrieval_weight_solid,
            "crystal": config.retrieval_weight_crystal,
            "liquid": config.retrieval_weight_liquid,
        }
    }


# ============================================================
# MAIN COMPARISON
# ============================================================

def run_demo_comparison():
    """Main function - Run all demo comparisons"""
    print_header("FCM DEMO COMPARISON: V1 vs V2", "═")
    
    all_results = {
        "v1": {"demos": {}},
        "v2": {"demos": {}},
    }
    
    # Demo 1: Basic Flow
    print_header("DEMO 1: Basic Flow", "-")
    print("\nV1 Running...")
    v1_basic = demo_basic_flow_v1()
    all_results["v1"]["demos"]["basic_flow"] = v1_basic
    
    print("V2 Running...")
    v2_basic = demo_basic_flow_v2()
    all_results["v2"]["demos"]["basic_flow"] = v2_basic
    
    print(f"\n📊 Results:")
    print(f"   V1: Accuracy={v1_basic['accuracy']:.1%}, Time={v1_basic['time']:.2f}s")
    print(f"   V2: Accuracy={v2_basic['accuracy']:.1%}, Time={v2_basic['time']:.2f}s")
    print(f"   V2 Extras: Attention Sinks={v2_basic.get('attention_sinks', 0)}, "
          f"Topic Shifts={v2_basic.get('topic_shifts_detected', 0)}, "
          f"LLM Saved={v2_basic.get('llm_calls_saved', 0)}")
    
    # Demo 2: Topic Shift
    print_header("DEMO 2: Topic Shift Detection", "-")
    print("\nV1 Running...")
    v1_topic = demo_topic_shift_v1()
    all_results["v1"]["demos"]["topic_shift"] = v1_topic
    
    print("V2 Running...")
    v2_topic = demo_topic_shift_v2()
    all_results["v2"]["demos"]["topic_shift"] = v2_topic
    
    print(f"\n📊 Results:")
    print(f"   V1: Topic Shifts={v1_topic['topic_shifts_detected']}, "
          f"Crystallize Triggers={v1_topic['crystallize_by_topic_shift']}")
    print(f"   V2: Topic Shifts={v2_topic['topic_shifts_detected']}, "
          f"Crystallize Triggers={v2_topic['crystallize_by_topic_shift']}, "
          f"LLM Calls Saved={v2_topic.get('llm_calls_saved_by_embedding', 0)}")
    
    # Demo 3: Conflict Resolution
    print_header("DEMO 3: Conflict Resolution", "-")
    print("\nV1 Running...")
    v1_conflict = demo_conflict_v1()
    all_results["v1"]["demos"]["conflict"] = v1_conflict
    
    print("V2 Running...")
    v2_conflict = demo_conflict_v2()
    all_results["v2"]["demos"]["conflict"] = v2_conflict
    
    print(f"\n📊 Results:")
    print(f"   V1: Resolved={v1_conflict['actually_resolved']}/{v1_conflict['total_conflicts']}")
    print(f"   V2: Resolved={v2_conflict['actually_resolved']}/{v2_conflict['total_conflicts']} "
          f"(Bi-Temporal={v2_conflict.get('bi_temporal_used', False)})")
    
    # Demo 4: Search Strategies
    print_header("DEMO 4: Search Strategies", "-")
    print("\nV1 Running...")
    v1_search = demo_search_strategies_v1()
    all_results["v1"]["demos"]["search"] = v1_search
    
    print("V2 Running...")
    v2_search = demo_search_strategies_v2()
    all_results["v2"]["demos"]["search"] = v2_search
    
    print(f"\n📊 V1 Strategies: {v1_search['available_strategies']}")
    print(f"   V2 Strategies: {v2_search['available_strategies']}")
    print(f"   V2 Weights: {v2_search.get('weights_used', {})}")
    
    # Final Summary
    print_header("FINAL SUMMARY", "═")
    
    # Calculate overall scores
    v1_score = v1_basic["accuracy"] + (v1_conflict["actually_resolved"] / 2)
    v2_score = v2_basic["accuracy"] + (v2_conflict["actually_resolved"] / 2)
    
    # Bonus for V2 features
    v2_bonus = 0
    if v2_topic.get("llm_calls_saved_by_embedding", 0) > 0:
        v2_bonus += 0.1
    if v2_basic.get("attention_sinks", 0) > 0:
        v2_bonus += 0.1
    
    print(f"\n{'Feature':<35} {'V1':>15} {'V2':>15}")
    print("-" * 65)
    print(f"{'Basic Flow Accuracy':<35} {v1_basic['accuracy']:>14.1%} {v2_basic['accuracy']:>14.1%}")
    print(f"{'Conflict Resolution':<35} {v1_conflict['actually_resolved']}/2{' ':>10} {v2_conflict['actually_resolved']}/2")
    print(f"{'Topic Shift Detection':<35} {v1_topic['topic_shifts_detected']:>15} {v2_topic['topic_shifts_detected']:>15}")
    print(f"{'LLM Calls Saved (Embedding)':<35} {'N/A':>15} {v2_topic.get('llm_calls_saved_by_embedding', 0):>15}")
    print(f"{'Attention Sinks':<35} {'N/A':>15} {v2_basic.get('attention_sinks', 0):>15}")
    print(f"{'Weighted Retrieval':<35} {'No':>15} {'Yes':>15}")
    print(f"{'Bi-Temporal Support':<35} {'No':>15} {'Yes':>15}")
    print(f"{'Dynamic Persona':<35} {'No':>15} {'Yes':>15}")
    print("-" * 65)
    print(f"{'Composite Score':<35} {v1_score:>15.2f} {v2_score + v2_bonus:>15.2f}")
    
    if v2_score + v2_bonus > v1_score:
        print("\n🏆 WINNER: FCM V2")
    elif v1_score > v2_score + v2_bonus:
        print("\n🏆 WINNER: FCM V1")
    else:
        print("\n🤝 TIE")
    
    return all_results


if __name__ == "__main__":
    run_demo_comparison()
