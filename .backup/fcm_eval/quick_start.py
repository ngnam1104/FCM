"""
FCM Quick Start Comparison
==========================

So sánh FCM V1 vs V2 với kịch bản Quick Start đơn giản.

Chạy: python -m fcm_eval.quick_start
"""

import os
import sys
import time
import logging

# Suppress logging
logging.basicConfig(level=logging.WARNING)
logging.getLogger("mem0").setLevel(logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fcm_eval.common import (
    print_header, print_subheader, print_comparison_table,
    check_answer, BenchmarkResult,
    QUICK_START_MESSAGES, QUICK_START_QUERIES
)


def run_v1_quick_start(user_id: str = "quick_start_v1") -> BenchmarkResult:
    """Run Quick Start với FCM V1"""
    from fcm import FCMAgent, FCMConfig
    
    print_subheader("FCM V1: Running...")
    start_time = time.time()
    
    config = FCMConfig(
        verbose=False,
        crystallize_threshold=3,
    )
    
    agent = FCMAgent(config=config, user_id=user_id)
    
    # Clear old memories
    try:
        agent.memory.delete_all(user_id=user_id)
    except:
        pass
    
    # Process messages
    print("  Ingesting messages...", end="", flush=True)
    for msg in QUICK_START_MESSAGES:
        agent.chat(msg)
        print(".", end="", flush=True)
    print(" Done.")
    
    # End session (crystallize + evolve)
    print("  Processing (Crystallize + Evolve)...", end="", flush=True)
    agent.end_session()
    print(" Done.")
    
    # Search and evaluate (using ENHANCED retrieval - same as V2 for fair comparison)
    print("  Evaluating search accuracy (Enhanced Retrieval)...")
    correct = 0
    for query, expected in QUICK_START_QUERIES:
        result = agent.search(query, strategy="enhanced")
        combined = result.get("combined", [])
        if combined:
            retrieved = combined[0].get("memory", "")
            if check_answer(retrieved, expected):
                correct += 1
                source = combined[0].get("source_layer", "?")
                print(f"    ✅ '{query}' → Found '{expected}' [{source}]")
            else:
                print(f"    ❌ '{query}' → Expected '{expected}', got '{retrieved[:50]}...'")
        else:
            print(f"    ❌ '{query}' → No results")
    
    accuracy = correct / len(QUICK_START_QUERIES)
    
    # Get stats
    stats = agent.get_stats()
    elapsed = time.time() - start_time
    
    return BenchmarkResult(
        version="V1",
        total_time=elapsed,
        messages_processed=len(QUICK_START_MESSAGES),
        facts_extracted=stats.get("crystal_count", 0) + stats.get("solid_count", 0),
        search_accuracy=accuracy,
        memory_stats={
            "Liquid Count": stats.get("liquid_count", 0),
            "Crystal Count": stats.get("crystal_count", 0),
            "Solid Count": stats.get("solid_count", 0),
        },
        additional_metrics={}
    )


def run_v2_quick_start(user_id: str = "quick_start_v2") -> BenchmarkResult:
    """Run Quick Start với FCM V2"""
    from fcm import FCMAgentV2, FCMConfigV2
    
    print_subheader("FCM V2: Running...")
    start_time = time.time()
    
    config = FCMConfigV2(
        verbose=False,
        crystallize_threshold=3,
        attention_sink_count=2,
        enable_active_forgetting=True,
        enable_dynamic_persona=True,
    )
    
    agent = FCMAgentV2(config=config, user_id=user_id)
    
    # Clear old memories
    agent.clear_all_memories()
    
    # Process messages
    print("  Ingesting messages...", end="", flush=True)
    attention_sinks = 0
    for msg in QUICK_START_MESSAGES:
        result = agent.chat(msg)
        if result.get("is_attention_sink"):
            attention_sinks += 1
        print(".", end="", flush=True)
    print(" Done.")
    
    # End session
    print("  Processing (Crystallize + Evolve)...", end="", flush=True)
    agent.end_session()
    print(" Done.")
    
    # Search and evaluate (using ENHANCED retrieval - new pipeline)
    print("  Evaluating search accuracy (Enhanced Retrieval)...")
    correct = 0
    for query, expected in QUICK_START_QUERIES:
        result = agent.search(query, strategy="enhanced")
        combined = result.combined_results
        if combined:
            retrieved = combined[0].get("memory", "")
            if check_answer(retrieved, expected):
                correct += 1
                source = combined[0].get("source_layer", "?")
                print(f"    ✅ '{query}' → Found '{expected}' [{source}]")
            else:
                print(f"    ❌ '{query}' → Expected '{expected}', got '{retrieved[:50]}...'")
        else:
            print(f"    ❌ '{query}' → No results")
    
    accuracy = correct / len(QUICK_START_QUERIES)
    
    # Get stats
    stats = agent.get_stats()
    elapsed = time.time() - start_time
    
    return BenchmarkResult(
        version="V2",
        total_time=elapsed,
        messages_processed=len(QUICK_START_MESSAGES),
        facts_extracted=stats.get("crystal_count", 0) + stats.get("solid_count", 0),
        search_accuracy=accuracy,
        memory_stats={
            "Liquid Count": stats.get("liquid_count", 0),
            "Crystal Count": stats.get("crystal_count", 0),
            "Solid Count": stats.get("solid_count", 0),
        },
        additional_metrics={
            "Attention Sinks": stats.get("attention_sinks_count", attention_sinks),
            "LLM Calls Saved": stats.get("llm_calls_saved", 0),
            "Persona Extracted": "Yes" if agent.get_user_persona() else "No",
        }
    )


def run_quick_start_comparison():
    """Main function - So sánh V1 vs V2 với Quick Start scenario"""
    print_header("FCM QUICK START COMPARISON", "═")
    print("""
    Kịch bản: Quick Start - 5 tin nhắn cơ bản về thông tin cá nhân
    
    Test: 
    - Ingest 5 messages
    - Crystallize + Evolve  
    - Search 3 queries
    
    Đánh giá: Search Accuracy, Processing Time, Memory Stats
    """)
    
    # Run V1
    v1_result = run_v1_quick_start()
    
    # Run V2
    v2_result = run_v2_quick_start()
    
    # Print comparison
    print_comparison_table(v1_result, v2_result)
    
    return v1_result, v2_result


if __name__ == "__main__":
    run_quick_start_comparison()
