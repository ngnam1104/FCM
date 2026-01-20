"""
FCM Evaluation Runner
=====================

Script chính để chạy tất cả benchmarks so sánh FCM V1 vs V2.

Chạy:
  python -m fcm_eval                    # Menu chọn
  python -m fcm_eval --all              # Chạy tất cả
  python -m fcm_eval --quick-start      # Chỉ Quick Start
  python -m fcm_eval --demo             # Chỉ Demo
  python -m fcm_eval --locomo           # Chỉ LoCoMo benchmark
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_banner():
    """Print FCM Evaluation banner"""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║     ███████╗ ██████╗███╗   ███╗    ███████╗██╗   ██╗ █████╗ ██╗  ║
║     ██╔════╝██╔════╝████╗ ████║    ██╔════╝██║   ██║██╔══██╗██║  ║
║     █████╗  ██║     ██╔████╔██║    █████╗  ██║   ██║███████║██║  ║
║     ██╔══╝  ██║     ██║╚██╔╝██║    ██╔══╝  ╚██╗ ██╔╝██╔══██║██║  ║
║     ██║     ╚██████╗██║ ╚═╝ ██║    ███████╗ ╚████╔╝ ██║  ██║███████╗
║     ╚═╝      ╚═════╝╚═╝     ╚═╝    ╚══════╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝
║                                                                  ║
║          Frequency-based Crystallizing Memory Evaluation         ║
║                         V1 vs V2 Comparison                      ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)


def print_summary():
    """Print summary of FCM V1 vs V2"""
    print("""
┌─────────────────────────────────────────────────────────────────┐
│                      ARCHITECTURE COMPARISON                     │
├─────────────────────────────────────────────────────────────────┤
│  Feature                      │  FCM V1      │  FCM V2          │
├───────────────────────────────┼──────────────┼──────────────────┤
│  Bi-Temporal Schema           │  ❌ No       │  ✅ Yes          │
│  Attention Sinks              │  ❌ No       │  ✅ Yes          │
│  Semantic Grouping            │  ❌ No       │  ✅ Yes (Embed)  │
│  Active Forgetting            │  ❌ No       │  ✅ Ebbinghaus   │
│  Dynamic Persona              │  ❌ No       │  ✅ Yes          │
│  Weighted Retrieval           │  ❌ No       │  ✅ Yes          │
│  Parallel Search              │  ❌ No       │  ✅ Yes          │
│  Cold Storage                 │  ❌ No       │  ✅ Yes          │
├───────────────────────────────┼──────────────┼──────────────────┤
│  Lines of Code                │  ~2,100      │  ~3,300          │
│  Modular Structure            │  1 file      │  6 folders       │
└───────────────────────────────┴──────────────┴──────────────────┘
    """)


def interactive_menu():
    """Interactive menu to select benchmark"""
    from fcm_eval.quick_start import run_quick_start_comparison
    from fcm_eval.demo import run_demo_comparison
    from fcm_eval.locomo import run_locomo_comparison
    
    while True:
        print("\n" + "=" * 50)
        print("  FCM EVALUATION MENU")
        print("=" * 50)
        print("  1. Quick Start Comparison")
        print("  2. Demo Comparison (4 scenarios)")
        print("  3. LoCoMo Benchmark")
        print("  4. Run All Benchmarks")
        print("  5. Show Architecture Summary")
        print("  0. Exit")
        print("-" * 50)
        
        choice = input("Select option (0-5): ").strip()
        
        if choice == "0":
            print("\nGoodbye! 👋")
            break
        elif choice == "1":
            run_quick_start_comparison()
        elif choice == "2":
            run_demo_comparison()
        elif choice == "3":
            run_locomo_comparison()
        elif choice == "4":
            print("\n🚀 Running all benchmarks...")
            run_quick_start_comparison()
            run_demo_comparison()
            run_locomo_comparison()
            print("\n✅ All benchmarks completed!")
        elif choice == "5":
            print_summary()
        else:
            print("Invalid option. Please try again.")


def main():
    parser = argparse.ArgumentParser(
        description="FCM Evaluation - Compare V1 vs V2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m fcm_eval                    # Interactive menu
  python -m fcm_eval --all              # Run all benchmarks
  python -m fcm_eval --quick-start      # Quick start only
  python -m fcm_eval --demo             # Demo scenarios only
  python -m fcm_eval --locomo           # LoCoMo benchmark only
        """
    )
    
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--quick-start", action="store_true", help="Run Quick Start comparison")
    parser.add_argument("--demo", action="store_true", help="Run Demo comparison")
    parser.add_argument("--locomo", action="store_true", help="Run LoCoMo benchmark")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument("--summary", action="store_true", help="Show architecture summary")
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.summary:
        print_summary()
        return
    
    # Check if any specific benchmark selected
    if args.all or args.quick_start or args.demo or args.locomo:
        from fcm_eval.quick_start import run_quick_start_comparison
        from fcm_eval.demo import run_demo_comparison
        from fcm_eval.locomo import run_locomo_comparison
        
        if args.all or args.quick_start:
            run_quick_start_comparison()
        
        if args.all or args.demo:
            run_demo_comparison()
        
        if args.all or args.locomo:
            run_locomo_comparison(verbose=not args.quiet)
        
        print("\n✅ Benchmarks completed!")
    else:
        # Interactive mode
        interactive_menu()


if __name__ == "__main__":
    main()
