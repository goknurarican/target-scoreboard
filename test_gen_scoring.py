#!/usr/bin/env python3
"""
Test script to verify genetics scoring fix.
Run this after implementing the changes to verify everything works.
"""
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


async def test_opentargets_integration():
    """Test Open Targets integration."""
    print("Testing Open Targets integration...")

    try:
        from app.data_access.opentargets import fetch_ot_association

        # Test EGFR - NSCLC association
        result = await fetch_ot_association("EFO_0000305", "EGFR")

        print(f"EGFR-NSCLC association:")
        print(f"  Overall score: {result.get('overall', 0):.3f}")
        print(f"  Genetics score: {result.get('genetics', 0):.3f}")
        print(f"  Evidence count: {result.get('evidence_count', 0)}")
        print(f"  Release: {result.get('release', 'unknown')}")
        print(f"  Cached: {result.get('cached', False)}")
        print(f"  Fetch time: {result.get('fetch_ms', 0):.1f}ms")

        # Test should pass if genetics score > 0
        if result.get('genetics', 0) > 0:
            print("✅ EGFR genetics score > 0 (PASS)")
        else:
            print("❌ EGFR genetics score = 0 (FAIL)")

        return result.get('genetics', 0) > 0

    except Exception as e:
        print(f"❌ Open Targets test failed: {e}")
        return False


async def test_genetics_scoring():
    """Test genetics scoring channel."""
    print("\nTesting genetics scoring channel...")

    try:
        from app.channels.genetics import compute_genetics_score
        from app.data_access.opentargets import fetch_ot_association

        # Get OT data
        ot_data = await fetch_ot_association("EFO_0000305", "EGFR")

        # Compute genetics score
        score, evidence = compute_genetics_score("EFO_0000305", "EGFR", ot_data)

        print(f"Genetics scoring:")
        print(f"  Score: {score:.3f}")
        print(f"  Evidence refs: {len(evidence)}")
        print(f"  Evidence: {evidence[:3]}")  # Show first 3

        if score > 0:
            print("✅ Genetics scoring works (PASS)")
        else:
            print("❌ Genetics scoring returns 0 (FAIL)")

        return score > 0

    except Exception as e:
        print(f"❌ Genetics scoring test failed: {e}")
        return False


async def test_full_scoring_pipeline():
    """Test full scoring pipeline."""
    print("\nTesting full scoring pipeline...")

    try:
        from app.scoring import score_targets
        from app.schemas import ScoreRequest

        # Create test request
        request = ScoreRequest(
            disease="EFO_0000305",
            targets=["EGFR", "KRAS"],
            weights={"genetics": 0.5, "ppi": 0.3, "pathway": 0.2}
        )

        # Score targets
        results = await score_targets(request)

        print(f"Full scoring results:")
        for result in results:
            print(f"  {result.target}:")
            print(f"    Total score: {result.total_score:.3f}")
            print(f"    Genetics: {result.breakdown.genetics:.3f}")
            print(f"    Data version: {result.data_version}")

        # Check if any genetics scores > 0
        genetics_scores = [r.breakdown.genetics for r in results if r.breakdown.genetics > 0]

        if genetics_scores:
            print(f"✅ Full pipeline works - found {len(genetics_scores)} targets with genetics scores (PASS)")
        else:
            print("❌ Full pipeline - no genetics scores found (FAIL)")

        return len(genetics_scores) > 0

    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        return False


async def test_cache_functionality():
    """Test caching functionality."""
    print("\nTesting cache functionality...")

    try:
        from app.data_access.opentargets import fetch_ot_association
        import time

        # First call (should be fresh)
        start = time.time()
        result1 = await fetch_ot_association("EFO_0000305", "EGFR")
        time1 = time.time() - start

        # Second call (should be cached)
        start = time.time()
        result2 = await fetch_ot_association("EFO_0000305", "EGFR")
        time2 = time.time() - start

        print(f"Cache test:")
        print(f"  First call: {time1 * 1000:.1f}ms, cached: {result1.get('cached', False)}")
        print(f"  Second call: {time2 * 1000:.1f}ms, cached: {result2.get('cached', False)}")

        # Second call should be faster and cached
        if result2.get('cached', False) and time2 < time1:
            print("✅ Cache functionality works (PASS)")
            return True
        else:
            print("⚠️  Cache might not be working as expected")
            return False

    except Exception as e:
        print(f"❌ Cache test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("=== Testing Genetics Scoring Fix ===\n")

    tests = [
        ("Open Targets Integration", test_opentargets_integration),
        ("Genetics Scoring", test_genetics_scoring),
        ("Full Scoring Pipeline", test_full_scoring_pipeline),
        ("Cache Functionality", test_cache_functionality)
    ]

    results = []
    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} test crashed: {e}")
            results.append((name, False))

    print("\n=== Test Summary ===")
    passed = 0
    for name, result in results:
        status = "PASS" if result else "FAIL"
        icon = "✅" if result else "❌"
        print(f"{icon} {name}: {status}")
        if result:
            passed += 1

    print(f"\nTests passed: {passed}/{len(results)}")

    if passed == len(results):
        print("🎉 All tests passed! Genetics scoring fix is working.")
    else:
        print("⚠️  Some tests failed. Check the implementation.")


if __name__ == "__main__":
    asyncio.run(main())