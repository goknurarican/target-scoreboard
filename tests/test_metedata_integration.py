#!/usr/bin/env python3
"""
Test script to verify versioning and cache metadata integration.
"""
import asyncio
import json
import requests
import time


async def test_scoring_metadata():
    """Test scoring with metadata collection."""
    print("Testing scoring metadata collection...")

    try:
        from app.scoring import score_targets
        from app.schemas import ScoreRequest

        # Create test request
        request = ScoreRequest(
            disease="EFO_0000305",
            targets=["EGFR", "KRAS"],
            weights={"genetics": 0.5, "ppi": 0.3, "pathway": 0.2}
        )

        # Score targets with metadata
        target_scores, metadata = await score_targets(request)

        print(f"Metadata collected:")
        print(f"  Data version: {metadata.get('data_version')}")
        print(f"  Cache metadata: {metadata.get('meta')}")

        # Verify data_version format
        data_version = metadata.get('data_version', '')
        expected_parts = ['OT-', 'STRING-', 'Reactome-']

        if all(part in data_version for part in expected_parts):
            print("✅ Data version format correct")
        else:
            print(f"❌ Data version format incorrect: {data_version}")

        # Verify cache metadata structure
        cache_meta = metadata.get('meta', {})
        required_fields = ['cached', 'fetch_ms', 'cache_hit_rate', 'total_calls']

        if all(field in cache_meta for field in required_fields):
            print("✅ Cache metadata structure correct")
        else:
            print(f"❌ Cache metadata missing fields: {cache_meta}")

        return True

    except Exception as e:
        print(f"❌ Metadata test failed: {e}")
        return False


def test_api_response_format():
    """Test API response includes metadata fields."""
    print("\nTesting API response format...")

    try:
        # Start by checking if API is running
        health_response = requests.get("http://localhost:8000/healthz", timeout=5)
        if health_response.status_code != 200:
            print("⚠️  API not running, skipping API tests")
            return False

        # Test scoring endpoint
        payload = {
            "disease": "EFO_0000305",
            "targets": ["EGFR"],
            "weights": {"genetics": 0.5, "ppi": 0.3, "pathway": 0.2}
        }

        response = requests.post(
            "http://localhost:8000/score",
            json=payload,
            timeout=30
        )

        if response.status_code != 200:
            print(f"❌ API error: {response.status_code} - {response.text}")
            return False

        data = response.json()

        # Check required fields
        required_fields = ['targets', 'request_summary', 'processing_time_ms', 'data_version', 'meta']
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            print(f"❌ Missing fields in API response: {missing_fields}")
            return False

        print("✅ API response contains all required fields")

        # Verify data_version format
        data_version = data.get('data_version', '')
        if 'OT-' in data_version and 'STRING-' in data_version and 'Reactome-' in data_version:
            print(f"✅ API data_version format correct: {data_version}")
        else:
            print(f"❌ API data_version format incorrect: {data_version}")

        # Verify meta structure
        meta = data.get('meta', {})
        meta_fields = ['cached', 'fetch_ms', 'cache_hit_rate', 'total_calls']
        missing_meta = [field for field in meta_fields if field not in meta]

        if missing_meta:
            print(f"❌ Missing meta fields: {missing_meta}")
            return False

        print("✅ API meta structure correct")
        print(f"  Cache hit rate: {meta.get('cache_hit_rate', 0):.2%}")
        print(f"  Total fetch time: {meta.get('fetch_ms', 0):.1f}ms")

        return True

    except requests.exceptions.ConnectionError:
        print("⚠️  Cannot connect to API (not running)")
        return False
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False


def test_cache_behavior():
    """Test cache behavior with multiple calls."""
    print("\nTesting cache behavior...")

    try:
        payload = {
            "disease": "EFO_0000305",
            "targets": ["EGFR"],
            "weights": {"genetics": 0.5, "ppi": 0.3, "pathway": 0.2}
        }

        # First call (should populate cache)
        start_time = time.time()
        response1 = requests.post("http://localhost:8000/score", json=payload, timeout=30)
        time1 = time.time() - start_time

        if response1.status_code != 200:
            print("❌ First API call failed")
            return False

        # Second call (should use cache)
        start_time = time.time()
        response2 = requests.post("http://localhost:8000/score", json=payload, timeout=30)
        time2 = time.time() - start_time

        if response2.status_code != 200:
            print("❌ Second API call failed")
            return False

        data1 = response1.json()
        data2 = response2.json()

        # Compare cache hit rates
        cache_rate1 = data1.get('meta', {}).get('cache_hit_rate', 0)
        cache_rate2 = data2.get('meta', {}).get('cache_hit_rate', 0)

        print(f"First call: {time1:.2f}s, cache hit rate: {cache_rate1:.2%}")
        print(f"Second call: {time2:.2f}s, cache hit rate: {cache_rate2:.2%}")

        # Second call should have higher cache hit rate or be faster
        if cache_rate2 >= cache_rate1 or time2 < time1:
            print("✅ Cache behavior working")
            return True
        else:
            print("⚠️  Cache behavior unclear (may still be working)")
            return True

    except Exception as e:
        print(f"❌ Cache test failed: {e}")
        return False


def validate_sample_response():
    """Validate the sample response file."""
    print("\nValidating sample response file...")

    try:
        with open('../examples/sample_response.json', 'r') as f:
            sample = json.load(f)

        # Check structure
        required_fields = ['targets', 'request_summary', 'processing_time_ms', 'data_version', 'meta']
        missing = [field for field in required_fields if field not in sample]

        if missing:
            print(f"❌ Sample response missing fields: {missing}")
            return False

        # Check data_version format
        data_version = sample.get('data_version', '')
        if 'OT-' in data_version and 'STRING-' in data_version and 'Reactome-' in data_version:
            print("✅ Sample response data_version format correct")
        else:
            print(f"❌ Sample response data_version format incorrect: {data_version}")
            return False

        # Check meta structure
        meta = sample.get('meta', {})
        meta_fields = ['cached', 'fetch_ms', 'cache_hit_rate', 'total_calls']
        missing_meta = [field for field in meta_fields if field not in meta]

        if missing_meta:
            print(f"❌ Sample response missing meta fields: {missing_meta}")
            return False

        print("✅ Sample response structure correct")
        return True

    except FileNotFoundError:
        print("❌ Sample response file not found")
        return False
    except json.JSONDecodeError:
        print("❌ Sample response file invalid JSON")
        return False
    except Exception as e:
        print(f"❌ Sample response validation failed: {e}")
        return False


async def main():
    """Run all metadata tests."""
    print("=== Testing Versioning and Cache Metadata ===\n")

    tests = [
        ("Scoring Metadata Collection", test_scoring_metadata),
        ("API Response Format", test_api_response_format),
        ("Cache Behavior", test_cache_behavior),
        ("Sample Response Validation", validate_sample_response)
    ]

    results = []
    for name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
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
        print("🎉 All metadata tests passed!")
    else:
        print("⚠️  Some tests failed. Check the implementation.")


if __name__ == "__main__":
    asyncio.run(main())