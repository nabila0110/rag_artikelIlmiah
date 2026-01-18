# test_api.py

"""
Script to test Flask API endpoints
"""

import requests
import json
import time

BASE_URL = 'http://localhost:5000'

def test_health():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("TEST 1: Health Check")
    print("="*60)
    
    response = requests.get(f'{BASE_URL}/health')
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200


def test_stats():
    """Test statistics endpoint"""
    print("\n" + "="*60)
    print("TEST 2: Statistics")
    print("="*60)
    
    response = requests.get(f'{BASE_URL}/api/stats')
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200


def test_search_without_generation():
    print("\n" + "-"*60)
    print("TEST 3: Search (No Generation)")
    print("-"*60)
    
    payload = {
        'query': 'Apa itu machine learning?',
        'top_k': 10,
        'generate_answer': False
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    
    start = time.time()
    response = requests.post(f'{BASE_URL}/api/search', json=payload)
    elapsed = time.time() - start
    
    print(f"\nStatus Code: {response.status_code}")
    print(f"Time: {elapsed:.2f}s")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nResults:")
        print(f"  - Query: {data['query']}")
        print(f"  - Num results: {data['num_results']}")
        print(f"\nTop 5 results:")
        for i, result in enumerate(data['results'][:5], 1):
            print(f"\n  {i}. {result['judul']}")
            print(f"     Similarity: {result['similarity']:.4f}")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200


def test_search_with_generation():
    print("\n" + "-"*60)
    print("TEST 4: Search + Generation (SLOW)")
    print("-"*60)
    
    payload = {
        'query': 'Apa itu machine learning?',
        'top_k': 10,
        'generate_answer': True
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    
    start = time.time()
    response = requests.post(f'{BASE_URL}/api/search', json=payload, timeout=60)
    elapsed = time.time() - start
    
    print(f"\nStatus Code: {response.status_code}")
    print(f"Time: {elapsed:.2f}s")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nResults:")
        print(f"  - Query: {data['query']}")
        print(f"  - Answer preview: {data['answer'][:200]}...")
        print(f"  - Context chunks used: {data['context_chunks_used']}")
        print(f"  - Cited references: {len(data['cited_references'])}")
        print(f"  - Additional references: {len(data['additional_references'])}")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200


def main():
    print("\n" + "-"*60)
    print("# FLASK API TESTING")
    print("-"*60)
    
    tests = [
        ('Health Check', test_health),
        ('Statistics', test_stats),
        ('Search (No Generation)', test_search_without_generation),
        ('Search + Generation', test_search_with_generation),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\nTest failed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "-"*60)
    print("# TEST SUMMARY")
    print("-"*60)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{status} - {name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"\nTotal: {passed}/{total} passed")
    
    return all(r for _, r in results)


if __name__ == '__main__':
    try:
        success = main()
        exit(0 if success else 1)
    except requests.exceptions.ConnectionError:
        print("\nCannot connect to Flask app")
        print("Make sure the app is running: python run.py")
        exit(1)
    except KeyboardInterrupt:
        print("\n\nUser ganggu cih")
        exit(1)