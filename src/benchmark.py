import argparse
import onnxruntime as ort
import numpy as np
import time
import json
import os

def benchmark(args):
    providers = ort.get_available_providers()
    print(f"Available providers: {providers}")
    
    results = {}
    
    for provider in providers:
        if 'ExecutionProvider' not in provider:
            continue
            
        try:
            session = ort.InferenceSession(args.model, providers=[provider])
        except Exception as e:
            print(f"Failed to load {provider}: {e}")
            continue
            
        input_name = session.get_inputs()[0].name
        shape = session.get_inputs()[0].shape
        
        # Replace dynamic batch size with 1
        shape = [1 if isinstance(d, str) else d for d in shape]
        dummy_input = np.random.randn(*shape).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            session.run(None, {input_name: dummy_input})
            
        times = []
        for _ in range(100):
            start = time.perf_counter()
            session.run(None, {input_name: dummy_input})
            end = time.perf_counter()
            times.append((end - start) * 1000) # ms
            
        p50 = np.percentile(times, 50)
        p95 = np.percentile(times, 95)
        p99 = np.percentile(times, 99)
        
        results[provider] = {
            'p50_ms': p50,
            'p95_ms': p95,
            'p99_ms': p99
        }
        print(f"Provider: {provider} | P50: {p50:.2f}ms | P95: {p95:.2f}ms | P99: {p99:.2f}ms")
        
    with open('benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    benchmark(args)
