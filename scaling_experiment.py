"""Run scaling experiments on synthetic ECMPP instances."""

import time
import pandas as pd
from instance_generator import ECMPPGenerator
from ecmpp_encoder import ECMPPEncoder


def run_scaling_experiment(sizes, timeout=3600):
    """Run ECMPP solver on instances of increasing size."""
    generator = ECMPPGenerator()
    results = []
    
    for size in sizes:
        print(f"Running instance with {size} blocks...")
        n_clauses = max(10, int(size * 0.1))
        blocks, clauses, npv_min = generator.generate_synthetic(size, n_clauses)
        
        start = time.time()
        encoder = ECMPPEncoder(blocks, clauses, npv_min)
        status, model, npv = encoder.solve(timeout=timeout)
        elapsed = time.time() - start
        
        results.append({
            'n_blocks': size,
            'n_clauses': len(clauses),
            'status': status,
            'solve_time': elapsed,
            'npv': npv if model else None
        })
        
        print(f"  Status: {status}, Time: {elapsed:.2f}s")
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    sizes = [25, 100, 500, 1000, 5000, 10000, 50000, 100000]
    df = run_scaling_experiment(sizes)
    df.to_csv('scaling_results.csv', index=False)
    print("\nResults saved to scaling_results.csv")
    print(df)