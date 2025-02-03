import time
import numpy as np
from memory_profiler import profile


def original_softmax(x):
    """Traditional 3-pass implementation of softmax."""

    # 1. compute the maximum value of the row.
    m_n = float("-inf")
    for val in x:
        m_n = max(m_n, val)

    # 2. compute the normalization factor.
    l_n = 0
    for val in x:
        l_n += np.exp(val - m_n)

    # 3. compute the softmax values.
    output = np.zeros_like(x)
    for i, val in enumerate(x):
        output[i] = np.exp(val - m_n) / l_n

    return output


def online_softmax(x):
    """Online (2-pass) implementation of softmax."""

    # 1. compute the maximum value and the normalization factor together.
    m_prev = float("-inf")
    l_prev = 0

    for val in x:
        m_curr = max(m_prev, val)
        l_curr = l_prev * np.exp(m_prev - m_curr) + np.exp(val - m_curr)
        m_prev = m_curr
        l_prev = l_curr

    # 2. compute the final softmax values.
    output = np.zeros_like(x)
    for i, val in enumerate(x):
        output[i] = np.exp(val - m_prev) / l_prev

    return output


@profile
def test_softmax(vector_size=1000000):
    """Test softmax implementations with different vector sizes."""
    print(f"\nTesting with vector size: {vector_size:,}")

    x = np.random.randn(vector_size) * 10

    x[0] = 1000  # large positive value.
    x[-1] = -1000  # large negative value.

    # 1. compute the traditional softmax.
    start_time = time.time()
    result1 = original_softmax(x)
    trad_time = time.time() - start_time

    print("\nTraditional Softmax:")
    print(f"First few values: {[round(val, 4) for val in result1[:3]]}...")
    print(f"Sum of probabilities: {np.sum(result1):.6f}")
    print(f"Time taken: {trad_time:.6f} seconds")

    # 2. compute the online softmax.
    start_time = time.time()
    result2 = online_softmax(x)
    online_time = time.time() - start_time

    print("\nOnline Softmax:")
    print(f"First few values: {[round(val, 4) for val in result2[:3]]}...")
    print(f"Sum of probabilities: {np.sum(result2):.6f}")
    print(f"Time taken: {online_time:.6f} seconds")

    max_diff = np.max(np.abs(result1 - result2))
    print(f"\nMaximum difference between implementations: {max_diff:.10f}")

    return trad_time, online_time


@profile
def main():
    sizes = [1000, 10000, 100000, 1000000]

    print("\nComparing softmax implementations with different vector sizes:")
    print("-" * 60)

    for size in sizes:
        trad_time, online_time = test_softmax(size)
        speedup = (trad_time - online_time) / trad_time * 100
        print(f"\nVector size {size:,}:")
        print(f"Speedup: {speedup:.2f}%")
        print("-" * 60)


if __name__ == "__main__":
    main()
