import numpy as np

def inspect_dataset(path, n=5):
    data = np.load(path)
    print(f"Keys: {list(data.keys())}")
    print(f"States shape: {data['states'].shape}")
    print(f"Actions shape: {data['actions'].shape}")
    print(f"Values shape: {data['values'].shape}")
    print("\nSample evaluation scores:")
    for i, val in enumerate(data["values"][:n]):
        print(f"{i:2d}: {val:.6f}")

# Example:
inspect_dataset("dataset.npz")