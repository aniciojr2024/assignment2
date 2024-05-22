import numpy as np

def rmse(predictions, targets):
    pred = np.array(predictions)
    tar = np.array(targets)
    rmse_value = np.sqrt(np.mean((pred - tar) ** 2))
    return rmse_value

def test_rmse():
    tests = [
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], 0),
        ([2, 3, 4, 5, 6], [1, 2, 3, 4, 5], 1)
    ]
    
    for i, (predictions, targets, expected) in enumerate(tests, 1):
        result = rmse(predictions, targets)
        print(f"Test {i} - Predictions: {predictions}, Targets: {targets}")
        print(f"Expected: {expected}, Actual: {result}")
        assert abs(result - expected) < 1e-9, f"Test {i} failed: Expected {expected}, got {result}"
        print(f"Test {i} passed!\n")

if __name__ == "__main__":
    test_rmse()
    print("All tests passed!")
