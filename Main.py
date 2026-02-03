#!/usr/bin/env python3
import numpy as np

from LinearSystems import LinearSystems

A = np.array([[1, 2, 3], [4, 5, 6], [5, 8, 9]], dtype=float)
b = np.array([1, 2, 3], dtype=float)

solution: np.ndarray = LinearSystems(A, b).gauss()
print(f"Linear System solved: {solution}")

sol_np: np.ndarray = np.linalg.solve(A, b)
print(f"Numpy solution: {sol_np}")

# A = np.array([[1, 2, 3], [4, 5, 6], [5, 8, 9]], dtype=float)
# A: np.ndarray = np.array([[1, 2, 0, 1], [1, 2, 1, 0], [0, 1, 0, 1], [1, 0, 0, 1]], dtype=float)
A = np.array([
    [4, 2, 0],
    [4, 4, 2],
    [2, 2, 3]
], dtype=float)
decomposition: np.ndarray = LinearSystems(A, None).LUDecomposition()
print(f"L: {decomposition[0]}")
print(f"U: {decomposition[1]}")

print(f"A = {decomposition[0] @ decomposition[1]}")