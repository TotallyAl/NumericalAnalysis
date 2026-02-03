import numpy as np

def backwardSubstitution(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    '''
    Direct Method.

    Solves the system of linear equations Ax = b using backward substitution.
    
    Parameters:
    A (np.ndarray): Coefficient matrix (assumed to be upper triangular, for now).
    b (np.ndarray): Right-hand side vector.

    Returns:
    x (np.ndarray): Solution vector.
    '''
    n: int = len(b)
    x: np.ndarray = np.zeros(n)
    for i in range(n-1, -1, -1):
        if i != n-1:
            x[i] = (b[i] - sum(A[i, j] * x[j] for j in range(i+1, n))) / A[i, i]
            continue
        x[i] = b[i]/A[i, i]
    return x.T
    
def swapLines(matrix: np.ndarray, line1: int, line2: int) -> None:
    temp: np.ndarray = matrix[line1].copy()
    matrix[line1] = matrix[line2]
    matrix[line2] = temp

def bestPivot(A: np.ndarray, b: np.ndarray , pivot: float, dimension: int) -> None:
    for line in range(pivot, dimension):
        if abs(A[line][pivot]) > abs(A[pivot][pivot]):
            swapLines(A, line, pivot)
            if b is not None:
                swapLines(b, line, pivot)

def triangularize(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    '''
    This method will transform a square matrix into an upper triangular matrix
    '''

    dimension: int = len(A)
    for pivot in range(0, dimension):
        bestPivot(A, b, pivot, dimension)
        for row in range(pivot+1, dimension):
            coeff: float = A[row][pivot] / A[pivot][pivot]
            A[row] = A[row] - coeff * A[pivot]
            b[row] = b[row] - coeff * b[pivot]
    return A, b

def triangularSystem(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    return backwardSubstitution(A, b)

def gauss(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    '''
    Solving a linear system with Gauss
    '''
    A, b = triangularize(A, b)
    return triangularSystem(A, b)

def LUDecomposition(A: np.ndarray) -> np.ndarray:
    '''
    Direct Method.
    '''

    U: np.ndarray = A.copy()
    dimension: int = len(U)
    L: np.ndarray = np.eye(dimension)
    
    for pivot in range(0, dimension):
        bestPivot(U, None, pivot, dimension)
        for row in range(pivot+1, dimension):
            coeff: float = U[row][pivot] / U[pivot][pivot]
            U[row] = U[row] - coeff * U[pivot]
            L[row][pivot] = coeff
    return L, U

def jacobi() -> None:
    '''
    Iterative Method.
    '''
    
    pass

def gaussSeidel() -> None:
    '''
    Iterative Method.
    '''
    pass