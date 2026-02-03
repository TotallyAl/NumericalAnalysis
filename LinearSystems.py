import numpy as np

class LinearSystems:

    def __init__(self, A:np.ndarray, b:np.ndarray) -> None:
        self.A = A
        self.b = b

    def _backwardSubstitution(self) -> np.ndarray:
        '''
        Direct Method.

        Solves the system of linear equations Ax = b using backward substitution.
        
        Parameters:
        A (np.ndarray): Coefficient matrix (assumed to be upper triangular, for now).
        b (np.ndarray): Right-hand side vector.

        Returns:
        x (np.ndarray): Solution vector.
        '''
        n: int = len(self.b)
        x: np.ndarray = np.zeros(n)
        for i in range(n-1, -1, -1):
            if i != n-1:
                x[i] = (self.b[i] - sum(self.A[i, j] * x[j] for j in range(i+1, n))) / self.A[i, i]
                continue
            x[i] = self.b[i]/self.A[i, i]
        return x.T
    
    def _swapLines(self, matrix: np.ndarray, line1: int, line2: int) -> None:
        temp: np.ndarray = matrix[line1].copy()
        matrix[line1] = matrix[line2]
        matrix[line2] = temp

    def _bestPivot(self, A: np.ndarray, b: np.ndarray , pivot: float, dimension: int) -> None:
        for line in range(pivot, dimension):
            if abs(A[line][pivot]) > abs(A[pivot][pivot]):
                self._swapLines(A, line, pivot)
                if b is not None:
                    self._swapLines(b, line, pivot)

    def _triangularize(self) -> np.ndarray:
        '''
        This method will transform a square matrix into an upper triangular matrix
        '''
        
        A: np.ndarray = self.A.copy()
        b: np.ndarray = self.b.copy()

        dimension: int = len(A)
        
        for pivot in range(0, dimension):
            self._bestPivot(A, b, pivot, dimension)
            for row in range(pivot+1, dimension):
                coeff: float = A[row][pivot] / A[pivot][pivot]
                A[row] = A[row] - coeff * A[pivot]
                b[row] = b[row] - coeff * b[pivot]
        return A, b

    def triangularSystem(self) -> np.ndarray:
        return self._backwardSubstitution()
    
    def gauss(self) -> np.ndarray:
        '''
        Solving a linear system with Gauss
        '''
        self.A, self.b = self._triangularize()
        return self.triangularSystem()

    def LUDecomposition(self) -> np.ndarray:
        '''
        Direct Method.
        '''

        U: np.ndarray = self.A.copy()
        dimension: int = len(U)
        L: np.ndarray = np.eye(dimension)
        
        for pivot in range(0, dimension):
            self._bestPivot(U, None, pivot, dimension)
            for row in range(pivot+1, dimension):
                coeff: float = U[row][pivot] / U[pivot][pivot]
                U[row] = U[row] - coeff * U[pivot]
                L[row][pivot] = coeff
        return L, U

    def jacobi(self) -> None:
        '''
        Iterative Method.
        '''
        
        pass

    def gaussSeidel(self) -> None:
        '''
        Iterative Method.
        '''
        pass