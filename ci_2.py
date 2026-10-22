
import numpy as np

n = int(input("Enter the number of equations and unknowns (n): "))

A = []
b = []

for i in range(n):
    row = []
    for j in range(n):
        value = float(input(f"Enter a[{i+1}][{j+1}]: "))
        row.append(value)
    A.append(row)

for i in range(n):
    value = float(input(f"Enter b[{i+1}]: "))
    b.append(value)

A = np.array(A, dtype=float)
b = np.array(b, dtype=float)

try:
    x = np.linalg.solve(A, b)

    print("\nMatrix A:")
    print(A)

    print("\nVector b:")
    print(b)

    print("\nSolution:")
    for i in range(n):
        print(f"x{i+1} = {x[i]:.5f}")

except np.linalg.LinAlgError:
    print("\nThe system has no unique solution.")
    print("Matrix A is singular.")
