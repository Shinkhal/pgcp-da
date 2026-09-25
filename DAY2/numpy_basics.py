import numpy as np
import sys
from pprint import pprint

def print_section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    
    
def print_value(label,value):
    print(f"{label:<32}:{value}")
    
vector1 = np.array([10,20,30,40,50])
matrix_2d = np.array([[10,20,30,40,50],[60,70,80,90,100]])

print_section("Introduction to NDARRAYS")
print_value("1D Array ",vector1)
print_value("Data Type", vector1.dtype)
print_value("Array type", type(vector1))
print_value("Shape", vector1.shape)
print_value("Python object size (bytes)",sys.getsizeof(vector1))
print_value("NumPY buffer size (bytes)", vector1.nbytes)

print("\n2D array: ")
print(matrix_2d)
print_value("Shape", matrix_2d.shape)
print_value("")
print_value("Python object size (bytes)",sys.getsizeof(matrix_2d))
print_value("NumPY buffer size (bytes)", matrix_2d.nbytes)



# Tensor Initialzation using factory methods

zero_matrix = np.zeros((3,4),dtype=int)
one_matrix = np.ones((3,4), np.float64)
const_matrix = np.full((3,4),5.5)
identity_matrix = np.eye(4)

print_section("2. Tensor Initialization (Factory Functions)")
