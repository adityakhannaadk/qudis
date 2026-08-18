import numpy as np

def print_as_polynomial(factor, poly, poly_name=""):
    poly_str=""
    for i in range(len(poly)):
        if(i>0): poly_str+="+"
        poly_str+=str(factor[i])+"*x^"+str(poly[i][0])+"*y^"+str(poly[i][1])
    print(f"{poly_name}: {poly_str}")
    return(f"{poly_name}: {poly_str}")

def to_array(matrix):
    if isinstance(matrix, np.ndarray):
        return matrix
    else:
        return matrix.toarray()