"""prune.py
This file checks equivalences and prunes generated codes to make sure 
that we aren't doing any unneccessary work. 
"""

def reduce_polynomial(pol):
    """
    Args:
        pol - A polynomial represented as List[Tuple]
    Returns:
        reduced form of the polynomial e.g. factorise out greatest monomial
    """
    x_list = []
    y_list = []
    for xy in pol:
        x_list.append(xy[0])
        y_list.append(xy[1])
    min_x = min(x_list)
    min_y = min(y_list)
    return [(xy[0]-min_x,xy[1]-min_y) for xy in pol]

def check_same_pol(A,B):
    return set(A)==set(B)


