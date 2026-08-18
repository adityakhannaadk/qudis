"""logical.py
This is a core file for the construction of CSS codes.
There is a lot to add here, currently it supports only 
bicycle codes and trivariate bicycle codes, 
but it should be extended to support more general constructions.
Adapted from https://github.com/txhaug/QuditLDPCCodes/blob/main/DecodeQuditMIP.py"""
import galois
import numpy as np


def qudit_css_code(qudit,HX,HZ,get_logicals=True):
    GF = galois.GF(qudit)
    HX_GF = GF(HX)
    HZ_GF = GF(HZ)
    n=np.shape(HX)[1]
    K=n-np.linalg.matrix_rank(HX_GF)-np.linalg.matrix_rank(HZ_GF)


    def get_logicals_z(X_kernel,Z_row_space):
        ##adapted from https://pypi.org/project/bposd/
        ##extended to qudits using galois package
        log_stack=np.vstack([Z_row_space,X_kernel])
        
        ##get row echlon form
        row_reduced=(log_stack.T).row_reduce()
        
        ##list of pivots (i.e. first non-zero column in each row)
        pivots=np.argmax(row_reduced!=0,axis=1)
        pivots=pivots[:np.argmax(pivots)+1]
        
        log_op_indices=[i for i in range(Z_row_space.shape[0],log_stack.shape[0]) if i in pivots]
        log_ops=log_stack[log_op_indices]
        return log_ops


    if(get_logicals==True and K>0):
        
        
        ##check that HX and HZ commute
        assert not (HZ_GF@(HX_GF.T)).any()
        assert not (HX_GF@(HZ_GF.T)).any()
        
        
        # Calculate kernels and row spaces (mod q)
        X_kernel = HX_GF.null_space()
        X_row_space = (HX_GF.T).column_space()
        Z_kernel = HZ_GF.null_space()
        Z_row_space = (HZ_GF.T).column_space()
        
        # print(X_kernel)
        # print(Z_row_space)
        
    
        
        
        lz=get_logicals_z(X_kernel,Z_row_space)
        lx=get_logicals_z(Z_kernel,X_row_space)
        
        assert n==HZ.shape[1]==lz.shape[1]==lx.shape[1]
        assert K==lz.shape[0]==lx.shape[0]
        
        ##check that logicals in kernel
        assert not (HZ_GF@lx.T).any()
        assert not (HX_GF@lz.T).any()
        
        ##check that logical operators span K logical qubits
        assert np.linalg.matrix_rank((lx@lz.T))==K
        
        
    else:
        lz=[]
        lx=[]
        
    return n,K,HZ,HX,lz,lx

#def get_bicycle_code(ell,m,poly_A,poly_B,qudit=2,factor_A_in=[],factor_B_in=[]):

def get_bicycle_code(code_dict):
    print(code_dict)
    ell=code_dict["ell"]
    m=code_dict["m"]
    poly_A=code_dict["A"]
    poly_B=code_dict["B"]
    if("qudit" in code_dict.keys()):
        qudit=code_dict["qudit"]
        assert qudit!=4 ##not defined for prime power!

    else:
        qudit=2
    
    if("g" in code_dict.keys()):
        g=code_dict["g"]
    else:
        g=1
        
        
    if("factorA" in code_dict.keys() and len(code_dict["factorA"])>0):

        factor_A=code_dict["factorA"]
        factor_B=code_dict["factorB"]
        assert len(poly_A)==len(factor_A)
        assert len(poly_B)==len(factor_B)
    else:
        factor_A=[1]*len(poly_A)
        factor_B=[1]*len(poly_A)
        
    ####factor in front of HX=gamma1*A | gamma2*B, HZ=delta1*B^T | delta2*A^T

    if("gamma1" in code_dict.keys()):
        gamma1=code_dict["gamma1"]
        gamma2=code_dict["gamma2"] 
        delta1=code_dict["delta1"]
        delta2=code_dict["delta2"]
        ##from CSS

        assert (gamma1 * delta1 + gamma2 * delta2)%qudit == 0
    else:
        gamma1=1
        gamma2=1 
        delta1=1
        delta2=-1 ##default -1 for qudit>2, can be set equivalently to 1 for qudit=2

        assert (gamma1 * delta1 + gamma2 * delta2)%qudit == 0
        

    n = 2*ell*m
    

    # define cyclic shift matrices 

    I_ell = np.identity(ell,dtype=np.int8)
    I_m = np.identity(m,dtype=np.int8)
    
    if(g>1):
        I_g = np.identity(g,dtype=np.int8)
    
    if(type(code_dict["A"][0])==str):
        ##old format as ("x2","y3",...)

        A_terms=[]
        for i in range(len(code_dict["A"])):
            if(code_dict["A"][i][0]=="x"):
                A_terms.append((int(code_dict["A"][i][1:]),0))
            elif(code_dict["A"][i][0]=="y"):
                A_terms.append((0,int(code_dict["A"][i][1:])))
            else:
                raise NameError("not defined")
                
        B_terms=[]
        for i in range(len(code_dict["B"])):
            if(code_dict["B"][i][0]=="x"):
                B_terms.append((int(code_dict["B"][i][1:]),0))
            elif(code_dict["B"][i][0]=="y"):
                B_terms.append((0,int(code_dict["B"][i][1:])))
            else:
                raise NameError("not defined")
                
    else:
        ##new format as list of [(x,y),(x,y),...]

        A_terms=code_dict["A"]
        B_terms=code_dict["B"]

    A=np.zeros([ell*m*g,ell*m*g],dtype=np.int8)
    for i in range(len(A_terms)):
      xp=np.kron(np.roll(I_ell, A_terms[i][0], axis=1), I_m)
      yp=np.kron(I_ell, np.roll(I_m, A_terms[i][1], axis=1))
      A+=(factor_A[i]*np.dot(xp,yp))%qudit

    B=np.zeros([ell*m*g,ell*m*g],dtype=np.int8)
    for i in range(len(B_terms)):
      xp=np.kron(np.roll(I_ell, B_terms[i][0], axis=1), I_m)
      yp=np.kron(I_ell, np.roll(I_m, B_terms[i][1], axis=1))
      B+=((factor_B[i])*np.dot(xp,yp))%qudit
    
    HX = np.hstack((gamma1*A, gamma2*B)).astype(np.int8) % qudit
    HZ = np.hstack((delta1*np.transpose(B), delta2*np.transpose(A))).astype(np.int8) % qudit
    
    return HX,HZ

def get_trivariate_bicycle_code(code_dict):
    print(code_dict)
    ell=code_dict["ell"]
    m=code_dict["m"]
    poly_A=code_dict["A"]
    poly_B=code_dict["B"]
    if("qudit" in code_dict.keys()):
        qudit=code_dict["qudit"]
        assert qudit!=4 ##not defined for prime power!

    else:
        qudit=2
    
    if("g" in code_dict.keys()):
        g=code_dict["g"]
    else:
        g=1

    if("s" in code_dict.keys()):
        s=code_dict["s"]
    else:
        s=1

    assert g >= 1 and s >= 1
        
    if("factorA" in code_dict.keys() and len(code_dict["factorA"])>0):

        factor_A=code_dict["factorA"]
        factor_B=code_dict["factorB"]
        assert len(poly_A)==len(factor_A)
        assert len(poly_B)==len(factor_B)
    else:
        print("factorA, factorB not found in dict")
        factor_A=[1]*len(poly_A)
        factor_B=[1]*len(poly_A)
        
    ####factor in front of HX=gamma1*A | gamma2*B, HZ=delta1*B^T | delta2*A^T

    if("gamma1" in code_dict.keys()):
        gamma1=code_dict["gamma1"]
        gamma2=code_dict["gamma2"] 
        delta1=code_dict["delta1"]
        delta2=code_dict["delta2"]
        ##from CSS

        assert (gamma1 * delta1 + gamma2 * delta2)%qudit == 0
    else:
        gamma1=1
        gamma2=1 
        delta1=1
        delta2=-1 ##default -1 for qudit>2, can be set equivalently to 1 for qudit=2

        assert (gamma1 * delta1 + gamma2 * delta2)%qudit == 0
        

    n = 2*ell*m*s*g
    

    # define cyclic shift matrices 

    I_ell = np.identity(ell,dtype=np.int8)
    I_m = np.identity(m,dtype=np.int8)
    I_s = np.identity(s,dtype=np.int8)
    I_g = np.identity(g,dtype=np.int8)

    def monomial_matrix(monomial):
        ##monomial is a tuple of (x,y,z) shifts
        xp=np.kron(np.kron(np.kron(np.roll(I_ell, monomial[0], axis=1), I_m), I_s), I_g)
        yp=np.kron(np.kron(np.kron(I_ell, np.roll(I_m, monomial[1], axis=1)), I_s), I_g)
        zp=np.kron(np.kron(np.kron(I_ell, I_m), np.roll(I_s, monomial[2], axis=1)), I_g)
        return xp @ yp @ zp

    if(type(code_dict["A"][0])==str):
        ##old format as ("x2","y3",...)

        A_terms=[]
        for i in range(len(code_dict["A"])):
            if(code_dict["A"][i][0]=="x"):
                A_terms.append((int(code_dict["A"][i][1:]),0))
            elif(code_dict["A"][i][0]=="y"):
                A_terms.append((0,int(code_dict["A"][i][1:])))
            else:
                raise NameError("not defined")
                
        B_terms=[]
        for i in range(len(code_dict["B"])):
            if(code_dict["B"][i][0]=="x"):
                B_terms.append((int(code_dict["B"][i][1:]),0))
            elif(code_dict["B"][i][0]=="y"):
                B_terms.append((0,int(code_dict["B"][i][1:])))
            else:
                raise NameError("not defined")
                
    else:
        ##new format as list of [(x,y),(x,y),...]

        A_terms=code_dict["A"]
        B_terms=code_dict["B"]

    A=np.zeros([ell*m*s*g,ell*m*s*g],dtype=np.int8)
    for i in range(len(A_terms)):
      monomial=A_terms[i]
      mon_mat=monomial_matrix(monomial)
    
      A+=(factor_A[i]*mon_mat)%qudit

    B=np.zeros([ell*m*s*g,ell*m*s*g],dtype=np.int8)
    for i in range(len(B_terms)):
        monomial=B_terms[i]
        mon_mat=monomial_matrix(monomial)
        try:
            B+=((factor_B[i])*mon_mat)%qudit
        except:
            print("factor B. then its length, then B terms, then its length, then i")
            print(factor_B)
            print(len(factor_B))
            print(B_terms)
            print(len(B_terms))
            print(i)
            raise ValueError

    
    HX = np.hstack((gamma1*A, gamma2*B)).astype(np.int8) % qudit
    HZ = np.hstack((delta1*np.transpose(B), delta2*np.transpose(A))).astype(np.int8) % qudit
    
    return HX,HZ
