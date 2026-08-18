import numpy as np
import time

try:
    from cuopt.linear_programming.problem import Problem, INTEGER, MINIMIZE
    from cuopt.linear_programming.solver_settings import SolverSettings
except ImportError:  # pragma: no cover - exercised in environments without CuOpt
    Problem = None
    INTEGER = None
    MINIMIZE = None
    SolverSettings = None


##Tip: add random stabilizer to logicOp, this greatly enhances stability and convergence
def heuristic_distance_test_qudit(stab, logicOp, qudit, max_time=None):
    
    assert qudit!=4 ##not properly impelemtned for non-prime qudit numbers...

    if Problem is None or SolverSettings is None:
        raise ImportError("CuOpt is required to solve distance problems with this implementation")

    logicOp_tmp=np.array(logicOp)
    
    stab_tmp=np.array(stab)
    
    start_time=time.time()
    # number of qubits

    n = stab_tmp.shape[1]
    # number of stabilizers

    m = stab_tmp.shape[0]

    # maximum stabilizer weight

    wstab = np.max([np.sum((stab_tmp[i,:]%qudit)!=0) for i in range(m)])
	# weight of the logical operator

    wlog = np.count_nonzero(logicOp_tmp)
	# how many slack varia33bles are needed to express orthogonality constraints modulo qudit

    
    ##does this need to increase for qudits?

    ##factor (qudit-1)**2 comes from fact that largest possible value is (qudit-1)**2 

    ##this is from stabilizer check weight qudit-1, and qudit itsself has qudit-1 as mixmal value

    ##as we take product, we get maximally (qudit-1)**2 over at most wstab qubits

    
    #print(stab)

    num_anc_stab = int(np.ceil(np.log((qudit-1)**2*wstab)/np.log(qudit)))
    num_anc_logical = int(np.ceil(np.log((qudit-1)**2*wlog)/np.log(qudit)))
    #print(num_anc_stab,num_anc_logical)

	# total number of variables

    num_var = n + m*num_anc_stab + num_anc_logical

    model = Problem("Qudit Distance Test")

    ##variables,
    ##restriction between 0 and qudit-1
    x = [model.addVariable(vtype=INTEGER, lb=0, ub=qudit-1, name=f"x_{i}") for i in range(num_var)]
    
    
    #x = [model.add_var(var_type=CONTINUOUS) for i in range(num_var)]

    ##minimze weight on non-slack variables

    ##we want to have the logical error with minimal support

    ##there is no difference between 1,2,..,qudit-1 values, so we need to add the !=0

    ##if x is 0, it is counted as False=0, else 1

    if(qudit==2):
        model.setObjective(sum(x[i] for i in range(n)), sense=MINIMIZE)
    
        
    else:
        # ##the constraint must be chosen such that when x[i]!=0, the minimziation yields 1, else 0

        # ##cannot use miniization over x[i] for qudits as it can take higher values!

        
        
        y=[[model.addVariable(vtype=INTEGER, lb=0, ub=1, name=f"y_{i}_{j}") for j in range(qudit-1) ] for i in range(n)]
        
        
        for i in range(n):
            ##ensures that ys are 0 when x=0, else one of the ys>0

            model.addConstraint(x[i] - sum((j+1)*y[i][j] for j in range(qudit-1)) == 0)
            
            ##this ensures that only one y is triggered at a time, so that we can use as a hamming weight

            model.addConstraint(sum(y[i][j] for j in range(qudit-1)) <= 1)
            
        ##this is 0 when x=0, else 1

        ##this is exactly the hamming weight, i.e. indicating whether x[i] differs from 0

        model.setObjective(sum(sum(y[i][j] for j in range(qudit-1)) for i in range(n)), sense=MINIMIZE)

        
        # ##the constraint must be chosen such that when x[i]!=0, the minimziation yields 1, else 0

        # ##cannot use miniization over x[i] for qudits as it can take higher values!

        
        


    weight_stab=[]
	# orthogonality to rows of stab constraints

    for row in range(m): ##go through stabilizers

        weight = np.zeros(num_var,dtype=int)#[0]*num_var

        # supp = np.nonzero(stab[row,:])[0] ##support of stabilizer

        # #print(supp)

        # for q in supp:

        #     weight[q] = stab[row,q]%qudit

            
        weight[:n]=stab_tmp[row,:]%qudit
        ##slack variables to account for modulo 2

        cnt = 1
        for q in range(num_anc_stab):
            ##slack variables which give modulo qudit

            weight[n + row*num_anc_stab +q] = -(qudit**cnt)#(1<<cnt)## -2**cnt

            cnt+=1
        ##should commute with stabilizer %qudit

        model.addConstraint(sum(weight[i] * x[i] for i in range(num_var)) == 0)
        weight_stab.append(list(weight))




    #print(logicOp_tmp)

	# non-zero overlap with logicOp constraint

    ##anti-commute with logical


    #weight = [0]*num_var

    weight = np.zeros(num_var,dtype=int)
    
    #    supp = np.nonzero(logicOp_tmp)[0]

    # for q in supp:

    #     weight[q] = logicOp_tmp[q]%qudit

        
        
    weight[:n]=logicOp_tmp
        
    ##slack variables to account for modulo 2

    cnt = 1
    for q in range(num_anc_logical):
        ##slack variables which give modulo qudit

        weight[n + m*num_anc_stab +q] = -(qudit**cnt)#-(1<<cnt)

        cnt+=1
    #print(weight)

    ##the anti-commutation condition, now must not be 0 %qudit

    model.addConstraint(sum(weight[i] * x[i] for i in range(num_var)) >= 1)
    model.addConstraint(sum(weight[i] * x[i] for i in range(num_var)) <= qudit-1)
        
    # for i in range(num_var):

    #     model += xsum([x[i]])>=0

    #     model += xsum([x[i]])<=1

    
    #max_time=0
    #using cuda cuopt:
    settings = SolverSettings()
    if max_time is None or max_time in (np.inf, float("inf")):
        max_time = 60
    settings.set_parameter("time_limit", int(max_time))
    model.solve(settings)


    
    #print(weight)

    #print(weight_stab)


    #print([x[i].x for i in range(n)])


    ##we want to have the logical error with minimal support

    ##there is no difference between 1,2,..,qudit-1 values, so we need to add the !=0

    ##if x is 0, it is counted as False=0, else 1

    opt_val = int(sum(1 for i in range(n) if x[i].getValue() is not None and float(x[i].getValue()) != 0.0))
    

    return opt_val
