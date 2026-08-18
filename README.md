<img width="403" height="172" alt="image" src="https://github.com/user-attachments/assets/39547158-3d06-44f0-bd5c-930026e67ed1" />

## Introduction
This package is dedicated to constructing, searching for and characterising qudit LDPC codes. Currently it supports bivariate-bicycle and trivariate-bicycle, but this is to be extended. There is an integer programming based exact distance finder which supports warm starts and heuristic solutions, and a BP-OSD based decoder and distance finder which finds the heuristic solutions. We may also use the mathematical techniques in "Sequences of Bivariate Bicycle codes from Covering Graphs" by Symons, Rajput and Browne to generate sequences of good candidates given one good base code, as demonstrated in coversearch.py which takes in such a base code. The formulas within that can be generalised to multivariate bicycle etc. codes. 


## More information/derivations etc. 
See the resources folder for handwritten documents explaining the non-binary BP-OSD, ILP formulation etc. Will be typeset in future. 


## Usage example:

This takes the [24,4,4]_3 base code and runs a parallelised search.

```
python3 coversearch.py\
    --qudit 3 \
    --ell 4 \
    --m 3 \
    --base-d 4 \
    --A "1,0;2,0" \
    --B "3,0;0,1;0,2" \
    --factorA 1 1 \
    --factorB 1 2 2 \
    --u 3 \
    --t 3
```
    
