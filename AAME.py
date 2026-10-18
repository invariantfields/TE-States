from ncpol2sdpa import *
from sympy import substitution

n = 2**4 # Number of variables
level = 3 # Requested level of relaxation

x = generate_variables('x', n)

obj = -sum([x[i]*x[i] for i in range(n)])

substitutions={}

inequalities=[]

inequalities.append(sum([x[i] for i in range(n)])>=1.0)
inequalities.append(1.0 >= sum([x[i] for i in range(n)]))

for i in range(n-1):
    inequalities.append(x[i]>= x[i+1])

inequalities.append(4*x[n-1]*x[n-3]
                    -x[n-2]*x[n-2]
                    +2*x[0]*x[n-2]
                    -x[0]*x[0] >=0)

for i in range(n):
    inequalities.append(x[i]>= 0)

sdp = SdpRelaxation(x)
sdp.get_relaxation(level, objective=obj, inequalities=inequalities,substitutions=substitutions)

sdp.solve(solver='mosek')
print(sdp.primal, sdp.dual, sdp.status)
