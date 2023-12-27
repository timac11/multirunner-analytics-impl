import numpy as np

from analytics.main import Equation, Distribution

distrib = Distribution([.4, .6])
_lambda = 2
_mu = 1

K = len(distrib)
M = 1

eq = Equation(
    M,
    distrib,
    _lambda,
    _mu,
    2
)

eq.build(last=True)
solution = eq.solve()

print(eq.matrix)

print(solution)

print(np.vstack(eq.matrices))
