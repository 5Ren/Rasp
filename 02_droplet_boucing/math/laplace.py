from sympy import *

init_printing(False)
(x, k) = symbols('x,k')

print(fourier_transform(exp(-x**2), x, k))
print(fourier_transform(1, x, k))
print(fourier_transform(DiracDelta(x), x, k))
