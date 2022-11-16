import numpy as np

def calc_k(gamma, lambdaR, lambda_u):
    return np.sqrt(2*((lambdaR*2*gamma**2)/lambda_u - 1))

def calc_lambdaR(gamma, lambda_u, k):
    return lambda_u/(2*gamma**2) * (1+k**2/2)

