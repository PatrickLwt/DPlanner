import numpy as np 

def threshold_type12(z, U, L):
    threshold = np.power((U*np.e/L), z)*L/np.e
    return threshold

def threshold_type3(z):
    threshold = 26.8 * np.power(z, 2)
    return threshold

def filter_type1(z, U, L, eps, eps0, sig):
    z0 = 1 / (1+np.log(U/L))
    if z<=z0:
        return True
    else:
        noise1 = np.random.laplace(0, 4/eps0, 1)
        noise2 = np.random.laplace(0, 2/eps0, 1)
        if sig/(eps+eps0) + noise1 > threshold_type12(z, U, L) + noise2:
            return True
    
    return False

def filter_type2(z, U, L, eps0, sig, gamma, lambd, sens):
    z0 = 1 / (1+np.log(U/L))
    if z<=z0:
        return True
    else:
        noise1 = np.random.laplace(0, 4/eps0, 1)
        noise2 = np.random.laplace(0, 2/eps0, 1)
        if sig*gamma*np.sqrt(1-lambd)/sens + noise1 > threshold_type12(z, U, L) + noise2:
            return True
    
    return False

_U = 100
_L = 0.000001
# _eps0 = 5
_beta = 1.5
_delta_g = 0.001

def schedule(schedule_type, arguments):
    initial_budget, consumed_budget, lambd, gamma, sensitivity, significance, eps0 = arguments
    # Sage
    if schedule_type == 0:
        epsilon = - sensitivity * np.log(1-lambd) / gamma
    # Infocom
    elif schedule_type == 1:
        epsilon = - sensitivity * np.log(1-lambd) / gamma
        z = consumed_budget / initial_budget
        if not filter_type1(z, _U, _L, epsilon, eps0, significance):
            epsilon = 0
    # RPBS     
    elif schedule_type == 2:
        b = (gamma / sensitivity) * (1/lambd -1 + np.sqrt((1/lambd)*(1/lambd -1)))
        epsilon = np.random.gamma(2, 1/b, 1)
        z = consumed_budget / initial_budget
        if not filter_type2(z, _U, _L, eps0, significance, gamma, lambd, sensitivity):
            epsilon = 0

    # Extended RPBS
    elif schedule_type == 3:
        c = gamma / sensitivity * significance
        z = consumed_budget / initial_budget
        T = threshold_type3(z)
        delta = (T-c)*(T-c) - 4*(_beta-1)*c*T
        if delta < 0:
            epsilon = 0
        else:
            b = (c-T+np.sqrt(delta))/(2*_beta-2)/significance
            print(b)
            if (b*(1-z)*initial_budget+1)*np.exp(-b*(1-z)*initial_budget) <= _delta_g:
                epsilon = np.random.gamma(2, 1/b, 1)
            else:
                epsilon = 0
                      
    return epsilon
        
        
