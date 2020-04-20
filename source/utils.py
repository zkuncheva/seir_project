import sys
import warnings
import numpy as np
import scipy.interpolate as sci
import itertools
flatten = itertools.chain.from_iterable


def steeper(V,I,K):
    '''Function to make exp steeper a decreasing function V in the interval [a,b]'''
    """
    :param array V: array composed of the function defined on a grid on the interval I=[a,b]
    :param array I: interval I=[a,b]
    :param float K: K is the steepness in exp(-K*x)
    :return: list of values for steeper V values
    :rtype: list
    """
    a = I[0]
    b = I[1]
    step = (b-a)/(len(V)-1)
    grid = np.arange(a,b+0.1*step,step).tolist()
    Va = [x-V[-1] for x in V]
    Vb = np.multiply(np.exp([-K*(x-a) for x in grid]).tolist(),Va)
    Vb = Vb+V[-1]
    return Vb

def interp_steeper(Nodes,paramNode,TS,SPLINE,K=0.4):
    """Function to interpolate by making steeper increasing or decreasing pieces of the spline.
        'pchip' method extrapolates with constant to the right    
    """
    '''    
    :param array Nodes: length-4 list composed of four dates [StartDate,DateOfImposedSanctions1,DateOfImposedSanctions2,EndDate0]
    :param array betaNode: length-4 list composed of estimated beta values
    :param array TS: time series of float values indicating all dates with available data
    :param str SPLINE: method for interpolation 'spline' or 'pchip' ! Currently developed only for 'pchip'
    :param float K: K=0.4 by default
    '''
    # obtain spline
    if SPLINE == 'spline':
        warnings.warn("The 'spline' calculation is not currently implemented. The method will default to 'pchip'.")
        SPLINE = 'pchip'
    if SPLINE == 'pchip':
        paramVal = sci.pchip_interpolate(Nodes,paramNode,TS)
        paramVal_new = paramVal.copy()
    else:
        sys.exit('SPLINE method not recognized: "spline" and "pchip" available only')
    time_interval = [x-Nodes[0] for x in Nodes]
    #start modifying all parts of the spline and make them steeper
    for i in range(len(Nodes)-1):
        if paramNode[i] > paramNode[i+1]:
            time_range = range(time_interval[i],time_interval[i+1]+1)
            I=[time_interval[i],time_interval[i+1]]
            steeper_segment=steeper(paramVal[time_range],I,K)
            paramVal_new[time_range]=steeper_segment #we allow overlap due to interpolation
        else:
            time_range = range(time_interval[i],time_interval[i+1]+1)
            I=[time_interval[i],time_interval[i+1]]
            steeper_segment = np.array([-x for x in steeper([-x for x in paramVal[time_range]], I, K)]);
            paramVal_new[time_range]=steeper_segment #we allow overlap due to interpolation
    return paramVal_new

# Functions F0, F1, F2 to calculate iteratively the SEIR values
def F0(t,X,N,beta,gamma,sigma):
    """F0 Function to calculate SEIR parameters from simultaneous equations F0,F1,F2"""
    """
    :param list X: X = [S0,E0,I0,R0]
    :param int t: time point index
    """
    S2 = X[0]; E2 = X[1]; I2 = X[2]; R2 = X[3]
    S1 = -beta[t]*S2*I2/N
    E1 = beta[t]*S2*I2/N - sigma*E2
    I1 = sigma*E2 - gamma[t]*I2
    R1 = gamma[t]*I2
    return S1,E1,I1,R1

def F1(t,X,N,beta,gamma,sigma):
    """F1 Function to calculate SEIR parameters from simultaneous equations F0,F1,F2"""
    """
    :param list X: X = [S0,E0,I0,R0]
    :param int t: time point index
    """
    S2 = X[0]; E2 = X[1]; I2 = X[2]; R2 = X[3]
    S1 = -0.5*(beta[t]+beta[t+1])*S2*I2/N
    E1 = 0.5*(beta[t]+beta[t+1])*S2*I2/N - sigma*E2
    I1 = sigma*E2 - 0.5*(gamma[t]+gamma[t+1])*I2
    R1 = 0.5*(gamma[t]+gamma[t+1])*I2
    return S1,E1,I1,R1

def F2(t,X,N,beta,gamma,sigma):
    """F2 Function to calculate SEIR parameters from simultaneous equations F0,F1,F2"""
    """
    :param list X: X = [S0,E0,I0,R0]
    :param int t: time point index
    """
    S2 = X[0]; E2 = X[1]; I2 = X[2]; R2 = X[3]
    S1 = -beta[t+1]*S2*I2/N
    E1 = beta[t+1]*S2*I2/N - sigma*E2
    I1 = sigma*E2 - 0.5*gamma[t+1]*I2
    R1 = 0.5*gamma[t+1]*I2
    return S1,E1,I1,R1

#SEIR prediction function
def seir_spline_predict(StartDate, EndDate0, EndDate1, Tbreak, ctrBeta, ctrGamma, Nodes, betaNode, gammaNode, N, S0, E0, I0, R0, SPLINE, Sigma, RK, **kwargs):
    """Function to predict SEIR model coefficients and gamma parameter"""
    '''
    :param float StartDate: first date available for modelling; date is modelled as ordinal number
    :param float EndDate0: the first end date where spline is modelled up to
    :param float EndDate1: end date, which is either current date or end date for prediction,i.e. EndDate1=EndDate0+50days
    :param float Tbreak: varying dates of interest; recommended to be EndDate0<Tbreak<=EndDate1
    :param array ctrBeta: length-2 list composed of starting Beta parameters
    :param array ctrGamma: length-2 list composed of starting Gamma parameters
    :param array Nodes: length-4 list composed of four dates [StartDate,DateOfImposedSanctions1,DateOfImposedSanctions2,EndDate0]
    :param array betaNode: length-4 list composed of estimated beta values
    :param array gammaNode: length-4 list composed of estimated gamma values
    :param float Sigma: sigma
    :param float N: size of population
    :param float S0: initial S value
    :param float E0: initial E value
    :param float I0: initial I value
    :param float R0: initial R value
    :param str SPLINE: type of interpolation 'spline' or 'pchip'
    :param int RK: method to calculate SEIR values - 0 for Euler method, 1 for Runge-Kutta 
    '''
    # capture **kwargs arguments and assign default values
    K=kwargs.get('K',0.4)
    tiny_constant=kwargs.get('tiny_constant',0.00000001)

    # create time series range
    TS = range(StartDate, EndDate1+1)
    T = len(TS) # length of time series
    S = [S0]; E = [E0]; I = [I0]; R = [R0];
    if SPLINE == 'spline':
        warnings.warn("The 'spline' calculation is not currently implemented. The method will default to 'pchip'.")
        SPLINE = 'pchip'
        """
        # code for 'spline' method
        #get beta starting coefficients incl ctrBeta values
        betaStart = np.asarray(list(flatten([betaNode , ctrBeta ])))
        betaStart[betaStart == 0] = tiny_constant # make sure no initial coefficients are zero
        betaStart = betaStart.tolist()
        #get gamma starting coefficients incl ctrGamma values
        gammaStart = np.asarray(list(flatten([gammaNode , ctrGamma ])))
        gammaStart[gammaStart == 0] = tiny_constant # make sure no initial coefficients are zero    
        gammaStart = gammaStart.tolist()
        #add Tbreak values and final EndDate to get interolation and its steeper version for beta and gamma
        allNodes = list(flatten([Nodes,[Tbreak],[EndDate1]]))
        beta =  np.exp(interp_steeper(allNodes,[log(x) for x in betaStart],TS,SPLINE,K))
        gamma = np.exp(interp_steeper(allNodes,[log(x) for x in gammaStart],TS,SPLINE,K))
        """
    if SPLINE == 'pchip':
        #get beta starting coefficients incl ctrBeta values
        betaStart = np.asarray(list(flatten([betaNode , ctrBeta ]))).tolist()
        #get gamma starting coefficients incl ctrGamma values
        gammaStart = np.asarray(list(flatten([gammaNode , ctrGamma ]))).tolist()
        #add Tbreak values and final EndDate to get interolation and its steeper version for beta and gamma
        allNodes = list(flatten([Nodes,[Tbreak],[EndDate1]]))
        beta = interp_steeper(allNodes,betaStart,TS,SPLINE,K) 
        gamma = interp_steeper(allNodes,gammaStart,TS,SPLINE,K) 
    else:
        sys.exit('SPLINE method not recognized: "spline" and "pchip" available only') 
    if RK == 0: #for euler method of calculating S,E,I,R values
        for time_ix in range(T-1):
            S.append(S[-1] - beta[time_ix]*S[-1]*I[-1]/N)
            E.append(E[-1] + beta[time_ix]*S[-2]*I[-1]/N - Sigma*E[-1])
            I.append(I[-1] + Sigma*E[-2] - gamma[time_ix]*I[-1])
            R.append(R[-1] + gamma[time_ix]*I[-1])
    else: #use Runge-Kutta method
        for time_ix in range(T-1):
            SEIR0 = F0(time_ix,[S[-1],E[-1],I[-1],R[-1]],N,beta,gamma,Sigma)
            SEIR11 = F1(time_ix,[x + y/2 for x,y in zip([S[-1],E[-1],I[-1],R[-1]],SEIR0)],N,beta,gamma,Sigma)
            SEIR12 = F1(time_ix,[x + y/2 for x,y in zip([S[-1],E[-1],I[-1],R[-1]],SEIR11)],N,beta,gamma,Sigma)
            SEIR2 = F2(time_ix,[x + y for x,y in zip([S[-1],E[-1],I[-1],R[-1]],SEIR12)],N,beta,gamma,Sigma)
            SEIRTotal =  [x + (1/6)*(y+2*z+2*u+v) for x,y,z,u,v in zip([S[-1],E[-1],I[-1],R[-1]],SEIR0,SEIR11,SEIR12,SEIR2)]  
            S.append(SEIRTotal[0])
            E.append(SEIRTotal[1])
            I.append(SEIRTotal[2])
            R.append(SEIRTotal[3])
    return S,E,I,R,gamma
    
