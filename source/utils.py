#import system packages
import os
import sys
import warnings
#import scientific packages
import math
import numpy as np
import scipy.interpolate as sci
import itertools
flatten = itertools.chain.from_iterable
#import datetime
from datetime import date,datetime
#import plotting tools
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# date functions
def sdate2ordinal(x):
    ''' Function to take date in the format 14-Mar-2020 and turn it into date(2020,3,14)'''
    return date.toordinal(datetime.strptime(x, '%d-%b-%Y').date())


# Spline functions

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
    
# Plotting tools
def seir_plot(S,E,I,R,TS,title_add='',fig_save=0,**kwargs):
    """Function to plot all four SEIR values over the entire data recorded"""
    """
    :param list TS: Time Series (in date.toordinal format) for the entire date range
    :param list S: S(usceptibility) population
    :param list E: E(exposed) population
    :param list I: I(nfected) population
    :param list R: R(ecovered) population
    :param str title_add: text to add to the title in the format ' extra text'
    :param int fig_save: binary variable indicating whether to save a figure (1) or not (0)
    :param str fig_text: further name to add to save fig in the format '_text_' 
    """
    fig, ax1 = plt.subplots(1,1,figsize=(10,6))

    # share the x-axis for both the axes
    ax2=ax1.twinx()

    # create a plot for all four components
    function1 = ax1.plot(TS,S,'b',label='Susceptible population')
    function2 = ax2.plot(TS,E,'y',label='Exposed population')
    function3 = ax2.plot(TS,I,'r',label='Infectious population')
    function4 = ax2.plot(TS,R,'g',label='Recovered population')

    #save the figure
    functions = function1+function2+function3+function4
    labels= [f.get_label() for f in functions]
    plt.legend(functions, labels, loc=0, fontsize=15)


    # add x-label (only one, since it is shared) and the y-labels
    ax1.set_xlabel('$date$',fontsize=15)
    ax1.set_ylabel('$S(t)$',fontsize=15)
    ax2.set_ylabel('$E(t), I(t), R(t)$',fontsize=15)
    plt.sca(ax1)
    plt.xticks(ticks=TS[0::5], labels=[date.fromordinal(x) for x in TS[0::5]], rotation=45,fontsize=15)

    plt.title('SEIR'+title_add,fontsize=25)

    if fig_save==1:
        fig_text = kwargs.get('fig_text','')
        plt.savefig('SEIR'+fig_text+'.png')

def gamma_plot(I_data,I_predicted,TimeSeries, DaysHistoryFromStart, StartDate, EndDate0, EndDate1, Nodes
               , Tbreak
               , SetDates1
               , coefB
               , coefBrx
               , coefG
               , coefGrx
               , NameState
               , FileOut_name
               , FileOut_savedir
               , fig_save):
    plt.figure(figsize=[20,12])
    ax = plt.gca()
    # plot the true I curve (both known and predicted section) including true raw I data
    plt.plot(TimeSeries[0:DaysHistoryFromStart], I_predicted[0:DaysHistoryFromStart], 'r', linewidth=7)
    plt.plot(TimeSeries[DaysHistoryFromStart-1:], I_predicted[DaysHistoryFromStart-1:], 'r', linewidth=3)
    plt.plot(TimeSeries[0:DaysHistoryFromStart], I_data[0:DaysHistoryFromStart], 'b*', markersize=12)

    # expand top y-limit to allow for important date labels
    bottom,top = plt.ylim()
    plt.ylim(bottom,top+0.5*(top-bottom)) 

    # add patch: green for present data and magenda for predicted 
    plt.axvspan(EndDate0, EndDate1, facecolor="magenta", alpha=0.1, zorder=-100)
    plt.axvspan(StartDate, EndDate0, facecolor="green", alpha=0.1, zorder=-100)

    # Important dates: date, label, color, linestyle, position of label on line
    label_list = [
        (StartDate, 'Start Date:\n %s'%(date.fromordinal(StartDate)), 'b','solid', 4),
        (Nodes[1], 'First restr. measure:\n %s'%(date.fromordinal(Nodes[1])), 'b','solid', 5),
        (Nodes[2], 'Second restr. measure:\n %s'%(date.fromordinal(Nodes[2])), 'b','solid', 6),
        (EndDate0, 'TODAY:\n %s'%(date.fromordinal(EndDate0)),'g','solid', 4),
        (Tbreak, 'Third restr. measure:\n %s'%(date.fromordinal(Tbreak)), 'magenta','dashed', 5),
        (EndDate1, '', 'b','solid', 1)    
    ]
    # Plot important dates as vertical lines with labels on them
    bottom,top = plt.ylim()
    for date_point, label, clr, lstyle, label_pos in label_list:
        plt.axvline(x=date_point, color=clr, linestyle=lstyle)
        plt.text(date_point, ax.get_ylim()[1]-((label_pos)/6)*(1/3)*(top-bottom), label, 
                horizontalalignment='center',
                verticalalignment='center',
                color=clr,
                bbox=dict(facecolor='white', alpha=0.9, edgecolor=clr),
                fontsize=15)
    #Another option would be to use vlines which get the labels of each vertical line automatically added to the legend 

    ### ADD DETAILS ###
    plt.ylabel('Daily number of infected people (spreading virus)', fontsize=20)
    plt.yticks(fontsize=15)
    # Get x-valued time labels to plot
    time_labels = TimeSeries[0::3]
    time_labels = [x for x in time_labels if [abs(x-Nodes[1])>3 for x in time_labels]]
    time_labels = [x for x in time_labels if [abs(x-Nodes[2])>3 for x in time_labels]]
    time_labels = [x for x in time_labels if [abs(x-EndDate0)>3 for x in time_labels]]
    time_labels = [x for x in time_labels if [abs(x-Tbreak)>3 for x in time_labels]]
    time_labels = list(flatten([time_labels,[Nodes[1]],[Nodes[2]],[EndDate0],[Tbreak]]))
    time_labels.sort()
    plt.xticks(ticks=time_labels, labels=[date.fromordinal(x) for x in time_labels], rotation=45, fontsize=15)
    # add patch: white for user specified coefficients
    # plt.axhspan(top-4, top, facecolor="white", zorder=100)

    # insert textbox with user specified parameters
    coefBbefore = 'Coef for Transmission rate (before third restr. measure) = %s'%(str(coefB))
    coefBafter = 'Coef for Removal rate (before third restr. measure) = %s'%(str(coefG))
    coefGbefore = 'Coef for Transmission rate (after third restr. measure) = %s'%(str(coefBrx+0.8))
    coefGafter = 'Coef for Removal rate (after third restr. measure) = %s'%(str(coefGrx+0.8))
    plt.text(StartDate+0.2*(EndDate0-StartDate), ax.get_ylim()[1]-(1.5/6)*(1/3)*(top-bottom), 
            'User specified coefficients:\n   %s\n   %s\n   %s\n   %s'%(coefBbefore,coefBafter,coefGbefore,coefGafter), 
            horizontalalignment='left',
            verticalalignment='center',
            color=clr,
            bbox=dict(facecolor='white', edgecolor='b'),
            fontsize=17)

    # insert section titles
    plt.text(StartDate+0.1*(EndDate0-StartDate), top+0.1, 
             'Fit model to data for %s'%(NameState)
             , fontsize=20
             , weight='bold'
            )
    plt.text(EndDate0, top+0.1, 'Prediction of TVBG-SEIR Model', fontsize=20, weight='bold')

    # saving figure
    time_date = float([i for i in range(len(SetDates1)) if [Tbreak-y for y in SetDates1][i]==0][0])
    coefB_ix = round(coefB/0.2,0)
    coefG_ix = round(coefG/0.2,0)
    coefBrx_ix = float(math.ceil(coefBrx/0.4))
    coefGrx_ix = float(math.ceil(coefGrx/0.4))
    ExperimentNumber = 49*3*3*time_date+7*3*3*(coefB_ix-1)+3*3*(coefG_ix-1)+3*(coefBrx_ix-1)+coefGrx_ix

    if fig_save == 1:
        plt.savefig(FileOut_savedir+"/"+str(FileOut_name)+str(int(ExperimentNumber))+'.jpg')
    else:
        divergent_ix = ExperimentNumber #
    plt.close()
    try:
        divergent_ix
    except NameError:
        pass        
    else: 
        return divergent_ix
