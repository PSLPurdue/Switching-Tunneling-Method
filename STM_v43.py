"""
===============
Originally based on:
    Adaptive Memory Programming for Global Optimization (AMPGO).
    
    added to lmfit by Renee Otten (2018)
    
    based on the Python implementation of Andrea Gavana
    (see: http://infinity77.net/global_optimization/)
    
    Implementation details can be found in this paper:
        http://leeds-faculty.colorado.edu/glover/fred%20pubs/416%20-%20AMP%20(TS)%20for%20Constrained%20Global%20Opt%20w%20Lasdon%20et%20al%20.pdf
===============

Significantly modified using techniques from:
    S. Gomez and C. Barron, “The exponential tunneling method,” Ser. Reportes Investig. IIMAS, vol. 1, no. 3, 1991.
    A. V. Levy and Susana, “The Tunneling Method Applied to Global Optimization,” in Numerical Optimization, P. T. Boggs, R. H. Byrd, and R. B. Schnabel, Eds. New York: Society for Industrial & Applied Mathematics, 1985, pp. 213–244.
    L. Castellanos and S. Gómez, “A new implementation of the tunneling methods for bound constrained global optimization,” Rep. Investig. IIMAS, vol. 10, no. 59, pp. 1–18, 2000.
    J. Nocedal and S. J. Wright, Numerical Optimization, 2nd ed. 2006.
==============
    
Modifications include:
    Use of exponential tunneling function rather than tabu   
    Tunneling zero search method implemented according to Gomez and Barron
    Wolfe criteria for step size
    Implementation of log barrier constraints with iterative reduction of mu and
        associated slack and tolerance factors
    Augmented Lagrangian Method to enforce equality constraints
    Iteration history plotting available
    Switching Method to use primary and secondary function transformations

Many of the parameters (minimum step sizes, tolerances, slack, etc) are problem
dependent and must be tested/adjusted accordingly.
    
See paper:
Inverse Design of Bistable Composite Laminates with Switching
Tunneling Method for Global Optimization
Katherine S. Riley, Mark H. Jhon, Hortense Le Ferrand, Dan Wang, Andres F. Arrieta

==============
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. 
==============

"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import minimize, newton, root, Bounds, fmin_l_bfgs_b,fmin_slsqp
from scipy.linalg import norm
import time
import switchcount2
from numba import jit

SCIPY_LOCAL_SOLVERS = ['Nelder-Mead', 'Powell', 'L-BFGS-B', 'TNC', 'SLSQP','Newton-CG'] #added newton-cg nye 2019
global f0, sc

def ampgo(objfun, x0, grad=None,hess=None,local='SLSQP',
          local_opts=None, boundsin=None,
          maxfunevals=None, totaliter=20, maxiter=10, alphastart=1, tuniter=100,
          tunphases=0,
          glbtol=1e-3, 
          disp=False,plot=False, 
          fixedvar=None,objargs=None,slack=0.0,
          minsec=None,minsecgrad=None,tunfun=None,
          tunfungrad=None,tunfunsec=None,tunfunsecgrad=None,
          f0fun=None,confun=None,initlambda=False,l0=1,lt0=0,maxrounds=1,
          plotfuns=[],plotfunscontranges=[],Lmin=np.inf,Leps=0,
          timelimit=100000,trackfunc=[],
          Lmultmin=10,Lmulttun=1,alphamu=1.05,tunstrategy=1,minrounds=5,
          stopfunc=None,almiters=1,cutoff=1.0,dm=1,ineq_min = [],
          grad_ineq_min = []):
    # added grad, hess 30 Dec 2019
    """Search for one or multilple global minima using the tunneling algorithm.

    Parameters
    ----------
    objfun: callable
        Objective function to be minimized. objfun(x, *args)
    x0: numpy.ndarray
         Initial guess as starting point
    grad: callable
        Gradient of objective function. grad(x,*args)
    hess: callable
        Hessian of objective function. hess(x,*args)
    local: str, optional
        Local minimization method. 'L-BFGS-B' or 'SLSQP' 
    bounds: sequence, optional
        List of tuples of the lower and upper bound for each
        variable [(`xl0`, `xu0`), (`xl1`, `xu1`), ...].
        Only used in minimization phase.
    maxfunevals: int, optional
        Maximum number of function evaluations. If None, the optimization will
        stop after `totaliter` number of iterations.
    totaliter: int, optional
        Maximum number of global iterations.
    tunphases: int, optional
        Maximum number of tunneling phases per start point, i.e. maximum 
        number of random start points to try
    glbtol: float, optional
        Tolerance whether or not to accept a solution after a tunneling phase.
    disp: bool, optional
        Set to True to print convergence messages.
    slack: tunneling slack variable
    objargs: tuple
        Arguments to be passed to objective function and its gradient in 
        minimization phase. Duplicate and change to lower min for tunneling.
    Lmin: minimum value for lambda to reach
    Leps: small epsilon value to increase lambda value
    timelimit: wall clock time limit in seconds, evaluated at minimization phase
    Lmultmin: multiplying factor to increase Lagrange multiplier each round for minimzation
    Lmulttun: multiplying factor to increase Lagrange multiplier each round for tunneling
    tunstrategy: 1-just alpha*r, 2 - alpha*r for 2N, then random to max
    almiters: whether or not to iterate alpha, muL

    Returns
 
    """
    timestart = time.time()
    # global f0, switchcount
    if local not in SCIPY_LOCAL_SOLVERS:
        raise Exception('Invalid local solver selected: {}'.format(local))

    x0 = np.atleast_1d(x0)
    print('x0 = {0}'.format(x0))
    n = len(x0)

    if boundsin is None:
        bounds = [(None, None)] * n
    else:
        bounds = boundsin
    if len(bounds) != n:
        raise ValueError('length of x0 != length of bounds')

    bounds = [b if b is not None else (None, None) for b in bounds]
    _bounds = [(-np.inf if l is None else l, np.inf if u is None else u)
               for l, u in bounds]
    low, up = np.array(_bounds).T
  
    if maxfunevals is None:
        maxfunevals = 1e6 #SLSQP doesn't set to np.inf 
    
    # Initial values
    x0_0 = x0.copy()
    Lbound1 =  objargs[1] #bounds[-1][0] #should match objargs[1]
    Lboundt =  objargs[1] #bounds[-1][0]
    # Lbound1start = bounds[-1][0] # not used
    # bounds[-1] = (Lbound1 + Leps,np.inf)
    if Lbound1 != objargs[1]:
        print('Warning: lambda bounds and argument vector do not match!')
    minlist = [x0]
    startptlist = [x0]
    tabulist = []
    startptlist_hist = [startptlist]
    best_f = np.array([np.inf])#np.array([objfun(x0,*objargs)])#np.array([np.inf])
    best_x = x0
    best_x_hist = [] # adding tracker of all best x found
    best_x_hist.append(best_x)
    best_f_hist = [] # adding tracker of all best f found
    best_f_hist.append(best_f[0])
    round_hist = [0]
    tunphase_hist = []
    trackfunc_hist = []
    trackfunc_hist = record_trackfuncs(x0,objargs,trackfunc,trackfunc_hist)
    mulist = []
    lambda_laglist = []
    rplist = []
    tuncount = 0 # count how many successful tunneling phases
    minhist = {} # dictionary to track minimization phase history
    minhist_all = {}

    global_iter = 0
    all_tunnel = success_tunnel = 0
    evaluations = 0

    local_tol = min(1e-2, glbtol)
    
    figcount = 0
    mumin = list(objargs)[0] #problem-dependent 
    mu = list(objargs)[0] #first value
    lambda_lag = list(objargs)[1] # lagrange multiplier lambda
    #rp = list(objargs)[2] # lagrange multiplier penalty rp
    mucount = 1
    
    # Different sets of args
    argsmin = objargs #input arguments - use in min phase
    argscopy = list(objargs)
    argstun = tuple(argscopy) # copy of args to use in tun phase
    
    warn_iter = 0 # Warn if tunneling found new pts that weren't used bc of max iter limit
    icount = 0
    if mu < mumin:
        print('Warning: starting mu value is less than minimum value.')
    # The outer loop iteratively decreases mu (log barrier multiplier)
    while check_stop(icount,maxrounds,timestart,timelimit,minlist,stopfunc,*argsmin):
        slack = slack # problem-dependent (how close are min to constraints?)
        # glbtol = 1*10**-(icount+3)# problem-dependent !!!
        # Update mu and lambda_min values in Min Phase args
        arglst = list(argsmin) 
        arglst[0] = mu
        arglst[1] = Lbound1 # SET NEW LOWER BOUND
        #arglst[2] = rp
        # arglst[3] = muLk1
        argsmin = tuple(arglst)
        # Update mu in tun args
        arglstt = list(argstun)
        arglstt[0] = mu
        argstun = tuple(arglstt)
        tp_total = tunphases
        tp_used = 0

        # Initialize lists for start points, minima, tabu pts
        startptlistnew = []

        if icount > 0: #if trying another random pt, keep minlist and tabulist(?), best_f
            boundlow = []
            boundhigh = []
            if boundsin == None:
                x0 = np.random.uniform(size=[1,len(x0_0)])[0]
            else:
                for bb in bounds:
                    boundlow.append(bb[0])
                    boundhigh.append(bb[1])
                x0 = np.random.uniform(low=boundlow,high=boundhigh,size=[1,len(x0_0)])[0]
            startptlist = [x0]
            for mm in minlist:
                startptlist.append(mm)
            startptlistnew = minlist.copy() #?
            # startptlist.append(-1*minlist)
            print('\nNew x0: ',x0)

        # '''
        
        startcount = 1
        print('\n{0}\nRound #{1:d} \n Current mu: {2} \n Current lambda_lag: {4:3.2f} \n '
              'Current startptlist = {3} \n{0}'
                          .format('='*50,icount,mu,np.round(startptlist,2),Lbound1))
        # Investigate each start point (points of interest identified at previous
        # mu iteration)
        for startpt in startptlist:
            nonewpt = 0;
            x0 = startpt
            stepcount = 1
            esc_attempt = 0; # how many times you can get repeat min
            findzero = (np.array([0]),np.inf,False) #initialize
            tun_total_iter = 0
            while nonewpt == 0 and check_stop(tun_total_iter,totaliter,timestart,timelimit,minlist,stopfunc,*argsmin):
                # minimization to find local minimum, either from initial values or
                # after a successful tunneling loop
                
                # ===============================================
                xstarn1, fstarn1, minhist, argsmin2, num_fun = perform_minimization(x0,\
                                     objfun,grad,minsec,minsecgrad,argsmin,\
                                     minlist,best_x,best_f,mucount,\
                                     startcount,stepcount,ineq_min,grad_ineq_min,\
                                     bounds=bounds,\
                                     minrounds=minrounds,disp=True,\
                                     maxfunevals=2000,confun=confun,alphamu=alphamu,\
                                     local=local,trackfunc=trackfunc)
                # ===============================================
                        
                minhist_all[global_iter] = minhist.copy()
                
                # Update f0
                f0_update = objfun(xstarn1,*objargs) # New f0
                arglst = list(argsmin) 
                arglst[2] = f0_update
                argsmin = tuple(arglst)
                x0copy = x0.copy()
                # x0copy[-1] = Lk2 + Leps
                xf, yf = choosemin(xstarn1,x0copy,objfun,argsmin2)

                if isinstance(yf, np.ndarray):
                    yf = yf[0]
        
                maxfunevals -= num_fun
                evaluations += num_fun
                # new min
                trytunnel = True
                best_x, best_f, best_x_hist, best_f_hist, minlist,\
                    startptlistnew,round_hist, trackfunc_hist, tp_total,\
                    nonewpt, trytunnel, success_tunnel, esc_attempt = evaluate_results(xf,yf,\
                                     best_x,best_f,\
                                     lt0,minlist,tabulist,startptlistnew, \
                                     argsmin2,trackfunc,trackfunc_hist,\
                                     best_x_hist,best_f_hist,round_hist,icount,\
                                     glbtol,tp_total,tp_used,findzero,\
                                     success_tunnel,tunphases,\
                                     esc_attempt,nonewpt,trytunnel)
                        
                # if maxfunevals <= 0: #!!!
                #     #nonewpt = 1
                #     if disp: print('='*30) #changed from 72 to 30            
                   
                # Tunneling
                if trytunnel == True and tunphases != 0 and check_stop(icount,maxrounds,timestart,timelimit,minlist,stopfunc,*argsmin):
                    # try:#!!!
    
                    topphases = str(mucount) + '-' + str(startcount) + '-' + str(stepcount)

                    x0tun = xf.copy()
                    minlisttun = minlist # use numpy copy for list of arrays
                    tabulisttun = np.copy(tabulist) # use numpy copy for list of arrays

                    # Update f0
                    if f0fun != None:
                        xf0 = best_x.copy()
                        # xf0[-1] = argstun[1] + 0.1
                        f0tun = f0fun(xf0,*argstun)
                        print('f0tun: ',f0tun)
                        arglst2 = list(argstun) 
                        arglst2[2] = f0tun #objfun(best_x,*argstun)
                        argstun = tuple(arglst2)
                    else:
                        arglst2 = list(argstun)
                        arglst2[2] = best_f
                        argstun = tuple(arglst2)
                    # Use tunneling args:
                    findzero = perform_tunneling(x0tun,tunfun,tunfungrad,
                                                 tunfunsec,tunfunsecgrad,
                                                 argstun,minlisttun,
                                                 bounds=bounds,
                                                 topphase=topphases,
                                                 disp=disp,maxiter=tuniter,
                                                 tabulist=tabulist,
                                                 slack=slack,alpha0=alphastart,
                                                 attempts=tunphases,
                                                 plot=plot,l0=l0*(1+esc_attempt),lt0=lt0,
                                                 plotfuns=plotfuns,
                                                 plotfunscontranges=plotfunscontranges,
                                                 tunstrat=tunstrategy,
                                                 dirmethod=dm,cutoff=cutoff)
                    tp_used = findzero[-1] #used tunneling phases
                    tunphase_hist.append(tp_used)
 
                    tun_total_iter += 1
                    
                    # Process results of tunnel
                    if findzero[2] == True: #successful tunnel
                        tuncount += 1
                        num_fun = 1
                        print('tunneled to x: ',findzero[0])
                        print('tunneled f(x) = ',objfun(np.around(findzero[0],2),*argsmin))
                        print('T(x) = ',findzero[1])
                        #improve = 1
                        xf = findzero[0]
                        yf = objfun(xf,*argsmin)
                        nonewpt = 0

                    else: #unsuccessful tunnel
                        #improve = 0
                        nonewpt = 1 #no further new points
            
                    # except: #if there was error in tunneling
                    #     test = 1     
                    #     print('Tunneling loop error')
                    #     break
                
                global_iter += 1
                '''
                if global_iter >= totaliter:
                    print('Maximum number of global iterations exceeded. \n')
                    print('best_x_hist: ',best_x_hist)
                    print('best_f_hist: ',best_f_hist)
                    print('Min list: ',minlist)
                    if nonewpt == 0:
                        print('Warning! New point added to start list but not tunneled from.')
                        warn_iter = 1
                       # if not np.any(np.isclose(xf,startptlistnew,atol=1e-2),axis=0).all():
                        startptlistnew.append(xf)
                    #print('\n converted minlist: ',minlist)
                    nonewpt = 1
                '''
#                    return (best_x, best_f, evaluations,
#                            'Maximum number of global iterations exceeded',
#                            (all_tunnel, success_tunnel),best_x_hist,best_f_hist)
                    
                if isinstance(yf, np.ndarray):
                            yf = yf[0]
        
                maxfunevals -= num_fun
                evaluations += num_fun
         
                yf = objfun(xf,*argsmin)
            
                if maxfunevals <= 0:
                    print('Maximum number of function evaluations exceeded.')
                    print('\n best_x_hist: ',best_x_hist)
                    print('\n best_f_hist: ',best_f_hist)
                    print('\n minlist: ',minlist)
                    
                x0 = xf
                stepcount += 1    
            if disp:
                print('='*30) #changed to 30 from 72
    
            global_iter += 1
            x0 = xf.copy()
            startcount += 1
        # Update constraint multipliers
        mulist.append(mu)
        lambda_laglist.append(lambda_laglist)
        #rplist.append(rp)
        mu = mu/20 #changed from /10 june 13 2022
        mucount += 1

        icount += 1
        try:
            startptlist = sortstartpts(startptlistnew,objfun,*argsmin2)
        except:
            print('could not sort')
        startptlist_hist.append(startptlist)
    for mustep in np.arange(0,len(mulist)):
        lst = []
        for pt in startptlist_hist[mustep]:
            lst.append(np.around(pt,3))
        print('Round: {0:3f} Start Point List: {1}'
              .format(mustep,lst))
        
    if stopfunc(minlist,*argsmin) == True:
        print('Stopping function evaluated to True')
    mlist = []
    for minimum in minlist:
        mlist.append(np.around(minimum,3))
    print('{0}\nLambda_lag: {1} \nmu: {2} \nTunneling points found: {3}\
          \nStart List: {4} \nTabulist: {5} \nMin list: {6}\
          \nBest f*: {7} \nRounds: {8} \nlt0: {9}\
          \nl0: {10}'.format('===== Results =====',argsmin[1],argsmin[0],\
              tuncount,startptlist,tabulist,mlist,np.around(best_f,5),\
                  icount,lt0,l0))

    if warn_iter == 1:
        print('Warning: Tunneled points not tested due to total iteration limit.\
              Suggested to increase totaliter and rerun.\nTotal iter: {0}'.format(global_iter))
    return (minlist, best_f, evaluations,
                'Maximum number of function evaluations exceeded',
                (tuncount, success_tunnel),best_x_hist,best_f_hist,trackfunc_hist,\
                argsmin,\
                    tabulist,round_hist,minhist_all,tunphase_hist)\
                

def tunnel_exp2(x,fixed,lstar,lm,lt,movelist=[],tabulist=[],*args):
    #tunnel_exp2(x,fixed,lstar,lm,lt,movelist=[],tabulist=[],*args,**kws):
    """
    Exponential tunneling function

    Uses exponential tunneling method to create poles.
    S. Gomez and C. Barron, “The exponential tunneling method,” Ser. Reportes 
    Investig. IIMAS, vol. 1, no. 3, 1991.
    
    x0 = x value to calculate T(x) at
    fixed = list, [objfun,gradfun,aspiration,slack,minlist,objargs] 
        objfun = callable, objective function
        gradfun = callable, gradient of objective function
        aspiration value = aspiration value of objective function (current best value)
        slack = slack value to relax where T(x) is 0
            0 of T(x) at f(x) - slack = 0
        minlist = list of global minima x*
        objargs = tuple, arguments passed to objfun and gradfun
    lstar = pole strength of x* values
    lm = pole strength of moveable poles
    lt = pole strength of tabulist poles
    movelist = list of moveable poles
    disp = boolean, prints output if True
    """
    # lstar,lm,lt,movelist,tabulist,fixed = args
    objfun, gradfun, aspiration, slack, minlist = fixed[0:5]
    objargs = tuple(fixed[5::])

    fdiff = (objfun(x,*objargs) - aspiration)
    
    ptlist = minlist + movelist + tabulist
    # polelist = np.array([])
    poles1 = lstar * np.ones(len(minlist))
    poles2 = lm * np.ones(len(movelist)) 
    poles3 = lt * np.ones(len(tabulist))
    polelist = np.concatenate((poles1,poles2,poles3))
        
    ytf = tunnel_exp2_sub(x,fdiff,aspiration,np.array(ptlist),polelist)

    return ytf 

@jit('f8(f8[:],f8,f8,f8[:,:],f8[:])',nopython=True,cache=True) 
def tunnel_exp2_sub(x,fdiff,aspiration,ptlist,polelist):
    #tunnel_exp2(x,fixed,lstar,lm,lt,movelist=[],tabulist=[],*args,**kws):
    """Tunneling objective function. Exponential version.
        Compiled sub-function to improve speed.
    """

    product = 1.0
    
    for p in range(0,len(ptlist)):
        xdiff = x - ptlist[p]
        pole = polelist[p]
        product = product * np.exp(pole/(np.linalg.norm(xdiff)))

    ytf = fdiff * product #- slack

    return ytf 

def grad_tunnel_exp2(x,fixed,lstar,lm,lt,movelist=[],tabulist=[],*args):
    #(x,fixed,lstar,lm,lt,movelist=[],tabulist=[],
                    # *args,**kws):
    """Gradient of exponential tunneling function
 
    Uses exponential tunneling method to create poles.
    S. Gomez and C. Barron, “The exponential tunneling method,” Ser. Reportes 
    Investig. IIMAS, vol. 1, no. 3, 1991.
    
    x0 = x value to calculate T(x) at
    fixed = list, [objfun,gradfun,aspiration,slack,tabulist,objargs] 
        objfun = callable, objective function
        gradfun = callable, gradient of objective function
        aspiration value = aspiration value of objective function (current best value)
        slack = slack value to relax where T(x) is 0
            0 of T(x) at f(x) - slack = 0
        tabulist = list of global minima x*
        objargs = tuple, arguments passed to objfun and gradfun
    lstar = pole strength of x* values
    lm = pole strength of moveable poles
    movelist = list of moveable poles
    disp = boolean, prints output if True
    """
    # lstar,lm,lt,movelist,tabulist,fixed = args
    objfun, gradfun, aspiration, slack, minlist = fixed[0:5]
    objargs = tuple(fixed[5::])

    gradfx = gradfun(x,*objargs)
    
    fx = objfun(x,*objargs)
    
    ptlist = minlist + movelist + tabulist
    # polelist = np.array([])
    poles1 = lstar * np.ones(len(minlist))
    poles2 = lm * np.ones(len(movelist)) 
    poles3 = lt * np.ones(len(tabulist))
    polelist = np.concatenate((poles1,poles2,poles3))
    # for xmin in minlist:
    #     ptlist.append(xmin)
    #     polelist = np.append(polelist,lstar)
    
    # for xm in movelist:
    #     ptlist.append(xm)
    #     polelist = np.append(polelist,lm)
    
    # for xt in tabulist:
    #     ptlist.append(xt)
    #     polelist = np.append(polelist,lt)
    
    grad_tunnel = grad_tunnel_exp2_sub(x,fx,gradfx,aspiration,np.array(ptlist),polelist)
    
    return grad_tunnel

@jit('f8[:](f8[:],f8,f8[:],f8,f8[:,:],f8[:])',nopython=True,cache=True) 
def grad_tunnel_exp2_sub(x,fx,gradfx,aspiration,ptlist,polelist):
    #(x,fixed,lstar,lm,lt,movelist=[],tabulist=[],
                    # *args,**kws):
    """Gradient of exponential tunneling function
        Compiled sub-function to improve speed
    """
    
    fdiff = (fx - aspiration)

    exp_product_all = 1.0
    
    # Calculate exponential multiplications
    for p in range(0,len(ptlist)):
        xdiff = x - ptlist[p]
        pole = polelist[p]
        exp_product_all = exp_product_all * np.exp(pole/np.linalg.norm(xdiff))

    
    grad_tunnel = gradfx * exp_product_all
    
    for p2 in range(0,len(ptlist)):
        xdiff = x - ptlist[p2]
        pole = polelist[p2]
        grad_tunnel += fdiff * -pole/(np.linalg.norm(xdiff)**2) * \
            (xdiff/np.linalg.norm(xdiff)) * exp_product_all

    return grad_tunnel

def perform_tunneling(xstar,f,gradf,f2,gradf2,argstun,minlist,bounds=None,
                      topphase='',disp=True,maxiter=100,tabulist=[],
                      slack=0,alpha0=1,attempts=None,tol=1e-3,
                      plot=False,dirmethod=1,eps1=1e-2,
                      l0=0.5,lt0=0,plotfuns=[],
                      plotfunscontranges=[],tunstrat=1,cutoff=1.0):
    """Find zeros using method from exponential tunneling paper.
    xstar = current best value, x*
    f = callable, objective function
    gradf = callable, gradient of objective function
    minlist = list of best points (all global min found so far)
    tabulist = list of almost minima to track
    slack = slack value, accept points where T(x) <= slack (rather than <= 0)
    objargs = tuple, args passed to objfun and gradfun
    maxiter = maximum number of tunneling steps
    numphases = number of tunneling phases (if 0, defualts to 2 x number of design variables)
    tol =
    topphase = top optimization phase number
    plot = boolean, plot iteration history or not
    dirmethod = method to calculate direction (0 = as in exp tunneling paper,
                                               1 = -gradient)
    eps1 = tolerance to count 2 points as being the same for moveable poles
    tunstrat = 1 or 2; 1 = all tunneling start pts are eps from xstar in 
                random direction, 2 = 2n start pts are eps from xstar in 
                random direction, rest are random points within bounds
                (2 requires bounds)
    """
    # Number of variables
    numvars = len(xstar)
    
    # Automatic number of tunneling phases
    if attempts == None: attempts = 2 * numvars 

    #aspiration value of objfun(x), f*
    fstar = f(minlist[0],*argstun) #in case xstar passed in is local min
    fixed1 = [f, gradf, fstar, slack, minlist] + list(argstun)

    f2star = f2(minlist[0],*argstun);
    fixed2 = [f2,gradf2,f2star,slack,minlist] + list(argstun)

    fixed = [fixed1,fixed2] # list of fixeds
    func = 0 #which function to use; start with primary

    fixed_pf = []
    for pf in plotfuns:
        asp_pf = pf(xstar,*argstun)
        fixed_pf.append([pf,[],asp_pf,slack,minlist] + list(argstun))

    T = tunnel_exp2 #tunnel function
    gT = grad_tunnel_exp2 #gradient tunnel function
    
    # Pole settings
    lstar, ltabu, deltaeta = l0, lt0, lt0
    
    # Wolfe conditions for step size - values from Nocedal Wright
    c1, c2 = 1e-4, 0.9
    
    # Generate r_matrix and x0_matrix according to tunstrat
    r_matrix, x0_matrix = generate_r_x0(xstar,lstar,attempts,tunstrat,bounds = [])
    epsilon0 = lstar/np.log(100)#max(lstar/np.log(100),1e-3)
    
    # Outer loop - total number of allowed attempts
    tp, found = 0, False

    # While you still have attempts left and haven't found a zero
    while tp < attempts and found == False:
        #
        x0, r = x0_matrix[tp], r_matrix[tp]
        nancheckct = 0 # check if stepping out of bounds
        epsilon = epsilon0
        # Recording lists
        xklist, Tklist, gTf1klist, gTf2klist, Tf1klist, Tf2klist, fklist, Hf1klist, \
        Hf2klist, alphalist,\
        xmlist,xmiterlist, which = [], [], [], [], [], [], [], [], [], [], [], [], []

        alphalist.append(np.array([0]))
        #
        # Reset values
        eta, xm = 0.0, [] # no moveable poles to start
        #
        # Print Status
        print('\n{0} \n Tunneling Phase {1}-{2}\
              \n Current x* = {3} \n Minlist = {4} \
              \n Aspiration = {5} \n Lambda = {6}\
              \n{0} \n Tunneling Steps:'.format('='*30,topphase,tp, \
              np.around(xstar,3),np.around(minlist,3),fstar,lstar))
        
        # Update r vector if it immediately steps out of bounds
        while np.isnan(T(x0,fixed[func],lstar,eta,ltabu,
                                   movelist=xm,tabulist=tabulist)) \
            and nancheckct < 20:
            r = np.random.uniform(-1.0, 1.0, size=(numvars, ))
            # Check if at bounds of variables
            for v in np.arange(0,len(xstar)):
                if xstar[v] <= bounds[v][0] + tol:
                    r[v] = np.abs(r[v]) #1.0
                elif xstar[v] >= bounds[v][1] - tol:
                    r[v] = -np.abs(r[v]) #-1.0
            r = r/np.linalg.norm(r)
            x0 = xstar + epsilon*r
            nancheckct += 1
            #if disp == True:
            print('Changed r: ',np.around(r,3))

        if nancheckct >= 20:
            print('No feasible step direction found')
            Tf2klist = [np.inf]
            Tk1 = np.inf
            xk1 = xstar
            break

        infcheckct = 0
        while np.linalg.norm(gT(x0,fixed[func],lstar,eta,ltabu,\
                                movelist=xm,tabulist=tabulist)) > 1e50 \
            and infcheckct < 25:
                epsilon += epsilon0*(infcheckct+1)
                x0 = epsilon * r
                infcheckct += 1
                print('Increased initial perturbation size')

        # Record fist step values
        # k, xk, Hf1k, Hf2k = 0, x0, np.identity(numvars), np.identity(numvars) #initial values
        k, xk = 0, x0
        # Hf1k = np.zeros([numvars,numvars])
        # Hf2k = np.zeros([numvars,numvars])
        Tk = T(x0,fixed[func],lstar,eta,ltabu,movelist=xm,tabulist=tabulist)
        Tf1k = T(x0,fixed[0],lstar,eta,ltabu,movelist=xm,tabulist=tabulist)
        Tf2k = T(x0,fixed[1],lstar,eta,ltabu,movelist=xm,tabulist=tabulist)
        gTf1k = gT(x0,fixed[0],lstar,eta,ltabu,movelist=xm,tabulist=tabulist)
        gTf2k = gT(x0,fixed[1],lstar,eta,ltabu,movelist=xm,tabulist=tabulist)
        
        xklist.append(x0), Tklist.append(Tk), gTf1klist.append(gTf1k),gTf2klist.append(gTf2k),
        Tf1klist.append(Tk), Tf2klist.append(Tf2k),fklist.append(f(x0,*argstun)), which.append(func)
        # Hf1klist.append(Hf1k),Hf2klist.append(Hf2k)
        
        # Print initial step info
        if disp == True: print_tun_step(k,x0,f(x0,*argstun),f2(x0,*argstun),gradf2(x0,*argstun),\
                                        Tf1k,Tf2k,gTf1k,gTf2k,0,0,func)
        #
        # Find good lambdastar, lstar (pole strength at x*)    
        d = direction(dirmethod,Tf1k,gTf1k,xk)#,Hf1k)
        while np.dot(d,gT(x0,fixed[func],lstar,eta,ltabu,movelist=xm,tabulist=tabulist)) >= -.5 and \
                lstar < l0*10: #check if another limit for lambdastar
                # Increase lambdastar
                lstar += l0 #1
                # Recalculate d        
                d = direction(dirmethod,Tf1k,gTf1k,xk)#,Hf1k)
                print('Increased lambda_star')

        print('lambda_star: ',lstar)
        # Begin zero search
        nan = False
        while k < maxiter and found == False and nan == False:
            # Check if we reached a good tunnel point (<= 0)
            if T(xk,fixed[func],lstar,0,ltabu,movelist=xm,tabulist=tabulist) <= slack:
                found == True
                # Print current step info
                print('Found good tunneling point: {0}\nTk: {1}\nIterations: {2}'\
                      .format(np.around(xk,5),np.around(Tk,3),k))
                # Plot code 1
                if plot == True:
                    pltrange = [-.03,.03]
                    plot_iter_history_contour(f,argstun,fixed[0],lstar,
                                              eta,ltabu,pltrange,pltrange,xklist,which,
                                              res=[100,100],decplaces=3,contrange=[0,10],
                                              background='contourtunnel',topphase=topphase,
                                              tunphase=tp,minlist=minlist,movelist=xmlist,tabulist=tabulist,
                                              msg='T(x) Function 1 - Zero found')
                    plot_iter_history_contour(f2,argstun,fixed[1],lstar,
                                              eta,ltabu,pltrange,pltrange,xklist,which,
                                              res=[100,100],decplaces=3,contrange=[0,10],
                                              background='contourtunnel',topphase=topphase,
                                              tunphase=tp,minlist=minlist,movelist=xmlist,tabulist=tabulist,
                                              msg='T(x) Function 2 - Zero found')
                    plot_iter_history_contour(f2,argstun,fixed[1],lstar,
                                              eta,ltabu,pltrange,pltrange,xklist,which,
                                              res=[100,100],decplaces=3,contrange=[0,1],
                                              background='contour',topphase=topphase,
                                              tunphase=tp,minlist=minlist,movelist=xmlist,tabulist=tabulist,
                                              msg='f2(x) Function 2 - Zero found')
                # Return found point
                return xk,T(xk,fixed[0],lstar,0,ltabu,movelist=xm,tabulist=tabulist),True,tp

           # Check if need to move moveable pole closer to current iteration
                '''
            if np.size(xm) != 0 and eta != 0:
                if np.linalg.norm(xk - xm) > 1.25*epsilon:
                    #print('Move pole closer')
                    xm = [xk - 100*eta*epsilon*((xk - xm[0])/np.linalg.norm(xk - xm[0]))]
                    if disp == True:
                        print('Move pole closer')
                '''
            # Find new step (size and direction)
            step_attempt, newstepfound, maxstep = 0, False, 5

            while newstepfound == False and step_attempt <= maxstep:
                # Default to primary function
                func = 0
                gTf2knorm = np.linalg.norm(gTf2k)
                gf2norm = np.linalg.norm(gradf2(xk,*argstun))  
                # If gradient is very flat, try using function 2 to
                # get direction instead
                if f2 != None and gf2norm >= cutoff: #gTf2knorm >= 2.0:
                    # Switch to secondary function
                    if disp == True: print('Switch on, using f2')  
                    func = 1
                    if eta == 0:
                        d = direction(dirmethod,Tf2k,gTf2k,xk)#,Hf2k)  
                        # d = direction(1,Tf2k,gTf2k,xk,Hf2k)  
                    else:
                        d = direction(1,Tf2k,gTf2k,xk)#,Hf2k)
                     
                else:
                    # Calculate direction
                    if eta == 0: 
                        d = direction(dirmethod,Tf1k,gTf1k,xk)#,Hf1k)
                        
                    else:
                        d = direction(1,Tf1k,gTf1k,xk)#,Hf1k)
                    # # Calculate gradient of T(xk)
                    # gTk = gT(xk,fixed[func],lstar,0,ltabu,
                    #                      movelist=xm,tabulist=tabulist)                        
                #
                # Check if d is nan
                if np.isnan(d).any():
                    print('Exit: Nan problem')
                    nanproblem, k = True, maxiter
                '''
                # Check if moveable pole can be turned off   - old position 
                '''
                # Compute step size alpha
                alpha,alphaiter,maxbisect = alpha0, 0, 7 #starting value may depend on problem
                #
                # if func == 1: maxbisect = 10
                # added 4/2022 - if infeasible value, try smaller step size
                xk1 = xk + alpha*d
                Tk = T(xk,fixed[func],lstar,eta,ltabu,movelist=xm,tabulist=tabulist)
                Tk1 = T(xk1,fixed[func],lstar,eta,ltabu,movelist=xm,tabulist=tabulist)
                gTk = gT(xk,fixed[func],lstar,eta,ltabu,movelist=xm,tabulist=tabulist)
                gTk1 = gT(xk1,fixed[func],lstar,eta,ltabu,movelist=xm,tabulist=tabulist)
                while (not(wolfe(Tk1,Tk,alpha,gTk1,gTk,d,c1,c2)) and alphaiter < maxbisect) \
                    or (np.isnan(Tk1).any() and alphaiter < maxbisect):
                    # Bisect alpha
                    alpha = alpha * 0.5 #bisect alpha
                    # Recalculate xk1, Tk1, gTk1
                    xk1 = xk + alpha*d
                    Tk1 = T(xk + alpha*d,fixed[func],lstar,eta,ltabu,movelist=xm,tabulist=tabulist)
                    gTk1 = gT(xk1,fixed[func],lstar,eta,ltabu,movelist=xm,tabulist=tabulist)
                    alphaiter += 1
       
                #Check if need moveable pole - if too many bisections or grad too flat or NAN
                if (alphaiter >= maxbisect or np.linalg.norm(gTk) < tol) and \
                (Tk1 >= 1e-6 or np.isnan(Tk1).any()):
                    # if there is already a moveable pole turned on
                    if eta != 0 and np.any(np.isclose(xk,xm,atol=eps1),axis=0).all(): 
                        eta += deltaeta #increase pole strength
                        if step_attempt > 1: eta += deltaeta 
                        step_attempt += 1
                        if disp == True: print('increase eta ===================== ')
                    # if there is no moveable pole already
                    else:
                        # Create moveable pole
                        xm, eta = [xk], deltaeta
                        # Depending on problem, can tell next iteration to
                        # move away from starting point
                        #d = (xm-xstar)/np.linalg.norm(xm-xstar)

                        # Generate new random point (Gomez 1991)
                        epsilonm = (np.finfo(np.float).eps*10**7)**(1/10) #* (1 + np.linalg.norm(xstar))
                        tolf = np.finfo(float).eps*10**7
                        epsilonm = 2*(tolf**(1/5))*(1+np.linalg.norm(xk))

                        # Generate random direction vector to move away from xm
                        d = np.random.normal(scale=1, size=(numvars, ))

                        d = d/np.linalg.norm(d)
                        alpha = min(epsilonm,1e-3)
                        # Check if steps out of bounds
                        x0m = xk + alpha*d
                        nancheckct = 0 
                        while np.isnan(T(x0m,fixed[func],lstar,eta,ltabu,
                                         movelist=xm,tabulist=tabulist)) and nancheckct < 7:
                            d = np.random.normal(scale=1, size=(numvars, ))
                            d = d/np.linalg.norm(d)
                            x0m = xk + alpha*d
                            nancheckct += 1  
                        # Check if increase eta moveable pole strength 
                        xk1 = xk + alpha*d
                        #'''
                        while np.dot(d,gT(xk1,fixed[func],lstar,eta,ltabu,movelist=xm,tabulist=tabulist)) >= -.5 and \
                                eta < 100*deltaeta: 
                                # Increase moveable pole strength
                                eta += deltaeta
                                print('increase eta ===================== ')
                        #'''
                        # Record new step
                        newstepfound = True
                        step_attempt += 1
                        xk1 = xk + alpha * d #new value
                        Tf1k1 = T(xk1,fixed[0],lstar,eta,ltabu,movelist=xm,tabulist=tabulist)
                        Tf2k1 = T(xk1,fixed[1],lstar,eta,ltabu,movelist=xm,tabulist=tabulist)
                        gTf1k1 = gT(xk1,fixed[0],lstar,eta,ltabu,movelist=xm,tabulist=tabulist)
                        gTf2k1 = gT(xk1,fixed[1],lstar,eta,ltabu,movelist=xm,tabulist=tabulist)
                        # Print current step info
                        if disp == True: #???????
                            print('Moveable pole added')
                            print_tun_step(k+1,xk1,f(xk1,*argstun),f2(xk1,*argstun),gradf2(xk1,*argstun),Tf1k1,\
                                           Tf2k1,gTf1k1,gTf2k1,eta,alphaiter,func)
                        xklist.append(xk1),Tf1klist.append(Tf1k1),Tf2klist.append(Tf2k1)
                        gTf1klist.append(gTf1k1),gTf2klist.append(gTf2k1),fklist.append(f(xk1,*argstun))
                        alphalist.append(alpha*d), xmlist.append(xm[0])
                        which.append(func)
 
                        # Set new step to be current step
                        xk, Tk, gTf1k, gTf2k = xk1, Tk1, gTf1k1, gTf2k1
                        # Hf1k, Hf2k = Hf1k1, Hf2k1
                        k += 1
                        print('Moveable pole added')

                # if not at singular point, can take step
                else:
                    newstepfound = True
                    # Update values
                    # xk1 = xk + alpha * d 
                    # Tk1 = T(xk1,fixed[func],lstar,eta,ltabu,movelist=xm,tabulist=tabulist)
                    Tf1k1 = T(xk1,fixed[0],lstar,eta,ltabu,movelist=xm,tabulist=tabulist)
                    Tf2k1 = T(xk1,fixed[1],lstar,eta,ltabu,movelist=xm,tabulist=tabulist)
                    gTf1k1 = gT(xk1,fixed[0],lstar,eta,ltabu,movelist=xm,tabulist=tabulist)
                    gTf2k1 = gT(xk1,fixed[1],lstar,eta,ltabu,movelist=xm,tabulist=tabulist)

                    # Print current step info
                    if disp == True: #???????
                        # Tk1_1 = T(xk1,fixed[0],lstar,0,ltabu,movelist=xm,tabulist=tabulist)
                        print_tun_step(k+1,xk1,f(xk1,*argstun),f2(xk1,*argstun),gradf2(xk1,*argstun),Tf1k1,\
                                       Tf2k1,gTf1k,gTf2k1,eta,alphaiter,func)
                    # Record new step
                    xklist.append(xk1),Tf1klist.append(Tf1k1),Tf2klist.append(Tf2k1)
                    gTf1klist.append(gTf1k1),gTf2klist.append(gTf2k1),fklist.append(f(xk1,*argstun))
                    alphalist.append(alpha*d)
                    which.append(func)
                    if eta != 0: # if moveable pole used, record
                        xmlist.append(xm[0]), xmiterlist.append(k)
                        # step_attempt += 1
                    # Check if moveable pole can be turned off
                    if eta != 0 and step_attempt == 0:
                        #if same sign
                        if np.dot(d,gT(xk,fixed[func],lstar,0,ltabu,movelist=xm,tabulist=tabulist)) * \
                            np.dot(d,gTk) > 0: 
                            eta = 0 #turn off moveable pole
                            if disp == True: print('Moveable pole turned off')
                        elif f(xk,*argstun) >= 0 and f(xk,*argstun) < 0.9*f(xm[0],*argstun): #positive
                            eta = 0
                            if disp == True: print('Moveable pole turned off')
                        elif f(xk,*argstun) < 0 and f(xk,*argstun) < 1.1*f(xm[0],*argstun): #negative
                            eta = 0 #turn off moveable pole
                            if disp == True: print('Moveable pole turned off')
                            # elif np.dot(np.dot(d,gT(xk,fixed,lstar=lambdastar,lm=0,movelist=xm,disp=False)),(xk - xm)) > 0:
                            #     eta = 0
                            #     print('Moveable pole turned off')      
                        
                    # Set new xk,Tk,gTk for next step
                    xk, Tk, gTf1k, gTf2k = xk1, Tk1, gTf1k1, gTf2k1

                    k += 1 

                step_attempt += 1
                if step_attempt > maxstep:
                    print('No good step further \nIterations: ',k)
                    # if plot == True:
                    #     plot_iter_history_contour(f,argstun,fixed[0],lstar,
                    #                               eta,ltabu,[-.1,.1],[-.1,.1],xklist,which,
                    #                               res=[100,100],decplaces=3,contrange=[0,10],
                    #                               background='contourtunnel',topphase=topphase,
                    #                               tunphase=tp,minlist=minlist,movelist=xmlist,tabulist=tabulist,
                    #                               msg='Function 1 - No zero found')
                    #     plot_iter_history_contour(f2,argstun,fixed[1],lstar,
                    #                               eta,ltabu,[-.1,.1],[-.1,.1],xklist,which,
                    #                               res=[100,100],decplaces=3,contrange=[0,5],
                    #                               background='contourtunnel',topphase=topphase,
                    #                               tunphase=tp,minlist=minlist,movelist=xmlist,tabulist=tabulist,
                    #                               msg='Function 2 - No zero found')
                    k = maxiter
                    # plot code 2
                    break
            #
            # Stop if max number of iterations reached    
            if k >= maxiter and step_attempt > maxstep:
                print('Maximum number of iterations reached. \n Iterations: {0}\
                      \nSlack: {1} \nMin val: {2:6.5f}'.format(k,slack,min(Tf2klist)))
                # plot code 3
        tp += 1
        # plot code 4

    ''' Turn on if you want to print iteration hist or 
    moveable pole history over a certain design variable
    for iter in xklist: 
        xklist1.append(iter[0])
    for iterm in xmlist:
        xmlist1.append(iterm[0])
    '''

    print('No new tunneling point found at this starting point.\
          \nSlack: {0} \nMin val: {1:6.5f}'.format(slack,min(Tf2klist)))
    return xk1, Tk1, False, tp

def generate_r_x0(xstar,lstar,attempts,tunstrat,bounds = []):
    numvars = len(xstar)
    # Tunneling strategy
    if tunstrat == 1: #regular, all pts from startpt
        attempts1 = attempts
        attempts2 = 0
    elif tunstrat == 2 and bounds == []:
        print('Tunneling strategy 2 requires bounds. Reverting to strategy 1.')
        tunstrat = 1
        attempts1 = attempts
        attempts2 = 0
    elif tunstrat == 2: # pts from startpt + random pts
        attempts1 = min(attempts,2*numvars)
        attempts2 = max(0,attempts - attempts1)
    
    r_matrix = np.random.normal(size=(attempts1,numvars))

    # Normalize random search vectors
    for row in np.arange(attempts1): r_matrix[row] = r_matrix[row]/np.linalg.norm(r_matrix[row]) #normalize
    
    # Check if duplicates
    dups = True
    r_iter = 0
    while dups == True and r_iter < 7:
        size1 = attempts1
        size2 = np.size(np.unique(np.around(r_matrix,decimals=1),axis=0),axis=0)
        if size1 == size2:
            dups = False #no duplicate search directions found
        else: #regenerate
            r_matrix = np.random.normal(size=(attempts1,numvars))
            for row in np.arange(attempts1): r_matrix[row] = r_matrix[row]/np.linalg.norm(r_matrix[row]) #normalize
        r_iter += 1
    
    # Make matrix of initial points in random search directions
    x0_matrix = np.zeros([attempts,numvars]) #initialize matrix
    epsilon0 = lstar/np.log(100)#max(lstar/np.log(100),1e-3)
    # Strategy 1 points
    for rr in range(0,attempts1):
        x0_matrix[rr] = xstar + r_matrix[rr]*epsilon0
    # Strategy 2 points
    if tunstrat == 2:
        boundlow = []
        boundhigh = []
        for bb in bounds:
            boundlow.append(bb[0])
            boundhigh.append(bb[1])
        for rr2 in range(0,attempts2):
            x0_matrix[attempts1 + rr2] = np.random.uniform(low=boundlow,high=boundhigh,size=[1,numvars])
    return r_matrix,x0_matrix

@jit('f8[:](i8,f8,f8[:],f8[:])',nopython=True,cache=True)         
def direction(method,Tk,gTk,xk): #,Hk=np.identity(1)):
    # Different ways to calculate step direction
    # Exponential paper or simple gradient
    '''
    d = -T(x0,fixed,lstar=lambdastar,lm=eta,movelist=xm,disp=False) \
        * gT(x0,fixed,lstar=lambdastar,lm=eta,movelist=xm,disp=False) / \
        (np.linalg.norm(gT(x0,fixed,lstar=lambdastar,lm=eta,movelist=xm,disp=False))**2)                    
    d = d/np.linalg.norm(d) #adding normalizing d
    d0 = -T(x0,fixed,lstar=lambdastar,lm=eta,movelist=xm,disp=False) \
        * gT(x0,fixed,lstar=lambdastar,lm=eta,movelist=xm,disp=False) / \
        (np.linalg.norm(gT(x0,fixed,lstar=lambdastar,lm=eta,movelist=xm,disp=False))**2)
    d2 = gT(x0,fixed,lstar=lambdastar,lm=eta,movelist=xm,disp=False) / \
        (np.linalg.norm(gT(x0,fixed,lstar=lambdastar,lm=eta,movelist=xm,disp=False))**2)
    d2 = -d2/np.linalg.norm(d2)
    d3 = -gT(x0,fixed,lstar=lambdastar,lm=eta,movelist=xm,disp=False)
    d3 = d3/np.linalg.norm(d3)
    '''
    if method == 0: # exponential tunneling paper direction
        d = - Tk * gTk / (np.linalg.norm(gTk)**2)
    elif method == 1: # gradient method
        d = - gTk
    #elif method == 2:
    #    d = - np.dot(Hk,gTk)
    # normalize
    nsum = 0
    for el in d: nsum += el**2
    n = np.sqrt(nsum)
    return d * (1/n)
    # elif method == 2: #BFGS
    #     dd = np.dot(Hk,gTk)
    #     n = 
    #     return -np.dot(dd,n)

def direction_nojit(method,Tk,gTk,xk,Hk):
    # Different ways to calculate step direction
    # Exponential paper or simple gradient
    '''
    d = -T(x0,fixed,lstar=lambdastar,lm=eta,movelist=xm,disp=False) \
        * gT(x0,fixed,lstar=lambdastar,lm=eta,movelist=xm,disp=False) / \
        (np.linalg.norm(gT(x0,fixed,lstar=lambdastar,lm=eta,movelist=xm,disp=False))**2)                    
    d = d/np.linalg.norm(d) #adding normalizing d
    d0 = -T(x0,fixed,lstar=lambdastar,lm=eta,movelist=xm,disp=False) \
        * gT(x0,fixed,lstar=lambdastar,lm=eta,movelist=xm,disp=False) / \
        (np.linalg.norm(gT(x0,fixed,lstar=lambdastar,lm=eta,movelist=xm,disp=False))**2)
    d2 = gT(x0,fixed,lstar=lambdastar,lm=eta,movelist=xm,disp=False) / \
        (np.linalg.norm(gT(x0,fixed,lstar=lambdastar,lm=eta,movelist=xm,disp=False))**2)
    d2 = -d2/np.linalg.norm(d2)
    d3 = -gT(x0,fixed,lstar=lambdastar,lm=eta,movelist=xm,disp=False)
    d3 = d3/np.linalg.norm(d3)
    '''
    if method == 0: # exponential tunneling paper direction
        d = - Tk * gTk / (np.linalg.norm(gTk)**2)
    elif method == 1: # gradient method
        d = - gTk
    elif method == 2:
        d = - np.dot(Hk,gTk)
    # normalize
    n = np.linalg.norm(d)
    return d * (1/n)
    
@jit('f8[:,:](f8[:],f8[:],f8[:],f8[:],f8[:,:])',nopython=True,cache=True)  
def calc_Hk(xk,xk1,gTk,gTk1,Hk):
    # dk = -Hk * gradf(xk)
    sk = xk1 - xk
    yk = gTk1 - gTk
    rhok = 1/np.dot(yk,sk)
    numvars = len(xk)
    I = np.identity(numvars)
    Hk1 = (I - rhok*sk*yk) * Hk * (I - rhok*yk*sk) + rhok*sk*sk
    return Hk1

def perform_minimization(x0,f,gradf,f2,gradf2,argsmin,minlist,best_x,best_f,mucount,\
                         startcount,stepcount,ineq_min,grad_ineq_min,\
                         bounds=None,minrounds=5,disp=True,\
                         maxfunevals=3000,confun=None,alphamu=1.05,local='SLSQP',\
                         trackfunc=None):
    '''
    Perform minimization phase of tunneling algorithm with switching.
    x0 = start point
    f = primary objective function
    gradf = gradient of f
    f2 = secondary objective function
    gradf2 = gradient of f2
    argsmin = args to be passed to f, f2, gradf, gradf2
    minlist = current list of global minima
    best_x = current x*
    best_f = current f*
    mucount = counter used in higher function (***FIX)
    startcount = start point counter
    stepcount = step counter (***)
    bounds = bounds to be passed to minimizer
    minrounds = how many rounds of minimization to perform, recalculating augmented
                Lagrange multiplier lambda (Lk) and penatly factor (muL) each time
    disp = turns detailed display on/off with True/False
    maxfunevals = max number of function evaluations allowed in minimizer (each call)
    confun = equality constraint function to calculate new lambda (Lk) value
    alphamu = multiplier used to update muL
    local = local minimizer to call, SLSQP or L-BFGS-B
    trackfunc = functions to track the values of at each round of minimization
    '''
    mu = argsmin[0] #???
    
    # Display phase information:
    print('\n{0}\nStarting MINIMIZATION Phase {1:d}-{2:d}-{3:d}-0 \n Current x* = {4} \n Current minlist: {7} \n Current best_f = {5} \n{0}'
          .format('='*30,mucount,startcount,stepcount,np.around(best_x,3),best_f,mu,np.around(minlist,3))) #changed to 30 from 72
    print('x0: ',x0) 
    print('Minlist: ',minlist)
    
    # MINIMIZE
    # Which function to use depends on problem
    # Use fmin_ functions to more easily access iteration history
    #res = fmin_slsqp(objfun, x0, args=objargs,fprime=grad,bounds=bounds,
    #               disp=2,iter=1000,full_output=True,acc=1e-6)
    
    # Set display options
    if disp == True:
        callbackfunc = None #printiter
        ipval = 1 #iprint value
    else:
        callbackfunc = None
        ipval = 0
        
    # Set initial values
    minroundsconv = 5 * minrounds
    bounds2 = bounds.copy()
    x02 = x0.copy()
    switch = 0 # start with switching off
    f0 = best_f.copy()
    ii = 1
    Lk1 = argsmin[1]
    muLk1 = argsmin[3]
    # minhist_entry = {}
    fminhist, xminhist, minhist = {}, {}, {}
    fminhist[0] = best_f #first minhist entry
    xminhist[0] = x0
    minhist[0] = {0:{"x*":list(x0),"y*":best_f,"lambda":0,\
                   "muL":muLk1,"switch":switch,"iter":0}}
   
    arglstmin2 = list(argsmin)
    total_num_fun = 0
    # Print initial
    print('\tjj: {0:2d}'.format(0))
    print('\tLambda: {0:5.3e}'.format(Lk1))
    print('\tmuL: {0:5.3e}'.format(muLk1))
    # print('\tf0: {0:5.5e}'.format(f0))
    print('\tc9: {0:5.3f}'.format(x0[8])) #*180/np.pi)
    print('\tc10: {0:5.3f}'.format(x0[9])) #*180/np.pi)
    print('\tth1: {0:5.3f}'.format(x0[-2]*180/np.pi))
    print('\tth2: {0:5.3f}'.format(x0[-1]*180/np.pi))
    # print('\tc11: ',x02[10])
    print('\tEnergy: {0:5.3e}'.format(confun(x0,*argsmin)))
    print('\tShape: ',trackfunc[1](x0,*argsmin))
    print('\ty*: {0:5.3e}'.format(best_f[-1]))
    print('\tobjfun: {0:5.3e}'.format(f(x0,*argsmin)))
    # Perform rounds of minimization
    # Outer for loop: Lk (lambda multiplier) and muLk (penalty factor) are 
    # updated
    for jj in range(0,minrounds):
        # Update Lk and muLk
        # arglstmin2 = list(argsmin2)
        # old values
        Lk1 = arglstmin2[1]
        muLk1 = arglstmin2[3]
        # new values
        Lk2 = Lk1 + muLk1 * confun(x02,*argsmin)
        muLk2 = alphamu * muLk1
        
        # Update mu
        mu2 = mu * 1/(20 ** (jj-1))
        arglstmin2[0] = mu2

        # save new values into args
        arglstmin2[1] = Lk2
        arglstmin2[3] = muLk2
        # argsmin2 = tuple(arglstmin2)

        # Set accuracy value - increases with increasing rounds (i.e. as you
        # approach a min)
        accslsqp = 1*10**-(10+jj) # make it get more accurate?
        
        converge = False
        
        minhist_entry = {}
        
        switch = 0 #? Start with switch off
        
        # Inner loop: repeat minimization with current params until convergence
        while converge == False and ii < minroundsconv:
            # Update f0 using current args
            f0 = f(x02,*tuple(arglstmin2))
            arglstmin2[2] = f0 # new f0
            argsmin2 = tuple(arglstmin2)
            
            # Minimize
            print('Minimization Results:')
            if switch == 0:
                if local == 'L-BFGS-B':
                    resb = fmin_l_bfgs_b(f,x02,args=argsmin2,fprime=gradf,iprint=ipval,
                                         bounds=bounds2,maxfun=maxfunevals,callback=callbackfunc,
                                         factr=10,m=15)
                    resbxf, resbyf, num_fun = resb[0], resb[1], resb[2]['funcalls']
                elif local == 'SLSQP':
                    resb = fmin_slsqp(f,x02,args=argsmin2,bounds=bounds2,
                                      fprime=gradf,iter=maxfunevals,acc=accslsqp,
                                      iprint=ipval,full_output=1,callback=callbackfunc,
                                      ieqcons=ineq_min,fprime_ieqcons=grad_ineq_min)
                    resbxf, resbyf, num_fun = resb[0], resb[1], resb[2]
            
            else: #If different minimization function specified (i.e. switching)
                if local == 'L-BFGS-B':
                    switchcount2.sc = 0
                    resb = fmin_l_bfgs_b(f2,x02,args=argsmin2,fprime=gradf2,iprint=ipval,
                                          bounds=bounds2,maxfun=maxfunevals,callback=callbackfunc,
                                          factr=10,m=15,maxls=20)
                    resbxf, resbyf, num_fun = resb[0], resb[1], resb[2]['funcalls']
                elif local == 'SLSQP':
                    switchcount2.sc = 0
                    resb = fmin_slsqp(f2,x02,args=argsmin2,bounds=bounds2,
                                      fprime=gradf2,iter=maxfunevals,acc=accslsqp,
                                      iprint=ipval,full_output=1,callback=callbackfunc,
                                      ieqcons=ineq_min,fprime_ieqcons=grad_ineq_min)
                    resbxf, resbyf, num_fun = resb[0], resb[1], resb[2]

            # Pick best point in case minimization did not improve                        
            x02, y02 = choosemin(resbxf,x02,f,argsmin2)
            
            # Record minhist
            fminhist[ii] = y02
            xminhist[ii] = x02
            minhist_entry[ii] = {"x*":list(x02),"y*":y02,"lambda":Lk2,\
                           "muL":muLk2,"switch":switch,"iter":num_fun}
            
            # Start values for next round
            xstarn1 = x02 #[:-1]
            xstarn = xminhist[ii-1] #[:-1]
            fstarn = f(xstarn,*argsmin2)
            fstarn1 = f(xstarn1,*argsmin2) #y02
            
            # Update num_fun
            total_num_fun += num_fun
            
            # Print results from this round
            print('\tjj: {0:2d}'.format(jj+1))
            print('\tLambda: {0:5.3e}'.format(Lk2))
            print('\tmuL: {0:5.3e}'.format(muLk2))
            print('\tf0: {0:5.5e}'.format(f0))
            print('\tc9: {0:5.3f}'.format(x02[8])) #*180/np.pi)
            print('\tc10: {0:5.3f}'.format(x02[9])) #*180/np.pi)
            print('\tth1: {0:5.3f}'.format(x02[-2]*180/np.pi))
            print('\tth2: {0:5.3f}'.format(x02[-1]*180/np.pi))
            # print('\tc11: ',x02[10])
            print('\tEnergy: {0:5.3e}'.format(confun(x02,*argsmin2)))
            print('\tShape: ',trackfunc[1](x02,*argsmin2))
            print('\titers: {0:5d}'.format(num_fun))
            print('\tswitch: {0}'.format(switch))
            print('\ty*: {0:5.3e}'.format(y02))
            print('\tobjfun: {0:5.3e}'.format(f(x02,*argsmin2)))
          
            # Check convergence and if switching turns on
            deltaf = np.abs(fstarn - fstarn1)
            if switch == 0:
                if (deltaf)/fstarn < 0.001 or deltaf < 1e-6:
                    switch = 1
                    converge = False
                else:
                    converge = False
            elif switch == 1: 
                if deltaf/fstarn < 0.001 or deltaf < 1e-6:
                    converge = True
                else: 
                    converge = False
            # Print convergence, increase iteration counter ii
            print('\tconverge: ',converge)
            ii += 1                    
        minhist[jj+1] = minhist_entry

    return xstarn1, fstarn1, minhist, argsmin2, num_fun
   

def plot_iter_history(topphase,tunphase,xmin,k,xklist,fklist,Tklist,xmiterlist,xmlist):
    ''' 
    Plot iteration history from tunneling
    topphase = top tunneling phase number
    tunphase = tunneling phase number
    xmin = current minimum x value, x*
    k = iteration number
    xklist = list of x values for each iteration
    fklist = list of objective function values for each iteration
    Tklist = list of tunneling function values for each iteration
    xmiterlist = list of iterations with moveable poles
    xmlist = list of moveable pole values for iterations with moveable poles
    '''
    # Max y-value for tunneling function value
    Tmax = min(max(Tklist),800)
    '''
    # Plot x* vs. iterations 
    fig, ax1 = plt.subplots()
    plt.title('Tunneling Phase {0}-{1} x* = {2}'.format(topphase,tunphase,round(xmin[0],3)))
    color = 'tab:orange'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('x_k',color=color)
    ax1.scatter(np.arange(0,k+1),xklist,color=color)
    ax1.tick_params(axis='y',labelcolor=color)
    color = 'tab:pink'
    ax1.scatter(xmiterlist,xmlist,color=color,marker='^')
    ax2 = ax1.twinx() # create second y axis with same x axis
    color = 'tab:blue'
    ax2.set_ylabel('T(x)',color=color)
    ax2.plot(np.arange(0,k+1),Tklist,color=color)
    ax2.set_ylim([0,Tmax])
    #ax2.set_yscale('log')
    ax2.tick_params(axis='y',labelcolor=color)
    fig.tight_layout()
    plt.show()
    '''
    # Plot with f(x)
    fig, ax1 = plt.subplots()
    plt.title('Tunneling Phase {0}-{1} x* = {2}'.format(topphase,tunphase,round(xmin[0],3)))
    color = 'tab:orange'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('T(x)',color=color)
    ax1.plot(np.arange(0,len(Tklist)),Tklist,color=color)
    ax1.tick_params(axis='y',labelcolor=color)
    ax1.set_ylim([0,Tmax])
    color = 'tab:pink'
    #ax1.scatter(xmiterlist,xmlist,color=color,marker='^')
    ax2 = ax1.twinx() # create second y axis with same x axis
    color = 'tab:purple'
    ax2.set_ylabel('f(x)',color=color)
    ax2.plot(np.arange(0,len(fklist)),fklist,color=color)
    #ax2.set_yscale('log')
    ax2.tick_params(axis='y',labelcolor=color)
    fig.tight_layout()
    plt.show()
    
def plot_contour_tunnel(fun,fixed,lambdastar,eta,lt,xrange,yrange,contrange=[],
                 minlist=[],movelist=[],tabulist=[],topphase=0,tunphase=0,msg=''):
    '''
    Contour plot - works for exponential tunneling function only
    fun = function to be plotted
    funargs = *() args to pass to fun
    xrange = [xmin xmax]
    yrange = [ymin ymax]
    contrange = bounds for contour scale      
    '''
    xplt = np.linspace(xrange[0],xrange[1],25)
    yplt = np.linspace(yrange[0],yrange[1],25)
    X,Y = np.meshgrid(xplt,yplt)
    Z = np.zeros([len(xplt),len(yplt)])
    for ii in range(0,len(xplt)):
        for jj in range(0,len(yplt)):
            Z[ii,jj] = tunnel_exp2(np.array([X[ii,jj],Y[ii,jj]]),fixed,
                 lstar=lambdastar,lm=eta,lt=lt,movelist=[],tabulist=tabulist)

    fig, ax = plt.subplots(1,1)
    if len(contrange) == 0:
        contrange = [Z.min(), Z.max()] # if no contour range given, use min and max vals
    cp = ax.contourf(X,Y,Z,np.linspace(contrange[0],contrange[1],51),cmap='viridis')
    fig.colorbar(cp)
    #labeling
    tlist = []
    movelistlabel = []
    # for jj in enumerate(minlist): plt.scatter(jj[1][0],jj[1][1],color='tab:orange')
    # for kk in enumerate(movelist): plt.scatter(kk[1][0],kk[1][1],color='tab:pink',marker='^')
    # for ll in enumerate(tabulist): plt.scatter(ll[1][0],ll[1][1],color='tab:blue',marker='*')
    if msg == '':
        plt.title('Tunneling Phase ({0}-{1:d}) \n Minlist:{2} \
                   \n Moveable Poles: {3}'.format(\
                topphase, tunphase,minlist,movelist))  
    else:
        plt.title('Tunneling Phase ({0}-{1:d}) \n Minlist:{2} \
                   \n{4}'.format(\
                topphase, tunphase,minlist,movelist,msg))                
    plt.show
    return fig, ax

def plot_contour(fun,fcnargs,xrange,yrange,contrange=[],res=[10,10],topphase=0,
                 tunphase=0,minlist=[],msg=''):
    '''
    Contour plot - works for exponential tunneling function only
    fun = function to be plotted
    funargs = *() args to pass to fun
    xrange = [xmin xmax]
    yrange = [ymin ymax]
    contrange = bounds for contour scale      
    '''
    xplt = np.linspace(xrange[0],xrange[1],res[0])
    yplt = np.linspace(yrange[0],yrange[1],res[1])
    X,Y = np.meshgrid(xplt,yplt)
    Z = np.zeros([len(xplt),len(yplt)])
    for ii in range(0,len(xplt)):
        for jj in range(0,len(yplt)):
            Z[ii,jj] = fun(np.array([X[ii,jj],Y[ii,jj]]),*fcnargs)

    fig, ax = plt.subplots(1,1)
    if len(contrange) == 0:
        contrange = [Z.min(), Z.max()] # if no contour range given, use min and max vals
    cp = ax.contourf(X,Y,Z,np.linspace(contrange[0],contrange[1],51))
    fig.colorbar(cp)
    if msg == '':
        plt.title('Tunneling Phase ({0}-{1:d}) \n Minlist:{2}'.format(\
                topphase, tunphase,minlist))  
    else:
        plt.title('Tunneling Phase ({0}-{1:d}) \n Minlist:{2} \n{3}'.format(\
                topphase, tunphase,minlist,msg))
    #labeling
    # tlist = []
    # movelistlabel = []
    # for jj in enumerate(tabulist): plt.scatter(jj[1][0],jj[1][1],color='tab:orange')
    # for kk in enumerate(movelist): plt.scatter(kk[1][0],kk[1][1],color='tab:pink',marker='^')
    #plt.title('Tunneling Phase ({1:d}-{2:d}) \n Tabulist:{3} \n Moveable Poles: {4}'.format(\
    #      '='*30, topphase, tunphase,tabulist,movelist))                
    plt.show
    return fig, ax

def plot_text(fun,fcnargs,xrange,yrange,ptslist=[],res=[10,10],decplaces=3,contrange=[]):
    '''
    Contour plot - works for exponential tunneling function only
    fun = function to be plotted
    funargs = *() args to pass to fun
    xrange = [xmin xmax]
    yrange = [ymin ymax]
    contrange = bounds for contour scale      
    '''
    xplt = np.linspace(xrange[0],xrange[1],res[0])
    yplt = np.linspace(yrange[0],yrange[1],res[1])
    X,Y = np.meshgrid(xplt,yplt)
    Z = np.zeros([len(yplt),len(xplt)])
    for ii in range(0,len(yplt)):
        for jj in range(0,len(xplt)):
            Z[ii,jj] = fun(np.array([X[ii,jj],Y[ii,jj]]),*fcnargs)
    Ztruncate = np.round(Z,decimals=decplaces) #round for string conversion
    Zstr = Ztruncate.astype(str)
    fig, ax = plt.subplots(1,1)
    for ii in range(0,len(yplt)):
        for jj in range(0,len(xplt)):
            plt.text(X[ii,jj],Y[ii,jj],Zstr[ii,jj],fontsize='small',ha='center',
                     va='center')
    plt.axis([xrange[0],xrange[1],yrange[0],yrange[1]])
    if len(contrange) == 0:
        contrange = [Z.min(), Z.max()] # if no contour range given, use min and max vals
    cp = ax.contourf(X,Y,Z,np.linspace(contrange[0],contrange[1],51),cmap='BuPu')
    fig.colorbar(cp)
    plt.show

def plot_text_tunnel(fun,fcnargs,xrange,yrange,fixed,lstar=1,lm=0.1,movelist=[],
                     ptslist=[],res=[10,10],decplaces=3,contrange=[]):
    '''
    Contour plot - works for exponential tunneling function only
    fun = function to be plotted
    funargs = *() args to pass to fun
    xrange = [xmin xmax]
    yrange = [ymin ymax]
    contrange = bounds for contour scale      
    '''
    xplt = np.linspace(xrange[0],xrange[1],res[0])
    yplt = np.linspace(yrange[0],yrange[1],res[1])
    X,Y = np.meshgrid(xplt,yplt)
    Z = np.zeros([len(yplt),len(xplt)])
    for ii in range(0,len(yplt)):
        for jj in range(0,len(xplt)):
            x0 = np.array([X[ii,jj],Y[ii,jj]])
            Z[ii,jj] = fun(x0,fixed,lstar=1,lm=0,movelist=[],disp=False)
    Ztruncate = np.round(Z,decimals=decplaces) #round for string conversion
    Zstr = Ztruncate.astype(str)
    fig, ax = plt.subplots(1,1)
    for ii in range(0,len(yplt)):
        for jj in range(0,len(xplt)):
            plt.text(X[ii,jj],Y[ii,jj],Zstr[ii,jj],fontsize='small',ha='center',
                     va='center')
    plt.axis([xrange[0],xrange[1],yrange[0],yrange[1]])
    if len(contrange) == 0:
        contrange = [Z.min(), Z.max()] # if no contour range given, use min and max vals
    cp = ax.contourf(X,Y,Z,np.linspace(contrange[0],contrange[1],51),cmap='BuPu')
    fig.colorbar(cp)
    plt.show
#tunnel_exp2(x0,fixed,lstar=1,lm=0.1,movelist=[],disp=False,
#                *args,**kws)

def plot_iter_history_contour(fun,fcnargs,fixed,lambdastar,eta,lt,xrange,yrange,
                              iterhist,whichlst,ptslist=[],
                              res=[10,10],decplaces=3,contrange=[],background='contour',
                              topphase=0,tunphase=0,minlist=[],movelist=[],tabulist=[],msg=''):
    #whichlst = which func used, determines which symbol
    if background == 'contour':
        fig, ax = plot_contour(fun,fcnargs,xrange,yrange,contrange=contrange,res=res,
                     topphase=topphase,tunphase=tunphase,msg=msg)
    elif background == 'text':
        plot_text(fun,fcnargs,xrange,yrange,ptslist=ptslist,res=res,decplaces=decplaces,
                  contrange=contrange)
    elif background == 'contourtunnel':
        fig,ax = plot_contour_tunnel(fun,fixed,lambdastar,eta,lt,xrange,yrange,contrange=contrange,
                 minlist=minlist,movelist=movelist,tabulist=tabulist,topphase=topphase,
                 tunphase=tunphase,msg=msg)
    for xm in enumerate(movelist): plt.scatter(xm[1][0],xm[1][1],color='tab:pink',marker='^')
    for xmin in enumerate(minlist): plt.scatter(xmin[1][0],xmin[1][1],color='tab:orange',marker='x')
    for xtabu in enumerate(tabulist): plt.scatter(xtabu[1][0],xtabu[1][1],color='tab:blue',marker='*')

    iterx = []
    itery = []
    markerlst = []
    for i in whichlst: # fill marker for primary func, empty for sec func
        if i == 0:
            markerlst.append('full')
        else:
            markerlst.append('none')
    for ii in np.arange(0,len(iterhist)):
        iterx.append(iterhist[ii][0])
        itery.append(iterhist[ii][1])
        color = ii/(len(iterhist) - 1) #1 - ii/(len(iterhist) - 1)
        if whichlst[ii] == 0:
            face = str(color)
        else:
            face = 'none'
        ax.scatter(iterhist[ii][0],iterhist[ii][1], facecolors=face,
                    edgecolors=str(color))
        ax.text(iterhist[ii][0]*1.01,iterhist[ii][1]*1.01,ii)
        # plt.plot(iterhist[ii][0],iterhist[ii][1],marker='o',
        #             fillstyle=markerlst[ii],c=[str(color)])
    cmap = mpl.cm.cool
    norm = mpl.colors.Normalize(vmin=0, vmax=len(iterhist)-1)
    # cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
    #                                 norm=norm,
    #                                 orientation='vertical')
    cm = plt.cm.get_cmap('RdYlBu')
    plt_iter=ax.scatter(iterx,itery,c=np.arange(0,len(iterhist)),cmap='Greys',marker="")
    fig.colorbar(plt_iter,format='%.0f')
    # plt.colorbar(cm)
    # sc = plt.scatter(xy, xy, c=z, vmin=0, vmax=20, s=35, cmap=cm)
    # plt.colorbar(fig)
    # cb1.set_label('Some Units')
    # fig.show()    
    # iternumber = np.arange(0,len(iterhist))
    # plt_iter = plt.scatter(iterx,itery,c=iternumber,cmap='Greys') 
    
    # plt.colorbar(plt_iter)
    fig.show
    filename = msg + '.svg'
    plt.savefig(filename,dpi=300)

def evaluate_results(xf,yf,best_x,best_f,lt0,minlist,tabulist,startptlistnew, 
                     argsmin2,trackfunc,trackfunc_hist,
                     best_x_hist,best_f_hist,round_hist,icount,
                     glbtol,tp_total,tp_used,findzero,success_tunnel,tunphases,
                     esc_attempt,nonewpt,trytunnel):
    # better min
    if yf < best_f - glbtol and \
        not np.any(np.isclose(xf,minlist,atol=1e-3),axis=0).all():# and\
        #not np.any(np.isclose(xf,startptlistnew,atol=1e-3),axis=0).all(): #any((int_to_ext(xf,lbs[0],ubs[0])==test).all() for test in int_to_ext(tabulist,lbs[0],ubs[0])): #not any(np.isclose(xf,tabulist,glbtol)):
        best_f = np.array([yf])
        best_x = xf.copy()
        best_x_hist.append(best_x)
        best_f_hist.append(best_f[0])
        round_hist.append(icount+1)
        trackfunc_hist = record_trackfuncs(best_x,argsmin2,trackfunc,trackfunc_hist)
        minlist = [best_x]
        
        # add to startptlist if not already in
        if len(startptlistnew) == 0 or not np.any(np.isclose(xf,startptlistnew,atol=1e-3),axis=0).all():
            startptlistnew = [xf]
        print('New global min')
        print('Found new best min: ', np.around(xf,3))
        if findzero[2] == True: success_tunnel += 1
        tp_total = tunphases
        
    # Check if same level min
    elif best_f - glbtol <= yf <= best_f + glbtol and \
        not np.any(np.isclose(xf,minlist,atol=1e-2),axis=0).all():# and \
        #not np.any(np.isclose(xf,startptlistnew,atol=1e-3),axis=0).all(): #!!! add check that xf not already in tabulist of minima
        if yf < best_f[0]:
            best_f = np.array([yf])
            best_x = xf.copy()
            best_x_hist.append(best_x)
            best_f_hist.append(yf)
            round_hist.append(icount+1)
            trackfunc_hist = record_trackfuncs(best_x,argsmin2,trackfunc,trackfunc_hist)

        minlist.append(xf) #add to minlist
        startptlistnew.append(xf)
        print('Same level global min: ', np.around(xf,3))
        print(np.around(minlist,3))
        # tp_total -= tp_used
        if findzero[2] == True: success_tunnel += 1
        tp_total = tunphases

    # not a min
    elif yf >= best_f + glbtol:
        print('Not a global min: ', np.around(xf,3))
        trytunnel = True # don't tunnel from non-min
        nonewpt = 1 # don't keep minimizing from this pt
        if len(startptlistnew) == 0: startptlistnew.append(xf)
        if not np.any(np.isclose(xf,startptlistnew,atol=1e-3),axis=0).all():
            startptlistnew.append(xf) #keep as start pt if tricks tunneling lol
            tabulist.append(xf) #add to toplevel tabulist
        else:
            lt0 += lt0 #increase tabulist if ending up again at it
        tp_total -= tp_used
        #change start pt to best x?
    # repeated min
    else:
        print('Repeat min point: ', np.around(xf,3))
        nonewpt = 1
        esc_attempt +=1 #try again some times to escape min
        print('Escape attempts: ',esc_attempt)
        # l0 += l0 # try 12/8/22
        tp_total -= tp_used
        if esc_attempt >= 5: # Pick limit???
            trytunnel = False #don't repeat tunneling
    return best_x, best_f, best_x_hist, best_f_hist, minlist,\
        startptlistnew,round_hist, trackfunc_hist, tp_total, nonewpt, trytunnel,\
            success_tunnel, esc_attempt

def printiter(X):
    '''
    Callback function for built in optimizers to display each iteration value
    '''
    print(X)
    return []

# Python program to copy or clone a list
# Using the Slice Operator
def clone(li):
    li_copy = li[:]
    return li_copy

def record_trackfuncs(x,objargs,funclist,current):
    newvals = np.zeros([len(funclist),1])
    ii = 0
    for f in funclist:
        newvals[ii] = f(x,*objargs)
        ii += 1
    if len(current) == 0:
        return newvals
    else:
        return np.append(current,newvals,axis=1)
    # np.append(trackfunc_hist,trackfuncvals,axis=1)

def choosemin(x1,x2,f,fargs):
    f1 = f(x1,*fargs)
    f2 = f(x2,*fargs)
    if f1 < f2:
        return x1, f1
    else:
        return x2, f2

def sortstartpts(xlst,f,*args):
    fvals = []
    for x in xlst:
        fvals.append(f(x,*args))
    sortlst = [xx for _,xx in sorted(zip(fvals,xlst))]
    return sortlst

def wolfe(fk1,fk,alphak,gradfk1,gradfk,dk,c1,c2):
    if (fk1 <= fk + c1*alphak*np.dot(gradfk,dk)) and (np.dot(gradfk1,dk) >= c2*np.dot(gradfk,dk)):
        return True
    else:
        return False

def print_tun_step(k,xk,f1k,f2k,gf2k,Tkf1,Tkf2,gTf1k,gTf2k,eta,alphaiter,func):
    print('Iter: ',k)
    print('\tSwitch:',func)
    print('\txk: ',np.around(xk,3))
    print('\teta: ',eta)
    print('\talpha iters: ',alphaiter)
    print('\tnorm(gradTk f1): {0:5.5e}'.format(np.around(np.linalg.norm(gTf1k),3)))
    print('\tnorm(gradTk f2): {0:5.5e}'.format(np.around(np.linalg.norm(gTf2k),3)))
    print('\tTk f1: {0:5.5f}'.format(Tkf1))
    print('\tTk f2: {0:5.5e}'.format(Tkf2))
    print('\tf1(xk): {0:5.5e}'.format(f1k))
    print('\tf2(xk): {0:5.5e}'.format(f2k))
    print('\tgf2(xk): {0:5.5e}'.format(np.around(np.linalg.norm(gf2k),3)))
    return 0

def check_stop(icount,maxrounds,tstart,tlimit,minlist,stopfunc,*argsmin):
    if icount > maxrounds:
        print('Max rounds exceeded')
        return False
    if (time.time() - tstart) > tlimit:
        print('Time limit exceeded')
        return False
    if stopfunc(minlist,*argsmin) == True:
        print('Stop func evaluated to True')
        return False
    else:
        return True