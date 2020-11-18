# importing necessary libraries
import gc
import pandas as pd
from scipy.optimize import minimize as minim
import seaborn as sns
import numpy as np 
import numpy.random as rnd
import scipy.stats as st
import math as mt
import matplotlib.pyplot as plt
from scipy.integrate import quad,dblquad
gc.collect()
plt.close("all")

###à######################### Functions ##########################
### avoiding the repetition
def cpow(x1,x2):
    return np.power(x1,x2)

def div(x1,x2):
    return np.divide(x1,x2)

# Estimating the parameters
def dist_plot(data,color,bins=100,hist=True,kde=True,norm_hist=False,xlabel="X axis",ylabel="Y axis", title="Plot title"):
    sns.distplot(data,bins=bins,hist=hist,kde=kde,color=color,norm_hist=norm_hist)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
def line_plot(data,time,xlabel="X axis",ylabel="Y axis", title="Plot title"):
    sns.lineplot(time,data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
def estimate(data,epsilon,delta_t):
    jumps=[val for val in data if abs(val)>epsilon]
    no_jumps=[val for val in data if abs(val)<=epsilon]
    _lambda=len(jumps)/(len(data)*delta_t)
    sigma_d=np.std(no_jumps)/np.sqrt(delta_t)
    mu_d=div(2*np.mean(no_jumps)+delta_t*cpow(sigma_d,2),2*delta_t)
    sigma_j=np.sqrt(cpow(np.std(jumps),2)-cpow(sigma_d,2)*delta_t)
    mu_j=np.mean(jumps)-(mu_d-div(cpow(sigma_d,2),2))*delta_t
    return _lambda,mu_d,sigma_d,mu_j,sigma_j

# Density function 
def norm_density(data,exp,var):
    res=div(np.exp(div(-cpow(np.subtract(data,exp),2),2*var)),np.sqrt(2*mt.pi*var))
    return res

def density(data,_lambda,mu_d,sigma_d,mu_j,sigma_j,delta_t,k_max):
    if(_lambda>0):
        result=np.zeros((len(data),k_max))
        for i in range(1,k_max+1):
            result[:,i-1]=st.poisson.pmf(i,_lambda*delta_t)*norm_density(data,(mu_d-div(sigma_d**2,2))*delta_t+mu_j*i,sigma_d**2*delta_t+sigma_j**2*i)
        result=result.sum(axis=1)
    else:
        result=st.norm.pdf(data,delta_t*(mu_d-div(cpow(sigma_d,2),2)),cpow(sigma_d,2)*delta_t)
    return result

# maximum likelihood function
def log_likelihood_MJ(params,data,delta_t,k_max=1000):
    _lambda,mu_d,sigma_d,mu_j,sigma_j=params
    result=-np.sum(np.log(density(data,_lambda,mu_d,sigma_d,mu_j,sigma_j,delta_t,k_max)))
    return result


# Log_returns simulation
def simulated(params,size):
    _lambda,mu,sigma,mu_j,sigma_j,delta_t=params
    brnm=np.multiply(abs(sigma),rnd.normal(loc=0.0, scale=np.sqrt(delta_t), size=size))
    pois=rnd.poisson(_lambda*delta_t,size=size)
    norm=rnd.normal(np.multiply(pois,mu_j),np.multiply(np.sqrt(pois),np.abs(sigma_j)))
    const=(mu-sigma**2/2)*delta_t
    res=np.add(const,np.add(brnm,norm))
    return res

# non central-Moments
def moment(data,order):
    res=div(np.sum(cpow(data,order)),len(data))
    return res

    
def cuml1(data):
    return st.kstat(log_returns,1)

def cuml2(data):
    return st.kstat(log_returns,2)

def cuml3(data):
    return st.kstat(log_returns,3)

def cuml4(data):
    return st.kstat(log_returns,4)

def cuml5(data):
    res=moment(data,5)-5*moment(data,4)*moment(data,1)-10*moment(data,3)*moment(data,2)\
        +20*moment(data,3)*cpow(moment(data,1),2)+30*moment(data,1)*cpow(moment(data,2),2)\
        -60*moment(data,2)*cpow(moment(data,1),3)+24*cpow(moment(data,1),5)
    return res

def cuml6(data):
    res=moment(data,6)-6*moment(data,5)*moment(data,1)-15*moment(data,4)*moment(data,2)\
        +30*moment(data,4)*cpow(moment(data,1),2)-10*cpow(moment(data,3),2)+\
        120*moment(data,3)*moment(data,2)*moment(data,1)-120*moment(data,3)*cpow(moment(data,1),3)\
        +30*cpow(moment(data,2),2)-270*cpow(moment(data,2),2)*cpow(moment(data,1),2)+\
        360*moment(data,2)*cpow(moment(data,1),4)-120*cpow(moment(data,1),6)
    return res
################################ Data processing and parameters estimation #############################
    
# Getting the exchange rate data DAI-ETHER
df = pd.read_csv('')

# Plotting the exchange rate of DAI / Ether from 19/11/2019 to 26/06/2020
plt.figure()
sns.lineplot(x=df.timestamp, y=df.high, data=df)

# Getting log returns from data
log_returns=np.diff(np.log(df.close))
sns.set_style("darkgrid")
plt.figure()
sns.distplot(log_returns);
#### Maximum likelihood method
# initial threshold for jumps
delta_t=1/1440# Define Delta t
num_points=15
epsilon_interval=np.linspace(0.01,0.15,num_points)


# initial estimate for the parameters
lambda_list,mu_list,sigma_list,mu_j_list,sigma_j_list=[],[],[],[],[]
for i in range(num_points):
    initial_guess=estimate(log_returns,epsilon_interval[i],delta_t)
    min_var=minim(log_likelihood_MJ, initial_guess,args=(log_returns,delta_t),method='Nelder-Mead')
    lambda_est,mu_est,sigma_est,mu_j_est,sigma_j_est=min_var.x
    lambda_list.append(lambda_est)
    mu_list.append(mu_est)
    sigma_list.append(sigma_est)
    mu_j_list.append(mu_j_est)
    sigma_j_list.append(sigma_j_est)
# verifying the result and robustness
plt.figure()
dist_plot(lambda_list,'b',None,True,False,"Value","Frequency","Estimated lambda distribution")
plt.figure()
dist_plot(mu_list,'b',None,True,False,"Value","Frequency","Estimated Mu distribution")
plt.figure()
dist_plot(np.abs(sigma_list),'b',None,True,False,"Value","Frequency","Estimated Sigma distribution")
plt.figure()
dist_plot(mu_j_list,'b',None,True,False,"Value","Frequency","Estimated Mu_j distribution")
plt.figure()
dist_plot(np.abs(sigma_j_list),'b',None,True,False,"Value","Frequency","Estimated Sigma_j distribution")
print("Lambda Mean: {m:8.8f},  std: {s:8.8f}". 
      format(m = np.mean(lambda_list), s= np.std(lambda_list)))
print("Mu Mean: {m:8.8f},  std: {s:8.8f}". 
      format(m=np.mean(mu_list), s=np.std(mu_list)))
print("Sigma Mean: {m:8.8f},  std: {s:8.8f}". 
      format(m=np.mean(np.abs(sigma_list)), s=np.std(np.abs(sigma_list))))
print("MU_j Mean: {m:8.8f},  std: {s:8.8f}". 
      format(m=np.mean(mu_j_list),s=np.std(mu_j_list)))
print("Sigma_j Mean: {m:8.8f},  std: {s:8.8f}". 
      format(m=np.mean(np.abs(sigma_j_list)),s=np.std(np.abs(sigma_j_list))))



###################### Cumulants ######
k1,k2,k3,k4,k5,k6=cuml1(log_returns),cuml2(log_returns),cuml3(log_returns),cuml4(log_returns),cuml5(log_returns),cuml6(log_returns)
# Just to reduce line length

###################à cumulants matching for MJD model ############
c4=27*cpow(k5,3)*cpow(delta_t,4)
c3=-27*k5*cpow(delta_t,3)*(cpow(k4,3)-6*k3*k4*k5+4*k2*cpow(k5,2))
c2=cpow(delta_t,2)*(72*cpow(k5,2)*cpow(k3,3)+225*cpow(k4*k3,2)*k5\
    -45*cpow(k4,4)*k3+486*k3*k2*k4*cpow(k5,2)+ 81*k2*k5*cpow(k4,3)+162*cpow(k2,2)*cpow(k5,3))
c1= delta_t*(168*k4*k5*cpow(k3,4)+40*cpow(k4*k3,3)-144*k2*cpow(k5*k3,2)\
        -450*k2*k5*cpow(k4*k3,2)+ 90*k3*k2*cpow(k4,4)+486*cpow(k2*k5,2)*k4*k3\
        -81*k5*cpow(k2,2)*cpow(k4,3)-108*cpow(k2*k5,3))
c0=16*k5*cpow(k3,6)+48*cpow(k4,2)*cpow(k3,5)-168*k2*k4*k5*cpow(k3,4)\
    + 72*cpow(k2*k5,2)*cpow(k3,3)-40*k2*cpow(k4*k3,3)\
    + 225*cpow(k2*k4*k3,2)*k5-45*cpow(k2,2)*cpow(k4,4)*k3\
    - 162*cpow(k2*k5,2)*k4*k3 + 27*cpow(k2*k5,3)*k5\
    + 27*cpow(k2,4)*cpow(k5,3)
    
#######à Parameters estimation #################
### tau=sigma_d squared
coef_tau=[c4,c3,c2,c1,c0]
roots_tau=np.roots(coef_tau)
tau_est=[i for i in roots_tau if i.real>0 and i.imag==0][0].real
mu_j_est=div(3*k5*(delta_t*tau_est-k2)+4*k3*k4,3*k4*(tau_est*delta_t-k2)+4*cpow(k3,2))

####### Gamma = sigma_j_squared
gamma=abs(div(k3*k5-cpow(k4,2),3*k4*(tau_est*delta_t-k2)+4*cpow(k3,2)))
mu_d_est=div(k3*(7*k4*(delta_t*tau_est-k2)-k5*(2*k1+delta_t*tau_est))\
                +cpow(k4,2)*(2*k1+delta_t*tau_est)+3*k5*cpow(k2-delta_t*tau_est,2)+4*cpow(k3,3)\
               ,2*delta_t*(cpow(k4,2)-k3*k5))
lambda_est=abs(div(div(k1,delta_t)-mu_d_est+div(tau_est,2),mu_j_est))

#################### Simplex minimisation for LNJD/MJD model ###############
sigma_j_est=np.sqrt(gamma)
sigma_d_est=np.sqrt(tau_est)
initial_guess=[ lambda_est,mu_d_est,sigma_d_est,mu_j_est,sigma_j_est]
min_simp=minim(log_likelihood_MJ, initial_guess,args=(log_returns,delta_t),method='Nelder-Mead')
lambda_opt,mu_opt,sigma_opt,mu_j_opt,sigma_j_opt=min_simp.x

min_pow=minim(log_likelihood_MJ, initial_guess,args=(log_returns,delta_t),method='Powell')
lambda_opt_p,mu_opt_p,sigma_opt_p,mu_j_opt_p,sigma_j_opt_p=min_pow.x

###################### PBJD Model ###############
### Parameters estimation#####################à
coef=[div(div(cpow(k5,2),5)-div(k4*k6,6),20),div(div(k4*k5,2)+div(k3*k6,3),10),\
      div(cpow(k4,2),4)-div(k3*k5,5)]
roots=np.roots(coef)
ru=roots[[i for i in range(len(roots)) if roots[i]>0].pop()]
coef_down=[k5,-5*k4,5*k4*ru-20*k3]
roots_down=np.roots(coef_down)
rd=roots_down[[i for i in range(len(roots_down)) if roots_down[i]>0].pop()]
lambda_d=div(cpow(rd,4)*(div(k4*ru,24*delta_t)-div(k3,6*delta_t)),ru+rd)
lambda_u=cpow(ru,3)*(div(k3,6*delta_t)+div(lambda_d,cpow(rd,3)))
sigma=np.sqrt(abs(div(k2,delta_t)-2*div(lambda_u,cpow(ru,2))-2*div(lambda_d,cpow(rd,2))))
mu_pb=div(k1,delta_t)+0.5*sigma-div(lambda_u,ru)+div(lambda_d,rd)

######### Density function for PBJD model #########################
## defining the 4 densities for easy readibility
def f00(data,mu,sigma,delta_t):
    return norm_density(data,mu*delta_t-0.5*cpow(sigma,2)*delta_t,cpow(sigma,2)*delta_t)

def int1(x,data,count,mu,rd,sigma,delta_t):
    return cpow(-x,count-1)*np.exp(rd*x-div(cpow(data-x-mu*delta_t+0.5*cpow(sigma,2)*delta_t,2),2*delta_t*cpow(sigma,2)))

def f0n(data,count,mu,rd,sigma,delta_t,max_iter,a,tol):
    citer,ev1,ev2=0,quad(int1,a,0,args=(data,count,mu,rd,sigma,delta_t))[0],quad(int1,a*1.5,0,args=(data,count,mu,rd,sigma,delta_t))[0]
    while(abs(ev1-ev2)>tol and (citer<max_iter) ):
        citer+=1
        a*=1.5
        ev1,ev2=ev2,quad(int1,a,0,args=(data,count,mu,rd,sigma,delta_t))[0]
    return div(cpow(rd,count),mt.factorial(count-1)*np.sqrt(2*mt.pi*delta_t*cpow(sigma,2)))*ev2
    
def int2(x,data,count,mu,ru,sigma,delta_t):
    return cpow(x,count-1)*np.exp(-ru*x-div(cpow(data-x-mu*delta_t+0.5*cpow(sigma,2)*delta_t,2),2*delta_t*cpow(sigma,2)))

def fm0(data,count,mu,ru,sigma,delta_t,max_iter,b,tol):
    citer,ev1,ev2=0,quad(int1,0,b,args=(data,count,mu,rd,sigma,delta_t))[0],quad(int1,0,b*1.5,args=(data,count,mu,rd,sigma,delta_t))[0]
    while(abs(ev1-ev2)>tol and (citer<max_iter) ):
        citer+=1
        b*=1.5
        ev1,ev2=ev2,quad(int1,0,b,args=(data,count,mu,rd,sigma,delta_t))[0]
    return div(cpow(ru,count),mt.factorial(count-1)*np.sqrt(2*mt.pi*delta_t*cpow(sigma,2)))*ev2

def betaexp(y,shift,count_m,count_n,ru,rd):
    return cpow(-y,count_n-1)*cpow(shift-y,count_m-1)*mt.exp((ru+rd)*y)

def int3(x,y,data,count_m,count_n,mu,ru,rd,sigma,delta_t):
    return betaexp(y, x, count_m, count_n, ru, rd)*mt.exp(-ru*x-div(cpow(data-x-mu*delta_t+0.5*cpow(sigma,2)*delta_t,2),2*delta_t*cpow(sigma,2)))

def fmn(data,count_m,count_n,mu,ru,rd,sigma,delta_t,max_iter,tol,a):
    citer,ev1,ev2=0,dblquad(int3,-a,a,lambda x: -a,lambda x: min(0,x),args=(data,count_m,count_n,mu,ru,rd,sigma,delta_t))[0],\
        dblquad(int3,-1.5*a,1.5*a,lambda x: -a,lambda x: min(0,x),args=(data,count_m,count_n,mu,ru,rd,sigma,delta_t))[0]
    while(abs(ev1-ev2)<tol and (citer<max_iter)):
        citer+=1
        a*=1.5
        ev1=ev2
        ev2=dblquad(int3,-a,a,lambda x: -a,lambda x: min(0,x),args=(data,count_m,count_n,mu,ru,rd,sigma,delta_t))[0]
    return div(cpow(ru,count_m)*cpow(rd,count_n),mt.factorial(count_m-1)*mt.factorial(count_n-1)*np.sqrt(2*mt.pi*delta_t*cpow(sigma,2)))*ev2

def density_pbjd(data,mu,ru,rd,lambda_u,lambda_d,sigma,delta_t,ran=10,a=-10,b=10,tol=1e-8,stol=1e-10,max_iter=1000):
    count,res_aux=0,0
    
    if(lambda_u>0):
        print("dens1")
        exran=ran
        lres1=[st.poisson.pmf(i,lambda_d*delta_t)*f0n(data,i,mu,rd,sigma,delta_t,max_iter,b,tol)for i in range(1,exran)]
        lres2=[st.poisson.pmf(i,lambda_d*delta_t)*f0n(data,i,mu,rd,sigma,delta_t,max_iter,b,tol) for i in range(1,exran+1)]
        while((count<max_iter) and (2*abs(lres2[-1])>stol*(abs(sum(lres1))+abs(sum(lres2))))):
              exran+=1
              count+=1
              lres1=lres2
              lres2.append(st.poisson.pmf(exran,lambda_d*delta_t)*f0n(data,exran,mu,rd,sigma,delta_t,max_iter,a,tol))
        res_aux+=mt.exp(-lambda_u*delta_t)*sum(lres2)
        
    if(lambda_d>0):
        print("dens2")
        count=0
        exran=ran
        lres1=[st.poisson.pmf(i,lambda_u*delta_t)*fm0(data,i,mu,ru,sigma,delta_t,max_iter,a,tol)for i in range(1,exran)]
        lres2=[st.poisson.pmf(i,lambda_u*delta_t)*fm0(data,i,mu,ru,sigma,delta_t,max_iter,a,tol) for i in range(1,exran+1)]
        while((count<max_iter) and (2*abs(lres2[-1])>stol*(abs(sum(lres1))+abs(sum(lres2))))):
              exran+=1
              count+=1
              lres1=lres2
              lres2.append(st.poisson(exran,lambda_u*delta_t)*f0n(data,exran,mu,rd,sigma,delta_t,max_iter,a,tol))
        res_aux+=mt.exp(-lambda_d*delta_t)*sum(lres2)
        
    if(lambda_u>0 and lambda_d >0):
        print("dens3")
        count=0
        print("first half")
        exran,exran1=ran,ran
        lres1=[st.poisson.pmf(i,lambda_u*delta_t)*st.poisson.pmf(j,lambda_d*delta_t)*fmn(data,i,j,mu,ru,rd,sigma,delta_t,max_iter,tol,a)for i in range(1,exran) for j in range(1,exran1)]
        print("first one")
        lres2=[st.poisson.pmf(i,lambda_u*delta_t)*st.poisson.pmf(j,lambda_d*delta_t)*fmn(data,i,j,mu,ru,rd,sigma,delta_t,max_iter,tol,a) for i in range(1,exran+1) for j in range(1,exran1)]
        print("second one")
        while(count<max_iter and 2*abs(lres2[-1])>stol*(abs(sum(lres1))+abs(sum(lres2)))):
            print("second half")
            exran1+=1
            count1=0
            lres1_aux=[st.poisson.pmf(i,lambda_u*delta_t)*st.poisson.pmf(j,lambda_d*delta_t)*fmn(data,i,j,mu,ru,rd,sigma,delta_t,max_iter,tol,a)for i in range(1,exran) for j in range(1,exran1)]
            lres2_aux=[st.poisson.pmf(i,lambda_u*delta_t)*st.poisson.pmf(j,lambda_d*delta_t)*fmn(data,i,j,mu,ru,rd,sigma,delta_t,max_iter,tol,a) for i in range(1,exran+1) for j in range(1,exran1)]
            while(count1<max_iter and (2*abs(lres2_aux[-1])>stol*(abs(sum(lres1_aux))+abs(sum(lres2_aux))))):
                exran1+=1
                count1+=1
                lres1_aux=lres2_aux
                lres2_aux=[st.poisson.pmf(i,lambda_u*delta_t)*st.poisson.pmf(j,lambda_d*delta_t)*fmn(data,i,j,mu,ru,rd,sigma,delta_t,max_iter,tol,a) for i in range(1,exran) for j in range(1,exran1)]
            exran+=1
            count+=1
            lres1=lres2
            lres2=[st.poisson.pmf(i,lambda_u*delta_t)*st.poisson.pmf(j,lambda_d*delta_t)*fmn(data,i,j,mu,ru,rd,sigma,delta_t,max_iter,tol,a) for i in range(1,exran) for j in range(1,exran1)]
        res_aux+=lres2
    res=mt.exp((-lambda_u-lambda_d)*delta_t)*f00(data,mu,sigma, delta_t)+res_aux
    return res

def pbjd_likelihood(params,delta_t,data):
    mu,ru,rd,lambda_u,lambda_d,sigma=params
    return -1*sum([np.log(density_pbjd(sample,mu,ru,rd,lambda_u,lambda_d,sigma,delta_t)) for sample in data])
initial_guess_pb=[mu_pb,ru,rd,lambda_u,lambda_d,sigma]
# res=pbjd_likelihood(initial_guess_pb, delta_t, log_returns)
min_pow_pbjd=minim(pbjd_likelihood, initial_guess_pb,args=(delta_t,log_returns),method='Powell')
lambda_opt_p,mu_opt_p,sigma_opt_p,mu_j_opt_p,sigma_j_opt_p=min_pow_pbjd.x

############################## Standard deviation of the estimated parameters of theMJD model ################à

def dlambda(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax):
    res=0
    for i in range(1,kmax+1):
       res+=st.poisson.pmf(i,lambda_opt*delta_t)*(div(i,lambda_opt)-delta_t)*norm_density(sample,(mu-div(sigma**2,2))*delta_t+mu_j*i,sigma**2*delta_t+sigma_j**2*i)
    return res

def dmu(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax):
    res=0
    for i in range(1,kmax+1):
        res+=st.poisson.pmf(i,delta_t*lambda_opt)*norm_density(sample,(mu-div(sigma**2,2))*delta_t+mu_j*i,sigma**2*delta_t+sigma_j**2*i)\
            *div(delta_t*(sample-mu*delta_t-mu_j*i+div(cpow(sigma,2),2)*delta_t),cpow(sigma,2)*delta_t+cpow(sigma_j,2)*i)
    return res

def dmuj(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax):
    res=0
    for i in range(1,kmax+1):
        res+=st.poisson.pmf(i,delta_t*lambda_opt)*norm_density(sample,(mu-div(sigma**2,2))*delta_t+mu_j*i,sigma**2*delta_t+sigma_j**2*i)\
            *div(i*(sample-mu*delta_t-mu_j*i+div(cpow(sigma,2),2)*delta_t),cpow(sigma,2)*delta_t+cpow(sigma_j,2)*i)
    return res

def dsigmaj(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax):
    res=0
    for i in range(1,kmax+1):
        res+=st.poisson.pmf(i,delta_t*lambda_opt)*norm_density(sample,(mu-div(sigma**2,2))*delta_t+mu_j*i,sigma**2*delta_t+sigma_j**2*i)*sigma_j*i\
            *cpow(div(sample-mu*delta_t-mu_j*i+div(cpow(sigma,2),2)*delta_t,cpow(sigma,2)*delta_t+cpow(sigma_j,2)*i),2)
    return res

def dsigma(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax):
    res=0
    for i in range(1,kmax+1):
        res+=st.poisson.pmf(i,delta_t*lambda_opt)*norm_density(sample,(mu-div(sigma**2,2))*delta_t+mu_j*i,sigma**2*delta_t+sigma_j**2*i)\
            *div(delta_t*sigma*(sample-mu*delta_t-mu_j*i+div(cpow(sigma,2),2)*delta_t)*(sample-mu*delta_t-mu_j*i-div(cpow(sigma,2),2)*delta_t-cpow(sigma_j,2)*i),cpow(cpow(sigma,2)*delta_t+cpow(sigma_j,2),2))
    return res
 
def dlambda2(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax):
    res=0
    for i in range(1,kmax+1):
        res+=st.poisson.pmf(i,lambda_opt*delta_t)*div(cpow(i,2)-i*(2*delta_t*lambda_opt+1)+cpow(delta_t*lambda_opt,2),cpow(lambda_opt,2))*norm_density(sample,(mu-div(sigma**2,2))*delta_t+mu_j*i,sigma**2*delta_t+sigma_j**2*i)
    return res

def dlambdadmu(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax):
    res=0
    for i in range(1,kmax+1):
        res+=st.poisson.pmf(i,lambda_opt*delta_t)*(div(i,lambda_opt)-delta_t)*norm_density(sample,(mu-div(sigma**2,2))*delta_t+mu_j*i,sigma**2*delta_t+sigma_j**2*i)\
            *div(delta_t*(sample-mu*delta_t-mu_j*i+div(cpow(sigma,2),2)*delta_t),cpow(sigma,2)*delta_t+cpow(sigma_j,2)*i)
    return res

def dlambdadmuj(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax):
    res=0
    for i in range(1,kmax+1):
        res+=st.poisson.pmf(i,lambda_opt*delta_t)*(div(i,lambda_opt)-delta_t)*norm_density(sample,(mu-div(sigma**2,2))*delta_t+mu_j*i,sigma**2*delta_t+sigma_j**2*i)\
            *div(i*(sample-mu*delta_t-mu_j*i+div(cpow(sigma,2),2)*delta_t),cpow(sigma,2)*delta_t+cpow(sigma_j,2)*i)
    return res

def dlambdadsigmaj(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax):
    res=0
    for i in range(1,kmax+1):
        res+=st.poisson.pmf(i,lambda_opt*delta_t)*(div(i,lambda_opt)-delta_t)*norm_density(sample,(mu-div(sigma**2,2))*delta_t+mu_j*i,sigma**2*delta_t+sigma_j**2*i)\
            *sigma_j*i*cpow(div(sample-mu*delta_t-mu_j*i+div(cpow(sigma,2),2)*delta_t,cpow(sigma,2)*delta_t+cpow(sigma_j,2)*i),2)
    return res

def dlambdadsigma(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax):
    res=0
    for i in range(1,kmax+1):
        res+=st.poisson.pmf(i,lambda_opt*delta_t)*(div(i,lambda_opt)-delta_t)*norm_density(sample,(mu-div(sigma**2,2))*delta_t+mu_j*i,sigma**2*delta_t+sigma_j**2*i)\
            *div(delta_t*sigma*(sample-mu*delta_t-mu_j*i+div(cpow(sigma,2),2)*delta_t)*(sample-mu*delta_t-mu_j*i-div(cpow(sigma,2),2)*delta_t-cpow(sigma_j,2)*i),cpow(cpow(sigma,2)*delta_t+cpow(sigma_j,2),2))
    return res

def dmu2(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax):
    res=0
    for i in range(1,kmax+1):
        res+=st.poisson.pmf(i,lambda_opt*delta_t)*norm_density(sample,(mu-div(sigma**2,2))*delta_t+mu_j*i,sigma**2*delta_t+sigma_j**2*i)\
            *(cpow(div(delta_t*(sample-mu*delta_t-mu_j*i+div(cpow(sigma,2),2)*delta_t),cpow(sigma,2)*delta_t+cpow(sigma_j,2)*i),2)-div(cpow(delta_t,2),cpow(sigma,2)*delta_t+cpow(sigma_j,2)*i))
    return res

def dmudmuj(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax):
    res=0
    for i in range(1,kmax+1):
        res+=st.poisson.pmf(i,lambda_opt*delta_t)*norm_density(sample,(mu-div(sigma**2,2))*delta_t+mu_j*i,sigma**2*delta_t+sigma_j**2*i)\
            *(delta_t*i*cpow(div(sample-mu*delta_t-mu_j*i+div(cpow(sigma,2),2)*delta_t,cpow(sigma,2)*delta_t+cpow(sigma_j,2)*i),2)-div(delta_t*i,cpow(sigma,2)*delta_t+cpow(sigma_j,2)*i))
    return res

def dmudsigma_j(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax):
    res=0
    for i in range(1,kmax+1):
        res+=st.poisson.pmf(i,lambda_opt*delta_t)*norm_density(sample,(mu-div(sigma**2,2))*delta_t+mu_j*i,sigma**2*delta_t+sigma_j**2*i)\
            *sigma_j*delta_t*i*div(sample-mu*delta_t-mu_j*i+div(cpow(sigma,2),2)*delta_t,cpow(cpow(sigma,2)*delta_t+cpow(sigma_j,i)*i,2))*(div(cpow(sample-mu*delta_t-mu_j*i+div(cpow(sigma,2),2)*delta_t,2),cpow(sigma,2)*delta_t+cpow(sigma_j,2)*i)-2)
    return res

def dmudsigma(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax):
    res=0
    for i in range(1,kmax+1):
       res+=st.poisson.pmf(i,delta_t*lambda_opt)*norm_density(sample,(mu-div(sigma**2,2))*delta_t+mu_j*i,sigma**2*delta_t+sigma_j**2*i)\
            *(div(delta_t*(sample-mu*delta_t-mu_j*i+div(cpow(sigma,2),2)*delta_t),cpow(sigma,2)*delta_t+cpow(sigma_j,2)*i)*div(delta_t*sigma*(sample-mu*delta_t-mu_j*i+div(cpow(sigma,2),2)*delta_t)*(sample-mu*delta_t-mu_j*i-div(cpow(sigma,2),2)*delta_t-cpow(sigma_j,2)*i),cpow(cpow(sigma,2)*delta_t+cpow(sigma_j,2),2))\
             +div(cpow(delta_t,2)*sigma*(cpow(sigma_j,2)*i-2*(sample-mu*delta_t-mu_j*i)),cpow(cpow(sigma,2)*delta_t+cpow(sigma_j,2)*i,2)))
    return res

def dmu_j2(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax):
    res=0
    for i in range(1,kmax+1):
        res+=st.poisson.pmf(i,lambda_opt*delta_t)*norm_density(sample,(mu-div(sigma**2,2))*delta_t+mu_j*i,sigma**2*delta_t+sigma_j**2*i)\
            *cpow(i,2)*(cpow(div((sample-mu*delta_t-mu_j*i+div(cpow(sigma,2),2)*delta_t),cpow(sigma,2)*delta_t+cpow(sigma_j,2)*i),2)-div(1,cpow(sigma,2)*delta_t+cpow(sigma_j,2)*i))
    return res

def dmu_jdsigma_j(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax):
    res=0
    for i in range(1,kmax+1):
        res+=st.poisson.pmf(i,lambda_opt*delta_t)*norm_density(sample,(mu-div(sigma**2,2))*delta_t+mu_j*i,sigma**2*delta_t+sigma_j**2*i)\
            *sigma_j*cpow(i,2)*div(sample-mu*delta_t-mu_j*i+div(cpow(sigma,2),2)*delta_t,cpow(cpow(sigma,2)*delta_t+cpow(sigma_j,i)*i,2))*(div(cpow(sample-mu*delta_t-mu_j*i+div(cpow(sigma,2),2)*delta_t,2),cpow(sigma,2)*delta_t+cpow(sigma_j,2)*i)-2)
    return res

def dmu_jdsigma(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax):
    res=0
    for i in range(1,kmax+1):
        res+=st.poisson.pmf(i,lambda_opt*delta_t)*norm_density(sample,(mu-div(sigma**2,2))*delta_t+mu_j*i,sigma**2*delta_t+sigma_j**2*i)\
           *(div(i*(sample-mu*delta_t-mu_j*i+div(cpow(sigma,2),2)*delta_t),cpow(sigma,2)*delta_t+cpow(sigma_j,2)*i)*div(delta_t*sigma*(sample-mu*delta_t-mu_j*i+div(cpow(sigma,2),2)*delta_t)*(sample-mu*delta_t-mu_j*i-div(cpow(sigma,2),2)*delta_t-cpow(sigma_j,2)*i),cpow(cpow(sigma,2)*delta_t+cpow(sigma_j,2),2))\
            +div(i*delta_t*sigma*(cpow(sigma_j,2)*i-2*(sample-mu*delta_t-mu_j*i)),cpow(cpow(sigma,2)*delta_t+cpow(sigma_j,2)*i,2)))
    return res

def dsigma_j2(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax):
    res=0
    for i in range(1,kmax+1):
        res+=st.poisson.pmf(i,lambda_opt*delta_t)*norm_density(sample,(mu-div(sigma**2,2))*delta_t+mu_j*i,sigma**2*delta_t+sigma_j**2*i)\
            *(cpow(sigma_j*i,2)*cpow(div(sample-sigma*delta_t-i*mu_j+div(cpow(sigma,2),2)*delta_t,cpow(sigma,2)*delta_t+cpow(sigma_j,2)*i),4)+cpow(sample-mu*delta_t-mu_j*i+div(cpow(sigma,2),2)*delta_t,2)*div(i*(cpow(sigma,2)*delta_t+cpow(sigma_j,2)*i)-2*cpow(sigma_j*i,2),cpow(cpow(sigma,2)*delta_t+cpow(sigma_j,2)*i,3)))
    return res

def dsigma_jdsigma(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax):
    res=0
    for i in range(1,kmax+1):
        res+=st.poisson.pmf(i,lambda_opt*delta_t)*norm_density(sample,(mu-div(sigma**2,2))*delta_t+mu_j*i,sigma**2*delta_t+sigma_j**2*i)*sigma_j*i*delta_t*sigma\
            *(cpow(div(sample-mu*delta_t-mu_j*i+div(cpow(sigma,2),2)*delta_t,cpow(sigma,2)*delta_t+cpow(sigma_j,2)*i),2)*div((sample-mu*delta_t-mu_j*i+div(cpow(sigma,2),2)*delta_t)*(sample-mu*delta_t-mu_j*i-div(cpow(sigma,2),2)*delta_t-cpow(sigma_j,2)*i),cpow(cpow(sigma,2)*delta_t+cpow(sigma_j,2),2))\
            +div(2*(sample-mu*delta_t-mu_j*i+div(cpow(sigma,2),2)*delta_t),cpow(cpow(sigma,2)*delta_t+cpow(sigma_j,2)*i,3))*(cpow(sigma_j,2)*i-2*(sample-mu*delta_t-mu_j*i)))
    return res

def dsigma2(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax):
    res=0
    for i in range(1,kmax+1):
        res+=st.poisson.pmf(i,lambda_opt*delta_t)*norm_density(sample,(mu-div(sigma**2,2))*delta_t+mu_j*i,sigma**2*delta_t+sigma_j**2*i)\
        *(cpow(div(delta_t*sigma*(sample-mu*delta_t-mu_j*i+div(cpow(sigma,2),2)*delta_t)*(sample-mu*delta_t-mu_j*i-div(cpow(sigma,2),2)*delta_t-cpow(sigma_j,2)*i),cpow(cpow(sigma,2)*delta_t+cpow(sigma_j,2)*i,2)),2)\
          +div((delta_t*(sample-mu*delta_t-mu_j*i+div(cpow(sigma,2),2)*delta_t)*(sample-mu*delta_t-mu_j*i-div(cpow(sigma,2),2)*delta_t-cpow(sigma_j,2)*i)-cpow(sigma*delta_t,2)*(cpow(sigma,2)*delta_t+cpow(sigma_j,2)*i))*(cpow(sigma,2)*delta_t+cpow(sigma_j,2)*i)\
               -4*sigma*delta_t*div(delta_t*sigma*(sample-mu*delta_t-mu_j*i+div(cpow(sigma,2),2)*delta_t)*(sample-mu*delta_t-mu_j*i-div(cpow(sigma,2),2)*delta_t-cpow(sigma_j,2)*i)*(cpow(sigma,2)*delta_t+cpow(sigma_j,2)*i),cpow(cpow(sigma,2)*delta_t+cpow(sigma_j,2)*i,2)),cpow(cpow(sigma,2)*delta_t+cpow(sigma_j,2)*i,3)))
    return res

def dsigmadsigmaj(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax):
    res=0
    for i in range(1,kmax+1):
        res+=st.poisson.pmf(i,lambda_opt*delta_t)*norm_density(sample,(mu-div(sigma**2,2))*delta_t+mu_j*i,sigma**2*delta_t+sigma_j**2*i)*sigma*delta_t*sigma_j*i\
            *(cpow(div(sample-mu*delta_t-mu_j*i+div(cpow(sigma,2),2)*delta_t,cpow(sigma,2)*delta_t+cpow(sigma_j,2)*i),2)*div((sample-mu*delta_t-mu_j*i+div(cpow(sigma,2),2)*delta_t)*(sample-mu*delta_t-mu_j*i-div(cpow(sigma,2),2)*delta_t-cpow(sigma_j,2)*i),cpow(cpow(sigma,2)*delta_t+cpow(sigma_j,2),2))\
              -2*div(sample-mu*delta_t-mu_j*i+div(cpow(sigma,2),2),cpow(cpow(sigma,2)*delta_t+cpow(sigma_j,2)*i,3))*(2*(sample-mu*delta_t-mu_j*i*delta_t)-cpow(sigma_j,2)*i))
    return res


def flambda2(sample,sample_dens,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax):
    return div(dlambda2(sample, lambda_opt, mu, sigma, mu_j, sigma_j, delta_t, kmax)*sample_dens-cpow(dlambda(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax),2),cpow(sample_dens,2))

def flambdamu(sample,sample_dens,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax):
    return div(dlambdadmu(sample, lambda_opt, mu, sigma, mu_j, sigma_j, delta_t, kmax)*sample_dens-dlambda(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax)*dmu(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax),cpow(sample_dens,2))

def flambdamuj(sample,sample_dens,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax):
    return div(dlambdadmuj(sample, lambda_opt, mu, sigma, mu_j, sigma_j, delta_t, kmax)*sample_dens-dlambda(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax)*dmuj(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax),cpow(sample_dens,2))

def flambdasigma(sample,sample_dens,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax):
    return div(dlambdadsigma(sample, lambda_opt, mu, sigma, mu_j, sigma_j, delta_t, kmax)*sample_dens-dlambda(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax)*dsigma(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax),cpow(sample_dens,2))

def flambdasigma_j(sample,sample_dens,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax):
    return div(dlambdadsigmaj(sample, lambda_opt, mu, sigma, mu_j, sigma_j, delta_t, kmax)*sample_dens-dlambda(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax)*dsigmaj(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax),cpow(sample_dens,2))


def fmu2(sample,sample_dens,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax):
    return div(dmu2(sample, lambda_opt, mu, sigma, mu_j, sigma_j, delta_t, kmax)*sample_dens-cpow(dmu(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax),2),cpow(sample_dens,2))

def fmumuj(sample,sample_dens,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax):
    return div(dmudmuj(sample, lambda_opt, mu, sigma, mu_j, sigma_j, delta_t, kmax)*sample_dens-dmu(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax)*dmuj(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax),cpow(sample_dens,2))

def fmusigma(sample,sample_dens,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax):
    return div(dmudsigma(sample, lambda_opt, mu, sigma, mu_j, sigma_j, delta_t, kmax)*sample_dens-dmu(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax)*dsigma(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax),cpow(sample_dens,2))

def fmusigma_j(sample,sample_dens,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax):
    return div(dmudsigma_j(sample, lambda_opt, mu, sigma, mu_j, sigma_j, delta_t, kmax)*sample_dens-dmu(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax)*dsigmaj(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax),cpow(sample_dens,2))


def fmuj2(sample,sample_dens,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax):
    return div(dmu_j2(sample, lambda_opt, mu, sigma, mu_j, sigma_j, delta_t, kmax)*sample_dens-cpow(dmuj(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax),2),cpow(sample_dens,2))

def fmujsigma(sample,sample_dens,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax):
    return div(dmu_jdsigma(sample, lambda_opt, mu, sigma, mu_j, sigma_j, delta_t, kmax)*sample_dens-dmuj(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax)*dsigma(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax),cpow(sample_dens,2))

def fmujsigma_j(sample,sample_dens,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax):
    return div(dmu_jdsigma_j(sample, lambda_opt, mu, sigma, mu_j, sigma_j, delta_t, kmax)*sample_dens-dmuj(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax)*dsigmaj(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax),cpow(sample_dens,2))


def fsigmaj2(sample,sample_dens,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax):
    return div(dsigma_j2(sample, lambda_opt, mu, sigma, mu_j, sigma_j, delta_t, kmax)*sample_dens-cpow(dsigma(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax),2),cpow(sample_dens,2))

def fsigmajsigma(sample,sample_dens,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax):
    return div(dsigma_jdsigma(sample, lambda_opt, mu, sigma, mu_j, sigma_j, delta_t, kmax)*sample_dens-dsigmaj(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax)*dsigma(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax),cpow(sample_dens,2))


def fsigma2(sample,sample_dens,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax):
    return div(dsigma2(sample, lambda_opt, mu, sigma, mu_j, sigma_j, delta_t, kmax)*sample_dens-cpow(dsigma(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax),2),cpow(sample_dens,2))

def fsigmasigma_j(sample,sample_dens,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax):
    return div(dsigmadsigmaj(sample, lambda_opt, mu, sigma, mu_j, sigma_j, delta_t, kmax)*sample_dens-dsigma(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax)*dsigmaj(sample,lambda_opt,mu,sigma,mu_j,sigma_j,delta_t,kmax),cpow(sample_dens,2))

#### information matrix ##############
dens=density(log_returns,lambda_opt,mu_opt,sigma_opt,mu_j_opt,sigma_j_opt,delta_t,100)
comp11=sum([flambda2(log_returns[i],dens[i],lambda_opt,mu_opt,sigma_opt,mu_j_opt, sigma_j_opt, delta_t,100) for i in range(len(log_returns)) if not mt.isnan(flambda2(log_returns[i],dens[i],lambda_opt,mu_opt,sigma_opt,mu_j_opt, sigma_j_opt, delta_t,100))])
comp12=sum([flambdamu(log_returns[i],dens[i],lambda_opt,mu_opt,sigma_opt,mu_j_opt, sigma_j_opt, delta_t,100) for i in range(len(log_returns))if not mt.isnan(flambdamu(log_returns[i],dens[i],lambda_opt,mu_opt,sigma_opt,mu_j_opt, sigma_j_opt, delta_t,100))])
comp13=sum([flambdamuj(log_returns[i],dens[i],lambda_opt,mu_opt,sigma_opt,mu_j_opt, sigma_j_opt, delta_t,100) for i in range(len(log_returns)) if not mt.isnan(flambdamuj(log_returns[i],dens[i],lambda_opt,mu_opt,sigma_opt,mu_j_opt, sigma_j_opt, delta_t,100))])
comp14=sum([flambdasigma_j(log_returns[i],dens[i],lambda_opt,mu_opt,sigma_opt,mu_j_opt, sigma_j_opt, delta_t,100) for i in range(len(log_returns)) if not mt.isnan(flambdasigma_j(log_returns[i],dens[i],lambda_opt,mu_opt,sigma_opt,mu_j_opt, sigma_j_opt, delta_t,100))])
comp15=sum([flambdasigma(log_returns[i],dens[i],lambda_opt,mu_opt,sigma_opt,mu_j_opt, sigma_j_opt, delta_t,100) for i in range(len(log_returns)) if not mt.isnan(flambdasigma(log_returns[i],dens[i],lambda_opt,mu_opt,sigma_opt,mu_j_opt, sigma_j_opt, delta_t,100))])
comp22=sum([fmu2(log_returns[i],dens[i],lambda_opt,mu_opt,sigma_opt,mu_j_opt, sigma_j_opt, delta_t,100) for i in range(len(log_returns)) if not mt.isnan(fmu2(log_returns[i],dens[i],lambda_opt,mu_opt,sigma_opt,mu_j_opt, sigma_j_opt, delta_t,100))])
comp23=sum([fmumuj(log_returns[i],dens[i],lambda_opt,mu_opt,sigma_opt,mu_j_opt, sigma_j_opt, delta_t,100) for i in range(len(log_returns)) if not mt.isnan(fmumuj(log_returns[i],dens[i],lambda_opt,mu_opt,sigma_opt,mu_j_opt, sigma_j_opt, delta_t,100))])
comp24=sum([fmusigma_j(log_returns[i],dens[i],lambda_opt,mu_opt,sigma_opt,mu_j_opt, sigma_j_opt, delta_t,100) for i in range(len(log_returns)) if not mt.isnan(fmumuj(log_returns[i],dens[i],lambda_opt,mu_opt,sigma_opt,mu_j_opt, sigma_j_opt, delta_t,100))])
comp25=sum([fmusigma(log_returns[i],dens[i],lambda_opt,mu_opt,sigma_opt,mu_j_opt, sigma_j_opt, delta_t,100) for i in range(len(log_returns)) if not mt.isnan(fmusigma(log_returns[i],dens[i],lambda_opt,mu_opt,sigma_opt,mu_j_opt, sigma_j_opt, delta_t,100))])
comp33=sum([fmuj2(log_returns[i],dens[i],lambda_opt,mu_opt,sigma_opt,mu_j_opt, sigma_j_opt, delta_t,100) for i in range(len(log_returns)) if not mt.isnan(fmuj2(log_returns[i],dens[i],lambda_opt,mu_opt,sigma_opt,mu_j_opt, sigma_j_opt, delta_t,100))])
comp34=sum([fmujsigma_j(log_returns[i],dens[i],lambda_opt,mu_opt,sigma_opt,mu_j_opt, sigma_j_opt, delta_t,100) for i in range(len(log_returns))if not mt.isnan(fmujsigma_j(log_returns[i],dens[i],lambda_opt,mu_opt,sigma_opt,mu_j_opt, sigma_j_opt, delta_t,100))])
comp35=sum([fmujsigma(log_returns[i],dens[i],lambda_opt,mu_opt,sigma_opt,mu_j_opt, sigma_j_opt, delta_t,100) for i in range(len(log_returns)) if not mt.isnan(fmujsigma(log_returns[i],dens[i],lambda_opt,mu_opt,sigma_opt,mu_j_opt, sigma_j_opt, delta_t,100))])
comp44=sum([fsigmaj2(log_returns[i],dens[i],lambda_opt,mu_opt,sigma_opt,mu_j_opt, sigma_j_opt, delta_t,100) for i in range(len(log_returns))if not mt.isnan(fsigmaj2(log_returns[i],dens[i],lambda_opt,mu_opt,sigma_opt,mu_j_opt, sigma_j_opt, delta_t,100))])
comp45=sum([fsigmajsigma(log_returns[i],dens[i],lambda_opt,mu_opt,sigma_opt,mu_j_opt, sigma_j_opt, delta_t,100) for i in range(len(log_returns))if not mt.isnan(fsigmajsigma(log_returns[i],dens[i],lambda_opt,mu_opt,sigma_opt,mu_j_opt, sigma_j_opt, delta_t,100))])
comp54=sum([fsigmasigma_j(log_returns[i],dens[i],lambda_opt,mu_opt,sigma_opt,mu_j_opt, sigma_j_opt, delta_t,100) for i in range(len(log_returns))if not mt.isnan(fsigmasigma_j(log_returns[i],dens[i],lambda_opt,mu_opt,sigma_opt,mu_j_opt, sigma_j_opt, delta_t,100))])
comp55=sum([fsigma2(log_returns[i],dens[i],lambda_opt,mu_opt,sigma_opt,mu_j_opt, sigma_j_opt, delta_t,100) for i in range(len(log_returns))if not mt.isnan(fsigma2(log_returns[i],dens[i],lambda_opt,mu_opt,sigma_opt,mu_j_opt, sigma_j_opt, delta_t,100))])

temp=np.matrix([[comp11,comp12,comp13,comp14,comp15],
                [comp12,comp22,comp23,comp24,comp25],
                [comp13,comp23,comp33,comp34,comp35],
                [comp14,comp24,comp34,comp44,comp45],
                [comp15,comp25,comp35,comp54,comp55]])
covmat=np.linalg.inv(np.multiply(div(temp,len(log_returns)),-1))
################### Log_likelihood ratio test #############
mu_bs=div(np.mean(log_returns),delta_t)+div(cpow(np.std(log_returns),2),2*delta_t)
sigma_bs=np.sqrt(div(cpow(np.std(log_returns),2),delta_t))
initial_guess_bs=[mu_bs,sigma_bs]
def log_bs(params,data,delta_t):
    mu_bs,sigma_bs=params
    res=-1*sum(np.log(norm_density(data, (mu_bs-div(cpow(sigma_bs,2),2))*delta_t, cpow(sigma_bs,2)*delta_t)))
    return res
min_bs=minim(log_bs,initial_guess_bs,args=(log_returns,delta_t),method='Powell')
p = st.chi2.sf(-2*(-min_bs.fun+min_simp.fun),3)

############# estimated log_returns
plt.figure()
dist_plot(log_returns,'r')
params=[lambda_opt_p,mu_opt_p,sigma_opt_p,mu_j_opt_p,sigma_j_opt_p,delta_t]
CMC_sn=1
log_returns_simulated=np.asarray([0 for i in range(len(log_returns))],dtype="float64")
for i in range(CMC_sn):
    log_returns_simulated=np.add(log_returns_simulated,simulated(params,size=len(log_returns)))     

plt.figure()
dist_plot(log_returns_simulated,'g', 100,True,True,True)

##### Goodness of fit test ##########\
test=st.ks_2samp(log_returns,np.divide(log_returns_simulated,CMC_sn))
print(" KS 2 sample test distance : {d:8.8f}, p-value:{p:8.8f}".format(d=test.statistic,p=test.pvalue))
# Test does not reject the null hypothesis that the two samples came from the same distribution
############## Simulation of the log returns for different Ethereum block times ##########

block_time=pd.read_csv("/home/mohamed/Desktop//Folders/Ether/export-BlockTime.csv")
BT_avg=float(block_time.tail(1).Value)

# simulation with different block time
mult=[1,2,5,10]
Delta_t=np.multiply(mult,BT_avg)
params=[lambda_list[0],mu_list[0],sigma_list[0],mu_j_list[0],sigma_j_list[0],delta_t]
params.pop()
size=255
for i in range(len(Delta_t)):
    print(i)
    params.append(Delta_t[i])
    if i==0:
        BT_simulated=simulated(params,size)
    else:
        temp=simulated(params,size)
        BT_simulated=np.vstack((BT_simulated,temp))
    params.pop()
time=[i for i in range(size)]

### plotting simulated log_returns ######
plt.figure()
line_plot(BT_simulated[0,:],time)
plt.figure()
line_plot(BT_simulated[1,:],time)
plt.figure()
line_plot(BT_simulated[2,:],time)
plt.figure()
line_plot(BT_simulated[3,:],time)
######### Normality test #################
testSW=st.shapiro(log_returns)
print("Chapiro wilk test, Statistic: {a:2.8f}, p-value: {b:2.8f}".format(a=testSW.statistic, b=testSW.pvalue))
## KS test
distribution = "norm"
distr = getattr(st, distribution)
params = distr.fit(log_returns)
testKS=st.kstest(log_returns,distribution,args=params)
print("Ks normality test test, Statistic: {a:2.8f}, p-value: {b:2.8f}".format(a=testKS.statistic, b=testKS.pvalue))
############## student t test #############
distribution = "t"
distr = getattr(st, distribution)
params = distr.fit(log_returns)
testT=st.kstest(log_returns,distribution,args=params)
print("Ks student test, Statistic: {a:2.8f}, p-value: {b:2.8f}".format(a=testT.statistic, b=testT.pvalue))

############ HS simulation VaR ##########

def VaR(data,pvalue):
    return np.percentile(data, pvalue, interpolation="linear")

#### Bootstrapping #####
def bsVaR(data,pvalue,size):
    res=[]
    for i in range(size):
        res.append(VaR(np.random.choice(data, len(data)),pvalue))
    return np.mean(res)
############ Weighted HS ###########
def weighted_var(data,lambda_var=0.5,pvalue=0.05):
    wgts = [div(cpow(lambda_var,i-1) * (1-lambda_var),1-cpow(lambda_var,len(data))) for i in range(1, len(data)+1)] 
    weights_dict = {'LogReturns':data, 'Weights':wgts}
    wts_returns = pd.DataFrame(weights_dict)
    sort_wts = wts_returns.sort_values(by='LogReturns')
    sort_wts['Cumulative'] = sort_wts.Weights.cumsum()
    sort_wts = sort_wts.reset_index().drop(columns=['index'])
    ind=sort_wts[sort_wts.Cumulative <= pvalue].tail(1).index[0]
    xp = sort_wts.loc[ind::1, 'Cumulative'].values
    fp = sort_wts.loc[ind::1, 'LogReturns'].values
    return np.interp(pvalue, xp, fp)
def kupiec_back_test(data,func,pval=0.05,eval_size=1000,lambda_var=0.5,size=10000):
    train=data[:len(data)-1000]
    test=data[len(data)-1000:]
    if func==VaR:
        val=VaR(train,pval)
    elif func==bsVaR:
        val=bsVaR(train,pval,size)
    else:
        val=weighted_var(train,lambda_var,pval)
    return len([i for i in test if i < val])

              
