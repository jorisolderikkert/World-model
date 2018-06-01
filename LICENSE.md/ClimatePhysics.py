#
#
#
#
"""     World and climate modeling script

        Abstract: 
            
         A study of the controllability of the global mean surface 
        temperature (GMST) is performed within a hierarchy of 
        energy balance models (EBM). Firstly, the models were 
        validated for their confirmation to approved GMST levels.
        Secondly, the system inputs (greenhouse gas (GHG) 
        concentrations or emissions) were tested for their
        controllability using the control matrix condition in the 
        steady states of the models. For the EBMS with a constant 
        or temperature dependent albedo, and a constant CO2
        concentration, it is shown that the GMST is fully 
        controllable by a CO2 concentration controller. As the
        GHG-concentrations can only be regulated through 
        controlling the GHG-emissions, the model was extended 
        with a RCP3, -4.5, -6.0, and -8.0 based 
        emission-concentration subsystem. In the extended, 
        non-bifurcated model, the emission input is found to have 
        full control over the concentrations; consecutively the
        concentration vector is demonstrated to have full over 
        the GMST. For creating policy or assumptions on more 
        complex and bifurcated models, such as the earth’s 
        climate, policymakers and climate scientist are warranted 
        to take controllability into consideration.
        
        
        Script and research by Joris Olde Rikkert 2017
        
        Full research is available on 
        http://jorisolderikkert.com/2017/climatecontrol.html
        
        To use download the climate data in txt format: RCP3.txt, 
        45.txt, 6.txt, 85.txt
        Uncleaned version of the used script
"""


import math
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy as sp
import datetime
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
#set variables
sy=31557600 #sec per year
p=1.25              #kg/m^3 atmospheric density
Cpa=pow(10,3)       #[J/kg/K] heat capacity
r0=6.3*pow(10,6)    #[m] radius of Earth
Ha=8.4*pow(10,3)   #[m] atmospheric scale height
E0=1.36*pow(10,3)   #[Wm^-2] solar constant
D0=3.1*pow(10,6)   #[m^2s^-1] eddy diffusivity
C0=0.43             #atmospheric non-absorption coefficient
Cref=0.27701467*10**3#278.05158      #[ppmv] reference carbon concentration Myhre page 678
Current=Cref
RCP=4.5
def Cmax(tau):
    return tau        #max Co2 conc in 2050
maxxie=400
sb=5.67*pow(10,-8)  #[Wm^-2K^-4] Stefan Boltzman Constant
epsilon=1.0         #emissivity 
Eh=0.273            #[K] Heaviside
gamma=5.35*1.48      #189     #Myhre et al 2013a,+-10% co2 voorfactor voor 284K
Tzero=283.1737    #Begin temperature                         
a0=0.7              #albedo when T<T0
a1=0.2           #albedo when T>T1
aM=0.3
T0=263              #[K] temperature for a0
T1=293              #[K] temperature for a1
thetaN=math.radians(90)     #angle northpole
thetaS=math.radians(-90)   #angle southpole
aS=0.241            #short-wave radiative heat flux

k=0
M=20
#correction term
correction=273.15#standard albedo and gmst 284


#integration parameters:
h=0.1
N  =100 #number of timeintervals
t0 =0      #t-begin in years
tfinal=2500-1765-5
t = np.linspace(t0, tfinal, N+1)
spy=tfinal/N

#array creation
dtheta=np.abs((thetaS-thetaN)/(M))

A=np.zeros((M+1, M+1))
x=np.squeeze(np.ones((M+1,1))*Tzero) #begin conditions and array size
x0=np.squeeze(np.ones((M+1,1))*Tzero)
x0carbon=np.squeeze(np.ones((M+5,1))*Tzero)
x0carbon[M+2:M+5]=0
x0carbon[M+1]=Cref
constant=np.zeros((M+1,1))
albedoInTime=np.ones((M+1,N+1))*1.1
carbon=np.squeeze(np.ones((4,1)))

#parameters for emission model
e0=0.217278
e1=0.224037
e2=0.282381
e3=0.276303
tau1=394.409
tau2=36.5393
tau3=4.30365
cf=2.123 #http://www.atmos-chem-phys.net/13/2793/2013/acp-13-2793-2013.pdf page 2798

conc=np.squeeze(np.zeros((4,1)))
carbon0=np.squeeze(np.zeros((4,1)))
carbon0[0]=Cref                      #begin condition
concval=np.squeeze(np.zeros((4,1)))
carbon=np.squeeze(np.zeros((4,1)))
emitted=np.zeros((tfinal,1))
def emission(h,dat):
    year=round(h)
    emitted=(dat[year,1]+dat[year,2])
    return emitted

def sumemission(dat,tfinal):
    te=np.zeros((tfinal,1))
    for i in range (0,tfinal):
        te[i]=np.sum(dat[:i,1]+dat[:i,2])
    return te
    

def concentration(carbon,h,dat):
    concval[0]=e0*emission(h,dat)
    concval[1]=e1*emission(h,dat)-1/tau1*carbon[1]
    concval[2]=e2*emission(h,dat)-1/tau2*carbon[2]
    concval[3]=e3*emission(h,dat)-1/tau3*carbon[3]
    return concval#*spy #correct for changes per year: *steps per year (spy)

def pathway(conc,h,dat): 
    carbon=conc
    dconcdt = np.squeeze(concentration(carbon,h,dat))
    return dconcdt
#%%
dval=np.ones((4,1))
#%%
dat3=np.genfromtxt('RCP3.txt',)
dat45=np.genfromtxt('45.txt')
dat6=np.genfromtxt('6.txt')
dat85=np.genfromtxt('8.txt')

talbedo=np.squeeze(np.ones((N+1,1)))*405
#%%
RCP3n=np.sum(sp.integrate.odeint(pathway,carbon0,t,args=(dat3,)).T,axis=0)
RCP45n=np.sum(sp.integrate.odeint(pathway,carbon0,t,args=(dat45,)).T,axis=0)
RCP6n=np.sum(sp.integrate.odeint(pathway,carbon0,t,args=(dat6,)).T,axis=0)
RCP85n=np.sum(sp.integrate.odeint(pathway,carbon0,t,args=(dat85,)).T,axis=0)

RCP3base=sp.integrate.odeint(pathway,carbon0,t,args=(dat3,)).T
RCP45base=sp.integrate.odeint(pathway,carbon0,t,args=(dat45,)).T
RCP6base=sp.integrate.odeint(pathway,carbon0,t,args=(dat6,)).T
RCP85base=sp.integrate.odeint(pathway,carbon0,t,args=(dat85,)).T
#%%
rftalbedo=gamma*np.log(talbedo/Cref)
#%%
rf3=gamma*np.log(RCP3n/Cref)
rf45=gamma*np.log(RCP45n/Cref)
rf6=gamma*np.log(RCP6n/Cref)
rf85=gamma*np.log(RCP85n/Cref)

#%%
summed=np.zeros((1,N+1))
def summation(con):
    for i in range (0,N+1):
        summed[0,i]=sum(con[0:4,i])
    return np.squeeze(summed)

#%%
timeframe=np.linspace(1765,tfinal+1765,N+1)
fig2 = plt.figure(figsize=(10,5.0))
fig3r=plt.subplot(2, 1, 1)
#line3,=plt.plot(timeframe,RCP3,'b')
#line45,=plt.plot(timeframe,RCP45,'g')
#line6,=plt.plot(timeframe,RCP6,'y')
#line8,=plt.plot(timeframe,RCP85,'r')
linetalbedo=plt.plot(timeframe,talbedo,'')
#plt.legend([line3,line45,line6,line8], ['RCP 3', 'RCP 4.5','RCP 6','RCP 85'])
plt.title('Concentration Pathways')
plt.ylabel('Global $CO_2$ concentration in ppm')
fig3r.axes.get_xaxis().set_visible(False)
ax=plt.subplot(2, 1, 2)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
liner3,=plt.plot(timeframe,rf3,'b')
liner45,=plt.plot(timeframe,rf45,'g')
liner6,=plt.plot(timeframe,rf6,'y')
liner8,=plt.plot(timeframe,rf85,'r')
#plt.tight_layout()
ax.set_title('Radiative forcing',y=1)
plt.ylabel('Radiative forcing [$W/m^2$]')
plt.xlabel('Year ') 
equilibriumtemperature=plt.gcf()
equilibriumtemperature.savefig('concentrations1765-2500try3.png',dpi=500)
plt.show()

#%%2100 values
vval=np.zeros((4,1))
bval=np.zeros((4,1))
vval[0]=RCP3n[N]
vval[1]=RCP45n[N]
vval[2]=RCP6n[N]
vval[3]=RCP85n[N]

#%%benchmarking concentrations
bm3=np.genfromtxt('3conc.txt')[:,3]
bm45=np.genfromtxt('45conc.txt')[:,3]
bm6=np.genfromtxt('6conc.txt')[:,3]
bm85=np.genfromtxt('85conc.txt')[:,3]
bval[0]=bm3[N]
bval[1]=bm45[N]
bval[2]=bm6[N]
bval[3]=bm85[N]

dval=vval/bval
print(dval)
#%%
timeframe2=np.linspace(1765,tfinal+1765,tfinal)
fig3 = plt.figure(figsize=(10,3))
emission3,=plt.plot(timeframe2,sumemission(dat3,tfinal),'b')
emission45,=plt.plot(timeframe2,sumemission(dat45,tfinal), 'g')
emission6,=plt.plot(timeframe2,sumemission(dat6,tfinal),'y')
emission85,=plt.plot(timeframe2,sumemission(dat85,tfinal),'r')
#plt.legend([emission3,emission45,emission6,emission85], ['3', '45','6','8'])
plt.title('Emissions ')
plt.tight_layout()
plt.ylabel('Cumulative CO2 emissions in GtCO2')
plt.xlabel('years ')
equilibriumtemperature=plt.gcf()
equilibriumtemperature.savefig('emissionsgood.png',dpi=500)

plt.show()


#%%
def espilon():         #calculation of mean T according to function in new draft (different analytical solution)
    absorption=(1/sb*((E0/4*(1-aM)*(1-C0)*(3/2))-3*aS*E0/4*(1-aM)*(1-C0)*(pow(np.sin(thetaN),3)-pow(np.sin(thetaS),3))/(thetaN-thetaS)))/(284)**4
    epsilon=absorption
    return absorption
print("epsilon",espilon(), "K")  
epsilon=1

def TMean():         #calculation of mean T according to function in new draft (different analytical solution)
    value=((1/sb*((E0/4*(1-aM)*(1-C0)*(3/2))-3*aS*E0/4*(1-aM)*(1-C0)*(pow(np.sin(thetaN),3)-pow(np.sin(thetaS),3))/(thetaN-thetaS)))/epsilon)**(1/4)
    return value
print("GMST_new=",TMean(), "K")  

#Important paramater now tweaked at a temperature giving a reasonable mean temperature
def A0(K):
    value=At#-270          #value should be around 216 Wm^-2  
    #value=216-150    #[Wm^-2] long-wave radiative flux correction constant linearisation
    #value=216      #From paper Dijkstra 2016
    #value=sb*epsilon*K**4 #Black body radiation not used for now
    return value
         #[Wm^-2K^-1] Long-wave radiative flux variable linear term 

def Cgeo(k):
    return C0

#%% Stefan Boltzman linearization
bound=310
temp0=260
yk=np.zeros((bound-temp0,1))
xk=np.zeros((bound-temp0,1))
zk=np.zeros((bound-temp0,1))
result=np.zeros((bound-temp0,1))
for i in range (0,bound-temp0):
    xk[i]=temp0+i
    yk[i]=sb*epsilon*(i+temp0)**4
    zk[i]=216+(i+temp0)*1.5

def line(ka, a, b):
    return b *ka + a  

popt, pcov = sp.optimize.curve_fit(line, tuple(np.squeeze(xk)), tuple(np.squeeze(yk[:,0])))
for i in range (0,bound-temp0):
    result[i]=popt[1] *(i+temp0) + popt[0] 
print(popt)
At,B0=popt
#B0=0


#line1, =plt.plot(xk,yk, 'r')
#line2, =plt.plot(xk,zk, 'y')
#line3, =plt.plot(xk,result)
#plt.legend([line1,line2,line3], ['Stefan Boltzman Law', 'Linearization v1','Optimized linearization'])
#plt.title('Linearisation of Stefan-Boltzman temperature law', y=1.5)
#plt.xlabel('Temperature [K]')
#plt.ylabel('Energy radiation [Wm^-2]')
#plt.show()



#%%
def H(x):         #Heaviside function not used for now, instead pure step function is used
    return 1/2*(1 + math.tanh((x)/Eh))

def albedo(T):           #albedo step function
    aVal=a0*H(T0-T)+a1*H(T-T1)+(a0+ (a1-a0)*(T-T0)/(T1-T0))*H(T-T0)*H(T1-T)
    #aVal=0.3
    return aVal
#%%

valuee=np.ones((151,1))
#%%

Eh=0.273
def valuate():
    for j in range (200,351):
        waarde=j*1.01
        valuee[j-200,0]=albedo(waarde)
    return valuee
hi=valuate()

xac=np.linspace(200,350,151)
figure1=plt.plot(xac,hi)
plt.title("Albedo(T)")
plt.xlabel('Temperature [K] ')
plt.ylabel('Albedo coefficient')
albedoplot=plt.gcf()
albedoplot.savefig('tdepalbedotry.png',dpi=500)
plt.show()

#%%

def D(theta):    #function describing the eddy diffusivity 
    return (0.9 + 1.5*np.exp(-12*pow(theta,2)/np.pi))

def S(theta):    #short-wave radiative heat flux
    return (3/2-aS*(pow(3*np.sin(theta),2)))


#%%
        
def ACal(M):    #calculation of diffusion matrix as part of A for linearization of x: dx~/dt= Ax~ + Bu
                #used to calculate controllability
    
    A.astype(float)
    for j in range (0,M+1): 
        thetaj=thetaN-dtheta*j
        Dtp=D(thetaj-dtheta/2)
        Dtm=D(thetaj+dtheta/2)
        if j==0:
            A[j,j]=-B0/(p*Ha*Cpa)+D0/(r0**2)*(-1)/pow(dtheta,2)*(Dtp)*2
            A[j,j+1]=D0/(r0**2)/pow(dtheta,2)*(Dtp)*2
        elif j==M:
            A[j,j]=-B0/(p*Ha*Cpa)+D0/pow(r0,2)*(-1)/pow(dtheta,2)*(Dtm)*2
            A[j,j-1]=D0/(r0**2)/pow(dtheta,2)*(Dtm)*2
        else:
            A[j,j]=-B0/(p*Ha*Cpa)+D0/pow(r0,2)*(-1)/pow(dtheta,2)*(Dtm+Dtp)
            A[j,j-1]=D0/(r0**2)/pow(dtheta,2)*(Dtm)
            A[j,j+1]=D0/(r0**2)/pow(dtheta,2)*(Dtp)
    return A

print(ACal(M))
print("B0", -B0*280)
print(D0/pow(r0,2)*(-1)/pow(dtheta,2)*(D(0.0)*2)*280*(p*Ha*Cpa))
print(D0/pow(r0,2)*(-1)/pow(dtheta,2)*(D(np.pi/2)*2)*280*(p*Ha*Cpa))
print(np.log(RCP45n[N]/Cref)*gamma)
print(np.dot(ACal(M),x))
print(E0/4*S(np.pi/2)*(1-albedo(280))*(1-Cgeo(0))*1)
print((correction-A0(280))*1/(p*Ha*Cpa))


#print(gamma*np.log(Cpath(control)/Cref))
#%%
def constantCal(x,h,mu,corr,RCP,carbon,dat):        
    for j in range (0,M+1):
        thetaj=thetaN-dtheta*j
        constant[j]=sy/(p*Ha*Cpa)*(corr-A0(x[j])+E0/4*S(thetaj)*(1-albedo(x[j]))*(1-Cgeo(0))+ gamma*np.log(dat[int(np.round(h))]/Cref))
    return tuple(np.squeeze(constant))

def dxdtCal(x,h,i,cor,RCP,carbon,dat):
    dxdt=sy*np.dot(ACal(M),x)+ constantCal(x,h,i,cor,RCP,carbon,dat)
    return dxdt


#%%Main programme:
#Integration:


#%%

def pend(y,h,tuf,corre,dat,rcp):    
    x= y[0:M+1]
    carbon=y[M+1:M+5]
    dydt = np.concatenate((dxdtCal(x,h,tuf,corre,RCP,carbon,rcp),concentration(carbon,h,dat)),axis=0)
    #print(maxxie)
    return dydt

#%%
    #%%
def integrator(rcp):
    integrand=sp.integrate.odeint(pend,x0carbon,t,args=(400,correction,dat3,rcp))[:,4:M+5].T
    return integrand
#RCP3n=RCP3[:]
GMSTRCP3=integrator(RCP3n)
RCP45n=RCP45[:]
RCP6n=RCP6[:]
RCP85n=RCP85[:]
GMSTRCP45=integrator(RCP45n)
GMSTRCP6=integrator(RCP6n)
GMSTRCP85=integrator(RCP85n)






#%% Averaging
def weight(M,time,rcp):
    for i in range (0,M+1):
        WAref[i]=np.cos(thetaN-dtheta*(i))
    return (tuple(WAref))
WA=np.ones((M+1,1))
WAref=np.ones((M+1,1))
def means(M,time,rcp):
    for i  in range (0,M+1):
        WA[i]=np.cos(thetaN-dtheta*(i))*rcp[i,time]
        WAref[i]=np.cos(thetaN-dtheta*(i))
    return tuple(WA)

def gmst(M,N,rcp):
    gmstt=np.zeros((N+1,1))
    for i in range (0,N+1):          
        WA=np.ones((M+1,1))
        WAref=np.ones((M+1,1))
        gmstt[i]=sum(means(M,i,rcp))/sum(weight(M,i,rcp))
    return gmstt
GMSTTRCP3n=gmst(M,N,GMSTRCP3n)
GMSTTRCP45n=gmst(M,N,GMSTRCP45n)
GMSTTRCP6n=gmst(M,N,GMSTRCP6n)
GMSTTRCP85n=gmst(M,N,GMSTRCP85n)
GMSTTtalbedo=gmst(M,N,GMSTtalbedo)

 #%%

def bifurcation (u,v,w,tbegin):
    t2=np.zeros((w))
    arrayke = np.linspace(u,v,w)
    for k in range (0,w):
        Cmax(k)
        print(k,t2[k])
        hoie=arrayke[k]
        t2[-1]=284
        hoi0=np.squeeze(np.ones((M+1,1))*t2[k-1])
        sol=sp.integrate.odeint(pend,hoi0,t,args=(hoie,correction)).T
        t2[k]=sum(means(M,sol))/sum(weight(M,sol))
        #print(t2[k])
        #plt.plot(Ccon(N,hoie))
        #plt.show()
    return(t2)
w=11
resultoplopend=bifurcation(200,1000,w,284)
tbegin=resultoplopend[w-1]
resultaflopend=bifurcation(1000,200,w,tbegin)
    #%%
diff=np.zeros((100,1))
for i in range (0,99):
    diff[i,0]=abs(resultaflopend[i]-resultaflopend[i+1])
print(diff.index(max(diff)))   


#%%
hoiek=np.linspace(0,10,11)
hoie2=np.linspace(10,0,11)

xa=np.linspace(200,1000,w)
xa2=np.linspace(1000,200,w)

print(hoiek,hoie2)
plt.figure(1)
plt.title("Bifurcation diagram")
plt.xlabel('CO2 concentration [CO2eq ppmv] ')
plt.ylabel('GMST [K]')
low,=plt.plot(xa, resultoplopend,'go')
high,=plt.plot(xa2, resultaflopend,'ro')
plt.legend([low,high], ['Increasing concentration ', 'Decreasing concentration'])
bifur=plt.gcf()
bifur.savefig('bifurcationdiafocus2.png',dpi=500)
plt.show()


    #%% plotting GMST

import matplotlib.pyplot as plt
import numpy as np

xac=np.linspace(90,-90,M+1)
tac=np.linspace(1765,tfinal+1765,N)
fig1=plt.matshow(GMSTRCP3[0:M+1,:], aspect='auto')
plt.title("GMST RCP3", y=1.5)
plt.xlabel('Time 1765-2500')
plt.ylabel('Meriodonal section M')
clb=plt.colorbar(fig1, orientation='horizontal') 
clb.set_label("Temperature scale [K]")
GMST=plt.gcf()
GMST.savefig('equilibriumtemperatureRCP3180try2.png',dpi=500)
plt.show()
#%%
fig4 = plt.figure(figsize=(10,5))
lgmstrcp3,=plt.plot(tac,-(283.2717-GMSTTRCP3n[1:N+1,0]),'b')
lgmstrcp45,=plt.plot(tac,-(283.2717-GMSTTRCP45n[1:N+1,0]),'g')
lgmstrcp6,=plt.plot(tac,-(283.2717-GMSTTRCP6n[1:N+1,0]),'y')
lgmstrcp85,=plt.plot(tac,-(283.2717-GMSTTRCP85n[1:N+1,0]),'r')
#plt.legend([lgmstrcp3,lgmstrcp45,lgmstrcp6,lgmstrcp85], ['RCP 3', 'RCP 4.5','RCP 6','RCP 85'])
plt.title('$\Delta$ GMST(t) for Concentration Pathways ')
plt.ylabel('$\Delta$GMST [K]')
plt.tight_layout()
plt.xlabel('Year ')
equilibriumtemperature=plt.gcf()
equilibriumtemperature.savefig('equilibriumtemperaturecombitry4.png',dpi=500)
plt.show()


    #%%

 #showing eq temperature of different meriodonal locations
plt.plot(GMSTRCP3[0:M+1,N],xac)
plt.title('Equilibrium temperature for all seperate meriodonal sections ')
plt.ylabel('Latitude [degrees]')
plt.xlabel('Temperature [K] ')
plt.tight_layout()
equilibriumtemperature=plt.gcf()
equilibriumtemperature.savefig('equilibriumtemperaturealbedotdeptry2.png',dpi=500)
plt.show()
#%%
for i in range(M):  #showing paths of different meriodonal locations
    plt.plot(temperaturegmst.T[:])
plt.title('T[t] for all seperate meriodonal sections')
plt.ylabel('Temperature[t]')
plt.tight_layout()
plt.xlabel('Time 1765-2500 ')
settlement=plt.gcf()
settlement.savefig('settlementalbedotdeptry2.png',dpi=500)
plt.show()


#%%
def controllability(M): #calculating the controllability according to the criterium
    BI=np.eye(M+1)                                  #(M+1)x(M+1) Identity Matrix
    R=BI                                            #begin condition R, first part of R = BI
    Ak=np.dot(np.linalg.matrix_power(A,2),BI) 
      
    for k in range (0,M):                           #construct controllability matrix
        Ak=np.dot(np.linalg.matrix_power(A,k),BI)   #Ak=(matrix A)^k * BI
        R=np.concatenate((R,Ak),axis=1)             #adds the new Ak to the right side of existing reachability matrix
    
    rank=np.linalg.matrix_rank(R)  #calculating rank of R
    if rank==M+1:                    #system is controllable if the rank of the reachability matrix equals the dim of the system A
        return print("System is controllable")
    else:
        return print("Rank does not match dimensions, so system is uncontrollable")
controllability(M)
from matplotlib.colors import LogNorm
figMatrixA=plt.matshow(ACal(M),norm=LogNorm(vmin=0.01, vmax=10))      #plotting the diffusion matrix
plt.title('Jacobian A, T-dep albedo')
plt.axis('off')
plt.colorbar(figMatrixA, orientation='horizontal') 
MatrixA=plt.gcf()
plt.tight_layout()
MatrixA.savefig('Matrixalbedoalbedotdep.png',dpi=500)
plt.show()
#%%

   
M=20
B=np.zeros((M+5,M+5))
B[0,0]=e0
B[1,1]=e1
B[2,2]=e2
B[3,3]=e3
BI=B
Ak=np.dot(np.linalg.matrix_power(e35,2),BI) 
R=BI
def controllability2(M,ex): #calculating the controllability according to the criterium
    Ak=np.dot(np.linalg.matrix_power(ex,2),BI) 
    R=BI
    for k in range (0,M+5):                           #construct controllability matrix
        Ak=np.dot(np.linalg.matrix_power(ex,k),BI)   #Ak=(matrix A)^k * BI
        R=np.concatenate((R,Ak),axis=1)             #adds the new Ak to the right side of existing reachability matrix
    rank=np.linalg.matrix_rank(R)  #calculating rank of R
    return rank
    #if rank==M+5:                    #system is controllable if the rank of the reachability matrix equals the dim of the system A
    #    return print("System is controllable")
    #else:
    #    return print("Rank does not match dimensions, so system is uncontrollable")
hi=controllability2(M,e35)
#%%
from pylab import figure, cm
from matplotlib.colors import *
figMatrixA=plt.figure(figsize=(10,5))
figMatrixA=plt.matshow(hi, fignum=1,norm=SymLogNorm(0.000001,0.0001,vmin=hi[0:25,0:25].min()-0.000001, vmax=e35[0:10,0:10].max(), clip=False))    #plotting the diffusion matrix
plt.axis('off')
plt.colorbar(figMatrixA, orientation='horizontal') 
MatrixA=plt.gcf()
MatrixA.savefig('Controllabilitymatrix.png',dpi=500)
plt.show()

#%%
#Jacobian A: first derivatives




   

#%%
B2=np.zeros((4,4))
B2[0,0]=e0
B2[1,1]=e1
B2[2,2]=e2
B2[3,3]=e3
I2=np.eye(4)                                 #(M+1)x(M+1) Identity Matrix
BI2=np.dot(B2,I2) 
A2=np.zeros((4,4))
A2[0,0]=0
A2[1,1]=-1/tau1
A2[2,2]=-1/tau2
A2[3,3]=-1/tau3
Ak=np.dot(np.linalg.matrix_power(A2,2),BI2) 
R2=BI2
def controllability2(M): #calculating the controllability according to the criterium
    Ak=np.dot(np.linalg.matrix_power(A2,2),BI2) 
    R2=BI2
    for k in range (0,4):                           #construct controllability matrix
        Ak=np.dot(np.linalg.matrix_power(A2,k),BI2)   #Ak=(matrix A)^k * BI
        print(R2)
        R2=np.concatenate((R2,Ak),axis=1)             #adds the new Ak to the right side of existing reachability matrix
    rank=np.linalg.matrix_rank(R2)  #calculating rank of R
    return R2
    #if rank==4:                    #system is controllable if the rank of the reachability matrix equals the dim of the system A
     #   return print("System is controllable")
    #else:
     #   return print("Rank does not match dimensions, so system is uncontrollable")
leftover=controllability2(M)
#%%
figMatrixA=plt.matshow(jacob    )      #plotting the diffusion matrix
plt.title('Jacobian A, T-dep albedo')
plt.axis('off')
plt.colorbar(figMatrixA, orientation='horizontal') 
MatrixA=plt.gcf()
plt.tight_layout()
MatrixA.savefig('Matrixalbedoalbedotdep.png',dpi=500)
plt.show()    
    
#%%
jacob2=jacob
#%%
nx,ny=np.shape(jacob)
CXY=np.zeros([ny, nx])
for i in range(ny):
    CXY[i]=jacob[i,:]

#Binary data
np.save('maximums.npy', CXY)
#%%
#Human readable data
np.savetxt('RCP3jacob.txt', jacob)


#%%
    
B=np.eye((M+1,M+1))                                 #(M+1)x(M+1) Identity Matrix
BI=np.dot(B,I) 
Ak=np.dot(np.linalg.matrix_power(jacob,2),BI) 
R=BI
def controllability2(M): #calculating the controllability according to the criterium                                  #(M+1)x(M+1) Identity Matrix
    BI=np.dot(B,I) 
    Ak=np.dot(np.linalg.matrix_power(jacob,2),BI) 
    R=BI
    for k in range (0,M):                           #construct controllability matrix
        Ak=np.dot(np.linalg.matrix_power(jacob,k),BI)   #Ak=(matrix A)^k * BI
        R=np.concatenate((R,Ak),axis=1)             #adds the new Ak to the right side of existing reachability matrix
    rank=np.linalg.matrix_rank(R)  #calculating rank of R
    return R
#%%    if rank==M+1:                    #system is controllable if the rank of the reachability matrix equals the dim of the system A
        return print("System is controllable")
    else:
        return print("Rank does not match dimensions, so system is uncontrollable")
controllability2(M)
#%%
figMatrixA=plt.matshow(jacob)      #plotting the diffusion matrix
plt.title('Jacobian A, T-dep albedo')
plt.axis('off')
#plt.colorbar(figMatrixA, orientation='horizontal') 
MatrixA=plt.gcf()
MatrixA.savefig('Matrixalbedoalbedotdep.png',dpi=500)
plt.show()
