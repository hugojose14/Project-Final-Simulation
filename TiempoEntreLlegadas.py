from scipy.stats import sem, t
from scipy import mean
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf,acf
import random
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from re import findall
from scipy import stats

#Nombre de las distribuciones
distributions = [
   
    "norm",            #Normal (Gaussian)
    "alpha",           #Alpha
    "anglit",          #Anglit
    "beta",            #Beta
    "betaprime",       #Beta Prime
    "bradford",        #Bradford
    "burr",            #Burr
    "cauchy",          #Cauchy
    "chi",             #Chi
    "chi2",            #Chi-squared
    "cosine",          #Cosine
    "dgamma",          #Double Gamma
    "dweibull",        #Double Weibull
    "expon",           #Exponential
    "exponweib",       #Exponentiated Weibull
    "exponpow",        #Exponential Power
    "fatiguelife",     #Fatigue Life (Birnbaum-Sanders)
    "foldcauchy",      #Folded Cauchy
    "fisk",            #Fisk
    "gamma",           #Gamma
    "gausshyper",      #Gauss Hypergeometric
    "genexpon",        #Generalized Exponential
    "genextreme",      #Generalized Extreme Value
    "gengamma",        #Generalized gamma
    "genlogistic",     #Generalized Logistic
    "genpareto",       #Generalized Pareto
    "genhalflogistic", #Generalized Half Logistic
    "gilbrat",         #Gilbrat
    "gompertz",        #Gompertz (Truncated Gumbel)
    "gumbel_l",        #Left Sided Gumbel, etc.
    "gumbel_r",        #Right Sided Gumbel
    "halfcauchy",      #Half Cauchy
    "halflogistic",    #Half Logistic
    "halfnorm",        #Half Normal
    "hypsecant",       #Hyperbolic Secant
    "invgamma",        #Inverse Gamma
    "invweibull",      #Inverse Weibull
    "johnsonsb",       #Johnson SB
    "johnsonsu",       #Johnson SU
    "laplace",         #Laplace
    "logistic",        #Logistic
    "loggamma",        #Log-Gamma
    "loglaplace",      #Log-Laplace (Log Double Exponential)
    "lognorm",         #Log-Normal
    "lomax",           #Lomax (Pareto of the second kind)
    "maxwell",         #Maxwell
    "mielke",          #Mielke's Beta-Kappa
    "nakagami",        #Nakagami
    "ncx2",            #Non-central chi-squared
    "ncf",             #Non-central F
    "nct",             #Non-central Student's T
    "pareto",          #Pareto
    "powerlaw",        #Power-function
    "powerlognorm",    #Power log normal
    "powernorm",       #Power normal
    "rdist",           #R distribution
    "reciprocal",      #Reciprocal
    "rayleigh",        #Rayleigh
    "rice",            #Rice
    "recipinvgauss",   #Reciprocal Inverse Gaussian
    "semicircular",    #Semicircular
    "t",               #Student's T
    "triang",          #Triangular
    "truncexpon",      #Truncated Exponential
    "tukeylambda",     #Tukey-Lambda
    "uniform",         #Uniform
    "wald",            #Wald
    "weibull_min",     #Minimum Weibull (see Frechet)
    "weibull_max",     #Maximum Weibull (see Frechet)
    "wrapcauchy",      #Wrapped Cauchy
    "ksone",           #Kolmogorov-Smirnov one-sided (no stats)
    "kstwobign"        #Kolmogorov-Smirnov two-sided test for Large N
    
    ]      
def read():#leemos los datos del data.txt que seran nuestros y
    data = []
    with open("times.txt") as d:
        for i in d.readlines():
            data.append(findall(r"[\d]+.[\d]+", i))
    return np.array(data, dtype=float)
    

def generar_x(y):#función que nos genera los valores de x 
    x=np.array([[i+1] for i in range(len(y))])
    return x

def Autocorrelation(x):
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')
    result = r/(variance*n)
    return result[result.size//2:]

def Best_distribution(data):
    dist_results = []
    params = {}
    for dist_name in distributions:
        dist = getattr(stats, dist_name)
        param = dist.fit(data)#devuelve parametros para estimar

        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = stats.kstest(data, dist_name, args=param)
        print("p value for "+dist_name+" = "+str(p))
        dist_results.append((dist_name, p))

    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value

    print("\nBest fitting distribution: "+str(best_dist))
    print("Best p value: "+ str(best_p))
    print("Parameters for the best fit: "+ str(params[best_dist]))

    return best_dist,params[best_dist]

def Random_servicio(data):
    
    dist_name,param=Best_distribution(data)#llama a la funcion y me retorna el nombre
                                            #de la mejor distribucion y sus parametros fit
    
    dist=getattr(stats,dist_name)
    return dist.rvs(*param[:-2], loc=param[-2], scale=param[-1],size=75)
    #con size pongo el numero de random que quiero que salgan
   # return dist.rvs(a=param[0],c=param[1],loc= param[2],size=10) 

def  intervalo(array):
    n = len(array)
    media = mean(array)#funcion para calcular la media
    error = sem(array)#funcion para calcular el error estandar
    h = error*t.ppf((1+0.95)/2,n-1)
    inicio = media-h
    final= media+h
    rango = [inicio,final]

    return rango

def generador(noma,noms,pa,ps):

    
    dist=getattr(stats,noma)
    dist2=getattr(stats,noms)
   
    ai=[]
    a=0
    while(a< 10800):
        #tiempos entre llegada
        r = dist.rvs(*pa[:-2], loc=pa[-2], scale=pa[-1])
        #sumando los tiempos entre llegada
        a += r
        ai.append(a)
    n= len(ai)
    si= [dist2.rvs(*ps[:-2], loc=ps[-2], scale=ps[-1]) for x in range(n)]    
    return ai,si

    
def Single_server_queue(ai,si):
    Co=0.0
    i=0
    di = []
    ci = []
    while(i<len(ai)):
        a = ai[i] 
        if(a < Co):
            d = Co - a
            di.append(d)
        else:
            d = 0.0
            di.append(d)
        Co = a + si[i] + d
        ci.append(Co)
        i = i + 1
    return di,ci

if __name__ == "__main__":

   

    data = read()
    y=data #datos y 
    datos_x=generar_x(y[:,1])
    x=datos_x#datos x
    
    mean_of_distribution = np.mean(y[:,1])
    variance_of_distribution = np.var(y[:,1])
    g_beta=mean_of_distribution/variance_of_distribution
    print(1/g_beta)
    
    print(y[:,1])
    
   # print(y)  ver datos y 
   # print(x) ver datos x 
    #funcion para calcular la regresion lineal por minimos cuadrados
    slope, intercept, r_value, p_value, std_err = stats.linregress(x[:,0], y[:,1])

    if p_value>0.05:
        print("\nP valor : ",p_value,"es ID")
    else: 
        print("\nLa muestra no es ID")

    plt.plot(x,y[:,1],'o',label='Datos')
    plt.plot(x, intercept + slope*x, 'r', label='Ajuste')
   # plt.xticks(x)
    plt.xlabel("Observation number(x)")
    plt.ylabel("Service time(y)")
    plt.legend()
    plt.show()

    #calculo de la autocorrelación
    auto_c = Autocorrelation(y[:,1])
   # print ("\n\tResultado de la autocorrelación\n")
   # print(auto_c)
    print("\n")
   # print(len(auto_c))
    plot_acf(y[:,1])#grafica
    plt.show()

    #calcular los tiempos entre llegadas
    r=[]
    a_llegada=y[:,0]
    print (a_llegada)
    r.append(0)
    for i in range (len(y[:,1])-1):
        r.append(a_llegada[i+1]-a_llegada[i])
   # print("tiempos entre llegada:" ,r)
    #print (len(r))
    #regresion lineal de los entre llegadas
    slope_r, intercept_r, r_value_r, p_value_r, std_err_r = stats.linregress(x[:,0], r)

    if p_value_r>0.05:
        print("\nEl p valor es igual a: ",p_value_r,"por lo tanto se acepta la hipotesis nula,es ID")
    else: 
        print("\nLa muestra no es ID")
    
    plt.plot(x,r,'o',label='Datos')
    plt.plot(x, intercept_r + slope_r*x, 'r', label='Ajuste')
   # plt.xticks(x)
    plt.xlabel("Observation number(x)")
    plt.ylabel("Service time(y)")
    plt.legend()
    plt.show()

    print("\n\tcalculamos la mejor distribucion para los tiempos entre llegada")
    da,pa=Best_distribution(r)
    print("\n\tcalculamos la mejor distribucion para los tiempos de llegada")

    ds,ps=Best_distribution(y[:,1])
   

    n=[]#numero de trabajos
    Tinterrival_t=[]#tiempo promedio de llegada
    Arrival_rate=[]#tasa de llegada
    q_x=[]#tiempo promedio de servicio
    Service_rate=[]#tasa de servicio
    S_delay=[]#retraso promedio
    W_wait=[]#promedio espera
    q_time=[]#tiempo promedio en la cola
    x_time=[]#tiempo promedio en servicio
    l_time=[]#tiempo promedio en el nodo
    r=[]#tiempos entre llegadas

    for i in range(30):
        a,s=generador(da,ds,pa,ps)#generamos los tiempos de lelgada y de servicios aleatorios
        tra=len(a)#cantidad trabajos
        n.append(len(a))#guardamos los trabajos
        d,c=Single_server_queue(a,s)
        #print ("numeros de servicio: ",a)
       
        #estadisticas de trabajo promedio

        Tinterrival_t.append(a[tra-1]/tra) 
        Arrival_rate.append(1/Tinterrival_t[i]) 
        q_x.append(sum(s)/tra) #promedo de servicios
        Service_rate.append(1/q_x[i]) 
        S_delay.append(sum(d)/tra) #promedio de retraso
        W_wait.append(S_delay[i]+ q_x[i]) 

        #estadisiticas de tiempo promedio

        q_time.append((tra/c[len(c)-1])*S_delay[i]) #tiempo promedio en la cola
        x_time.append( (tra/c[len(c)-1])*q_x[i]) #tiempo promedio en servicio
        l_time.append(q_time[i]+x_time[i]) #tiempo promedio en el nodo 
    
    print("\n\t estadisticas de trabajo promedio\n")

    print(" promedio de tiempo de llegadas: ",intervalo(Tinterrival_t))
    print(" tasa de llegada: ",intervalo(Arrival_rate))
    print(" tiempo promedio de servicio: ",intervalo(q_x))
    print(" tasa de servicios: ",intervalo(Service_rate))
    print (" tiempo promedio de retraso en la cola: ",intervalo(S_delay))
    print("tiempo promedio de espera en el nodo: ",intervalo(W_wait))

    print("\n\t Estadisticas de tiempo promedio\n")

    print("numero del tiempo promedio en la cola: ",intervalo(q_time))
    print("numero del tiempo promedio en el servicio: ",intervalo(x_time))
    print(" numero del tiempo promedio en el nodo: ",intervalo(l_time ))

