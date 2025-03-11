import time
import numpy as np

from src.distribution import euler_gamma, hcauchy_mean, harmonic_mean, landau


for m,x in [(2,0.2),(2,2),(2,10),(2,50),(10,1),(10,4),(10,10),(10,50),(100,2),(100,5),(100,10),(100,50),(1000,4),(1000,7),(1000,10),(1000,50)]:
  start_time = time.perf_counter()
  res=hcauchy_mean.pdf(x, m, precision=True)
  end_time = time.perf_counter()
  execution_time_ms = (end_time - start_time) * 1000 
  approx=landau.pdf(x,2/np.pi*(1-euler_gamma+np.log(m)),1)
  print(m,x,"PDF=",f'{res[0]:.9f}',"Err",res[1],"Time=", execution_time_ms, "ms","Approx=",f'{approx:.9f}')
  start_time = time.perf_counter()
  res=hcauchy_mean.cdf(x, m, precision=True)
  end_time = time.perf_counter()
  execution_time_ms = (end_time - start_time) * 1000 
  approx=landau.cdf(x,2/np.pi*(1-euler_gamma+np.log(m)),1)
  print(m,x,"CDF=",f'{res[0]:.9f}',"Err",res[1],"Time=", execution_time_ms, "ms","Approx=",f'{approx:.9f}')

for m,x in [(2,0.2),(2,2),(2,10),(2,50),(10,1),(10,4),(10,10),(10,50),(100,2),(100,5),(100,10),(100,50),(1000,4),(1000,7),(1000,10),(1000,50)]:
  start_time = time.perf_counter()
  res=harmonic_mean.pdf(x, m, precision=True)
  end_time = time.perf_counter()
  execution_time_ms = (end_time - start_time) * 1000 
  approx=landau.pdf(x,(1-euler_gamma+np.log(m)),np.pi/2)
  print(m,x,"PDF=",f'{res[0]:.9f}',"Err",res[1],"Time=", execution_time_ms, "ms","Approx=",f'{approx:.9f}')
  start_time = time.perf_counter()
  res=harmonic_mean.cdf(x, m, precision=True)
  end_time = time.perf_counter()
  execution_time_ms = (end_time - start_time) * 1000 
  approx=landau.cdf(x,(1-euler_gamma+np.log(m)),np.pi/2)
  print(m,x,"CDF=",f'{res[0]:.9f}',"Err",res[1],"Time=", execution_time_ms, "ms","Approx=",f'{approx:.9f}')

