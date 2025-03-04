import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal

from src.meta_analysis import meta_analysis_1d


def cor_ar_1(dim, rho) -> np.ndarray:
  power_mat = np.abs(np.arange(dim).reshape(-1,1)-np.arange(dim))
  return rho**power_mat

def cor_fixed(dim, rho):
  return rho*np.ones((dim,dim))+(1-rho)*np.eye(dim)


np.random.seed(2025)

'''
plot 1 & 2 curve of coverage versus rho
1: AR correlation normal  2: fixed correlation normal
each plot two lines 99% and 95% with bands

plot 3 & 4 comparison of type I error versus rho
3: AR correlation 4: fixed correlation
5% only normal only
methods include ...

plot 5 & 6 density of width
5: AR correlation 6: fixed correlation
95% only normal only
rho = 0, 0.3, 0.6, 0.9

plot 7 & 8 power versus rho
7: AR correlation 8: fixed correlation
noise level and setup: see references
significance level 5% only
methods include ...
'''



run_107 = meta_analysis_1d(500, "HCauchy", level=0.05)
run_108 = meta_analysis_1d(500, "HCauchy", level=0.05)

num_run=10000
num_study = 500
rho =0.4
rho_s = np.array([0,0.3,0.6,0.9])
width=np.zeros((2,len(rho_s),num_run))


for i, rho in enumerate(rho_s):
  cor_1=cor_ar_1(num_study, rho)
  cor_2=cor_fixed(num_study, rho)
  point = np.zeros(num_study)
  theta_hat_1= multivariate_normal.rvs(mean=point, cov=cor_1, size=num_run)
  theta_hat_2= multivariate_normal.rvs(mean=point, cov=cor_2, size=num_run)
  for j in range(num_run):
    low_1, high_1, _ = run_107.confidence_interval(theta_hat_1[j], 1.0)
    low_2, high_2, _ = run_108.confidence_interval(theta_hat_2[j], 1.0)
    width[0,i,j] = high_1-low_1
    width[1,i,j]= high_2-low_2

# os.makedirs('./tmp', exist_ok=True)
# np.save(f"./tmp/run_107_108_{datetime.datetime.now().strftime('%m%d%H%M%S')}", width)

# width= np.load("good/width.npy")

fig, ax = plt.subplots()
sns.set_style('whitegrid')
data_preproc = pd.DataFrame({
    r'$\rho=0$': width[0,0,:],
    r'$\rho=0.3$': width[0,1,:],
    r'$\rho=0.6$': width[0,2,:],
    r'$\rho=0.9$': width[0,3,:]})
sns.kdeplot(data_preproc, ax=ax)
ax.set_xlim(0.2,3)
ax.set_xlabel('width')
ax.set_ylabel('density')
plt.show()
fig.savefig(f"./fig/run_107_{datetime.datetime.now().strftime('%m%d%H%M%S')}.pdf",format='pdf')

fig, ax = plt.subplots()
sns.set_style('whitegrid')
data_preproc = pd.DataFrame({
    r'$\rho=0$': width[1,0,:],
    r'$\rho=0.3$': width[1,1,:],
    r'$\rho=0.6$': width[1,2,:],
    r'$\rho=0.9$': width[1,3,:]})
sns.kdeplot(data_preproc, ax=ax)
ax.set_xlim(0.4,4.1)
ax.set_xlabel('width')
ax.set_ylabel('density')
plt.show()
fig.savefig(f"./fig/run_108_{datetime.datetime.now().strftime('%m%d%H%M%S')}.pdf",format='pdf')
