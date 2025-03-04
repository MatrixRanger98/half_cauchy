import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

from src.meta_analysis import meta_analysis_md


# load data
dim=9
num_study=28
treatments = np.array(['acar', 'benf', 'metf', 'migl', 'piog', 'rosi', 'sita', 'sulf', 'vild'])
mu_hat= np.array([-1.90, -0.82, -0.20, -1.34, -1.10, -1.30, -0.77, 0.16, 0.10, -1.30, -1.09, -1.50, -0.14, -1.20, -0.40, -0.80, -0.57, -0.70, -0.37, -0.74, -1.41, 0.00, -0.68, -0.40, -0.23, -1.01, -1.20, -1.00])
sigma_hat= np.array([0.1414, 0.0992, 0.3579, 0.1435, 0.1141, 0.1268, 0.1078, 0.0849, 0.1831, 0.1014, 0.2263, 0.1624, 0.2239, 0.1436, 0.1549, 0.1432, 0.1291, 0.1273, 0.1184, 0.1839, 0.2235, 0.2339, 0.2828, 0.4356, 0.3467, 0.1366, 0.3758, 0.4669])
projs=np.array([
  [0,0,1,0,0,0,0,0,0],
  [0,0,1,0,0,0,0,0,0],
  [-1,0,1,0,0,0,0,0,0],
  [0,0,0,0,0,1,0,0,0],
  [0,0,0,0,0,1,0,0,0],
  [0,0,0,0,1,0,0,0,0],
  [0,0,0,0,0,1,0,0,0],
  [0,0,-1,0,1,0,0,0,0],
  [0,0,0,0,1,-1,0,0,0],
  [0,0,0,0,0,1,0,0,0],
  [0,0,0,0,0,1,0,0,0],
  [0,0,0,0,0,1,0,0,0],
  [0,0,-1,0,0,1,0,0,0],
  [0,0,0,0,0,1,0,-1,0],
  [1,0,0,0,0,0,0,-1,0],
  [1,0,0,0,0,0,0,0,0],
  [0,0,0,0,0,0,1,0,0],
  [0,0,0,0,0,0,0,0,1],
  [0,0,1,0,0,0,0,-1,0],
  [0,0,0,1,0,0,0,0,0],
  [0,0,0,1,0,0,0,0,0],
  [0,0,-1,0,0,1,0,0,0],
  [0,0,0,1,0,0,0,0,0],
  [0,0,1,0,0,0,0,0,0],
  [0,1,0,0,0,0,0,0,0],
  [0,1,0,0,0,0,0,0,0],
  [0,0,1,0,0,0,0,0,0],
  [1,0,0,0,0,0,0,0,0]
])

# check if there is any inconsistent studies
run_401 = meta_analysis_md(num_study, dim=dim, method="HCauchy", level=0.05)
p_estimate = run_401.find_minimizer(xi_hat=mu_hat, Sigma=sigma_hat**2, sub_dim=1, df=None, projs=projs)
print(p_estimate)
print(run_401.check_cover(point=p_estimate,xi_hat=mu_hat, Sigma=sigma_hat**2, sub_dim=1, df=None, projs=projs))
# false, empty confidence region
u=np.argmax((mu_hat-projs@p_estimate)**2/sigma_hat**2)
print(u)
# u=0, exclude the first study from now on
num_study_1 = num_study - 1
mu_hat_1 = np.array([j for i,j in enumerate(mu_hat) if i not in [u,]])
sigma_hat_1 = np.array([j for i,j in enumerate(sigma_hat) if i not in [u,]])
projs_1 = np.array([j for i,j in enumerate(projs) if i not in [u,]])
run_401 = meta_analysis_md(num_study_1, dim=dim, method="HCauchy", level=0.05)
p_estimate_1 = run_401.find_minimizer(xi_hat=mu_hat_1, Sigma=sigma_hat_1**2, sub_dim=1, df=None, projs=projs_1)
print(p_estimate_1)
print(run_401.check_cover(point=p_estimate_1,xi_hat=mu_hat_1, Sigma=sigma_hat_1**2, sub_dim=1, df=None, projs=projs_1))
# false, empty confidence region
u_1=np.argmax((mu_hat_1-projs_1@p_estimate_1)**2/sigma_hat_1**2)
# u=5, exclude this study as well
num_study_2 = num_study_1 - 1
mu_hat_2 = np.array([j for i,j in enumerate(mu_hat_1) if i not in [u_1,]])
sigma_hat_2 = np.array([j for i,j in enumerate(sigma_hat_1) if i not in [u_1,]])
projs_2 = np.array([j for i,j in enumerate(projs_1) if i not in [u_1,]])
run_401 = meta_analysis_md(num_study_2, dim=dim, method="HCauchy", level=0.05)
p_estimate_2 = run_401.find_minimizer(xi_hat=mu_hat_2, Sigma=sigma_hat_2**2, sub_dim=1, df=None, projs=projs_2)
print(p_estimate_2)
print(run_401.check_cover(point=p_estimate_2,xi_hat=mu_hat_2, Sigma=sigma_hat_2**2, sub_dim=1, df=None, projs=projs_2))

run_401.simultaneous_interval(direction=np.array([0,0,1,-1,0,0,0,0,0]), xi_hat=mu_hat_2,Sigma=sigma_hat_2**2, sub_dim=1, df=None, projs=projs_2)

heatmap_data_1 = np.zeros((dim,dim))
for i in range(dim):
  for j in range(dim):
    if j==i:
      drt = np.zeros(dim)
      drt[i] = 1
      tmp = run_401.simultaneous_interval(direction=drt, xi_hat=mu_hat_2,Sigma=sigma_hat_2**2, sub_dim=1, df=None, projs=projs_2)
      heatmap_data_1[i,i]=tmp[1]-tmp[0]
    else:
      drt = np.zeros(dim)
      drt[i] = 1
      drt[j] = -1
      tmp = run_401.simultaneous_interval(direction=drt, xi_hat=mu_hat_2,Sigma=sigma_hat_2**2, sub_dim=1, df=None, projs=projs_2)
      heatmap_data_1[i,j]=tmp[1]-tmp[0]

# np.save('./tmp/heatmap_data_1.npy', heatmap_data_1)
heatmap_data_1=pd.DataFrame(heatmap_data_1, columns=treatments, index=treatments)
sns.heatmap(heatmap_data_1, annot=True, cmap='viridis', vmin=0, vmax=1.5)
plt.savefig(f'./fig/real_heatmap_1_{datetime.datetime.now().strftime('%m%d%H%M%S')}.pdf',bbox_inches="tight", format='pdf')



tmp = pd.read_feather("./tmp/ci_width.feather")
indices=[i for i in range(10) if i != 5]
heatmap_data_2 = tmp.values[indices][:,indices]
for i in range(9):
  if i<5:
    heatmap_data_2[i,i]=tmp.values[5,i]
  else:
    heatmap_data_2[i,i]=tmp.values[5,i+1]
heatmap_data_2=pd.DataFrame(heatmap_data_2, columns=treatments, index=treatments)
sns.heatmap(heatmap_data_2, annot=True, cmap='viridis', vmin=0, vmax=1.5)
plt.savefig(f'./fig/real_heatmap_2_{datetime.datetime.now().strftime('%m%d%H%M%S')}.pdf',bbox_inches="tight", format='pdf')

# heatmap_data_1 = np.load('./tmp/heatmap_data_1.npy')
sns.heatmap(heatmap_data_2*norm.ppf(.025/45)/norm.ppf(.025), annot=True, cmap='viridis', vmin=0, vmax=1.5)
plt.savefig(f'./fig/real_heatmap_3_{datetime.datetime.now().strftime('%m%d%H%M%S')}.pdf',bbox_inches="tight", format='pdf')

# Custom formatter to remove trailing zeros
def format_value(x):
    return f"{x:.2f}".rstrip('0').rstrip('.')  # Formats to 3 decimals and removes trailing zeros/dots


# Generate annotations dynamically
data=heatmap_data_1-heatmap_data_2*norm.ppf(.025/45)/norm.ppf(.025)
annotations = np.vectorize(format_value)(data)

# Create the heatmap
sns.set_style({'font.family':'sans-serif', 'font.sans-serif':'Helvetica'})
sns.set_theme(font_scale=1.15)
sns.heatmap(data, annot=annotations, fmt="",cmap='coolwarm',vmin=-0.5, vmax=0.5)
plt.savefig(f'./fig/real_heatmap_dif_{datetime.datetime.now().strftime('%m%d%H%M%S')}.pdf',bbox_inches="tight", format='pdf')


# plot of width wrt number of intervals/estimates
for i in [1,2,6,5]:
  width = np.array([norm.ppf(.025/k)/norm.ppf(.025)*heatmap_data_2.values[i,i] for k in range(1,46)])
  fig, ax = plt.subplots()
  ax.plot(np.arange(1,46), width, label='WLS', linewidth=1.5)
  ax.plot(np.arange(1,46), heatmap_data_1.values[i,i]*np.ones(45), ls='-', label='HCCT', linewidth=1.5)
  ax.legend()
  ax.set_xlabel('number of comparisons')
  ax.set_ylabel("width of simultaneous CI")
  # ax.set_ylim(0,0.35)
  plt.show()
  fig.savefig(f'./fig/real_tr_{i+1}_{datetime.datetime.now().strftime('%m%d%H%M%S')}.pdf',bbox_inches="tight", format='pdf')


