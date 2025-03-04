import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from scipy.interpolate import make_interp_spline
from cycler import cycler

from src.meta_analysis import combination_test


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


num_run = 10000
num_study = 500
d_rho = 0.1
rho = 0.4

v_1 = 0
v_2 = 0.3
r_1 = 0.1
r_2 = 0.3

def num_pos(num_study, v): 
  return int(num_study**(1-v))

def val_pos(num_study, r):
  return np.sqrt(2*r*np.log(num_study))

test_list=['HCauchy', 'EHMP', 'HMP', 'Cauchy', 'Levy', 'Fisher', 'Stouffer', 'Bonferroni', 'Simes']

point_0 = np.zeros(num_study)
point_1 = np.zeros(num_study)
point_2 = np.zeros(num_study)
point_1[:num_pos(num_study, v_1)]=val_pos(num_study, r_1)
point_2[:num_pos(num_study, v_2)]=val_pos(num_study, r_2)

power=np.zeros((len(test_list),int(1/d_rho),2))
for i, rho in enumerate(np.arange(0,1,d_rho)):
  cor=cor_fixed(num_study, rho)
  for k, method in enumerate(test_list):
    theta_hat_1= multivariate_normal.rvs(mean=point_1, cov=cor, size=num_run)
    theta_hat_2= multivariate_normal.rvs(mean=point_2, cov=cor, size=num_run)
    p_vector_1 = 2 * np.maximum(norm.sf(np.abs(point_0 - theta_hat_1)), 1e-150)
    p_vector_2 = 2 * np.maximum(norm.sf(np.abs(point_0 - theta_hat_2)), 1e-150)
    run_109 = combination_test(num_study, method, level=0.05)
    run_110 = combination_test(num_study, method, level=0.05)
    count_109 = 0
    count_110 = 0
    for j in range(num_run):
      count_109 += run_109.make_decision(p_vector_1[j])
      count_110 += run_110.make_decision(p_vector_2[j])
    power[k,i,0]=count_109/num_run
    power[k,i,1]=count_110/num_run

# os.makedirs('./tmp', exist_ok=True)
# np.save(f"./tmp/run_109_110_{datetime.datetime.now().strftime('%m%d%H%M%S')}",power)


num_run =10000
num_study=500
rho=0
# num_pos = int(np.sqrt(num_study))
# num_pos =500
# val_pos = np.sqrt(np.log(num_study))
# val_pos=np.sqrt(0.2*np.log(num_study))

# def val_pos(num_study, r):
#   return np.sqrt(2*r*np.log(num_study))
# def num_pos(num_study, v): 
#   return int(num_study**(1-v))
# cor_1=cor_ar_1(num_study, rho)
cor_2=cor_fixed(num_study, rho)
point_true = np.zeros(num_study)
v=0.5
r=0.5
point_true[:num_pos(num_study,v)]=val_pos(num_study,r)
point_0 = np.zeros(num_study)
# theta_hat_1= multivariate_normal.rvs(mean=point_true, cov=cor_1, size=num_run)
theta_hat_2= multivariate_normal.rvs(mean=point_true, cov=cor_2, size=num_run)
# p_vector_1 = 2 * np.maximum(norm.sf(np.abs(point_0 - theta_hat_1)), 1e-150)
p_vector_2 = 2 * np.maximum(norm.sf(np.abs(point_0 - theta_hat_2)), 1e-150)
# run_105 = combination_test(num_study, 'Fisher', level=0.05)
run_106 = combination_test(num_study, 'HCauchy', level=0.05)
# count_105 = 0
count_106 = 0
for j in range(num_run):
  # count_105 += run_105.make_decision(p_vector_1[j])
  count_106 += run_106.make_decision(p_vector_2[j])
# print(count_105/num_run)
print(count_106/num_run)

# power = np.load('')



x_new=np.arange(0,.9,d_rho/10)

curves_1 = np.zeros((len(test_list),x_new.shape[0]))
curves_2 = np.zeros((len(test_list),x_new.shape[0]))
for i in range(len(test_list)):
  curves_1[i,:]=make_interp_spline(np.arange(0,1,d_rho),power[i,:,0], k=3)(x_new)
  curves_2[i,:]=make_interp_spline(np.arange(0,1,d_rho),power[i,:,1], k=3)(x_new)



fig, ax = plt.subplots()
ax.set_prop_cycle(
        cycler('color',
               list(plt.rcParams["axes.prop_cycle"].by_key()["color"])))
ax.plot(x_new, curves_1[0], label=test_list[0], linewidth=1.5, zorder=2)
ax.plot(x_new, curves_1[1], label=test_list[1], linewidth=1.5, zorder=1)
for values, legend in zip(curves_1[3:], test_list[3:]):
  ax.plot(x_new, values, label=legend, linewidth=1.5,zorder=0)
ax.legend()
ax.set_xlabel(r'$\rho$')
ax.set_ylabel("power")
plt.show()
fig.savefig(f"./fig/run_109_{datetime.datetime.now().strftime('%m%d%H%M%S')}.pdf",format='pdf')


fig, ax = plt.subplots()
ax.set_prop_cycle(
        cycler('color',
               list(plt.rcParams["axes.prop_cycle"].by_key()["color"])))
ax.plot(x_new, curves_2[0], label=test_list[0], linewidth=1.5, zorder=2)
ax.plot(x_new, curves_2[1], label=test_list[1], linewidth=1.5, zorder=1)
for values, legend in zip(curves_2[3:], test_list[3:]):
  ax.plot(x_new, values, label=legend, linewidth=1.5,zorder=0)
ax.legend()
ax.set_xlabel(r'$\rho$')
ax.set_ylabel("power")
plt.show()
fig.savefig(f"./fig/run_110_{datetime.datetime.now().strftime('%m%d%H%M%S')}.pdf",format='pdf')


