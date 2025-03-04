import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, chi2
from scipy.interpolate import make_interp_spline

from src.meta_analysis import meta_analysis_md

'''
1. coverage multivariate normal
2. contour plot + scatter plot for 2d
'''
# generate the correlation matrix
def cor_fixed(dim, rho):
  return rho*np.ones((dim,dim))+(1-rho)*np.eye(dim)

np.random.seed(2025)

repeat = 10
num_run = 10000
num_study = 500
point=np.zeros(num_study)
dims=np.array([2,5,10,25])
d_rho =0.1
rho = 0.3

coverage=np.zeros((repeat,len(dims),int(1/d_rho),2))
x_old=np.arange(0,1,d_rho)

for k, dim in enumerate(dims):
  run_301=meta_analysis_md(num_study, dim=dim, method="HCauchy", level=.05)
  run_302=meta_analysis_md(num_study, dim=dim, method="HCauchy", level=.01)
  for i, rho in enumerate(x_old):
    cor = cor_fixed(num_study, rho)
    for rep in range(repeat):
      theta_hat= np.swapaxes(multivariate_normal.rvs(mean=point, cov=cor, size=(num_run,dim)),1,2)
      chi2_score = (theta_hat**2).sum(-1)
      del theta_hat
      p_vector = np.maximum(chi2.sf(chi2_score, dim),1e-150)
      count_301 = 0
      count_302 = 0
      for j in range(num_run):
        count_301+=run_301.make_decision(p_vector[j])
        count_302+=run_302.make_decision(p_vector[j])
      del p_vector
      coverage[rep, k,i,0]=1-count_301/num_run
      coverage[rep, k,i,1]=1-count_302/num_run

coverage =coverage.mean(0)

# np.save(f"./tmp/run_301_302_m_500_{datetime.datetime.now().strftime('%m%d%H%M%S')}",coverage)

# coverage= np.load("good/coverage_2.npy")

x_new=np.arange(0,.9,d_rho/5)
curves=np.zeros((len(dims),x_new.shape[0],2))
for i in range(len(dims)):
  for j in (0,1):
    curves[i,:,j]=make_interp_spline(x_old,coverage[i,:,j], k=3)(x_new)


fig, ax = plt.subplots()
# Save the output of 'plot', as we need it later
l1, = ax.plot(x_new, curves[0,:,0],c='coral')
l2, = ax.plot(x_new, curves[1,:,0],c='darkseagreen')
l3, = ax.plot(x_new, curves[2,:,0],c='orchid')
l4, = ax.plot(x_new, curves[3,:,0],c='steelblue')
l5, = ax.plot(x_new, curves[0,:,1],c='coral',linestyle=(0, (5, 1)))
l6, = ax.plot(x_new, curves[1,:,1],c='darkseagreen',linestyle=(0, (5, 1)))
l7, = ax.plot(x_new, curves[2,:,1],c='orchid',linestyle=(0, (5, 1)))
l8, = ax.plot(x_new, curves[3,:,1],c='steelblue',linestyle=(0, (5, 1)))

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

ax.set_xlabel(r'$\rho$')
ax.set_ylabel("coverage")
# Put a legend to the right of the current axis
ax.legend([l1,l2,l3,l4,l5,l6,l7,l8],[f"dim={dims[0]}   p=.05", f"dim={dims[1]}   p=.05", f"dim={dims[2]} p=.05", f"dim={dims[3]} p=.05",f"dim={dims[0]}   p=.01", f"dim={dims[1]}   p=.01", f"dim={dims[2]} p=.01", f"dim={dims[3]} p=.01"],loc='lower left', bbox_to_anchor=(0, 0.45),fontsize="8")
plt.show()

fig.savefig(f"./fig/run_301_302_m_500_{datetime.datetime.now().strftime('%m%d%H%M%S')}.pdf",format='pdf',bbox_inches="tight")




np.random.seed(2025)

repeat = 10
num_run = 10000
num_study = 10
point=np.zeros(num_study)
dims=np.array([2,5,10,25])
d_rho =0.1
rho = 0.3

coverage=np.zeros((repeat,len(dims),int(1/d_rho),2))
x_old=np.arange(0,1,d_rho)

for k, dim in enumerate(dims):
  run_301=meta_analysis_md(num_study, dim=dim, method="HCauchy", level=.05)
  run_302=meta_analysis_md(num_study, dim=dim, method="HCauchy", level=.01)
  for i, rho in enumerate(x_old):
    cor = cor_fixed(num_study, rho)
    for rep in range(repeat):
      theta_hat= np.swapaxes(multivariate_normal.rvs(mean=point, cov=cor, size=(num_run,dim)),1,2)
      chi2_score = (theta_hat**2).sum(-1)
      del theta_hat
      p_vector = np.maximum(chi2.sf(chi2_score, dim),1e-150)
      count_301 = 0
      count_302 = 0
      for j in range(num_run):
        count_301+=run_301.make_decision(p_vector[j])
        count_302+=run_302.make_decision(p_vector[j])
      del p_vector
      coverage[rep, k,i,0]=1-count_301/num_run
      coverage[rep, k,i,1]=1-count_302/num_run

coverage =coverage.mean(0)

# np.save(f"./tmp/run_301_302_m_10_{datetime.datetime.now().strftime('%m%d%H%M%S')}",coverage)

# coverage= np.load("good/coverage_2.npy")

x_new=np.arange(0,.9,d_rho/5)
curves=np.zeros((len(dims),x_new.shape[0],2))
for i in range(len(dims)):
  for j in (0,1):
    curves[i,:,j]=make_interp_spline(x_old,coverage[i,:,j], k=3)(x_new)


fig, ax = plt.subplots()
# Save the output of 'plot', as we need it later
l1, = ax.plot(x_new, curves[0,:,0],c='coral')
l2, = ax.plot(x_new, curves[1,:,0],c='darkseagreen')
l3, = ax.plot(x_new, curves[2,:,0],c='orchid')
l4, = ax.plot(x_new, curves[3,:,0],c='steelblue')
l5, = ax.plot(x_new, curves[0,:,1],c='coral',linestyle=(0, (5, 1)))
l6, = ax.plot(x_new, curves[1,:,1],c='darkseagreen',linestyle=(0, (5, 1)))
l7, = ax.plot(x_new, curves[2,:,1],c='orchid',linestyle=(0, (5, 1)))
l8, = ax.plot(x_new, curves[3,:,1],c='steelblue',linestyle=(0, (5, 1)))

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

ax.set_xlabel(r'$\rho$')
ax.set_ylabel("coverage")
# Put a legend to the right of the current axis
ax.legend([l1,l2,l3,l4,l5,l6,l7,l8],[f"dim={dims[0]}   p=.05", f"dim={dims[1]}   p=.05", f"dim={dims[2]} p=.05", f"dim={dims[3]} p=.05",f"dim={dims[0]}   p=.01", f"dim={dims[1]}   p=.01", f"dim={dims[2]} p=.01", f"dim={dims[3]} p=.01"],loc='lower left', bbox_to_anchor=(0, 0.45),fontsize="8")
plt.show()

fig.savefig(f"./fig/run_301_302_m_10_{datetime.datetime.now().strftime('%m%d%H%M%S')}.pdf",format='pdf',bbox_inches="tight")