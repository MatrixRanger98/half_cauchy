import numpy as np
import datetime
from scipy.stats import multivariate_normal

from src.meta_analysis import meta_analysis_md
import matplotlib.pyplot as plt

np.random.seed(2024)

# generate the correlation matrix
def cor_fixed(dim, rho):
  return rho*np.ones((dim,dim))+(1-rho)*np.eye(dim)

rhos =[0,0.3,0.6,0.9]
dim=2
num_study=500
rho=0.6

run_303 = meta_analysis_md(500, dim=dim, method="HCauchy", level=0.05)
for i, rho in enumerate(rhos):
  cor = cor_fixed(num_study, rho)
  theta_hat= multivariate_normal.rvs(mean=np.zeros(500), cov=cor, size=dim).T
  fig, ax = plt.subplots()
  run_303.sim_confidence_area(point=(0,0), direction_1=(1,0), direction_2=(0,1), xi_hat=theta_hat, Sigma=np.eye(dim), sub_dim=dim, df=None, projs=None, xrange=(-4,4.1), yrange=(-3,3.1), ax=ax, colors=["darkmagenta","royalblue","cadetblue","darkseagreen","darkkhaki"])
  ax.scatter(theta_hat.T[0],theta_hat.T[1],s=0.5, c='orange', label='Estimates from\n Each Study')
  ax.scatter([0],[0], s=50, c='red', marker='*', label='True Value')
  ax.legend(loc='upper left')
  ax.set_xlim(-4,4.1)
  ax.set_ylim(-3,3.1)
  # ax.annotate('True Value', xy=(0,0), xytext=(-1,0.2))
  # ax.annotate('Observed Values', xy=(0,0), xytext=(-2.3,-2.5))
  plt.show()

  fig.savefig(f"./fig/run_303_rho_{rho}_{datetime.datetime.now().strftime('%m%d%H%M%S')}.pdf", format="pdf")


rhos =[0,0.3,0.6,0.9]
dim=10
num_study=500
rho=0.6

run_304 = meta_analysis_md(500, dim=dim, method="HCauchy", level=0.05)
for i, rho in enumerate(rhos):
  cor = cor_fixed(num_study, rho)
  theta_hat= multivariate_normal.rvs(mean=np.zeros(500), cov=cor, size=dim).T
  fig, ax = plt.subplots()
  run_304.sim_confidence_area(point=np.zeros(10), direction_1=[1]+[0]*9, direction_2=[0,1]+[0]*8, xi_hat=theta_hat, Sigma=np.eye(dim), sub_dim=dim, df=None, projs=None, xrange=(-4,4.1), yrange=(-3,3.1), ax=ax, colors=["darkmagenta","royalblue","cadetblue","darkseagreen","darkkhaki"])
  ax.scatter([0],[0], s=50, c='red', marker='*', label='True Value')
  ax.scatter(theta_hat.T[0],theta_hat.T[1],s=0.5, c='orange', label='Estimates from\n Each Study')
  ax.legend(loc='upper left')
  ax.set_xlim(-4,4.1)
  ax.set_ylim(-3,3.1)
  # ax.annotate('True Value', xy=(0,0), xytext=(-1,0.2))
  # ax.annotate('Observed Values', xy=(0,0), xytext=(-2.3,-2.5))
  plt.show()

  fig.savefig(f"./fig/run_304_rho_{rho}_{datetime.datetime.now().strftime('%m%d%H%M%S')}.pdf", format="pdf")




