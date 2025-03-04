import os
import datetime
import numpy as np
from scipy.stats import multivariate_normal

from src.meta_analysis import meta_analysis_md
import matplotlib.pyplot as plt


np.random.seed(2025)


# generate the correlation matrix
def cor_fixed(dim, rho):
  return rho * np.ones((dim, dim)) + (1 - rho) * np.eye(dim)


# basic setup and fix sample
dim = 100
rho = 0.6
num_study = 100
num_sample = 1000
mu = np.zeros(dim) + 0.0
cor = cor_fixed(dim, rho)
x_sample = multivariate_normal.rvs(mean=mu, cov=cor, size=num_sample)

##### contour plots for the plane slice (passing through the point estimates)


# 1-d substudies, (theta_1, theta_51)
sub_dim = 1
run_305 = meta_analysis_md(num_study, dim=dim, method="HCauchy", level=0.05)
origin = np.zeros(dim) + 0.0
direction = np.zeros(dim) + 0.0
direction[0] = 1.0
direction_2 = np.zeros(dim) + 0.0
direction_2[50] = 1
mu_hat = x_sample.mean(0)
# note: here we need std of the mean rather than std of x
sigma_hat = x_sample.std(0) / np.sqrt(num_sample)
# point estimate
p_estimate = run_305.find_minimizer(
  xi_hat=mu_hat,
  Sigma=sigma_hat**2,
  sub_dim=1,
  df=num_sample - 1,
  projs=None,
  method="Powell",
)
print(p_estimate)
# interval slice passing through origin
int_slice = run_305.interval_slice(
  point=p_estimate,
  direction=direction,
  xi_hat=mu_hat,
  Sigma=sigma_hat**2,
  sub_dim=1,
  df=num_sample - 1,
  projs=None,
)[:2]
print(int_slice)
# contour plot for the plane slice passing through origin
fig, ax = plt.subplots()
run_305.area_slice(
  point=p_estimate,
  direction_1=direction,
  direction_2=direction_2,
  xi_hat=mu_hat,
  Sigma=sigma_hat**2,
  sub_dim=1,
  df=num_sample - 1,
  projs=None,
  xrange=(-0.305, 0.305),
  yrange=(-0.22, 0.22),
  ax=ax,
  colors=["darkmagenta", "royalblue", "cadetblue", "darkseagreen", "darkkhaki"],
)
ax.scatter(
  [p_estimate[0]], [p_estimate[50]], s=30, c="blue", marker="^", label="Estimate"
)
ax.scatter([0], [0], s=50, c="red", marker="*", label="Projected True Value")
ax.legend()
plt.show()
os.makedirs("./fig", exist_ok=True)
fig.savefig(
  f"./fig/run_305_norm_{sub_dim}_1_51_{datetime.datetime.now().strftime('%m%d%H%M%S')}.pdf",
  format="pdf",
)
# simultaneous interval
sim_interval = run_305.simultaneous_interval(
  direction=direction,
  xi_hat=mu_hat,
  Sigma=sigma_hat**2,
  sub_dim=sub_dim,
  df=num_sample - 1,
  projs=None,
  method="Powell",
  print_res=True,
)
print(sim_interval)


# 1-d substudies, (theta_1, theta_2)
sub_dim = 1
run_312 = meta_analysis_md(num_study, dim=dim, method="HCauchy", level=0.05)
origin = np.zeros(dim) + 0.0
direction = np.zeros(dim) + 0.0
direction[0] = 1.0
direction_2 = np.zeros(dim) + 0.0
direction_2[1] = 1
mu_hat = x_sample.mean(0)
sigma_hat = x_sample.std(0) / np.sqrt(num_sample)
p_estimate = run_312.find_minimizer(
  xi_hat=mu_hat, Sigma=sigma_hat**2, sub_dim=1, df=num_sample - 1, projs=None
)
print(p_estimate)
fig, ax = plt.subplots()
run_312.area_slice(
  point=p_estimate,
  direction_1=direction,
  direction_2=direction_2,
  xi_hat=mu_hat,
  Sigma=sigma_hat**2,
  sub_dim=1,
  df=num_sample - 1,
  projs=None,
  xrange=(-0.305, 0.305),
  yrange=(-0.22, 0.22),
  ax=ax,
  colors=["darkmagenta", "royalblue", "cadetblue", "darkseagreen", "darkkhaki"],
)
ax.scatter(
  [p_estimate[0]], [p_estimate[1]], s=30, c="blue", marker="^", label="Estimate"
)
ax.scatter([0], [0], s=50, c="red", marker="*", label="Projected True Value")
ax.legend()
plt.show()
os.makedirs("./fig", exist_ok=True)
fig.savefig(
  f"./fig/run_312_norm_{sub_dim}_1_2_{datetime.datetime.now().strftime('%m%d%H%M%S')}.pdf",
  format="pdf",
)


# 5-d substudies, (theta_1, theta_51)
num_study = 20
sub_dim = 5
origin = np.zeros(dim) + 0.0
direction = np.zeros(dim) + 0.0
direction[0] = 1.0
direction_2 = np.zeros(dim) + 0.0
direction_2[50] = 1.0
x_sample_2 = x_sample.reshape(-1, num_study, sub_dim)
run_306 = meta_analysis_md(num_study, dim=dim, method="HCauchy", level=0.05)
mu_hat_2 = x_sample_2.mean(0)
Sigma_hat_2 = (
  np.transpose(x_sample_2 - mu_hat_2, (1, 2, 0))
  @ np.transpose(x_sample_2 - mu_hat_2, (1, 0, 2))
  / num_sample**2
)
p_estimate = run_306.find_minimizer(
  xi_hat=mu_hat_2, Sigma=Sigma_hat_2, sub_dim=sub_dim, df=num_sample - 1, projs=None
)
print(p_estimate)
int_slice = run_306.interval_slice(
  point=p_estimate,
  direction=direction,
  xi_hat=mu_hat_2,
  Sigma=Sigma_hat_2,
  sub_dim=sub_dim,
  df=num_sample - 1,
  projs=None,
)[:2]
print(int_slice)
fig, ax = plt.subplots()
run_306.area_slice(
  point=p_estimate,
  direction_1=direction,
  direction_2=direction_2,
  xi_hat=mu_hat_2,
  Sigma=Sigma_hat_2,
  sub_dim=sub_dim,
  df=num_sample - 1,
  projs=None,
  xrange=(-0.305, 0.305),
  yrange=(-0.22, 0.22),
  ax=ax,
  colors=["darkmagenta", "royalblue", "cadetblue", "darkseagreen", "darkkhaki"],
)
ax.scatter(
  [p_estimate[0]], [p_estimate[50]], s=30, c="blue", marker="^", label="Estimate"
)
ax.scatter([0], [0], s=50, c="red", marker="*", label="Projected True Value")
ax.legend()
plt.show()
os.makedirs("./fig", exist_ok=True)
fig.savefig(
  f"./fig/run_306_norm_{sub_dim}_1_51_{datetime.datetime.now().strftime('%m%d%H%M%S')}.pdf",
  format="pdf",
)


# 5-d substudies, (theta_1, theta_2)
num_study = 20
sub_dim = 5
origin = np.zeros(dim) + 0.0
direction = np.zeros(dim) + 0.0
direction[0] = 1.0
direction_2 = np.zeros(dim) + 0.0
direction_2[1] = 1.0
x_sample_2 = x_sample.reshape(-1, num_study, sub_dim)
run_308 = meta_analysis_md(num_study, dim=dim, method="HCauchy", level=0.05)
mu_hat_2 = x_sample_2.mean(0)
Sigma_hat_2 = (
  np.transpose(x_sample_2 - mu_hat_2, (1, 2, 0))
  @ np.transpose(x_sample_2 - mu_hat_2, (1, 0, 2))
  / num_sample**2
)
p_estimate = run_308.find_minimizer(
  xi_hat=mu_hat_2, Sigma=Sigma_hat_2, sub_dim=sub_dim, df=num_sample - 1, projs=None
)
print(p_estimate)
fig, ax = plt.subplots()
run_308.area_slice(
  point=p_estimate,
  direction_1=direction,
  direction_2=direction_2,
  xi_hat=mu_hat_2,
  Sigma=Sigma_hat_2,
  sub_dim=sub_dim,
  df=num_sample - 1,
  projs=None,
  xrange=(-0.305, 0.305),
  yrange=(-0.22, 0.22),
  ax=ax,
  colors=["darkmagenta", "royalblue", "cadetblue", "darkseagreen", "darkkhaki"],
)
ax.scatter(
  [p_estimate[0]], [p_estimate[1]], s=30, c="blue", marker="^", label="Estimate"
)
ax.scatter([0], [0], s=50, c="red", marker="*", label="Projected True Value")
ax.legend()
plt.show()
os.makedirs("./fig", exist_ok=True)
fig.savefig(
  f"./fig/run_308_norm_{sub_dim}_1_2_{datetime.datetime.now().strftime('%m%d%H%M%S')}.pdf",
  format="pdf",
)


# 25-d substudies, (theta_1, theta_51)
num_study = 4
sub_dim = 25
origin = np.zeros(dim) + 0.0
direction = np.zeros(dim) + 0.0
direction[0] = 1.0
direction_2 = np.zeros(dim) + 0.0
direction_2[50] = 1.0
x_sample_2 = x_sample.reshape(-1, num_study, sub_dim)
run_307 = meta_analysis_md(num_study, dim=dim, method="HCauchy", level=0.05)
mu_hat_2 = x_sample_2.mean(0)
Sigma_hat_2 = (
  np.transpose(x_sample_2 - mu_hat_2, (1, 2, 0))
  @ np.transpose(x_sample_2 - mu_hat_2, (1, 0, 2))
  / num_sample**2
)
p_estimate = run_307.find_minimizer(
  xi_hat=mu_hat_2,
  Sigma=Sigma_hat_2,
  sub_dim=sub_dim,
  df=num_sample - 1,
  projs=None,
  x0=x_sample.mean(0),
)
print(p_estimate)
int_slice = run_307.interval_slice(
  point=p_estimate,
  direction=direction,
  xi_hat=mu_hat_2,
  Sigma=Sigma_hat_2,
  sub_dim=sub_dim,
  df=num_sample - 1,
  projs=None,
)[:2]
print(int_slice)
fig, ax = plt.subplots()
run_307.area_slice(
  point=p_estimate,
  direction_1=direction,
  direction_2=direction_2,
  xi_hat=mu_hat_2,
  Sigma=Sigma_hat_2,
  sub_dim=sub_dim,
  df=num_sample - 1,
  projs=None,
  xrange=(-0.305, 0.305),
  yrange=(-0.22, 0.22),
  ax=ax,
  colors=["darkmagenta", "royalblue", "cadetblue", "darkseagreen", "darkkhaki"],
)
ax.scatter(
  [p_estimate[0]], [p_estimate[50]], s=30, c="blue", marker="^", label="Estimate"
)
ax.scatter([0], [0], s=50, c="red", marker="*", label="Projected True Value")
ax.legend()
plt.show()
os.makedirs("./fig", exist_ok=True)
fig.savefig(
  f"./fig/run_307_norm_{sub_dim}_1_51_{datetime.datetime.now().strftime('%m%d%H%M%S')}.pdf",
  format="pdf",
)


# 25-d substudies, (theta_1, theta_2)
num_study = 4
sub_dim = 25
origin = np.zeros(dim) + 0.0
direction = np.zeros(dim) + 0.0
direction[0] = 1.0
direction_2 = np.zeros(dim) + 0.0
direction_2[1] = 1.0
x_sample_2 = x_sample.reshape(-1, num_study, sub_dim)
run_309 = meta_analysis_md(num_study, dim=dim, method="HCauchy", level=0.05)
mu_hat_2 = x_sample_2.mean(0)
Sigma_hat_2 = (
  np.transpose(x_sample_2 - mu_hat_2, (1, 2, 0))
  @ np.transpose(x_sample_2 - mu_hat_2, (1, 0, 2))
  / num_sample**2
)
p_estimate = run_309.find_minimizer(
  xi_hat=mu_hat_2,
  Sigma=Sigma_hat_2,
  sub_dim=sub_dim,
  df=num_sample - 1,
  projs=None,
  x0=x_sample.mean(0),
)
print(p_estimate)
int_slice = run_309.interval_slice(
  point=p_estimate,
  direction=direction,
  xi_hat=mu_hat_2,
  Sigma=Sigma_hat_2,
  sub_dim=sub_dim,
  df=num_sample - 1,
  projs=None,
)[:2]
print(int_slice)
fig, ax = plt.subplots()
run_309.area_slice(
  point=p_estimate,
  direction_1=direction,
  direction_2=direction_2,
  xi_hat=mu_hat_2,
  Sigma=Sigma_hat_2,
  sub_dim=sub_dim,
  df=num_sample - 1,
  projs=None,
  xrange=(-0.305, 0.305),
  yrange=(-0.22, 0.22),
  ax=ax,
  colors=["darkmagenta", "royalblue", "cadetblue", "darkseagreen", "darkkhaki"],
)
ax.scatter(
  [p_estimate[0]], [p_estimate[1]], s=30, c="blue", marker="^", label="Estimate"
)
ax.scatter([0], [0], s=50, c="red", marker="*", label="Projected True Value")
ax.legend()
plt.show()
os.makedirs("./fig", exist_ok=True)
fig.savefig(
  f"./fig/run_309_norm_{sub_dim}_1_2_{datetime.datetime.now().strftime('%m%d%H%M%S')}.pdf",
  format="pdf",
)


# one single study, (theta_1, theta_51)
num_study = 1
sub_dim = 100
origin = np.zeros(dim) + 0.0
direction = np.zeros(dim) + 0.0
direction[0] = 1.0
direction_2 = np.zeros(dim) + 0.0
direction_2[50] = 1.0
x_sample_2 = x_sample.reshape(-1, num_study, sub_dim)
run_311 = meta_analysis_md(num_study, dim=dim, method="HCauchy", level=0.05)
mu_hat_2 = x_sample_2.mean(0)
Sigma_hat_2 = (
  np.transpose(x_sample_2 - mu_hat_2, (1, 2, 0))
  @ np.transpose(x_sample_2 - mu_hat_2, (1, 0, 2))
  / num_sample**2
)
p_estimate = run_311.find_minimizer(
  xi_hat=mu_hat_2, Sigma=Sigma_hat_2, sub_dim=sub_dim, df=num_sample - 1, projs=None
)
print(p_estimate)
int_slice = run_311.interval_slice(
  point=p_estimate,
  direction=direction,
  xi_hat=mu_hat_2,
  Sigma=Sigma_hat_2,
  sub_dim=sub_dim,
  df=num_sample - 1,
  projs=None,
)[:2]
print(int_slice)
fig, ax = plt.subplots()
run_311.area_slice(
  point=p_estimate,
  direction_1=direction,
  direction_2=direction_2,
  xi_hat=mu_hat_2,
  Sigma=Sigma_hat_2,
  sub_dim=sub_dim,
  df=num_sample - 1,
  projs=None,
  xrange=(-0.305, 0.305),
  yrange=(-0.22, 0.22),
  ax=ax,
  colors=["darkmagenta", "royalblue", "cadetblue", "darkseagreen", "darkkhaki"],
)
ax.scatter(
  [p_estimate[0]], [p_estimate[50]], s=30, c="blue", marker="^", label="Estimate"
)
ax.scatter([0], [0], s=50, c="red", marker="*", label="Projected True Value")
ax.legend()
plt.show()
os.makedirs("./fig", exist_ok=True)
fig.savefig(
  f"./fig/run_311_norm_{sub_dim}_1_51_{datetime.datetime.now().strftime('%m%d%H%M%S')}.pdf",
  format="pdf",
)



# one single study, (theta_1, theta_2)
num_study = 1
sub_dim = 100
origin = np.zeros(dim) + 0.0
direction = np.zeros(dim) + 0.0
direction[0] = 1.0
direction_2 = np.zeros(dim) + 0.0
direction_2[1] = 1.0
x_sample_2 = x_sample.reshape(-1, num_study, sub_dim)
run_310 = meta_analysis_md(num_study, dim=dim, method="HCauchy", level=0.05)
mu_hat_2 = x_sample_2.mean(0)
Sigma_hat_2 = (
  np.transpose(x_sample_2 - mu_hat_2, (1, 2, 0))
  @ np.transpose(x_sample_2 - mu_hat_2, (1, 0, 2))
  / num_sample**2
)
p_estimate = run_310.find_minimizer(
  xi_hat=mu_hat_2, Sigma=Sigma_hat_2, sub_dim=sub_dim, df=num_sample - 1, projs=None
)
print(p_estimate)
int_slice = run_310.interval_slice(
  point=p_estimate,
  direction=direction,
  xi_hat=mu_hat_2,
  Sigma=Sigma_hat_2,
  sub_dim=sub_dim,
  df=num_sample - 1,
  projs=None,
)[:2]
print(int_slice)
fig, ax = plt.subplots()
run_310.area_slice(
  point=p_estimate,
  direction_1=direction,
  direction_2=direction_2,
  xi_hat=mu_hat_2,
  Sigma=Sigma_hat_2,
  sub_dim=sub_dim,
  df=num_sample - 1,
  projs=None,
  xrange=(-0.305, 0.305),
  yrange=(-0.22, 0.22),
  ax=ax,
  colors=["darkmagenta", "royalblue", "cadetblue", "darkseagreen", "darkkhaki"],
)
ax.scatter(
  [p_estimate[0]], [p_estimate[1]], s=30, c="blue", marker="^", label="Estimate"
)
ax.scatter([0], [0], s=50, c="red", marker="*", label="Projected True Value")
ax.legend()
plt.show()
os.makedirs("./fig", exist_ok=True)
fig.savefig(
  f"./fig/run_310_norm_{sub_dim}_1_2_{datetime.datetime.now().strftime('%m%d%H%M%S')}.pdf",
  format="pdf",
)




# widths = np.zeros_like(np.arange(2,26))+0.
# for i, sub_dim in enumerate(np.arange(2,26)):
#   num_study = np.int_(np.ceil(dim/sub_dim))
#   origin = np.zeros(dim) + 0.0
#   direction = np.zeros(dim) + 0.0
#   direction[0] = 1/np.sqrt(2)
#   direction[1] = 1/np.sqrt(2)
#   x_sample_2 = np.concatenate([x_sample,x_sample[:,0:sub_dim*num_study-dim]], axis=1).reshape(-1, num_study, sub_dim)
#   run_317 = meta_analysis_md(num_study, dim=dim, method="HCauchy", level=0.05)
#   mu_hat_2 = x_sample_2.mean(0)
#   Sigma_hat_2 = (
#     np.transpose(x_sample_2 - mu_hat_2, (1, 2, 0))
#     @ np.transpose(x_sample_2 - mu_hat_2, (1, 0, 2))
#     / num_sample**2
#   )
#   p_estimate = run_317.find_minimizer(
#     xi_hat=mu_hat_2, Sigma=Sigma_hat_2, sub_dim=sub_dim, df=num_sample - 1, projs=None, x0=x_sample.mean(0)
#   )
#   # print(p_estimate)
#   res = run_317.simultaneous_interval(direction=direction, xi_hat=mu_hat_2, Sigma=Sigma_hat_2, sub_dim=sub_dim, df= num_sample-1, projs=None, x0=p_estimate, method='Powell', find_min=False, iteration=20)
#   print(res[1],res[0],res[1]-res[0])
#   widths[i]=res[1]-res[0]






# check_coverage
count_313 = 0
count_314 = 0
count_315 = 0
count_316 = 0
total = 2000
for repeat in range(total):
  x_sample = multivariate_normal.rvs(mean=mu, cov=cor, size=num_sample)
  num_study = 100
  sub_dim = 1
  run_313 = meta_analysis_md(num_study, dim=dim, method="HCauchy", level=0.05)
  mu_hat = x_sample.mean(0)
  sigma_hat = x_sample.std(0)/np.sqrt(num_sample)
  count_313 += run_313.check_cover(
    point=mu,
    xi_hat=mu_hat,
    Sigma=sigma_hat**2,
    sub_dim=1,
    df=num_sample - 1,
    projs=None,
  )
  num_study = 20
  sub_dim = 5
  x_sample_2 = x_sample.reshape(-1, num_study, sub_dim)
  run_314 = meta_analysis_md(num_study, dim=dim, method="HCauchy", level=0.05)
  mu_hat_2 = x_sample_2.mean(0)
  Sigma_hat_2 = (
    np.transpose(x_sample_2 - mu_hat_2, (1, 2, 0))
    @ np.transpose(x_sample_2 - mu_hat_2, (1, 0, 2))
    / num_sample**2
  )
  count_314 += run_314.check_cover(
    point=mu,
    xi_hat=mu_hat_2,
    Sigma=Sigma_hat_2,
    sub_dim=sub_dim,
    df=num_sample - 1,
    projs=None,
  )
  num_study = 4
  sub_dim = 25
  x_sample_2 = x_sample.reshape(-1, num_study, sub_dim)
  run_315 = meta_analysis_md(num_study, dim=dim, method="HCauchy", level=0.05)
  mu_hat_2 = x_sample_2.mean(0)
  Sigma_hat_2 = (
    np.transpose(x_sample_2 - mu_hat_2, (1, 2, 0))
    @ np.transpose(x_sample_2 - mu_hat_2, (1, 0, 2))
    / num_sample**2
  )
  count_315 += run_315.check_cover(
    point=mu,
    xi_hat=mu_hat_2,
    Sigma=Sigma_hat_2,
    sub_dim=sub_dim,
    df=num_sample - 1,
    projs=None,
  )
  num_study = 1
  sub_dim = 100
  x_sample_2 = x_sample.reshape(-1, num_study, sub_dim)
  run_316 = meta_analysis_md(num_study, dim=dim, method="HCauchy", level=0.05)
  mu_hat_2 = x_sample_2.mean(0)
  Sigma_hat_2 = (
    np.transpose(x_sample_2 - mu_hat_2, (1, 2, 0))
    @ np.transpose(x_sample_2 - mu_hat_2, (1, 0, 2))
    / num_sample**2
  )
  count_316 += run_316.check_cover(
    point=mu,
    xi_hat=mu_hat_2,
    Sigma=Sigma_hat_2,
    sub_dim=sub_dim,
    df=num_sample - 1,
    projs=None,
  )
print(count_313 / total)
print(count_314 / total)
print(count_315 / total)
print(count_316 / total)
