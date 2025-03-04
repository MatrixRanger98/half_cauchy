import os
import datetime
import numpy as np
from scipy.stats import multivariate_normal, norm
from scipy.optimize import minimize_scalar, root_scalar, minimize
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt

from src.meta_analysis import meta_analysis_md

np.random.seed(2025)

design_mat = np.array(
  [
    [0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0],
    [-1, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, -1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, -1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, -1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, -1, 0],
    [1, 0, 0, 0, 0, 0, 0, -1, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 1, 0, 0, 0, 0, -1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, -1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
  ]
)

theta = np.array([0, -0.5, -1, 0, -0.5, -1, 0, -0.5, -1])
eta = design_mat @ theta


# generate the correlation matrix
def cor_fixed(dim, rho):
  return rho * np.ones((dim, dim)) + (1 - rho) * np.eye(dim)

num_sample = 100
dim = 9
num_study = 28
sub_dim = 1

for rho in [0, 0.3, 0.6, 0.9]:
  cor = cor_fixed(num_study, rho)
  x_sample = multivariate_normal.rvs(mean=eta, cov=cor, size=num_sample)

  run_401 = meta_analysis_md(num_study, dim=dim, method="HCauchy", level=0.05)
  zeta_hat = x_sample.mean(0)
  sigma_hat = x_sample.std(0) / np.sqrt(num_sample)
  p_estimate = run_401.find_minimizer(
    xi_hat=zeta_hat,
    Sigma=sigma_hat**2,
    sub_dim=1,
    df=num_sample - 1,
    projs=design_mat,
    method="Powell",
  )
  print("estimate from hcct is", p_estimate)


direction = np.zeros(dim) + 0.0
direction[0] = 1.0
direction1 = np.zeros(dim) + 0.0
direction1[1] = 1.0

width_wls_0 = np.zeros(10)
width_wls_1 = np.zeros(10)
width_hcct_0 = np.zeros(10)
width_hcct_1 = np.zeros(10)
coverage_wls = np.zeros(10)
coverage_hcct = np.zeros(10)
repeat = 500


for k, rho in enumerate(np.arange(0, 1, 0.1)):
  print("current rho", rho)
  for i in range(repeat):
    cor = cor_fixed(num_study, rho)
    x_sample = multivariate_normal.rvs(mean=eta, cov=cor, size=num_sample)

    run_401 = meta_analysis_md(num_study, dim=dim, method="HCauchy", level=0.05)
    zeta_hat = x_sample.mean(0)
    sigma_hat = x_sample.std(0) / np.sqrt(num_sample)
    p_estimate = run_401.find_minimizer(
      xi_hat=zeta_hat,
      Sigma=sigma_hat**2,
      sub_dim=1,
      df=num_sample - 1,
      projs=design_mat,
      method="Powell",
    )
    print("estimate from hcct is", p_estimate)
    a = run_401.simultaneous_interval(
      direction=direction,
      xi_hat=zeta_hat,
      Sigma=sigma_hat**2,
      df=num_sample - 1,
      sub_dim=1,
      projs=design_mat,
      method="Powell",
    )
    print("hcct result 1", a[0], a[1], "width", a[1] - a[0])
    width_hcct_0[k] += a[1] - a[0]
    a1 = run_401.simultaneous_interval(
      direction=direction1,
      xi_hat=zeta_hat,
      Sigma=sigma_hat**2,
      df=num_sample - 1,
      sub_dim=1,
      projs=design_mat,
      method="Powell",
    )
    print("hcct result 2", a1[0], a1[1], "width", a1[1] - a1[0])
    width_hcct_1[k] += a1[1] - a1[0]
    check = run_401.check_cover(
      point=theta,
      xi_hat=zeta_hat,
      Sigma=sigma_hat**2,
      df=num_sample - 1,
      sub_dim=1,
      projs=design_mat,
    )
    coverage_hcct[k] += check
    # if check == 0:
    #   np.savez(
    #     f'./tmp/{datetime.datetime.now().strftime('%m%d%H%M%S')}.npz',
    #     point=theta,
    #     xi_hat=zeta_hat,
    #     Sigma=sigma_hat**2,
    #     projs=design_mat,
    #   )
    print("current coverage of hcct", coverage_hcct[k] / (i + 1))
    b = np.sqrt(np.diag(np.linalg.inv((design_mat.T / sigma_hat**2) @ design_mat))) * (
      -norm.ppf(0.025 / dim)
    )
    width_wls_0[k] += b[0] * 2
    width_wls_1[k] += b[1] * 2
    mat_tmp = design_mat
    x0 = np.linalg.pinv(mat_tmp.T @ mat_tmp) @ mat_tmp.T @ zeta_hat.reshape(-1)
    if np.all(x0 - b < theta) and np.all(x0 + b > theta):
      coverage_wls[k] += 1
    print("current coverage of wls", coverage_wls[k] / (i + 1))

coverage_hcct = coverage_hcct / repeat
coverage_wls = coverage_wls / repeat
width_hcct_0 = width_hcct_0 / repeat
width_hcct_1 = width_hcct_1 / repeat
width_wls_0 = width_wls_0 / repeat
width_wls_1 = width_wls_1 / repeat

np.savez(
  "./tmp/last_sim.npz",
  coverage_hcct=coverage_hcct,
  coverage_wls=coverage_wls,
  width_hcct_0=width_hcct_0,
  width_hcct_1=width_hcct_1,
  width_wls_0=width_wls_0,
  width_wls_1=width_wls_1,
)


c1 = coverage_hcct
c2 = coverage_wls
c3 = np.ones(10) * 0.95


y1 = width_hcct_0
y2 = width_hcct_1
x1 = width_wls_0
x2 = width_wls_1


# The following z1 and z2 are results from manual adjustment 
# b = np.sqrt(np.diag(np.linalg.inv((design_mat.T / sigma_hat**2) @ design_mat))) * (
#       -norm.ppf(0.025 / dim)
#     )
# dim = 10,37,200,800,5000,35000,200000,1500000,8000000,30000000
# this ensures that the coverage is always roughly 0.95
z1 = np.array(
  [
    0.30123266,
    0.34405365,
    0.39331598,
    0.42974317,
    0.47478013,
    0.518090934,
    0.554759427,
    0.59404881,
    0.62516734,
    0.64964069,
  ]
)
z2 = np.array(
  [
    0.39257832,
    0.44821173,
    0.51174065,
    0.55973089,
    0.61830234,
    0.674804275,
    0.722225097,
    0.77327019,
    0.81336679,
    0.84543146,
  ]
)

x_old = np.arange(0, 1, 0.1)
x_new = np.arange(0, 0.91, 0.02)
curves = np.zeros((3, x_new.shape[0]))
curves[0] = make_interp_spline(x_old, c1, k=3)(x_new)
curves[1] = make_interp_spline(x_old, c2, k=3)(x_new)
curves[2] = make_interp_spline(x_old, c3, k=3)(x_new)


fig, ax = plt.subplots()
(l1,) = ax.plot(x_new, curves[0], c="steelblue")
ax.scatter(x_old, c1, c="steelblue")
(l2,) = ax.plot(x_new, curves[1], c="coral")
ax.scatter(x_old, c2, c="coral")
(l3,) = ax.plot(x_new, curves[2], c="orchid")
ax.scatter(x_old, c3, c="orchid")

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

ax.set_xlabel(r"$\rho$")
ax.set_ylabel("coverage")
# Put a legend to the right of the current axis
ax.legend(
  [l1, l2, l3],
  ["HCCT", "WLS", "WLS-MA"],
  loc="lower left",
  bbox_to_anchor=(0, 0.45),
  fontsize="8",
)
plt.show()

fig.savefig(
  f"./fig/run_401_coverage_{datetime.datetime.now().strftime('%m%d%H%M%S')}.pdf",
  format="pdf",
  bbox_inches="tight",
)


x_old = np.arange(0, 1, 0.1)
x_new = np.arange(0, 0.91, 0.02)
curves = np.zeros((3, x_new.shape[0]))
curves[0] = make_interp_spline(x_old, y1, k=3)(x_new)
curves[1] = make_interp_spline(x_old, x1, k=3)(x_new)
curves[2] = make_interp_spline(x_old, z1, k=3)(x_new)


fig, ax = plt.subplots()
(l1,) = ax.plot(x_new, curves[0], c="steelblue")
ax.scatter(x_old, y1, c="steelblue")
(l2,) = ax.plot(x_new, curves[1], c="coral")
ax.scatter(x_old, x1, c="coral")
(l3,) = ax.plot(x_new, curves[2], c="orchid")
ax.scatter(x_old, z1, c="orchid")

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

ax.set_xlabel(r"$\rho$")
ax.set_ylabel("widths of simultaneous interval")
# Put a legend to the right of the current axis
ax.legend([l1, l2, l3], ["HCCT", "WLS", "WLS-MA"], fontsize="8")
plt.show()

fig.savefig(
  f"./fig/run_401_theta_1_{datetime.datetime.now().strftime('%m%d%H%M%S')}.pdf",
  format="pdf",
  bbox_inches="tight",
)


x_old = np.arange(0, 1, 0.1)
x_new = np.arange(0, 0.91, 0.02)
curves = np.zeros((3, x_new.shape[0]))
curves[0] = make_interp_spline(x_old, y2, k=3)(x_new)
curves[1] = make_interp_spline(x_old, x2, k=3)(x_new)
curves[2] = make_interp_spline(x_old, z2, k=3)(x_new)


fig, ax = plt.subplots()
(l1,) = ax.plot(x_new, curves[0], c="steelblue")
ax.scatter(x_old, y2, c="steelblue")
(l2,) = ax.plot(x_new, curves[1], c="coral")
ax.scatter(x_old, x2, c="coral")
(l3,) = ax.plot(x_new, curves[2], c="orchid")
ax.scatter(x_old, z2, c="orchid")

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

ax.set_xlabel(r"$\rho$")
ax.set_ylabel("widths of simultaneous interval")
# Put a legend to the right of the current axis
ax.legend([l1, l2, l3], ["HCCT", "WLS", "WLS-MA"], fontsize="8")
plt.show()

fig.savefig(
  f"./fig/run_401_theta_2_{datetime.datetime.now().strftime('%m%d%H%M%S')}.pdf",
  format="pdf",
  bbox_inches="tight",
)
