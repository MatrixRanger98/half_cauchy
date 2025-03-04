import pathlib
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t, multivariate_normal, multivariate_t
from scipy.interpolate import make_interp_spline

from src.meta_analysis import combination_test


def cor_ar_1(dim, rho) -> np.ndarray:
  power_mat = np.abs(np.arange(dim).reshape(-1, 1) - np.arange(dim))
  return rho**power_mat


def cor_fixed(dim, rho):
  return rho * np.ones((dim, dim)) + (1 - rho) * np.eye(dim)


np.random.seed(2025)

"""
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
"""


# run_101_102: multivariate normal with AR(1) correlation, hcauchy only

repeat = 50
num_run = 10000
num_study = 500
d_rho = 0.1
rho = 0.4

coverage = np.zeros((repeat, int(1 / d_rho), 2))

for i, rho in enumerate(np.arange(0, 1, d_rho)):
  for k in range(repeat):
    cor = cor_ar_1(num_study, rho)
    point = np.zeros(num_study)
    theta_hat = multivariate_normal.rvs(mean=point, cov=cor, size=num_run)
    p_vector = 2 * np.maximum(norm.sf(np.abs(point - theta_hat)), 1e-150)
    run_101 = combination_test(num_study, "HCauchy", level=0.05)
    run_102 = combination_test(num_study, "HCauchy", level=0.01)
    count_101 = 0
    count_102 = 0
    for j in range(num_run):
      count_101 += run_101.make_decision(p_vector[j])
      count_102 += run_102.make_decision(p_vector[j])
    coverage[k, i, 0] = 1 - count_101 / num_run
    coverage[k, i, 1] = 1 - count_102 / num_run

# pathlib.Path('./tmp').mkdir(parents=True, exist_ok=True)
# np.save(f"./tmp/run_101_102_{datetime.datetime.now().strftime('%m%d%H%M%S')}",coverage)
# coverage = np.load('')

fig, ax = plt.subplots()

x_new = np.arange(0, 0.9, d_rho / 5)

band_up = np.quantile(coverage, 0.975, axis=0)
band_up_new = np.zeros((x_new.shape[0], 2))
band_up_new[:, 0] = make_interp_spline(np.arange(0, 1, d_rho), band_up[:, 0], k=3)(
  x_new
)
band_up_new[:, 1] = make_interp_spline(np.arange(0, 1, d_rho), band_up[:, 1], k=3)(
  x_new
)
band_down = np.quantile(coverage, 0.025, axis=0)
band_down_new = np.zeros((x_new.shape[0], 2))
band_down_new[:, 0] = make_interp_spline(np.arange(0, 1, d_rho), band_down[:, 0], k=3)(
  x_new
)
band_down_new[:, 1] = make_interp_spline(np.arange(0, 1, d_rho), band_down[:, 1], k=3)(
  x_new
)
mean_curve = coverage.mean(0)
mean_curve_new = np.zeros((x_new.shape[0], 2))
mean_curve_new[:, 0] = make_interp_spline(
  np.arange(0, 1, d_rho), mean_curve[:, 0], k=3
)(x_new)
mean_curve_new[:, 1] = make_interp_spline(
  np.arange(0, 1, d_rho), mean_curve[:, 1], k=3
)(x_new)

# Save output of 'fill_between' (note there's no comma here)
band_1 = ax.fill_between(
  x_new, band_up_new[:, 0], band_down_new[:, 0], color="skyblue", alpha=0.5
)
band_2 = ax.fill_between(
  x_new, band_up_new[:, 1], band_down_new[:, 1], color="bisque", alpha=0.5
)

# Save the output of 'plot', as we need it later
(lmean_1,) = ax.plot(x_new, mean_curve_new[:, 0], c="cornflowerblue")
(lmean_2,) = ax.plot(x_new, mean_curve_new[:, 1], c="darkgoldenrod")

# Create the legend, combining the yellow rectangle for the
# uncertainty and the 'mean line'  as a single item
# ax.legend([lwalker, (lsigma, lmean)], ["Walker position", "Mean + 1sigma range"], loc=2)
ax.set_xlabel(r"$\rho$")
ax.set_ylabel("coverage")
ax.legend([lmean_1, lmean_2], ["p=.05", "p=.01"], loc="lower right", fontsize="8")
fig.savefig(
  f"./fig/run_101_102_{datetime.datetime.now().strftime('%m%d%H%M%S')}.pdf",
  format="pdf",
)
plt.close(fig)


# run_103_104: multivariate normal with fixed correlation

repeat = 50
num_run = 10000
num_study = 500
d_rho = 0.1
rho = 0.4

coverage = np.zeros((repeat, int(1 / d_rho), 2))

for i, rho in enumerate(np.arange(0, 1, d_rho)):
  for k in range(repeat):
    cor = cor_fixed(num_study, rho)
    point = np.zeros(num_study)
    theta_hat = multivariate_normal.rvs(mean=point, cov=cor, size=num_run)
    p_vector = 2 * np.maximum(norm.sf(np.abs(point - theta_hat)), 1e-150)
    run_103 = combination_test(num_study, "HCauchy", level=0.05)
    run_104 = combination_test(num_study, "HCauchy", level=0.01)
    count_103 = 0
    count_104 = 0
    for j in range(num_run):
      count_103 += run_103.make_decision(p_vector[j])
      count_104 += run_104.make_decision(p_vector[j])
    coverage[k, i, 0] = 1 - count_103 / num_run
    coverage[k, i, 1] = 1 - count_104 / num_run

# pathlib.Path("./tmp").mkdir(parents=True, exist_ok=True)
# np.save(
  # f"./tmp/run_103_104_{datetime.datetime.now().strftime('%m%d%H%M%S')}", coverage
# )
# coverage = np.load('')


fig, ax = plt.subplots()

x_new = np.arange(0, 0.9, d_rho / 5)

band_up = np.quantile(coverage, 0.975, axis=0)
band_up_new = np.zeros((x_new.shape[0], 2))
band_up_new[:, 0] = make_interp_spline(np.arange(0, 1, d_rho), band_up[:, 0], k=3)(
  x_new
)
band_up_new[:, 1] = make_interp_spline(np.arange(0, 1, d_rho), band_up[:, 1], k=3)(
  x_new
)
band_down = np.quantile(coverage, 0.025, axis=0)
band_down_new = np.zeros((x_new.shape[0], 2))
band_down_new[:, 0] = make_interp_spline(np.arange(0, 1, d_rho), band_down[:, 0], k=3)(
  x_new
)
band_down_new[:, 1] = make_interp_spline(np.arange(0, 1, d_rho), band_down[:, 1], k=3)(
  x_new
)
mean_curve = coverage.mean(0)
mean_curve_new = np.zeros((x_new.shape[0], 2))
mean_curve_new[:, 0] = make_interp_spline(
  np.arange(0, 1, d_rho), mean_curve[:, 0], k=3
)(x_new)
mean_curve_new[:, 1] = make_interp_spline(
  np.arange(0, 1, d_rho), mean_curve[:, 1], k=3
)(x_new)

# Save output of 'fill_between' (note there's no comma here)
band_1 = ax.fill_between(
  x_new, band_up_new[:, 0], band_down_new[:, 0], color="skyblue", alpha=0.5
)
band_2 = ax.fill_between(
  x_new, band_up_new[:, 1], band_down_new[:, 1], color="bisque", alpha=0.5
)

# Save the output of 'plot', as we need it later
(lmean_1,) = ax.plot(x_new, mean_curve_new[:, 0], c="cornflowerblue")
(lmean_2,) = ax.plot(x_new, mean_curve_new[:, 1], c="darkgoldenrod")

# Create the legend, combining the yellow rectangle for the
# uncertainty and the 'mean line'  as a single item
# ax.legend([lwalker, (lsigma, lmean)], ["Walker position", "Mean + 1sigma range"], loc=2)
ax.set_xlabel(r"$\rho$")
ax.set_ylabel("coverage")
ax.legend([lmean_1, lmean_2], ["p=.05", "p=.01"], loc="lower right", fontsize="8")
fig.savefig(
  f"./fig/run_103_104_{datetime.datetime.now().strftime('%m%d%H%M%S')}.pdf",
  format="pdf",
)
plt.close(fig)