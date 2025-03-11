import datetime
import numpy as np
from scipy.stats import norm, cauchy
import matplotlib.pyplot as plt

from src.distribution import euler_gamma, hcauchy_mean, landau


func_landau = lambda x: landau.pdf(x, 2 / np.pi * (1 - euler_gamma), 1)  # noqa: E731
func_10 = np.vectorize(lambda x: hcauchy_mean.pdf(x, 10))
func_100 = np.vectorize(lambda x: hcauchy_mean.pdf(x, 100))
func_1000 = np.vectorize(lambda x: hcauchy_mean.pdf(x, 1000))


fig_1, ax_1 = plt.subplots()
ax_1.plot(
  np.arange(-2.5, 0, 0.01),
  np.zeros_like(np.arange(-2.5, 0, 0.01)),
  color="b",
  label="Half Cauchy",
)
ax_1.plot(np.arange(0, 8, 0.01), 2 * cauchy.pdf(np.arange(0, 8, 0.01)), color="b")
ax_1.axvline(0, linewidth=1, ls="--")
ax_1.plot(
  np.arange(-2.5, 8, 0.01),
  func_landau(np.arange(-2.5, 8, 0.01)),
  color="r",
  label=r"Landau$(\frac{2}{\pi}(1-\gamma),1)$",
)
ax_1.legend()
fig_1.savefig(
  f"./fig/landau_1_{datetime.datetime.now().strftime('%m%d%H%M%S')}.pdf", format="pdf"
)


fig_10, ax_10 = plt.subplots()
ax_10.plot(
  np.arange(-1, 0, 0.01),
  np.zeros_like(np.arange(-1, 0, 0.01)),
  color="b",
  label=r"Half Cauchy mean, $m=10$",
)
ax_10.plot(np.arange(0, 9, 0.01), func_10(np.arange(0, 9, 0.01)), color="b")
ax_10.plot(
  np.arange(-1, 9, 0.01),
  func_landau(np.arange(-1, 9, 0.01) - 2 / np.pi * (np.log(10))),
  color="r",
  label=r"Landau$\left(\frac{2}{\pi}(\log 10+1-\gamma),1\right)$",
)
ax_10.legend()
fig_10.savefig(
  f"./fig/landau_10_{datetime.datetime.now().strftime('%m%d%H%M%S')}.pdf", format="pdf"
)


fig_100, ax_100 = plt.subplots()
ax_100.plot(
  np.arange(0, 1, 0.01),
  np.zeros_like(np.arange(0, 1, 0.01)),
  color="b",
  label=r"Half Cauchy mean, $m=100$",
)
ax_100.plot(np.arange(1, 10, 0.01), func_100(np.arange(1, 10, 0.01)), color="b")
ax_100.plot(
  np.arange(0, 10, 0.01),
  func_landau(np.arange(0, 10, 0.01) - 2 / np.pi * (np.log(100))),
  color="r",
  label=r"Landau$\left(\frac{2}{\pi}(\log 100+1-\gamma),1\right)$",
)
ax_100.legend()
fig_100.savefig(
  f"./fig/landau_100_{datetime.datetime.now().strftime('%m%d%H%M%S')}.pdf", format="pdf"
)


fig_1000, ax_1000 = plt.subplots()
ax_1000.plot(
  np.arange(1, 2, 0.01),
  np.zeros_like(np.arange(1, 2, 0.01)),
  color="b",
  label=r"Half Cauchy mean, $m=1000$",
)
ax_1000.plot(np.arange(2, 11, 0.01), func_1000(np.arange(2, 11, 0.01)), color="b")
ax_1000.plot(
  np.arange(1, 11, 0.01),
  func_landau(np.arange(1, 11, 0.01) - 2 / np.pi * (np.log(1000))),
  color="r",
  label=r"Landau$\left(\frac{2}{\pi}(\log 1000+1-\gamma),1\right)$",
)
ax_1000.legend()
fig_1000.savefig(
  f"./fig/landau_1000_{datetime.datetime.now().strftime('%m%d%H%M%S')}.pdf", format="pdf"
)
