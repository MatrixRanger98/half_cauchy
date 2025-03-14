import os
import warnings
import contextlib
import numpy as np
from scipy.special import sici, expi
from scipy.integrate import quad
from scipy.optimize import root_scalar
from collections.abc import Sequence

euler_gamma = 0.5772156649015329

def expi_over_exp(x: float, n: int=150):
  init = np.exp(-x) * (euler_gamma + np.log(x))
  coef = (
    (np.tri(n // 2 + 1, n // 2 + 1, 0) @ (1 / (2 * np.arange(n // 2 + 1) + 1))).repeat(
      2
    )[0:n]
    * 2
    * (np.arange(1, n + 1) % 2 * 2 - 1)
  )
  expo = (
    np.arange(1, n + 1) * np.expand_dims(np.log(x / 2), axis=-1)
    - np.expand_dims(x, axis=-1) / 2
    - np.tri(n, n, 0) @ np.log(np.arange(1, n + 1))
  )
  return init + np.inner(coef, np.exp(expo))


class hcauchy_mean(object):
  def __init__(self) -> None:
    pass

  @staticmethod
  def __pre_cdf(z: float | np.float_, x: float | np.float_, w: np.ndarray) -> float | np.float_:
    w = w[w != 0]
    loc = (-sum(w * np.log(w)) + 1 - euler_gamma) * 2 / np.pi
    if x > loc + 4:
      n = (loc + 4) * 10 / x
    else:
      n = 16
    f_star = (
      -2
      / np.pi
      * (
        np.cos(n * z * w) * (sici(n * z * w)[0] - np.pi / 2)
        - np.sin(n * z * w) * sici(n * z * w)[1]
      )
    )
    f_comp = -f_star + 2 * np.cos(n * z * w) + 2j * np.sin(n * z * w)
    f_sum = sum(np.log(f_comp))
    f_int = 2 * np.exp(f_sum - x * n * z).imag / (2 * np.pi * z)
    return f_int

  @staticmethod
  def __pre_pdf(z: float | np.float_, x: float | np.float_, w: np.ndarray) -> float | np.float_:
    w = w[w != 0]
    loc = (-sum(w * np.log(w)) + 1 - euler_gamma) * 2 / np.pi
    if x > loc + 4:
      n = (loc + 4) * 10 / x
    else:
      n = 16
    f_star = (
      -2
      / np.pi
      * (
        np.cos(n * z * w) * (sici(n * z * w)[0] - np.pi / 2)
        - np.sin(n * z * w) * sici(n * z * w)[1]
      )
    )
    f_comp = -f_star + 2 * np.cos(n * z * w) + 2j * np.sin(n * z * w)
    f_sum = sum(np.log(f_comp))
    f_int = 2 * np.exp(f_sum - x * n * z).imag * n / (2 * np.pi)
    return f_int

  @classmethod
  def sf(cls, x: float | np.float_, w:int | np.int_ | float | np.float_ | Sequence[float | np.float_] | np.ndarray, precision: bool=False, check_w: bool=True) -> float | np.float_ | tuple:
    if check_w:
      if isinstance(w, int | np.int_ | float | np.float_):
        weight = np.array([1 / w] * int(np.int_(w)))
      else:
        weight = np.array(w)
    else:
      weight = w
    location = (-sum(weight * np.log(weight)) + 1 - euler_gamma) * 2 / np.pi
    if x <= location - 2.5:
      if precision:
        return 1.0, 1e-8
      else:
        return 1.0
    with warnings.catch_warnings():
      warnings.simplefilter("error")
      try:
        res = quad(cls.__pre_cdf, 0, np.inf, args=(x, weight))
        if precision:
          return res
        else:
          return res[0]
      except Exception as error:
        print(error)
        if precision:
          return 1.0, 1e-8
        else:
          return 1.0

  @classmethod
  def cdf(cls, x: float | np.float_, w: int | np.int_ | float | np.float_ | Sequence[float | np.float_] | np.ndarray, precision: bool=False, check_w: bool= True) -> float | np.float_ | tuple:
    if check_w:
      if isinstance(w, int | np.int_ | float | np.float_):
        weight = np.array([1 / w] * int(np.int_(w)))
      else:
        weight = np.array(w)
    else:
      weight = w
    comp = cls.sf(x, weight, precision, check_w=False)
    if precision:
      return 1 - comp[0], comp[1]
    else:
      return 1 - comp

  @classmethod
  def pdf(cls, x: float | np.float_, w:int | np.int_ | float | np.float_ | Sequence[float | np.float_] | np.ndarray, precision: bool=False, check_w: bool=True) -> float | np.float_ | tuple:
    if check_w:
      if isinstance(w, int | np.int_ | float | np.float_):
        weight = np.array([1 / w] * int(np.int_(w)))
      else:
        weight = np.array(w)
    else:
      weight=w
    location = (-sum(weight * np.log(weight)) + 1 - euler_gamma) * 2 / np.pi
    if x <= location - 2.5:
      if precision:
        return 0.0, 1e-8
      else:
        return 0.0
    with warnings.catch_warnings():
      warnings.simplefilter("error")
      try:
        res = quad(cls.__pre_pdf, 0, np.inf, args=(x, weight))
        if precision:
          return res
        else:
          return res[0]
      except Exception as error:
        print(error)
        if precision:
          return 0.0, 1e-8
        else:
          return 0.0

  @classmethod
  def ppf(cls, alpha: float | np.float_, w:int | np.int_ | float | np.float_ | Sequence[float | np.float_] | np.ndarray, check_w: bool=True) -> float | np.float_:
    if check_w:
      if isinstance(w, int | np.int_ | float | np.float_):
        weight = np.array([1 / w] * int(np.int_(w)))
      else:
        weight = np.array(w)
    else:
      weight=w
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
      low = 0
      high = 100
      while True:
        if cls.cdf(high, weight, check_w=False) < alpha:
          low = high
          high = 2 * high
        else:
          break
      return root_scalar(
        lambda x: cls.cdf(x, weight, check_w=False) - alpha, bracket=[low, high], method="brentq"
      ).root


class harmonic_mean(object):
  def __init__(self) -> None:
    pass

  @staticmethod
  def __pre_cdf(z: float | np.float_, x: float | np.float_, w: np.ndarray) -> float | np.float_:
    w = w[w != 0]
    loc = (-sum(w * np.log(w)) + 1 - euler_gamma) * 2 / np.pi
    if x > loc + 4:
      n = (loc + 4) * 10 / x
    else:
      n = 16
    ei_2_over_exp = (n * z * w) * expi_over_exp(n * z * w) - 1
    f_comp = -ei_2_over_exp + 1j * np.pi * np.exp(np.log(n * z * w) - n * z * w)
    f_sum = sum(np.log(f_comp) + n * z * w)
    f_int = 2 * np.exp(f_sum - x * n * z).imag / (2 * np.pi * z)
    return f_int

  @staticmethod
  def __pre_pdf(z: float | np.float_, x: float | np.float_, w: np.ndarray) -> float | np.float_:
    w = w[w != 0]
    loc = (-sum(w * np.log(w)) + 1 - euler_gamma) * 2 / np.pi
    if x > loc + 4:
      n = (loc + 4) * 10 / x
    else:
      n = 16
    ei_2_over_exp = (n * z * w) * expi_over_exp(n * z * w) - 1
    f_comp = -ei_2_over_exp + 1j * np.pi * np.exp(np.log(n * z * w) - n * z * w)
    f_sum = sum(np.log(f_comp) + n * z * w)
    f_int = 2 * np.exp(f_sum - x * n * z).imag * n / (2 * np.pi)
    return f_int

  @classmethod
  def sf(cls, x: float | np.float_, w: int | np.int_ | float | np.float_ | Sequence[float | np.float_] | np.ndarray, precision: bool=False, check_w: bool=True) -> float | np.float_ | tuple:
    if check_w:
      if isinstance(w, int | np.int_ | float | np.float_):
        weight = np.array([1 / w] * int(np.int_(w)))
      else:
        weight = np.array(w)
    else:
      weight = w
    if x <= 1:
      if precision:
        return 1.0, 1e-8
      else:
        return 1.0
    with warnings.catch_warnings():
      warnings.simplefilter("error")
      try:
        res = quad(cls.__pre_cdf, 1e-150, np.inf, args=(x, weight))
        if precision:
          return res
        else:
          return res[0]
      except Exception as error:
        print(error)
        if precision:
          return 1.0, 1e-8
        else:
          return 1.0

  @classmethod
  def cdf(cls, x: float | np.float_, w: int | np.int_ | float | np.float_ | Sequence[float | np.float_] | np.ndarray, precision: bool=False, check_w: bool=True) -> float | np.float_ | tuple:
    if check_w:
      if isinstance(w, int | np.int_ | float | np.float_):
        weight = np.array([1 / w] * int(np.int_(w)))
      else:
        weight = np.array(w)
    else:
      weight = w
    comp = cls.sf(x, weight, precision, check_w=False)
    if precision:
      return 1 - comp[0], comp[1]
    else:
      return 1 - comp

  @classmethod
  def pdf(cls, x: float | np.float_, w: int | np.int_ | float | np.float_ | Sequence[float | np.float_] | np.ndarray, precision: bool=False, check_w: bool=True) -> float | np.float_ | tuple:
    if check_w:
      if isinstance(w, int | np.int_ | float | np.float_):
        weight = np.array([1 / w] * int(np.int_(w)))
      else:
        weight = np.array(w)
    else:
      weight = w
    if x <= 1:
      if precision:
        return 0.0, 1e-8
      else:
        return 0.0
    with warnings.catch_warnings():
      warnings.simplefilter("error")
      try:
        res = quad(cls.__pre_pdf, 1e-150, np.inf, args=(x, weight))
        if precision:
          return res
        else:
          return res[0]
      except Exception as error:
        print(error)
        if precision:
          return 0.0, 1e-8
        else:
          return 0.0

  @classmethod
  def ppf(cls, alpha: float | np.float_, w: int | np.int_ | float | np.float_ | Sequence[float | np.float_] | np.ndarray, check_w: bool=True):
    if check_w:
      if isinstance(w, int | np.int_ | float | np.float_):
        weight = np.array([1 / w] * int(np.int_(w)))
      else:
        weight = np.array(w)
    else:
      weight = w
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
      low = 1
      high = 100
      while True:
        if cls.cdf(high, weight, check_w=False) < alpha:
          low = high
          high = 2 * high
        else:
          break
      return root_scalar(
        lambda x: cls.cdf(x, weight, check_w=False) - alpha, bracket=[low, high], method="brentq"
      ).root


class landau(object):
  def __init__(self) -> None:
    pass

  @classmethod
  def pdf(cls, x: float | np.float_ | np.ndarray, loc: float | np.float_ | np.ndarray=0.0, scale: float | np.float_ | np.ndarray=1.0) -> float | np.float_ | np.ndarray:
    return (
      cls.__base_pdf(
        (x - loc) * np.pi / (2 * scale)
        + np.log(np.pi / (2 * scale))
      )
      * np.pi
      / (2 * scale)
    )

  @classmethod
  def cdf(cls, x: float | np.float_ | np.ndarray, loc: float | np.float_ | np.ndarray=0.0, scale: float | np.float_ | np.ndarray=1.0) -> float | np.float_ | np.ndarray:
    return cls.__base_cdf(
      (x - loc) * np.pi / (2 * scale) + np.log(np.pi / (2 * scale))
    )

  @classmethod
  def sf(cls, x: float | np.float_ | np.ndarray, loc: float | np.float_ | np.ndarray=0.0, scale: float | np.float_ | np.ndarray=1.0) -> float | np.float_ | np.ndarray:
    return 1 - cls.cdf(x, loc, scale)

  @classmethod
  def ppf(cls, x: float | np.float_ | np.ndarray, loc: float | np.float_ | np.ndarray=0.0, scale: float | np.float_ | np.ndarray=1.0) -> float | np.float_ | np.ndarray:
    return (
      cls.__base_ppf(x) * 2 * scale / np.pi
      + loc
      + 2 * scale / np.pi * np.log(2 * scale / np.pi)
    )
  
  # def convert_x_mpv_to_x0(x_mpv, xi):
  #   return (
  #     x_mpv + 0.22278298 * xi
  #   )  # This number I took from Root's langauss implementation: https://root.cern.ch/doc/master/langaus_8C.html and basically it gives the correct MPV value.


  # def convert_x0_to_x_mpv(x0, xi):
  #   return x0 - 0.22278298 * xi

  @staticmethod
  def __base_pdf(x: float | np.float_ | np.ndarray) -> float | np.float_ | np.ndarray:
    """Calculates the "basic" Landau distribution, i.e. the distribution when the location parameter is 0 and the scale parameter is pi/2. The algorithm was adapted from [the Root implementation](https://root.cern.ch/doc/master/PdfFuncMathCore_8cxx_source.html).

    Parameters
    ----------
    x: float, numpy array
            Point in which to calculate the function.

    Returns
    -------
    landau_pdf: float, numpy array
            The value of the Landau distribution.

    Error handling
    --------------
    Rises `TypeError` if the parameters are not within the accepted types.
    """

    def denlan_1(x):
      """Calculates denlan when x < -5.5. If x is outside this range, NaN value is returned."""
      a1 = (0.04166666667, -0.01996527778, 0.02709538966)
      u = np.exp(x + 1)
      denlan = (
        0.3989422803
        * (np.exp(-1 / u) / u**0.5)
        * (1 + (a1[0] + (a1[1] + a1[2] * u) * u) * u)
      )
      denlan[u < 1e-10] = 0
      denlan[x >= -5.5] = float("NaN")
      return denlan

    def denlan_2(x):
      """Calculates denlan when -5.5 <= x < -1. If x is outside this range, NaN value is returned."""
      p1 = (
        0.4259894875,
        -0.1249762550,
        0.03984243700,
        -0.006298287635,
        0.001511162253,
      )
      q1 = (1.0, -0.3388260629, 0.09594393323, -0.01608042283, 0.003778942063)
      u = np.exp(-x - 1)
      denlan = (
        np.exp(-u)
        * np.sqrt(u)
        * (p1[0] + (p1[1] + (p1[2] + (p1[3] + p1[4] * x) * x) * x) * x)
        / (q1[0] + (q1[1] + (q1[2] + (q1[3] + q1[4] * x) * x) * x) * x)
      )
      denlan[(x < -5.5) | (x >= -1)] = float("NaN")
      return denlan

    def denlan_3(x):
      """Calculates denlan when -1 <= x < 1. If x is outside this range, NaN value is returned."""
      p2 = (
        0.1788541609,
        0.1173957403,
        0.01488850518,
        -0.001394989411,
        0.0001283617211,
      )
      q2 = (1.0, 0.7428795082, 0.3153932961, 0.06694219548, 0.008790609714)
      denlan = (p2[0] + (p2[1] + (p2[2] + (p2[3] + p2[4] * x) * x) * x) * x) / (
        q2[0] + (q2[1] + (q2[2] + (q2[3] + q2[4] * x) * x) * x) * x
      )
      denlan[(x < -1) | (x >= 1)] = float("NaN")
      return denlan

    def denlan_4(x):
      """Calculates denlan when 1 <= x < 5. If x is outside this range, NaN value is returned."""
      p3 = (
        0.1788544503,
        0.09359161662,
        0.006325387654,
        0.00006611667319,
        -0.000002031049101,
      )
      q3 = (1.0, 0.6097809921, 0.2560616665, 0.04746722384, 0.006957301675)
      denlan = (p3[0] + (p3[1] + (p3[2] + (p3[3] + p3[4] * x) * x) * x) * x) / (
        q3[0] + (q3[1] + (q3[2] + (q3[3] + q3[4] * x) * x) * x) * x
      )
      denlan[(x < 1) | (x >= 5)] = float("NaN")
      return denlan

    def denlan_5(x):
      """Calculates denlan when 5 <= x < 12. If x is outside this range, NaN value is returned."""
      p4 = (0.9874054407, 118.6723273, 849.2794360, -743.7792444, 427.0262186)
      q4 = (1.0, 106.8615961, 337.6496214, 2016.712389, 1597.063511)
      u = 1 / x
      denlan = (
        u
        * u
        * (p4[0] + (p4[1] + (p4[2] + (p4[3] + p4[4] * u) * u) * u) * u)
        / (q4[0] + (q4[1] + (q4[2] + (q4[3] + q4[4] * u) * u) * u) * u)
      )
      denlan[(x < 5) | (x >= 12)] = float("NaN")
      return denlan

    def denlan_6(x):
      """Calculates denlan when 12 <= x < 50. If x is outside this range, NaN value is returned."""
      p5 = (1.003675074, 167.5702434, 4789.711289, 21217.86767, -22324.94910)
      q5 = (1.0, 156.9424537, 3745.310488, 9834.698876, 66924.28357)
      u = 1 / x
      denlan = (
        u
        * u
        * (p5[0] + (p5[1] + (p5[2] + (p5[3] + p5[4] * u) * u) * u) * u)
        / (q5[0] + (q5[1] + (q5[2] + (q5[3] + q5[4] * u) * u) * u) * u)
      )
      denlan[(x < 12) | (x >= 50)] = float("NaN")
      return denlan

    def denlan_7(x):
      """Calculates denlan when 50 <= x < 300. If x is outside this range, NaN value is returned."""
      p6 = (1.000827619, 664.9143136, 62972.92665, 475554.6998, -5743609.109)
      q6 = (1.0, 651.4101098, 56974.73333, 165917.4725, -2815759.939)
      u = 1 / x
      denlan = (
        u
        * u
        * (p6[0] + (p6[1] + (p6[2] + (p6[3] + p6[4] * u) * u) * u) * u)
        / (q6[0] + (q6[1] + (q6[2] + (q6[3] + q6[4] * u) * u) * u) * u)
      )
      denlan[(x < 50) | (x >= 300)] = float("NaN")
      return denlan

    def denlan_8(x):
      """Calculates denlan when x >= 300. If x is outside this range, NaN value is returned."""
      a2 = (-1.845568670, -4.284640743)
      u = 1 / (x - x * np.log(x) / (x + 1))
      denlan = u * u * (1 + (a2[0] + a2[1] * u) * u)
      denlan[x <= 300] = float("NaN")
      return denlan

    if not isinstance(x, (int, float, np.ndarray)):
      raise TypeError(
        f'Require float or numpy.ndarray, but received object of type {type(x)}.'
      )
    x_was_just_a_number = isinstance(x, (float, int))
    x = np.atleast_1d(x).astype(float)

    result = x * float("NaN")  # Initialize
    x_is_finite_indices = np.isfinite(x)
    with warnings.catch_warnings():
      warnings.simplefilter(
        "ignore"
      )  # I don't want to see the warnings of numpy, anyway it will fill with `NaN` values so it is fine.
      denlan = x[x_is_finite_indices] * float("NaN")  # Initialize.
      limits = (-float("inf"), -5.5, -1, 1, 5, 12, 50, 300, float("inf"))
      formulas = (
        denlan_1,
        denlan_2,
        denlan_3,
        denlan_4,
        denlan_5,
        denlan_6,
        denlan_7,
        denlan_8,
      )
      for k, formula in enumerate(formulas):
        indices = (limits[k] <= x[x_is_finite_indices]) & (
          x[x_is_finite_indices] < limits[k + 1]
        )
        denlan[indices] = formula(x[x_is_finite_indices][indices])
      result[x_is_finite_indices] = denlan
    result[np.isinf(x)] = 0

    result = np.squeeze(result)
    if x_was_just_a_number:
      result = float(result)
    return result

  @staticmethod
  def __base_cdf(x: float | np.float_ | np.ndarray) -> float | np.float_ | np.ndarray:
    """Calculates the CDF of the "basic" Landau distribution, i.e. the distribution when the location parameter is 0 and the scale parameter is pi/2. The algorithm was adapted from [the Root implementation](https://root.cern.ch/doc/master/PdfFuncMathCore_8cxx_source.html).

    Parameters
    ----------
    x: float, numpy array
            Point in which to calculate the function.

    Returns
    -------
    landau_cdf: float, numpy array
            The value of the Landau distribution.

    Error handling
    --------------
    Rises `TypeError` if the parameters are not within the accepted types.
    """

    def denlan_1(x):
      """Calculates denlan when x < -5.5. If x is outside this range, NaN value is returned."""
      a1 = (0, -0.4583333333, 0.6675347222, -1.641741416)
      u = np.exp(x + 1)
      denlan = (
        0.3989422803
        * np.exp(-1 / u)
        * u**0.5
        * (1 + (a1[0] + (a1[1] + (a1[2] + a1[3] * u) * u) * u) * u)
      )
      denlan[u < 1e-10] = 0
      denlan[x >= -5.5] = float("NaN")
      return denlan

    def denlan_2(x):
      """Calculates denlan when -5.5 <= x < -1. If x is outside this range, NaN value is returned."""
      p1 = (
        0.2514091491,
        -0.06250580444,
        0.0145838123,
        -0.002108817737,
        0.000741124729,
      )
      q1 = (1.0, -0.005571175625, 0.06225310236, -0.003137378427, 0.001931496439)
      u = np.exp(-x - 1)
      denlan = (
        np.exp(-u)
        / np.sqrt(u)
        * (p1[0] + (p1[1] + (p1[2] + (p1[3] + p1[4] * x) * x) * x) * x)
        / (q1[0] + (q1[1] + (q1[2] + (q1[3] + q1[4] * x) * x) * x) * x)
      )
      denlan[(x < -5.5) | (x >= -1)] = float("NaN")
      return denlan

    def denlan_3(x):
      """Calculates denlan when -1 <= x < 1. If x is outside this range, NaN value is returned."""
      p2 = (0.2868328584, 0.3564363231, 0.1523518695, 0.02251304883)
      q2 = (1.0, 0.6191136137, 0.1720721448, 0.02278594771)
      denlan = (p2[0] + (p2[1] + (p2[2] + p2[3] * x) * x) * x) / (
        q2[0] + (q2[1] + (q2[2] + q2[3] * x) * x) * x
      )
      denlan[(x < -1) | (x >= 1)] = float("NaN")
      return denlan

    def denlan_4(x):
      """Calculates denlan when 1 <= x < 4. If x is outside this range, NaN value is returned."""
      p3 = (0.2868329066, 0.3003828436, 0.09950951941, 0.008733827185)
      q3 = (1.0, 0.4237190502, 0.1095631512, 0.008693851567)
      denlan = (p3[0] + (p3[1] + (p3[2] + p3[3] * x) * x) * x) / (
        q3[0] + (q3[1] + (q3[2] + q3[3] * x) * x) * x
      )
      denlan[(x < 1) | (x >= 4)] = float("NaN")
      return denlan

    def denlan_5(x):
      """Calculates denlan when 4 <= x < 12. If x is outside this range, NaN value is returned."""
      p4 = (1.00035163, 4.503592498, 10.8588388, 7.536052269)
      q4 = (1.0, 5.539969678, 19.33581111, 27.21321508)
      u = 1 / x
      denlan = (p4[0] + (p4[1] + (p4[2] + p4[3] * u) * u) * u) / (
        q4[0] + (q4[1] + (q4[2] + q4[3] * u) * u) * u
      )
      denlan[(x < 4) | (x >= 12)] = float("NaN")
      return denlan

    def denlan_6(x):
      """Calculates denlan when 12 <= x < 50. If x is outside this range, NaN value is returned."""
      p5 = (1.000006517, 49.09414111, 85.05544753, 153.2153455)
      q5 = (1.0, 50.09928881, 139.9819104, 420.0002909)
      u = 1 / x
      denlan = (p5[0] + (p5[1] + (p5[2] + p5[3] * u) * u) * u) / (
        q5[0] + (q5[1] + (q5[2] + q5[3] * u) * u) * u
      )
      denlan[(x < 12) | (x >= 50)] = float("NaN")
      return denlan

    def denlan_7(x):
      """Calculates denlan when 50 <= x < 300. If x is outside this range, NaN value is returned."""
      p6 = (1.000000983, 132.9868456, 916.2149244, -960.5054274)
      q6 = (1.0, 133.9887843, 1055.990413, 553.2224619)
      u = 1 / x
      denlan = (p6[0] + (p6[1] + (p6[2] + p6[3] * u) * u) * u) / (
        q6[0] + (q6[1] + (q6[2] + q6[3] * u) * u) * u
      )
      denlan[(x < 50) | (x >= 300)] = float("NaN")
      return denlan

    def denlan_8(x):
      """Calculates denlan when x >= 300. If x is outside this range, NaN value is returned."""
      a2 = (0, 1.0, -0.4227843351, -2.043403138)
      u = 1 / (x - x * np.log(x) / (x + 1))
      denlan = 1 - (a2[1] + (a2[2] + a2[3] * u) * u) * u
      denlan[x <= 300] = float("NaN")
      return denlan

    if not isinstance(x, (int, float, np.ndarray)):
      raise TypeError(
        f'Require float or numpy.ndarray, but received object of type {type(x)}.'
      )
    x_was_just_a_number = isinstance(x, (float, int))
    x = np.atleast_1d(x).astype(float)

    result = x * float("NaN")  # Initialize
    x_is_finite_indices = np.isfinite(x)
    with warnings.catch_warnings():
      warnings.simplefilter(
        "ignore"
      )  # I don't want to see the warnings of numpy, anyway it will fill with `NaN` values so it is fine.
      denlan = x[x_is_finite_indices] * float("NaN")  # Initialize.
      limits = (-float("inf"), -5.5, -1, 1, 4, 12, 50, 300, float("inf"))
      formulas = (
        denlan_1,
        denlan_2,
        denlan_3,
        denlan_4,
        denlan_5,
        denlan_6,
        denlan_7,
        denlan_8,
      )
      for k, formula in enumerate(formulas):
        indices = (limits[k] <= x[x_is_finite_indices]) & (
          x[x_is_finite_indices] < limits[k + 1]
        )
        denlan[indices] = formula(x[x_is_finite_indices][indices])
      result[x_is_finite_indices] = denlan
    result[np.isinf(x) & (x < 0)] = 0
    result[np.isinf(x) & (x > 0)] = 1

    result = np.squeeze(result)
    if x_was_just_a_number:
      result = float(result)
    return result

  @staticmethod
  def __base_ppf(x: float | np.float_ | np.ndarray) -> float | np.float_ | np.ndarray:
    f = np.array([
      0,
      0,
      0,
      0,
      0,
      -2.244733,
      -2.204365,
      -2.168163,
      -2.135219,
      -2.104898,
      -2.076740,
      -2.050397,
      -2.025605,
      -2.002150,
      -1.979866,
      -1.958612,
      -1.938275,
      -1.918760,
      -1.899984,
      -1.881879,
      -1.864385,
      -1.847451,
      -1.831030,
      -1.815083,
      -1.799574,
      -1.784473,
      -1.769751,
      -1.755383,
      -1.741346,
      -1.727620,
      -1.714187,
      -1.701029,
      -1.688130,
      -1.675477,
      -1.663057,
      -1.650858,
      -1.638868,
      -1.627078,
      -1.615477,
      -1.604058,
      -1.592811,
      -1.581729,
      -1.570806,
      -1.560034,
      -1.549407,
      -1.538919,
      -1.528565,
      -1.518339,
      -1.508237,
      -1.498254,
      -1.488386,
      -1.478628,
      -1.468976,
      -1.459428,
      -1.449979,
      -1.440626,
      -1.431365,
      -1.422195,
      -1.413111,
      -1.404112,
      -1.395194,
      -1.386356,
      -1.377594,
      -1.368906,
      -1.360291,
      -1.351746,
      -1.343269,
      -1.334859,
      -1.326512,
      -1.318229,
      -1.310006,
      -1.301843,
      -1.293737,
      -1.285688,
      -1.277693,
      -1.269752,
      -1.261863,
      -1.254024,
      -1.246235,
      -1.238494,
      -1.230800,
      -1.223153,
      -1.215550,
      -1.207990,
      -1.200474,
      -1.192999,
      -1.185566,
      -1.178172,
      -1.170817,
      -1.163500,
      -1.156220,
      -1.148977,
      -1.141770,
      -1.134598,
      -1.127459,
      -1.120354,
      -1.113282,
      -1.106242,
      -1.099233,
      -1.092255,
      -1.085306,
      -1.078388,
      -1.071498,
      -1.064636,
      -1.057802,
      -1.050996,
      -1.044215,
      -1.037461,
      -1.030733,
      -1.024029,
      -1.017350,
      -1.010695,
      -1.004064,
      -0.997456,
      -0.990871,
      -0.984308,
      -0.977767,
      -0.971247,
      -0.964749,
      -0.958271,
      -0.951813,
      -0.945375,
      -0.938957,
      -0.932558,
      -0.926178,
      -0.919816,
      -0.913472,
      -0.907146,
      -0.900838,
      -0.894547,
      -0.888272,
      -0.882014,
      -0.875773,
      -0.869547,
      -0.863337,
      -0.857142,
      -0.850963,
      -0.844798,
      -0.838648,
      -0.832512,
      -0.826390,
      -0.820282,
      -0.814187,
      -0.808106,
      -0.802038,
      -0.795982,
      -0.789940,
      -0.783909,
      -0.777891,
      -0.771884,
      -0.765889,
      -0.759906,
      -0.753934,
      -0.747973,
      -0.742023,
      -0.736084,
      -0.730155,
      -0.724237,
      -0.718328,
      -0.712429,
      -0.706541,
      -0.700661,
      -0.694791,
      -0.688931,
      -0.683079,
      -0.677236,
      -0.671402,
      -0.665576,
      -0.659759,
      -0.653950,
      -0.648149,
      -0.642356,
      -0.636570,
      -0.630793,
      -0.625022,
      -0.619259,
      -0.613503,
      -0.607754,
      -0.602012,
      -0.596276,
      -0.590548,
      -0.584825,
      -0.579109,
      -0.573399,
      -0.567695,
      -0.561997,
      -0.556305,
      -0.550618,
      -0.544937,
      -0.539262,
      -0.533592,
      -0.527926,
      -0.522266,
      -0.516611,
      -0.510961,
      -0.505315,
      -0.499674,
      -0.494037,
      -0.488405,
      -0.482777,
      -0.477153,
      -0.471533,
      -0.465917,
      -0.460305,
      -0.454697,
      -0.449092,
      -0.443491,
      -0.437893,
      -0.432299,
      -0.426707,
      -0.421119,
      -0.415534,
      -0.409951,
      -0.404372,
      -0.398795,
      -0.393221,
      -0.387649,
      -0.382080,
      -0.376513,
      -0.370949,
      -0.365387,
      -0.359826,
      -0.354268,
      -0.348712,
      -0.343157,
      -0.337604,
      -0.332053,
      -0.326503,
      -0.320955,
      -0.315408,
      -0.309863,
      -0.304318,
      -0.298775,
      -0.293233,
      -0.287692,
      -0.282152,
      -0.276613,
      -0.271074,
      -0.265536,
      -0.259999,
      -0.254462,
      -0.248926,
      -0.243389,
      -0.237854,
      -0.232318,
      -0.226783,
      -0.221247,
      -0.215712,
      -0.210176,
      -0.204641,
      -0.199105,
      -0.193568,
      -0.188032,
      -0.182495,
      -0.176957,
      -0.171419,
      -0.165880,
      -0.160341,
      -0.154800,
      -0.149259,
      -0.143717,
      -0.138173,
      -0.132629,
      -0.127083,
      -0.121537,
      -0.115989,
      -0.110439,
      -0.104889,
      -0.099336,
      -0.093782,
      -0.088227,
      -0.082670,
      -0.077111,
      -0.071550,
      -0.065987,
      -0.060423,
      -0.054856,
      -0.049288,
      -0.043717,
      -0.038144,
      -0.032569,
      -0.026991,
      -0.021411,
      -0.015828,
      -0.010243,
      -0.004656,
      0.000934,
      0.006527,
      0.012123,
      0.017722,
      0.023323,
      0.028928,
      0.034535,
      0.040146,
      0.045759,
      0.051376,
      0.056997,
      0.062620,
      0.068247,
      0.073877,
      0.079511,
      0.085149,
      0.090790,
      0.096435,
      0.102083,
      0.107736,
      0.113392,
      0.119052,
      0.124716,
      0.130385,
      0.136057,
      0.141734,
      0.147414,
      0.153100,
      0.158789,
      0.164483,
      0.170181,
      0.175884,
      0.181592,
      0.187304,
      0.193021,
      0.198743,
      0.204469,
      0.210201,
      0.215937,
      0.221678,
      0.227425,
      0.233177,
      0.238933,
      0.244696,
      0.250463,
      0.256236,
      0.262014,
      0.267798,
      0.273587,
      0.279382,
      0.285183,
      0.290989,
      0.296801,
      0.302619,
      0.308443,
      0.314273,
      0.320109,
      0.325951,
      0.331799,
      0.337654,
      0.343515,
      0.349382,
      0.355255,
      0.361135,
      0.367022,
      0.372915,
      0.378815,
      0.384721,
      0.390634,
      0.396554,
      0.402481,
      0.408415,
      0.414356,
      0.420304,
      0.426260,
      0.432222,
      0.438192,
      0.444169,
      0.450153,
      0.456145,
      0.462144,
      0.468151,
      0.474166,
      0.480188,
      0.486218,
      0.492256,
      0.498302,
      0.504356,
      0.510418,
      0.516488,
      0.522566,
      0.528653,
      0.534747,
      0.540850,
      0.546962,
      0.553082,
      0.559210,
      0.565347,
      0.571493,
      0.577648,
      0.583811,
      0.589983,
      0.596164,
      0.602355,
      0.608554,
      0.614762,
      0.620980,
      0.627207,
      0.633444,
      0.639689,
      0.645945,
      0.652210,
      0.658484,
      0.664768,
      0.671062,
      0.677366,
      0.683680,
      0.690004,
      0.696338,
      0.702682,
      0.709036,
      0.715400,
      0.721775,
      0.728160,
      0.734556,
      0.740963,
      0.747379,
      0.753807,
      0.760246,
      0.766695,
      0.773155,
      0.779627,
      0.786109,
      0.792603,
      0.799107,
      0.805624,
      0.812151,
      0.818690,
      0.825241,
      0.831803,
      0.838377,
      0.844962,
      0.851560,
      0.858170,
      0.864791,
      0.871425,
      0.878071,
      0.884729,
      0.891399,
      0.898082,
      0.904778,
      0.911486,
      0.918206,
      0.924940,
      0.931686,
      0.938446,
      0.945218,
      0.952003,
      0.958802,
      0.965614,
      0.972439,
      0.979278,
      0.986130,
      0.992996,
      0.999875,
      1.006769,
      1.013676,
      1.020597,
      1.027533,
      1.034482,
      1.041446,
      1.048424,
      1.055417,
      1.062424,
      1.069446,
      1.076482,
      1.083534,
      1.090600,
      1.097681,
      1.104778,
      1.111889,
      1.119016,
      1.126159,
      1.133316,
      1.140490,
      1.147679,
      1.154884,
      1.162105,
      1.169342,
      1.176595,
      1.183864,
      1.191149,
      1.198451,
      1.205770,
      1.213105,
      1.220457,
      1.227826,
      1.235211,
      1.242614,
      1.250034,
      1.257471,
      1.264926,
      1.272398,
      1.279888,
      1.287395,
      1.294921,
      1.302464,
      1.310026,
      1.317605,
      1.325203,
      1.332819,
      1.340454,
      1.348108,
      1.355780,
      1.363472,
      1.371182,
      1.378912,
      1.386660,
      1.394429,
      1.402216,
      1.410024,
      1.417851,
      1.425698,
      1.433565,
      1.441453,
      1.449360,
      1.457288,
      1.465237,
      1.473206,
      1.481196,
      1.489208,
      1.497240,
      1.505293,
      1.513368,
      1.521465,
      1.529583,
      1.537723,
      1.545885,
      1.554068,
      1.562275,
      1.570503,
      1.578754,
      1.587028,
      1.595325,
      1.603644,
      1.611987,
      1.620353,
      1.628743,
      1.637156,
      1.645593,
      1.654053,
      1.662538,
      1.671047,
      1.679581,
      1.688139,
      1.696721,
      1.705329,
      1.713961,
      1.722619,
      1.731303,
      1.740011,
      1.748746,
      1.757506,
      1.766293,
      1.775106,
      1.783945,
      1.792810,
      1.801703,
      1.810623,
      1.819569,
      1.828543,
      1.837545,
      1.846574,
      1.855631,
      1.864717,
      1.873830,
      1.882972,
      1.892143,
      1.901343,
      1.910572,
      1.919830,
      1.929117,
      1.938434,
      1.947781,
      1.957158,
      1.966566,
      1.976004,
      1.985473,
      1.994972,
      2.004503,
      2.014065,
      2.023659,
      2.033285,
      2.042943,
      2.052633,
      2.062355,
      2.072110,
      2.081899,
      2.091720,
      2.101575,
      2.111464,
      2.121386,
      2.131343,
      2.141334,
      2.151360,
      2.161421,
      2.171517,
      2.181648,
      2.191815,
      2.202018,
      2.212257,
      2.222533,
      2.232845,
      2.243195,
      2.253582,
      2.264006,
      2.274468,
      2.284968,
      2.295507,
      2.306084,
      2.316701,
      2.327356,
      2.338051,
      2.348786,
      2.359562,
      2.370377,
      2.381234,
      2.392131,
      2.403070,
      2.414051,
      2.425073,
      2.436138,
      2.447246,
      2.458397,
      2.469591,
      2.480828,
      2.492110,
      2.503436,
      2.514807,
      2.526222,
      2.537684,
      2.549190,
      2.560743,
      2.572343,
      2.583989,
      2.595682,
      2.607423,
      2.619212,
      2.631050,
      2.642936,
      2.654871,
      2.666855,
      2.678890,
      2.690975,
      2.703110,
      2.715297,
      2.727535,
      2.739825,
      2.752168,
      2.764563,
      2.777012,
      2.789514,
      2.802070,
      2.814681,
      2.827347,
      2.840069,
      2.852846,
      2.865680,
      2.878570,
      2.891518,
      2.904524,
      2.917588,
      2.930712,
      2.943894,
      2.957136,
      2.970439,
      2.983802,
      2.997227,
      3.010714,
      3.024263,
      3.037875,
      3.051551,
      3.065290,
      3.079095,
      3.092965,
      3.106900,
      3.120902,
      3.134971,
      3.149107,
      3.163312,
      3.177585,
      3.191928,
      3.206340,
      3.220824,
      3.235378,
      3.250005,
      3.264704,
      3.279477,
      3.294323,
      3.309244,
      3.324240,
      3.339312,
      3.354461,
      3.369687,
      3.384992,
      3.400375,
      3.415838,
      3.431381,
      3.447005,
      3.462711,
      3.478500,
      3.494372,
      3.510328,
      3.526370,
      3.542497,
      3.558711,
      3.575012,
      3.591402,
      3.607881,
      3.624450,
      3.641111,
      3.657863,
      3.674708,
      3.691646,
      3.708680,
      3.725809,
      3.743034,
      3.760357,
      3.777779,
      3.795300,
      3.812921,
      3.830645,
      3.848470,
      3.866400,
      3.884434,
      3.902574,
      3.920821,
      3.939176,
      3.957640,
      3.976215,
      3.994901,
      4.013699,
      4.032612,
      4.051639,
      4.070783,
      4.090045,
      4.109425,
      4.128925,
      4.148547,
      4.168292,
      4.188160,
      4.208154,
      4.228275,
      4.248524,
      4.268903,
      4.289413,
      4.310056,
      4.330832,
      4.351745,
      4.372794,
      4.393982,
      4.415310,
      4.436781,
      4.458395,
      4.480154,
      4.502060,
      4.524114,
      4.546319,
      4.568676,
      4.591187,
      4.613854,
      4.636678,
      4.659662,
      4.682807,
      4.706116,
      4.729590,
      4.753231,
      4.777041,
      4.801024,
      4.825179,
      4.849511,
      4.874020,
      4.898710,
      4.923582,
      4.948639,
      4.973883,
      4.999316,
      5.024942,
      5.050761,
      5.076778,
      5.102993,
      5.129411,
      5.156034,
      5.182864,
      5.209903,
      5.237156,
      5.264625,
      5.292312,
      5.320220,
      5.348354,
      5.376714,
      5.405306,
      5.434131,
      5.463193,
      5.492496,
      5.522042,
      5.551836,
      5.581880,
      5.612178,
      5.642734,
      5.673552,
      5.704634,
      5.735986,
      5.767610,
      5.799512,
      5.831694,
      5.864161,
      5.896918,
      5.929968,
      5.963316,
      5.996967,
      6.030925,
      6.065194,
      6.099780,
      6.134687,
      6.169921,
      6.205486,
      6.241387,
      6.277630,
      6.314220,
      6.351163,
      6.388465,
      6.426130,
      6.464166,
      6.502578,
      6.541371,
      6.580553,
      6.620130,
      6.660109,
      6.700495,
      6.741297,
      6.782520,
      6.824173,
      6.866262,
      6.908795,
      6.951780,
      6.995225,
      7.039137,
      7.083525,
      7.128398,
      7.173764,
      7.219632,
      7.266011,
      7.312910,
      7.360339,
      7.408308,
      7.456827,
      7.505905,
      7.555554,
      7.605785,
      7.656608,
      7.708035,
      7.760077,
      7.812747,
      7.866057,
      7.920019,
      7.974647,
      8.029953,
      8.085952,
      8.142657,
      8.200083,
      8.258245,
      8.317158,
      8.376837,
      8.437300,
      8.498562,
      8.560641,
      8.623554,
      8.687319,
      8.751955,
      8.817481,
      8.883916,
      8.951282,
      9.019600,
      9.088889,
      9.159174,
      9.230477,
      9.302822,
      9.376233,
      9.450735,
      9.526355,
      9.603118,
      9.681054,
      9.760191,
      9.840558,
      9.922186,
      10.005107,
      10.089353,
      10.174959,
      10.261958,
      10.350389,
      10.440287,
      10.531693,
      10.624646,
      10.719188,
      10.815362,
      10.913214,
      11.012789,
      11.114137,
      11.217307,
      11.322352,
      11.429325,
      11.538283,
      11.649285,
      11.762390,
      11.877664,
      11.995170,
      12.114979,
      12.237161,
      12.361791,
      12.488946,
      12.618708,
      12.751161,
      12.886394,
      13.024498,
      13.165570,
      13.309711,
      13.457026,
      13.607625,
      13.761625,
      13.919145,
      14.080314,
      14.245263,
      14.414134,
      14.587072,
      14.764233,
      14.945778,
      15.131877,
      15.322712,
      15.518470,
      15.719353,
      15.925570,
      16.137345,
      16.354912,
      16.578520,
      16.808433,
      17.044929,
      17.288305,
      17.538873,
      17.796967,
      18.062943,
      18.337176,
      18.620068,
      18.912049,
      19.213574,
      19.525133,
      19.847249,
      20.180480,
      20.525429,
      20.882738,
      21.253102,
      21.637266,
      22.036036,
      22.450278,
      22.880933,
      23.329017,
      23.795634,
      24.281981,
      24.789364,
      25.319207,
      25.873062,
      26.452634,
      27.059789,
      27.696581,
      28.365274,
      29.068370,
      29.808638,
      30.589157,
      31.413354,
      32.285060,
      33.208568,
      34.188705,
      35.230920,
      36.341388,
      37.527131,
      38.796172,
      40.157721,
      41.622399,
      43.202525,
      44.912465,
      46.769077,
      48.792279,
      51.005773,
      53.437996,
      56.123356,
      59.103894,
    ])

    def ranlan_1(x):
      v = np.log(x)
      u = 1 / v
      ranlan = (
        (0.99858950 + (34.5213058 + 17.0854528 * u) * u)
        / (1 + (34.1760202 + 4.01244582 * u) * u)
      ) * (-np.log(-0.91893853 - v) - 1)
      ranlan[x <= 0] = -np.inf
      ranlan[x >= 0.007] = float("NaN")
      return ranlan

    def ranlan_2(x):
      y = np.array(1000 * x, dtype=int)
      z = 1000 * x - y
      y= np.clip(y,7,980)
      ranlan = f[y - 1] + z * (
        f[y] - f[y - 1] - 0.25 * (1 - z) * (f[y + 1] - f[y] - f[y - 1] + f[y - 2])
      )
      ranlan[(x < 0.007) | ((x>=0.07) &(x < 0.8)) | (x >= 0.98)] = float("NaN")
      return ranlan

    def ranlan_3(x):
      y = np.array(1000 * x, dtype=int)
      z = 1000 * x - y
      y= np.clip(y,7,980)
      ranlan = f[y - 1] + z * (f[y] - f[y - 1])
      ranlan[(x < 0.07) | (x >= 0.8)] = float("NaN")
      return ranlan

    def ranlan_4(x):
      u = 1 - x
      v = u * u
      ranlan = (1.00060006 + 263.991156 * u + 4373.20068 * v) / (
        (1 + 257.368075 * u + 3414.48018 * v) * u
      )
      ranlan[(x < 0.98) | (x >= 0.999)] = float("NaN")
      return ranlan

    def ranlan_5(x):
      u = 1 - x
      v = u * u
      ranlan = (1.00001538 + 6075.14119 * u + 734266.409 * v) / (
        (1 + 6065.11919 * u + 694021.044 * v) * u
      )
      ranlan[x < 0.999] = float("NaN")
      return ranlan

    if not isinstance(x, (int, float, np.ndarray)):
      raise TypeError(
        f'Require float or numpy.ndarray, but received object of type {type(x)}.'
      )
    x_was_just_a_number = isinstance(x, (float, int))
    x = np.atleast_1d(x).astype(float)

    result = x * float("NaN")  # Initialize
    x_is_finite_indices = np.isfinite(x)
    with warnings.catch_warnings():
      warnings.simplefilter(
        "ignore"
      )  # I don't want to see the warnings of numpy, anyway it will fill with `NaN` values so it is fine.
      denlan = x[x_is_finite_indices] * float("NaN")  # Initialize.
      limits = (-float("inf"), 0.007, 0.07, 0.8, 0.98, 0.999, float("inf"))
      formulas = (
        ranlan_1,
        ranlan_2,
        ranlan_3,
        ranlan_2,
        ranlan_4,
        ranlan_5,
      )
      for k, formula in enumerate(formulas):
        indices = (limits[k] <= x[x_is_finite_indices]) & (
          x[x_is_finite_indices] < limits[k + 1]
        )
        denlan[indices] = formula(x[x_is_finite_indices][indices])
      result[x_is_finite_indices] = denlan
    result[np.isinf(x) & (x < 0)] = -np.inf
    result[np.isinf(x) & (x > 0)] = np.inf

    result = np.squeeze(result)
    if x_was_just_a_number:
      result = float(result)
    return result
