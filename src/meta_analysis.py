import warnings
import numpy as np
import matplotlib.pyplot as plt
from typing import Any
from collections.abc import Iterable, Sequence
from scipy.optimize import minimize_scalar, root_scalar, minimize
from scipy.stats import cauchy, gamma, norm, halfnorm, t, chi2, f

from src.distribution import euler_gamma, hcauchy_mean, harmonic_mean, landau

# this is very very important: due to constraints in float accuracy, np.tan(np.pi/2) is NOT infty, which will lead to serious issues in optimization. one workaround is to use manually defined cotangent function.
def cot(x: Any):
  return 1 / np.tan(x)


class combination_test(object):
  def __init__(
    self,
    arg: int | np.int_ | float | np.float_ | Sequence[float | np.float_] | np.ndarray,
    method: str = "HCauchy",
    level: float = 0.05,
    **kwargs,
  ) -> None:
    # automatically get the number of tests and weight vector from arg
    if isinstance(arg, int | np.int_ | float | np.float_):
      if arg <= 0:
        raise ValueError("Number of tests should at least be one.")
      arg = int(np.int_(arg))
      self.__num_test = arg
      self.__weights = np.ones(arg) / arg
    elif isinstance(arg, Iterable):
      if (arg <= 0).sum() != 0:
        raise ValueError("Weights should be positive.")
      if np.isclose(arg.sum(), 1):
        raise ValueError("Weights should sum to one.")
      self.__num_test = len(arg)
      self.__weights = np.array(arg)
    else:
      raise ValueError("Please provide the number of tests or a vector of weights.")
    # method
    self.__method = method
    if self.__method == "Fisher" and self.__num_test != 1:
      warnings.warn("Equal weights are used for Fisher's test.")
    # significance level
    self.__level = level
    # initialize the threshold based on level, weights, and method
    self.__init_threshold()

  @property
  def num_test(self):
    return self.__num_test

  @property
  def weights(self):
    return self.__weights

  @property
  def method(self):
    return self.__method

  @property
  def level(self):
    return self.__level

  @property
  def threshold(self):
    return self.__threshold

  # get global score threshold from significance level
  def get_threshold_from_p(self, level: float) -> np.float_:
    if self.__num_test == 1:
      return -level
    elif self.__method == "HCauchy":
      if self.__num_test <= 1000:
        return hcauchy_mean.ppf(1 - level, self.__weights)
      else:
        location = (
          (-sum(self.__weights * np.log(self.__weights)) + 1 - euler_gamma) * 2 / np.pi
        )
        return landau.ppf(1 - level, location, 1)
    elif self.__method == "EHMP":
      if self.__num_test <= 1000:
        return harmonic_mean.ppf(1 - level, self.__weights)
      else:
        location = -sum(self.__weights * np.log(self.__weights)) + 1 - euler_gamma
        return landau.ppf(1 - level, location, np.pi / 2)
    elif self.__method == "HMP":
      location = -sum(self.__weights * np.log(self.__weights)) + 1 - euler_gamma
      return landau.ppf(1 - level, location, np.pi / 2)
    elif self.__method == "Cauchy":
      return cauchy.ppf(1 - level)
    elif self.__method == "Levy":
      return 1 / norm.ppf((1 + level) / 2) ** 2
    elif self.__method == "Fisher":
      return gamma.ppf(1 - level, self.__num_test)
    elif self.__method == "Stouffer":
      return norm.ppf(1 - level)
    elif self.__method == "Bonferroni":
      return -level / self.__num_test
    elif self.__method == "Simes":
      return -level / self.__num_test
    else:
      raise NotImplementedError("Method not implemented.")

  # part of initialization
  def __init_threshold(self) -> None:
    self.__threshold = self.get_threshold_from_p(self.__level)

  # change method
  def change_method(self, new_method: str) -> None:
    self.__method = new_method
    if self.__method == "Fisher" and self.__num_test != 1:
      warnings.warn("Equal weights are used for Fisher's test.")
    self.__init_threshold()

  # change level
  def change_level(self, new_level: float) -> None:
    self.__level = new_level
    self.__init_threshold()

  # get p-value from score for the global test
  def get_p_from_score(
    self, score: Sequence[float | np.float_] | np.ndarray
  ) -> np.float_:
    score = np.array(score)
    if self.__num_test == 1:
      return -score
    elif self.__method == "HCauchy":
      if self.__num_test <= 1000:
        return hcauchy_mean.sf(score, self.__weights)
      else:
        location = (
          (-sum(self.__weights * np.log(self.__weights)) + 1 - euler_gamma) * 2 / np.pi
        )
        return landau.sf(score, location, 1)
    elif self.__method == "EHMP":
      if self.__num_test <= 1000:
        return harmonic_mean.sf(score, self.__weights)
      else:
        location = -sum(self.__weights * np.log(self.__weights)) + 1 - euler_gamma
        return landau.sf(score, location, np.pi / 2)
    elif self.__method == "HMP":
      location = -sum(self.__weights * np.log(self.__weights)) + 1 - euler_gamma
      return landau.sf(score, location, np.pi / 2)
    elif self.__method == "Cauchy":
      return cauchy.sf(score)
    elif self.__method == "Levy":
      return halfnorm.cdf(1 / np.sqrt(score))
    elif self.__method == "Fisher":
      return gamma.sf(score, self.__num_test)
    elif self.__method == "Stouffer":
      return norm.sf(score)
    elif self.__method == "Bonferroni":
      return min(1, -score * self.__num_test)
    elif self.__method == "Simes":
      return min(1, -score * self.__num_test)
    else:
      raise NotImplementedError("Method not implemented.")

  # get global score from individual p-values
  def get_global_score(
    self, p_vector: Sequence[float | np.float_] | np.ndarray
  ) -> np.float_:
    p_vector = np.array(p_vector)
    # if (p_vector<1e-149).sum()>0:
    #   warnings.warn('Overflow encountered. No reliable optimizer can be found.')
    if self.__num_test == 1:
      return -np.squeeze(p_vector, axis=-1)
    elif self.__method == "HCauchy":
      return cot(p_vector * np.pi / 2) @ self.__weights
    elif self.__method == "EHMP":
      return 1 / p_vector @ self.__weights
    elif self.__method == "HMP":
      return 1 / p_vector @ self.__weights
    elif self.__method == "Cauchy":
      return cot(p_vector * np.pi) @ self.__weights
    elif self.__method == "Levy":
      return (
        1
        / norm.ppf((1 + p_vector) / 2) ** 2
        @ self.__weights
        / (np.sum(np.sqrt(self.__weights))) ** 2
      )
    elif self.__method == "Fisher":
      return -np.sum(np.log(p_vector))
    elif self.__method == "Stouffer":
      return -norm.ppf(p_vector) @ self.__weights / np.sqrt(np.sum(self.__weights**2))
    elif self.__method == "Bonferroni":
      return -p_vector.min()
    elif self.__method == "Simes":
      return -(np.sort(p_vector) / np.arange(1, self.__num_test + 1)).min()
    else:
      raise NotImplementedError("Method not implemented.")

  # get global p from individual p-values
  def get_global_p(
    self, p_vector: Sequence[float | np.float_] | np.ndarray
  ) -> np.float_:
    p_vector = np.array(p_vector)
    return self.get_p_from_score(self.get_global_score(p_vector))

  # make decision for the global hypothesis test
  def make_decision(self, p_vector: Sequence[float | np.float_] | np.ndarray) -> bool:
    p_vector = np.array(p_vector)
    return self.get_global_score(p_vector) >= self.__threshold


class meta_analysis_1d(combination_test):
  def __init__(
    self,
    arg: int | np.int_ | float | np.float_ | Sequence[float | np.float_] | np.ndarray,
    method: str = "HCauchy",
    level: float = 0.05,
    **kwargs,
  ) -> None:
    if method != "HCauchy" and method != "EHMP" and self.__num_test != 1:
      warnings.warn(
        "Please note that confidence intervals are only guaranteed to be correct for HCauchy or EHMP."
      )
    super().__init__(arg, method, level, **kwargs)
    self.__dim = 1

  @property
  def dim(self):
    return self.__dim

  # formatting input; please note that we define theta_hat, sigma or df as arguments for class methods rather than class members because in repeated experiments with different theta_hat, sigma or df's, reinitializing class instances takes a lot of time
  def __format_input(
    self,
    theta_hat: Sequence[float | np.float_] | np.ndarray,
    sigma: Sequence[float | np.float_] | np.ndarray | float | np.float_,
    df: Sequence[float | np.float_] | np.ndarray | float | np.float_,
  ) -> tuple:
    # check and format theta_hat
    if isinstance(theta_hat, Iterable):
      theta_hat = np.array(theta_hat)
      if theta_hat.shape[0] != self.num_test:
        raise ValueError("The length of theta_hat should be the same as num_test.")
    else:
      raise TypeError("theta_hat should be an array of estimates.")

    # check and format sigma
    if isinstance(sigma, Iterable):
      sigma = np.array(sigma)
      if sigma.shape[0] != self.num_test:
        raise ValueError("The length of sigma should be the same as num_test.")
    elif isinstance(sigma, (float, np.float_)):
      sigma = np.ones(self.num_test) * sigma
    else:
      raise TypeError("sigma should be either a float or an array of floats.")

    # check and format df
    if df is None:
      pass
    elif isinstance(df, Iterable):
      if df.shape[0] != self.num_test:
        raise ValueError("The length of df should be the same as num_test.")
      df = np.int_(df)
    elif isinstance(df, (float, int, np.float_, np.int_)):
      df = np.ones(self.num_test) * np.int_(df)
    else:
      raise TypeError("df should be either an int or an array of ints.")

    # return formatted theta_hat, sigma and df
    return theta_hat, sigma, df

  # get p-values from estimates for each study
  def get_p_vector(
    self,
    point: float | np.float_,
    theta_hat: Sequence[float | np.float_] | np.ndarray,
    sigma: Sequence[float | np.float_] | np.ndarray | float | np.float_,
    df: Sequence[float | np.float_] | np.ndarray | float | np.float_ | None = None,
    format_input: bool = True,
  ) -> np.ndarray:
    if format_input:
      theta_hat, sigma, df = self.__format_input(theta_hat, sigma, df)
    # if df is None, use normal distribution
    if df is None:
      return 2 * np.maximum(norm.sf(np.abs(point - theta_hat) / sigma), 1e-150)
    # if df is given, use student t distribution
    else:
      return 2 * np.maximum(t.sf(np.abs(point - theta_hat) / sigma, df), 1e-150)

  # check if a point or points are covered by the confidence interval
  def check_cover(
    self,
    point: float | np.float_,
    theta_hat: Sequence[float] | np.ndarray,
    sigma: Sequence[float] | np.ndarray | float | np.float_,
    df: Sequence[float] | np.ndarray | float | np.float_ | None = None,
    format_input: bool = True,
  ) -> bool:
    # format input as the default
    if format_input:
      theta_hat, sigma, df = self.__format_input(theta_hat, sigma, df)
    return (
      self.get_global_score(
        self.get_p_vector(point, theta_hat, sigma, df, format_input=False)
      )
      <= self.threshold
    )

  # find the minimizer in the confidence region (as a global point estimate and to be used in getting the confidence interval)
  def find_minimizer(
    self,
    theta_hat: Sequence[float | np.float_] | np.ndarray,
    sigma: Sequence[float | np.float_] | np.ndarray | float | np.float_,
    df: Sequence[float | np.float_] | np.ndarray | float | np.float_ | None = None,
    format_input: bool = True,
  ) -> float | np.float_:
    # format input as the default
    if format_input:
      theta_hat, sigma, df = self.__format_input(theta_hat, sigma, df)
    func = lambda x: self.get_global_score(  # noqa: E731
      self.get_p_vector(x, theta_hat, sigma, df, format_input=False)
    )
    # use the min and max of observations as the lower and upper bounds
    # low = min(theta_hat)
    # high = max(theta_hat)
    return minimize_scalar(func, method="Brent").x

  # return the confidence interval
  def confidence_interval(
    self,
    theta_hat: Sequence[float | np.float_] | np.ndarray,
    sigma: Sequence[float | np.float_] | np.ndarray | float | np.float_,
    df: Sequence[float | np.float_] | np.ndarray | float | np.float_ | None = None,
    method: str = "brentq",
    format_input: bool = True,
  ) -> tuple:
    # format input as the default
    if format_input:
      theta_hat, sigma, df = self.__format_input(theta_hat, sigma, df)
    func = lambda x: self.get_global_score(  # noqa: E731
      self.get_p_vector(x, theta_hat, sigma, df, format_input=False)
    )
    low = min(theta_hat)
    high = max(theta_hat)
    # fing the minimizer
    minimizer = self.find_minimizer(theta_hat, sigma, df, format_input=False)
    # deal with the case that the solution set is empty
    if func(minimizer) > self.threshold:
      print(f"{1-self.level} confidence interval is empty.")
      root_low = minimizer
      root_high = minimizer
    # when the solution set is not empty
    else:
      if method == "brentq":
        # find the correct lower and upper bounds
        low_1 = high_1 = minimizer
        while True:
          if func(low) < self.threshold:
            low_1 = low
            low = 2 * low - minimizer
          else:
            break
        while True:
          if func(high) < self.threshold:
            high_1 = high
            high = 2 * high - minimizer
          else:
            break
        # find the two roots
        root_low = root_scalar(
          lambda x: func(x) - self.threshold, bracket=[low, low_1], method="brentq"
        ).root
        root_high = root_scalar(
          lambda x: func(x) - self.threshold, bracket=[high_1, high], method="brentq"
        ).root
      else:
        raise NotImplementedError("The root finding method is not implemented.")

    return root_low, root_high, minimizer


class meta_analysis_md(combination_test):
  def __init__(
    self,
    arg: int | np.int_ | float | np.float_ | Sequence[float | np.float_] | np.ndarray,
    dim: int | np.int_ | float | np.float_,
    method: str = "HCauchy",
    level: float = 0.05,
    **kwargs,
  ) -> None:
    if method != "HCauchy" and method != "EHMP":
      warnings.warn(
        "Please note that confidence regions and simultaneous confidence intervals are only guaranteed to be correct for HCauchy or EHMP."
      )
    super().__init__(arg, method, level, **kwargs)
    if dim <= 1:
      raise ValueError(
        "dim should at least be 2. If dim=1, please consider using meta_analysis_1d instead."
      )
    self.__dim = int(np.int_(dim))

  @property
  def dim(self):
    return self.__dim

  # check if the dimensions of sub studies are the same
  def __check_equal_sub_dim(
    self, xi_hat: Sequence[float | np.float_ | np.ndarray] | np.ndarray
  ) -> bool:
    # format xi_hat and decide if the dimensions of estimates in all studies are the same
    equal_sub_dim = True
    try:
      xi_hat = np.array(xi_hat)
    except ValueError:
      equal_sub_dim = False
    return equal_sub_dim

  # check and format input variables. note that sub_dim could originally be either an integer, a 1d-array of integers, list of indexes or a 2d-array of indexes (indexing the row of projs). but the output sub_dim is either a list of length num_test, or a 2d-array of shape (num_test, len(sub_dim[0])
  def __format_input(
    self,
    xi_hat: Sequence[float | np.float_ | np.ndarray] | np.ndarray,
    Sigma: Sequence[float | np.float_ | np.ndarray] | np.ndarray,
    sub_dim: int | np.int_ | float | np.float_ | Sequence[int | np.int_ | float | np.float_ | np.ndarray] | np.ndarray,
    df: int | np.int_ | float | np.float_ | Sequence[int | np.int_ | float | np.float_] | np.ndarray | None,
    projs: Sequence[np.ndarray] | np.ndarray | None,
    equal_sub_dim: bool | None = None,
  ) -> tuple:
    # check format of projections. it is of shape (num_proj, dim)
    if projs is None:
      projs = np.eye(self.__dim)
    else:
      projs = np.array(projs)
    if projs.shape[1] != self.__dim:
      raise ValueError("The dimenion in projs do not match that of the parameter.")
    # get the number of projections
    num_proj = projs.shape[0]

    # check if the dimensions in all studies are the same
    if equal_sub_dim is None:
      equal_sub_dim = self.__check_equal_sub_dim(xi_hat)
    # case 1: the dimensions in all studies are the same
    if equal_sub_dim:
      # format xi_hat and Sigma into np.ndarray. xi_hat is of shape (num_test, sub_dim_value)
      xi_hat = np.array(xi_hat)
      # deal with the case where all subdimensions are 1
      if xi_hat.ndim == 1:
        xi_hat = np.expand_dims(xi_hat, axis=-1)
      Sigma = np.array(Sigma)
      # if all Sigma matrices are the same, insert a dimension at the beginning so that matrix operations are correctly broadcasted
      if Sigma.ndim == 2:
        Sigma = np.expand_dims(Sigma, axis=0)
      # deal with the case where subdimensions are all 1
      elif Sigma.ndim == 1:
        Sigma = np.expand_dims(Sigma, axis=(-2, -1))
      # if sub_dim is an integer, get a 2d-array of indexes indicating the rows of projections used in each study. the array is obtained in a cyclic manner.
      if isinstance(sub_dim, (int, float, np.int_, np.float_)):
        sub_dim = (
          np.arange(self.num_test * int(np.int_(sub_dim))).reshape(self.num_test, -1) % num_proj
        )
      # if sub_dim is already a 2d-array of indexes, everything is all set
      elif isinstance(sub_dim, Iterable):
        sub_dim = np.array(sub_dim)
        # dealing with the corner case where sub_dim is a 1d-array of the same integer
        if sub_dim.ndim == 1:
          sub_dim = (
            np.arange(self.num_test * int(np.int_(sub_dim[0]))).reshape(self.num_test, -1)
            % num_proj
          )
      # catch illegal input of sub_dim
      else:
        raise TypeError(
          "sub_dim should be either an int, a 1d-array or 2d-array of ints."
        )

    # case 2: the dimension of studies are not the same
    else:
      # xi_hat and Sigma are both list of length num_test
      xi_hat = [np.array(x) for x in xi_hat]
      Sigma = [np.array(x) for x in Sigma]
      # if sub_dim is a list of integers, get the list of projection indexes used in each study in a cyclic manner. the output is a list of 1d-arrays of integers.
      try:
        sub_dim = np.array(sub_dim)
        sub_dim = np.split(np.arange(sum(sub_dim)), np.cumsum(sub_dim)[:-1])
        sub_dim = [x % num_proj for x in sub_dim]
      # if sub_dim is already a list of projection indexes, everything is all set.
      except ValueError:
        sub_dim = [np.array(x) for x in sub_dim]

    # format the degrees of freedom. if df is None, do nothing.
    if df is None:
      pass
    # if df is a list of integers, check the length
    elif isinstance(df, Iterable):
      if df.shape[0] != self.num_test:
        raise ValueError("The length of df should be the same as num_test.")
      df = np.array(df)
    # if df is a single integer, repeat it to get a list (1d-array)
    elif isinstance(df, (float, int, np.float_, np.int_)):
      df = np.ones(self.num_test) * np.int_(df)
    # catch illegal input of df
    else:
      raise TypeError("df should be either an int or an array of ints.")

    # return formatted variables along with the judgment
    return xi_hat, Sigma, sub_dim, df, projs, equal_sub_dim

  # get p-values from estimates for each study
  def get_p_vector(
    self,
    point: Sequence[float | np.float_] | np.ndarray,
    xi_hat: Sequence[float | np.float_ | np.ndarray] | np.ndarray,
    Sigma: Sequence[float | np.float_ | np.ndarray] | np.ndarray,
    sub_dim: int | np.int_ | float | np.float_ | Sequence[int | np.int_ | float | np.float_ | np.ndarray] | np.ndarray,
    df: int | np.int_ | float | np.float_ | Sequence[int | np.int_ | float | np.float_] | np.ndarray | None = None,
    projs: Sequence[np.ndarray] | np.ndarray | None = None,
    format_input: bool = True,
    equal_sub_dim: bool | None = None,
  ) -> np.ndarray:
    # format point
    point = np.array(point)
    # by default we need to check if the variable inputs are illegal and if the sub dimensions in all studies are equal.
    if format_input:
      xi_hat, Sigma, sub_dim, df, projs, equal_sub_dim = self.__format_input(
        xi_hat, Sigma, sub_dim, df, projs
      )
    # need to check if the sub dimensions are equal as the functions would be slightly different due to tensorization (it is automatically done in format_input if format_input=True)
    elif equal_sub_dim is None:
      equal_sub_dim = self.__check_equal_sub_dim(xi_hat)

    # to distinguish sub_dim_index with sub_dim_value
    sub_dim_index = sub_dim

    # prepare the projections of points in each study and then get chi2 or hotelling scores. note that in calculating xi, we have taken into account the case where point could be an nd-array with the last dimension being the dimension of the parameter
    point_projs = projs @ np.expand_dims(point, axis=-1)
    if equal_sub_dim:
      # for each point, xi is a 2d-array of shape (num_test, sub_dim_value), where sub_dim_value is an integer. together xi is of shape (*point.shape[:-1], num_test, sub_dim_value). the ellipsis allows for the flexity of point.shape.
      xi = np.squeeze(point_projs, axis=-1)[..., sub_dim_index]
      sub_dim_value = len(sub_dim_index[0])
      # chi2_f_score is of shape (*point.shape[:-1], num_test)
      chi2_f_score = np.squeeze(
        np.expand_dims(xi - xi_hat, axis=-2)
        @ np.linalg.inv(Sigma)
        @ np.expand_dims(xi - xi_hat, axis=-1),
        (-1, -2),
      )
    else:
      # xi is a list of nd-arrays, and is of length num_test. xi[i] is of shape (*point.shape[:-1], sub_dim_value[i]). here sub_dim_value is a 1d-array of integers, and is of length num_test
      xi = [point_projs[..., x] for x in sub_dim_index]
      sub_dim_value = np.array([len(x) for x in sub_dim_index])
      # all elements in this list have the same shape point.shape[:-1]
      chi2_f_score_list = [
        np.squeeze(
          np.expand_dims(x - y, axis=-2)
          @ np.linalg.inv(z)
          @ np.expand_dims(x - y, axis=-1),
          (-1, -2),
        )
        for x, y, z in zip(xi, xi_hat, Sigma)
      ]
      # chi2_f_score is of shape (*point.shape[:-1], num_test)
      chi2_f_score = np.moveaxis(np.array(chi2_f_score_list), 0, -1)

    # if df is not provided, we use chi-squared distribution for each individual study
    if df is None:
      # sub_dim_value is broadcasted in chi2.sf
      return np.maximum(chi2.sf(chi2_f_score, sub_dim_value), 1e-150)
    # if df is provided, we use hotelling T-squared for each individual study
    else:
      # the arguments in f.sf are broadcasted
      return np.maximum(
        f.sf(
          chi2_f_score * (df + 1 - sub_dim_value) / (sub_dim_value * df),
          sub_dim_value,
          df + 1 - sub_dim_value,
        ),
        1e-150,
      )
    
  # get p-values from estimates for each study
  def get_dif_vector(
    self,
    point: Sequence[float | np.float_] | np.ndarray,
    xi_hat: Sequence[float | np.float_ | np.ndarray] | np.ndarray,
    Sigma: Sequence[float | np.float_ | np.ndarray] | np.ndarray,
    sub_dim: int | np.int_ | float | np.float_ | Sequence[int | np.int_ | float | np.float_ | np.ndarray] | np.ndarray,
    df: int | np.int_ | float | np.float_ | Sequence[int | np.int_ | float | np.float_] | np.ndarray | None = None,
    projs: Sequence[np.ndarray] | np.ndarray | None = None,
    format_input: bool = True,
    equal_sub_dim: bool | None = None,
  ) -> np.ndarray:
    # format point
    point = np.array(point)
    # by default we need to check if the variable inputs are illegal and if the sub dimensions in all studies are equal.
    if format_input:
      xi_hat, Sigma, sub_dim, df, projs, equal_sub_dim = self.__format_input(
        xi_hat, Sigma, sub_dim, df, projs
      )
    # need to check if the sub dimensions are equal as the functions would be slightly different due to tensorization (it is automatically done in format_input if format_input=True)
    elif equal_sub_dim is None:
      equal_sub_dim = self.__check_equal_sub_dim(xi_hat)

    # to distinguish sub_dim_index with sub_dim_value
    sub_dim_index = sub_dim

    # prepare the projections of points in each study and then get chi2 or hotelling scores. note that in calculating xi, we have taken into account the case where point could be an nd-array with the last dimension being the dimension of the parameter
    point_projs = projs @ np.expand_dims(point, axis=-1)
    if equal_sub_dim:
      # for each point, xi is a 2d-array of shape (num_test, sub_dim_value), where sub_dim_value is an integer. together xi is of shape (*point.shape[:-1], num_test, sub_dim_value). the ellipsis allows for the flexity of point.shape.
      xi = np.squeeze(point_projs, axis=-1)[..., sub_dim_index]
      sub_dim_value = len(sub_dim_index[0])
      # chi2_f_score is of shape (*point.shape[:-1], num_test)
      chi2_f_score = np.squeeze(
        np.expand_dims(xi - xi_hat, axis=-2)
        @ np.linalg.inv(Sigma)
        @ np.expand_dims(xi - xi_hat, axis=-1),
        (-1, -2),
      )
    else:
      # xi is a list of nd-arrays, and is of length num_test. xi[i] is of shape (*point.shape[:-1], sub_dim_value[i]). here sub_dim_value is a 1d-array of integers, and is of length num_test
      xi = [point_projs[..., x] for x in sub_dim_index]
      sub_dim_value = np.array([len(x) for x in sub_dim_index])
      # all elements in this list have the same shape point.shape[:-1]
      chi2_f_score_list = [
        np.squeeze(
          np.expand_dims(x - y, axis=-2)
          @ np.linalg.inv(z)
          @ np.expand_dims(x - y, axis=-1),
          (-1, -2),
        )
        for x, y, z in zip(xi, xi_hat, Sigma)
      ]
      # chi2_f_score is of shape (*point.shape[:-1], num_test)
      chi2_f_score = np.moveaxis(np.array(chi2_f_score_list), 0, -1)

    # if df is not provided, we use chi-squared distribution for each individual study
    if df is None:
      # sub_dim_value is broadcasted in chi2.sf
      return np.maximum(chi2.pdf(chi2_f_score, sub_dim_value), 1e-150)
    # if df is provided, we use hotelling T-squared for each individual study
    else:
      # the arguments in f.sf are broadcasted
      return np.maximum(
        f.pdf(
          chi2_f_score * (df + 1 - sub_dim_value) * (df + 1) / (sub_dim_value * df),
          sub_dim_value,
          df + 1 - sub_dim_value,
        ),
        1e-150,
      )

  # check if a point or points are covered by the confidence region
  def check_cover(
    self,
    point: Sequence[float | np.float_] | np.ndarray,
    xi_hat: Sequence[float | np.float_ | np.ndarray] | np.ndarray,
    Sigma: Sequence[float | np.float_ | np.ndarray] | np.ndarray,
    sub_dim: int | np.int_ | float | np.float_ | Sequence[int | np.int_ | float | np.float_ | np.ndarray] | np.ndarray,
    df: int | np.int_ | float | np.float_ | Sequence[int | np.int_ | float | np.float_] | np.ndarray | None = None,
    projs: Sequence[np.ndarray] | np.ndarray | None = None,
    format_input: bool = True,
    equal_sub_dim: bool | None = None,
  ) -> bool:
    # format point
    point = np.array(point)
    # by default we need to check if the variable inputs are illegal and if the sub dimensions in all studies are equal.
    if format_input:
      xi_hat, Sigma, sub_dim, df, projs, equal_sub_dim = self.__format_input(
        xi_hat, Sigma, sub_dim, df, projs
      )
    # need to check if the sub dimensions are equal as the functions would be slightly different due to tensorization (it is automatically done in format_input if format_input=True)
    elif equal_sub_dim is None:
      equal_sub_dim = self.__check_equal_sub_dim(xi_hat)
    current = self.get_global_score(
        self.get_p_vector(
          point,
          xi_hat,
          Sigma,
          sub_dim,
          df,
          projs,
          format_input=False,
          equal_sub_dim=equal_sub_dim,
        )
      )
    return (current <= self.threshold)

  # find the minimizer in the confidence region (as a global point estimate)
  def find_minimizer(
    self,
    xi_hat: Sequence[float | np.float_ | np.ndarray] | np.ndarray,
    Sigma: Sequence[float | np.float_ | np.ndarray] | np.ndarray,
    sub_dim: int | np.int_ | float | np.float_ | Sequence[int | np.int_ | float | np.float_ | np.ndarray] | np.ndarray,
    df: int | np.int_ | float | np.float_ | Sequence[int | np.int_ | float | np.float_] | np.ndarray | None = None,
    projs: Sequence[np.ndarray] | np.ndarray | None = None,
    format_input: bool = True,
    equal_sub_dim: bool | None = None,
    x0: np.ndarray | None = None,
    method: str = "Powell",
    require_grad: bool = False,
    **kwargs,
  ) -> np.ndarray:
    # by default we need to check if the variable inputs are illegal and if the sub dimensions in all studies are equal.
    if format_input:
      xi_hat, Sigma, sub_dim, df, projs, equal_sub_dim = self.__format_input(
        xi_hat, Sigma, sub_dim, df, projs
      )
    # need to check if the sub dimensions are equal as the functions would be slightly different due to tensorization (it is automatically done in format_input if format_input=True)
    elif equal_sub_dim is None:
      equal_sub_dim = self.__check_equal_sub_dim(xi_hat)

    func = lambda x: self.get_global_score(  # noqa: E731
      self.get_p_vector(
        x,
        xi_hat,
        Sigma,
        sub_dim,
        df,
        projs,
        format_input=False,
        equal_sub_dim=equal_sub_dim,
      )
    )

    if require_grad:

      def dfunc(xi):
        p_vector = self.get_p_vector(
          xi,
          xi_hat,
          Sigma,
          sub_dim,
          df,
          projs,
          format_input=False,
          equal_sub_dim=equal_sub_dim,
        )
        dif_vector = self.get_dif_vector(
          xi,
          xi_hat,
          Sigma,
          sub_dim,
          df,
          projs,
          format_input=False,
          equal_sub_dim=equal_sub_dim,
        )
        new_weights = self.weights * dif_vector / np.sin(p_vector)**2 * np.pi
        return np.concatenate([w * np.linalg.inv(z) @ (x - y) for w, z, x, y in zip(new_weights, Sigma, xi, xi_hat)])
      
      jac = dfunc
    
    else:
      jac = None

    # available method: "L-BFGS-B", "BFGS", "Powell", "TNC", "COBYLA", "COBYQA", "SLSQP", "Nelder-Mead"

    if x0 is None:
      if equal_sub_dim:
        mat_tmp = projs[sub_dim.reshape(-1), :]
        wgt_tmp = (np.ones_like(sub_dim) * self.weights.reshape(-1,1)/sub_dim.shape[-1]).reshape(-1)
      else:
        mat_tmp = projs[np.array(sum(sub_dim,[])),:]
        wgt_tmp = np.concatenate([np.ones_like(sub_dim[i])*self.weights[i]/sub_dim[i].shape for i in range(len(sub_dim))])
      x0 = np.linalg.pinv((mat_tmp.T * wgt_tmp) @ mat_tmp) @ (mat_tmp.T * wgt_tmp) @ xi_hat.reshape(-1)
      # print(f'initial search point for the minizer: {x0}.')

    if "xtol" not in kwargs:
      kwargs["xtol"]=1E-7

    res = minimize(func, x0=x0, method=method, jac=jac, options=kwargs).x
    
    return res

  # get simultaneous confidence interval
  def simultaneous_interval(
    self,
    direction: Sequence[float | np.float_] | np.ndarray,
    xi_hat: Sequence[float | np.float_ | np.ndarray] | np.ndarray,
    Sigma: Sequence[float | np.float_ | np.ndarray] | np.ndarray,
    sub_dim: int | np.int_ | float | np.float_ | Sequence[int | np.int_ | float | np.float_ | np.ndarray] | np.ndarray,
    df: int | np.int_ | float | np.float_ | Sequence[int | np.int_ | float | np.float_] | np.ndarray | None = None,
    projs: Sequence[np.ndarray] | np.ndarray | None = None,
    format_input: bool = True,
    equal_sub_dim: bool | None = None,
    x0: np.ndarray | None = None,
    find_min: bool = True,
    method: str = "Powell",
    lambda_inv: float = np.exp(-20),
    **kwargs,
  ) -> tuple:
    # format point and direction
    direction = np.array(direction)

    # DONT normalize direction
    # direction = direction / np.sqrt((direction**2).sum())

    # by default we need to check if the variable inputs are illegal and if the sub dimensions in all studies are equal.
    if format_input:
      xi_hat, Sigma, sub_dim, df, projs, equal_sub_dim = self.__format_input(
        xi_hat, Sigma, sub_dim, df, projs
      )
    # need to check if the sub dimensions are equal as the functions would be slightly different due to tensorization (it is automatically done in format_input if format_input=True)
    elif equal_sub_dim is None:
      equal_sub_dim = self.__check_equal_sub_dim(xi_hat)

    func = lambda x: self.get_global_score(  # noqa: E731
      self.get_p_vector(
        x,
        xi_hat,
        Sigma,
        sub_dim,
        df,
        projs,
        format_input=False,
        equal_sub_dim=equal_sub_dim,
      )
    )-self.threshold

    if x0 is None:
      if equal_sub_dim:
        mat_tmp = projs[sub_dim.reshape(-1), :]
      else:
        mat_tmp = projs[np.array(sum(sub_dim,[])),:]
      x0 = np.linalg.pinv((mat_tmp.T * self.weights) @ mat_tmp) @ (mat_tmp.T * self.weights) @ xi_hat.reshape(-1)
    
    if find_min:
      x1 = self.find_minimizer(xi_hat=xi_hat, Sigma=Sigma, sub_dim=sub_dim, df=df, projs=projs, format_input = False,equal_sub_dim=equal_sub_dim, x0=x0, method=method)
    else:
      x1 = x0

    if "xtol" not in kwargs.keys():
      kwargs["xtol"]=1E-7

    x_min = minimize(
      lambda x: lambda_inv*np.dot(direction, x)+np.maximum(func(x),0), x0=x1, method=method, options=kwargs
    ).x
    proj_min=x_min @ direction

    x_max = minimize(
      lambda x: -lambda_inv*np.dot(direction, x)+np.maximum(func(x),0), x0=x1, method=method, options=kwargs
    ).x
    proj_max = x_max @ direction

    return proj_min, proj_max, x_min, x_max

  # get confidence interval slice given a point and a direction
  def interval_slice(
    self,
    point: Sequence[float | np.float_] | np.ndarray,
    direction: Sequence[float | np.float_] | np.ndarray,
    xi_hat: Sequence[float | np.float_ | np.ndarray] | np.ndarray,
    Sigma: Sequence[float | np.float_ | np.ndarray] | np.ndarray,
    sub_dim: int | np.int_ | float | np.float_ | Sequence[int | np.int_ | float | np.float_ | np.ndarray] | np.ndarray,
    new_point: np.float_ | float | None=None,
    df: int | np.int_ | float | np.float_ | Sequence[int | np.int_ | float | np.float_] | np.ndarray | None = None,
    projs: Sequence[np.ndarray] | np.ndarray | None = None,
    format_input: bool = True,
    equal_sub_dim: bool | None = None,
    method: str = "brentq",
  ) -> tuple:
    # format point and direction
    point = np.array(point)
    direction = np.array(direction)

    # DONT normalize direction
    # direction = direction / np.sqrt((direction**2).sum())

    if new_point is None:
      new_point = point @ direction

    # by default we need to check if the variable inputs are illegal and if the sub dimensions in all studies are equal.
    if format_input:
      xi_hat, Sigma, sub_dim, df, projs, equal_sub_dim = self.__format_input(
        xi_hat, Sigma, sub_dim, df, projs
      )
    # need to check if the sub dimensions are equal as the functions would be slightly different due to tensorization (it is automatically done in format_input if format_input=True)
    elif equal_sub_dim is None:
      equal_sub_dim = self.__check_equal_sub_dim(xi_hat)

    # print(xi_hat.shape, Sigma.shape, sub_dim.shape, df.shape, projs.shape)
    func = lambda x: self.get_global_score(  # noqa: E731
      self.get_p_vector(
        point + (x - new_point) * direction,
        xi_hat,
        Sigma,
        sub_dim,
        df,
        projs,
        format_input=False,
        equal_sub_dim=equal_sub_dim,
      )
    )
    # get estimates of lower and upper bounds
    if equal_sub_dim:
      rebuild = np.squeeze(
        np.linalg.pinv(projs[sub_dim, :]) @ np.expand_dims(xi_hat, axis=-1), -1
      )
    else:
      rebuild = np.array(
        [np.linalg.pinv(projs[x, :]) @ y for x, y in zip(sub_dim, xi_hat)]
      )
    low = min((rebuild - point) @ direction)+new_point
    high = max((rebuild - point) @ direction)+new_point
    # print(low, high)

    # find minimizer of the func
    minimizer = minimize_scalar(func, method="Brent").x
    if low >= minimizer:
      low = minimizer - 1
    if high <= minimizer:
      high = minimizer + 1
    # print(minimizer, low, high)
    # deal with the case that the solution set is empty
    if func(minimizer) > self.threshold:
      print(f"{1-self.level} confidence interval is empty.")
      root_low = minimizer
      root_high = minimizer
    # when the solution is not empty
    else:
      if method == "brentq":
        # get correct lower and upper bounds
        low_1 = high_1 = minimizer
        while True:
          if func(low) < self.threshold:
            low_1 = low
            low = 2 * low - minimizer
          else:
            break
        while True:
          if func(high) < self.threshold:
            high_1 = high
            high = 2 * high - minimizer
          else:
            break
        # find the two roots
        root_low = root_scalar(
          lambda x: func(x) - self.threshold, bracket=[low, low_1], method="brentq"
        ).root
        root_high = root_scalar(
          lambda x: func(x) - self.threshold, bracket=[high_1, high], method="brentq"
        ).root
      else:
        raise NotImplementedError("The root finding method is not implemented.")

    return root_low, root_high, minimizer, direction

  # plot confidence area slice given a point and two directions
  def area_slice(
    self,
    point: Sequence[float | np.float_] | np.ndarray,
    direction_1: Sequence[float | np.float_] | np.ndarray,
    direction_2: Sequence[float | np.float_] | np.ndarray,
    xi_hat: Sequence[float | np.float_ | np.ndarray] | np.ndarray,
    Sigma: Sequence[float | np.float_ | np.ndarray] | np.ndarray,
    sub_dim: int | np.int_ | float | np.float_ | Sequence[int | np.int_ | float | np.float_ | np.ndarray] | np.ndarray,
    new_point_x: np.float_ | float | None=None,
    new_point_y: np.float_ | float | None=None,
    df: int | np.int_ | float | np.float_ | Sequence[int | np.int_ | float | np.float_] | np.ndarray | None = None,
    projs: Sequence[np.ndarray] | np.ndarray | None = None,
    format_input: bool = True,
    equal_sub_dim: bool | None = None,
    xrange: Sequence[float | np.float_] | np.ndarray | None = None,
    yrange: Sequence[float | np.float_] | np.ndarray | None = None,
    levels: Sequence[float] = [0.005, 0.01, 0.025, 0.05, 0.1],
    delta: float | None = None,
    ax: Any = None,
    clabel: bool = True,
    strs: Sequence[str] = None,
    **kwargs,
  ) -> Any:
    # format point and directions
    point = np.array(point)
    direction_1 = np.array(direction_1)
    direction_2 = np.array(direction_2)
    # directions DONT need to be orthonormal
    # direction_1 = direction_1 / np.sqrt((direction_1**2).sum())
    # direction_2 = direction_2 - (direction_2 @ direction_1) * direction_1
    # direction_2 = direction_2 / np.sqrt((direction_2**2).sum())

    if new_point_x is None:
      new_point_x = point @ direction_1
    if new_point_y is None:
      new_point_y = point @ direction_2

    # by default we need to check if the variable inputs are illegal and if the sub dimensions in all studies are equal.
    if format_input:
      xi_hat, Sigma, sub_dim, df, projs, equal_sub_dim = self.__format_input(
        xi_hat, Sigma, sub_dim, df, projs
      )
    # need to check if the sub dimensions are equal as the functions would be slightly different due to tensorization (it is automatically done in format_input if format_input=True)
    elif equal_sub_dim is None:
      equal_sub_dim = self.__check_equal_sub_dim(xi_hat)

    # if xrange or yrange is not given, we provide some suggestions of lower, upper and center of x and y based on observed data. but note that we do not automatically proceed with the plot after providing the information.
    if xrange is None or yrange is None:
      if equal_sub_dim:
        rebuild = np.squeeze(
          np.linalg.pinv(projs[sub_dim, :]) @ np.expand_dims(xi_hat, axis=-1), -1
        )
      else:
        rebuild = np.array(
          [np.linalg.pinv(projs[x, :]) @ y for x, y in zip(sub_dim, xi_hat)]
        )
      xlow = min((rebuild - point) @ direction_1)+new_point_x
      x0 = self.weights @ ((rebuild - point) @ direction_1)+new_point_x
      xhigh = max((rebuild - point) @ direction_1)+new_point_x
      if xlow >= x0:
        xlow = x0 - 1
      if xhigh <= x0:
        xhigh = x0 + 1
      ylow = min((rebuild - point) @ direction_2)+new_point_y
      y0 = self.weights @ ((rebuild - point) @ direction_2)+new_point_y
      yhigh = max((rebuild - point) @ direction_2)+new_point_y
      if ylow >= y0:
        ylow = y0 - 1
      if yhigh <= y0:
        yhigh = y0 + 1
      print(
        f"The centers of x, y are approximately {x0}, {y0}. Possible xrange: [{xlow}, {xhigh}]. Possible yrange: [{ylow}, {yhigh}]"
      )
    # otherwise, use the xrange and yrange provided by user
    else:
      xlow, xhigh = xrange
      ylow, yhigh = yrange

    # create meshgrid
    if delta is None:
      delta = min(xhigh - xlow, yhigh - ylow) / 100
    x_seq = np.arange(xlow, xhigh, delta)
    y_seq = np.arange(ylow, yhigh, delta)
    x_grid, y_grid = np.meshgrid(x_seq, y_seq)
    # print(x_grid, y_grid)
    # calculate global p_scores for the meshgrid
    func = lambda x, y: self.get_global_score(  # noqa: E731
      self.get_p_vector(
        point
        + np.expand_dims(x-new_point_x, axis=-1) * direction_1
        + np.expand_dims(y-new_point_y, axis=-1) * direction_2,
        xi_hat,
        Sigma,
        sub_dim,
        df,
        projs,
        format_input=False,
        equal_sub_dim=equal_sub_dim,
      )
    )
    z_grid = func(x_grid, y_grid)
    # print(x_grid.shape, y_grid.shape, z_grid.shape)

    # if the figure ax is given, use it. otherwise, create a new figure and return it later. but we should record this distinction.
    ax_given = True
    if ax is None:
      ax_given = False
      fig, ax = plt.subplots()
    # calculate score levels from p-value levels
    score_levels = [self.get_threshold_from_p(level) for level in levels]
    zipped = list(zip(score_levels, levels))
    zipped.sort(key=lambda x: x[0])
    score_levels, levels = list(zip(*zipped))
    # create contour plot
    CS = ax.contour(x_grid, y_grid, z_grid, score_levels, **kwargs)
    # if clabel, put labels onto the curves. if not, you can still add curve labels outside this function since we will return the figure and ax anyway.
    if clabel:
      fmt = {}
      # if strs is not provided to indicate levels, use the p-value levels as the default
      if strs is None:
        strs = [str(1 - level) for level in levels]
      # create fmt, a dictionary storing the formatting of labels
      for level, string in zip(CS.levels, strs):
        fmt[level] = string
      # create the curve labels
      ax.clabel(CS, fmt=fmt, inline=True)
    # if ax was given, returning CS is enough. otherwise, return the newly created fig and ax as well.
    if ax_given:
      return CS
    else:
      return fig, ax, CS
