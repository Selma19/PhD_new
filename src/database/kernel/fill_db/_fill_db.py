from typing import Literal, Dict, List, Union, Tuple
import numpy as np
from scipy.optimize import curve_fit
from sklearn import linear_model
from nautilus import Sampler, Prior
import json

from ...stimulus.loadData import load_list_block_data, get_list_block
from .utils import (
	Js_Mat_for_blocks, Dataset, exponential_Fred,
	vectorized_exp_Fred, exp_Fred_inv, vectorized_exp_Fred_inv, exp_Fred_jumps
)
from ...stimulus import Stimulus_db

__all__ = [
	"load_dataset", "crossVal", "fit_kernel",
	"evaluate", "read_kernel", "read_fit_fct", "load_fragments"
]

class Fit_model:
	"""Abstract base class for a model that will be tuned to approximate a kernel
	modulus as a function on [0, 1].

	Attributes
	----------
	out : Any
		stores the outcome of `self.fit` ;
		must be JSON serializable and contain enough information
		to recover the kernel function
	method : str
		stores the training method passed as input to `self.fit` ;
		used to recover the kernel function from `self.out`
	"""

	def __init__(self):
		self.out = None
		self.method = None

	def fit(self, kernel: np.ndarray, method: str):
		"""Fits `kernel` using the optimization algorithm specified by `method`.
		"""
		self.out = None
		self.method = method

	def read_kernel_from_out(self):
		"""Returns the complex-valued kernel as inferred from `self.out`,
		which has been set by the last call to `self.fit`.
		"""

class Fit_param1(Fit_model):
	r"""Defines a model used to approximate a kernel (list of 300 `float`)
	as a function defined on [0, 1].

	The model function writes:

	.. math:: f(t; \tau_1, \tau_2, \alpha, d, A) =
	   \begin{cases}
	   	0 \text{ if $t>d$}
	   	\\
	   	A \exp\left[-\frac{t}{\tau_1}\right]\left( 1 - np.exp\left[-\frac{t}{\tau_2}\right]
		\right)^{\alpha} \text{ else}
	   \end{cases}
	"""

	def __init__(self):
		super().__init__()
		self.param_names = ['tau1', 'tau2', 'alpha', 'd', 'A']

	def _fit_with_curve_fit(self, kernel: np.ndarray):
		popt, _ = curve_fit(
			exponential_Fred,
			np.linspace(0, 1, len(kernel)),
			np.abs(kernel),
			bounds =([1e-4, 1e-4, 0, 0, 0], [5, 5, 10, 1, 10])
		)
		self.out = dict(zip(self.param_names, popt))

	def _fit_with_nested_sampling(self, kernel: np.ndarray):
		# define the parameters we want to determine: tau1, tau2, alpha, d, A
		prior = Prior()
		prior.add_parameter('tau1', dist=(1e-4, 5))
		prior.add_parameter('tau2', dist=(1e-4, 5))
		prior.add_parameter('alpha', dist=(0, 5))
		prior.add_parameter('d', dist=(0, 1))
		prior.add_parameter('A', dist=(0, 10))

		# define the likelihood (vectorized for performance):
		# it is the fct to maximize, i.e. minus the error fct
		def vec_likelihood(dict_param: Dict[str, List[float]]):
			"""Keys should be the names of the parameters and each value is a `float`.
			"""
			params = [dict_param[key] for key in self.param_names]
			mat = vectorized_exp_Fred(np.linspace(0, 1, 300), *params)
			tgts = np.tile(
				np.abs(kernel).reshape(1, 300),
				(len(mat), 1)
			)
			return -np.sum( (mat - tgts) ** 2, axis=1 )

		# find the parameters that maximize the likelihood
		sampler = Sampler(prior, vec_likelihood, n_live=3000, vectorized=True, pass_dict=True)
		sampler.run(verbose=False, n_like_max=2e5)

		points, log_w, _ = sampler.posterior(return_as_dict=True)
		# change the data format to make self.out JSON serializable
		for key, value in points.items():
			points[key] = value.tolist()
		self.out = (points, log_w.tolist())

	def fit(self, kernel: np.ndarray, method: Literal['curve_fit', 'nested_sampling']):
		if method == 'curve_fit':
			self._fit_with_curve_fit(kernel)

		elif method == 'nested_sampling':
			self._fit_with_nested_sampling(kernel)
		
		else:
			raise ValueError("check the value of 'method'")
		self.method = method

	def read_kernel_from_out(self):
		if self.method == 'curve_fit':
			popt = [self.out[key] for key in self.param_names]

		elif self.method == 'nested_sampling':
			ind = np.argmax(self.out[1])
			popt = [self.out[0][key][ind] for key in self.param_names]

		return exponential_Fred(np.linspace(0, 1, 300), *popt)

class Fit_param2(Fit_model):
	r"""Defines a model used to approximate a kernel (list of 300 `float`)
	as a function defined on [0, 1].

	The model function writes:

	.. math:: f(t; \omega_1, \omega_2, \alpha, d, A, B_1, B_2) =
	   \begin{cases}
	   	B_1 \text{ if $t>d$}
	   	\\
	   	A \exp[-\omega_1 t]\left( 1 - np.exp[- \omega_2 t] \right)^{\alpha} + B_2 \text{ else}
	   \end{cases}
	"""

	def __init__(self):
		super().__init__()
		self.param_names = ['omega1', 'omega2', 'alpha', 'd', 'A', 'B1', 'B2']

	def _fit_with_curve_fit(self, kernel: np.ndarray):
		popt, _ = curve_fit(
			exp_Fred_inv,
			np.linspace(0, 1, len(kernel)),
			np.abs(kernel),
			bounds =([0, 0, 0, 0, 0, -10, -10], [100, 100, 10, 1, 10, 10, 10])
		)
		self.out = dict(zip(self.param_names, popt))

	def _fit_with_nested_sampling(self, kernel: np.ndarray):
		# define the parameters we want to determine: tau1, tau2, alpha, d, A
		prior = Prior()
		prior.add_parameter('omega1', dist=(0, 100))
		prior.add_parameter('omega2', dist=(0, 100))
		prior.add_parameter('alpha', dist=(0, 10))
		prior.add_parameter('d', dist=(0, 1))
		prior.add_parameter('A', dist=(0, 10))
		prior.add_parameter('B1', dist=(-10, 10))
		prior.add_parameter('B2', dist=(-10, 10))

		# define the likelihood (vectorized for performance):
		# it is the fct to maximize, i.e. minus the error fct
		def vec_likelihood(dict_param: Dict[str, List[float]]):
			"""Keys should be the names of the parameters and each value is a `float`.
			"""
			params = [dict_param[key] for key in self.param_names]
			mat = vectorized_exp_Fred_inv(np.linspace(0, 1, 300), *params)
			tgts = np.tile(
				np.abs(kernel).reshape(1, 300),
				(len(mat), 1)
			)
			return -np.sum( (mat - tgts) ** 2, axis=1 )

		# find the parameters that maximize the likelihood
		sampler = Sampler(prior, vec_likelihood, n_live=3000, vectorized=True, pass_dict=True)
		sampler.run(verbose=False, n_like_max=2e5)

		points, log_w, _ = sampler.posterior(return_as_dict=True)
		# change the data format to make self.out JSON serializable
		for key, value in points.items():
			points[key] = value.tolist()
		self.out = (points, log_w.tolist())

	def fit(self, kernel: np.ndarray, method: Literal['curve_fit', 'nested_sampling']):
		if method == 'curve_fit':
			self._fit_with_curve_fit(kernel)
		
		elif method == 'nested_sampling':
			self._fit_with_nested_sampling(kernel)

		else:
			raise ValueError("check the value of 'method'")
		self.method = method

	def read_kernel_from_out(self):
		if self.method == 'curve_fit':
			popt = [self.out[key] for key in self.param_names]

		elif self.method == 'nested_sampling':
			ind = np.argmax(self.out[1])
			popt = [self.out[0][key][ind] for key in self.param_names]

		return exp_Fred_inv(np.linspace(0, 1, 300), *popt)

class Fit_param3(Fit_model):
	r"""Defines a model used to approximate a kernel (list of 300 `float`)
	as a function defined on [0, 1].

	The implemented function is the same as `Fit_param2` but takes into
	account that the extracted kernels may show discontinuous jumps at first
	and last time steps.
	"""

	def __init__(self):
		super().__init__()
		self.param_names = [
			'omega1', 'omega2', 'alpha', 'd', 'A', 'V0', 'V1'
		]
	
	def _clean_kernel(self, kernel: np.ndarray):
		y = kernel[1:-1]
		diff_kernel = np.abs(y[1:] - y[:-1])

		# check whether there is significant noise
		if np.max(diff_kernel) / np.max(y) < 0.125:
			return kernel

		hist, bin_edges = np.histogram(diff_kernel, bins=20)
		for ind in range(1, len(hist)):
			if hist[ind] >= hist[ind - 1]:
				break
		threshold = (bin_edges[ind - 1] + bin_edges[ind]) / 2

		z = kernel
		diff_kernel = np.abs(z[1:] - z[:-1])
		indices = (diff_kernel >= threshold).nonzero()

		clean_kernel = z.copy()
		clean_kernel[indices] = 300
		clean_kernel[0] = z[0]
		clean_kernel[-1] = z[-1]
		return clean_kernel

	def _fitTrialCurve_fit(self, tab: np.ndarray):
		"""Cleans `tab` once and try to fit it."""
		indices = (tab < 250).nonzero()
		xdata = np.linspace(0, 1, len(tab))[indices]
		ydata = tab[indices]
		popt, _ = curve_fit(
			exp_Fred_jumps,
			xdata,
			ydata,
			p0=[10, 0.3, 1.3, 0, 8, 1, 1],
			bounds =([0, 0, 0, 0, 0, 0, 0], [100, 100, 10, 0.5, 10, 10, 10])
		)
		return popt

	def _fit_with_curve_fit(self, kernel: np.ndarray):
		n_trials = 5
		trial = 0

		tab = self._clean_kernel(np.abs(kernel))
		while trial < n_trials:
			try:
				popt = self._fitTrialCurve_fit(tab)
				break

			except RuntimeError:
				trial += 1
				tab = self._clean_kernel(tab)

		self.out = dict(zip(self.param_names, popt))

	def fit(self, kernel: np.ndarray, method: Literal['curve_fit', 'nested_sampling']):
		if method == 'curve_fit':
			self._fit_with_curve_fit(kernel)
		
		else:
			raise ValueError("check the value of 'method'")
		self.method = method

	def read_kernel_from_out(self):
		if self.method == 'curve_fit':
			popt = [self.out[key] for key in self.param_names]

		return exp_Fred_jumps(np.linspace(0, 1, 300), *popt)

class Kernel:
	"""Abstract base class that defines a kernel model.
	
	A kernel is characterized by a complex-valued function defined on [0, 1].
	It also defines a mapping between complex-valued time series, via
	the convolution product of the kernel function and the input time series.

	Notes
	-----
	Training a kernel means finding its kernel function that minimizes the gap
	between two times series:
	one is the target time series and the other is obtained by convolution of the kernel
	with an input time series.

	Attributes
	----------
	out : Any
		stores the outcome of `self.train` ;
		must be JSON serializable and contain enough information
		to recover the kernel function
	method : str
		stores the training method passed as input to `self.train` ;
		used to recover the kernel function from `self.out`
	"""

	def __init__(self):
		self.out = None
		self.method = None
	
	def _format_dataset(self, dataset: Dataset):
		"""Reorganizes the data contained in `dataset`
		so that it can be easily passed to any of the training methods of `self`.
		"""

	def train(self, train_set: Dataset, method: str, *args, **kwargs):
		"""Finds the best kernel function within the search space
		defined implicitly by `self`.
		Additional useful information may be computed and stored within `self.out`
		(e.g. uncertainty about the function found).
		However, make sure that `self.out` is JSON serializable.

		Notes
		-----
		The optional and keyword arguments may not be used by a child class.
		If they are, they correspond to parameters passed to the training method
		(e.g. the value of lambda in the case of Lasso or Ridge regression).
		"""
		self.out = None
		self.method = method

	def read_kernel_from_out(self):
		"""Returns the complex-valued kernel as inferred from `self.out`,
		which has been set by the last call to `self.train`.
		"""

class Raw_kernel(Kernel):
	"""The parameters of a raw kernel are exactly the 300 values its function
	takes on [0, 1].
	"""

	def _format_dataset(self, dataset: Dataset):
		Mat, Js = dataset
		n, p = Mat.shape
		Mat_real = np.zeros( (2 * n, 2 * p) )
		Mat_real[:n,:p] = np.real(Mat)
		Mat_real[:n,p:] = -np.imag(Mat)
		Mat_real[n:,:p] = np.imag(Mat)
		Mat_real[n:,p:] = np.real(Mat)
		Js_real = np.zeros(2 * n)
		Js_real[:n] = np.real(Js)
		Js_real[n:] = np.imag(Js)
		return Mat_real, Js_real

	def _train_lasso(self, train_set: Dataset, lamb: float):
		Mat, Js = self._format_dataset(train_set)
		
		clf = linear_model.Lasso(alpha=lamb, fit_intercept=True)
		clf.fit(Mat, Js)
		kernel = clf.coef_
		self.out = [kernel[:300].tolist(), kernel[300:].tolist()]
	
	def _train_ridge(self, train_set: Dataset, lamb: float):
		Mat, Js = self._format_dataset(train_set)
		
		clf = linear_model.Ridge(alpha=lamb, fit_intercept=True)
		clf.fit(Mat, Js)
		kernel = clf.coef_
		self.out = [kernel[:300].tolist(), kernel[300:].tolist()]

	def train(
		self,
		train_set: Dataset,
		method: Literal['linear_reg', 'lasso', 'ridge'],
		method_param: Union[str, Tuple[float, ...]]='no_param'
	):
		if method == 'linear_reg':
			Mat, Js = train_set
			x = np.conjugate(Mat.T)
			kernel = np.dot( np.dot(np.linalg.inv( np.dot(x, Mat) ), x), Js )
			self.out = [kernel.real.tolist(), kernel.imag.tolist()]

		elif method == 'lasso':
			self._train_lasso(train_set, method_param)

		elif method == 'ridge':
			self._train_ridge(train_set, method_param)
		
		else:
			raise ValueError("check the value of 'method'")
		self.method = method

	def read_kernel_from_out(self):
		return np.array(self.out[0]) + 1j * np.array(self.out[1])

class Param1_kernel(Kernel):
	"""The kernel function of this model is given by `utils.exponential_Fred`.

	During training, the search of the kernel function is restricted to the parameter space
	of `utils.exponential_Fred` (so a 5 dimensional space).
	"""

	def __init__(self):
		Kernel.__init__(self)
		self.param_names = ['tau1', 'tau2', 'alpha', 'd', 'A']

	def _format_dataset(self, dataset: Dataset):
		xdata, ydata = dataset
		xdata = np.concatenate( (xdata.real, xdata.imag), axis=1 )
		xdata = np.tile(xdata, (2, 1))
		ydata = np.concatenate( (ydata.real, ydata.imag) )
		return xdata, ydata

	def _train_curve_fit(self, train_set: Dataset):
		# set the dot and joystick complex time series at the correct format
		xdata, ydata = self._format_dataset(train_set)

		# define the function to pass to curve_fit
		def fct_to_fit(dot_vec, *params):
			"""The function whose parameters are fitted.
			
			It takes a vector of size `kernel_size` and returns a complex number.
			If `dot_vec` is of shape (n, kernel_size), then the result is of shape (n,).
			"""
			ker = exponential_Fred(np.linspace(0, 1, 300), *params)
			real_part = np.dot(dot_vec[:len(dot_vec) // 2, :300], ker)
			imag_part = np.dot(dot_vec[:len(dot_vec) // 2, 300:], ker)
			return np.concatenate( (real_part, imag_part) )

		popt, _ = curve_fit(
			fct_to_fit,
			xdata,
			ydata,
			p0=[1] * 5,
			bounds =([1e-4, 1e-4, 0, 0, 0], [5, 5, 10, 1, 10])
		)
		self.out = dict(zip(self.param_names, popt))

	def _train_NS(self, train_set: Dataset):
		# define the parameters we want to determine: tau1, tau2, alpha, d, A
		prior = Prior()
		prior.add_parameter('tau1', dist=(1e-4, 5))
		prior.add_parameter('tau2', dist=(1e-4, 5))
		prior.add_parameter('alpha', dist=(0, 5))
		prior.add_parameter('d', dist=(0, 1))
		prior.add_parameter('A', dist=(0, 10))

		# define the likelihood (vectorized for performance):
		# it is the fct to maximize, i.e. minus the error fct
		# first we need to define the model, i.e. the mapping from
		# the xdata (dot direction) to the ydata (joystick direction)
		def vec_fct(dot_vec, *params):
			"""The function whose parameters are fitted, in a vectorized version.
			
			It takes a vector of size `kernel_size` and returns a complex number.
			If `dot_vec` is of shape (n, kernel_size), then the result is of shape (n_params, n),
			where n_params if the length of any element of `params`.
			"""
			ker = vectorized_exp_Fred(np.linspace(0, 1, 300), *params).T

			real_part = np.dot(dot_vec[:len(dot_vec) // 2, :300], ker)
			imag_part = np.dot(dot_vec[:len(dot_vec) // 2, 300:], ker)
			return np.concatenate( (real_part, imag_part) ).T

		# second we need to build the xdata and ydata
		xdata, ydata = self._format_dataset(train_set)

		targets = np.tile( ydata.reshape(1, -1), (100, 1) )
		def vec_likelihood(dict_param: Dict[str, List[float]]):
			"""Keys should be the names of the parameters and each value is a `float`.
			"""
			params = [dict_param[key] for key in self.param_names]
			mat = vec_fct(xdata, *params)
			return -np.sum( (mat - targets) ** 2, axis=1 )

		# find the parameters that maximize the likelihood
		sampler = Sampler(
			prior, vec_likelihood,
			n_live=3000, vectorized=True, pass_dict=True
		)
		sampler.run(verbose=False, n_like_max=2e5)

		points, log_w, _ = sampler.posterior(return_as_dict=True)
		# change the data format to make self.out JSON serializable
		for key, value in points.items():
			points[key] = value.tolist()
		self.out = (points, log_w.tolist())

	def train(
		self,
		train_set: Dataset,
		method: Literal['curve_fit', 'nested_sampling'],
		*args, **kwargs
	):
		if method == 'curve_fit':
			self._train_curve_fit(train_set)

		elif method == 'nested_sampling':
			self._train_NS(train_set)
		
		else:
			raise ValueError("check the value of 'method'")
		self.method = method

	def read_kernel_from_out(self):
		if self.method == 'curve_fit':
			popt = [self.out[key] for key in self.param_names]

		elif self.method == 'nested_sampling':
			ind = np.argmax(self.out[1])
			popt = [self.out[0][key][ind] for key in self.param_names]

		return exponential_Fred(np.linspace(0, 1, 300), *popt)

def _filter_remove_after(
	joystick: List[complex],
	dot: List[complex],
	ts: List[float],
	nom_ts: List[float],
	tgt_ts: List[float]
):
	"""Splits the time series `joystick` and `dot` into
	a list of time series following the rules:
	- a time series ends when a target appears
	- after it has ended, a new time series starts at a change in nominal direction
	"""
	joysticks_ = []
	dots_ = []

	# the time steps that have been browsed so far range from 0 to ind
	ind = 0

	# the target times of appearance that have been browsed so far range from 0 to ind_tgt
	ind_tgt = 0

	# the times of change in nominal direction that have been browsed
	# so far range from 0 to ind_nom
	ind_nom = 0

	while True:
		# this current time is the starting time of the next time series
		current_time = ts[ind]

		# time of appearance of the next target (if any)
		next_tgt = None
		for num, t in enumerate(tgt_ts[ind_tgt:], start=ind_tgt):
			if t > current_time:
				next_tgt = t
				ind_tgt = num
				break

		# if there is no target anymore, then no more splits
		if next_tgt is None:
			joysticks_.append(joystick[ind:])
			dots_.append(dot[ind:])
			break

		# the time series spans from current_time to next_tgt
		# (it stops just before the target appears)
		start = ind
		for num, t in enumerate(ts[ind:], start=ind):
			if t >= next_tgt:
				end = num
				break
		joysticks_.append(joystick[start: end])
		dots_.append(dot[start: end])
		current_time = ts[end]

		# now we determine the starting time of the next time series, if any:
		# this is the time of the next change in nominal direction
		# (it starts just after the change)
		next_nom = None
		for num, t in enumerate(nom_ts[ind_nom:], start=ind_nom):
			if t >= current_time:
				next_nom = t
				ind_nom = num
				break

		# if there is no change anymore, then no more splits
		if next_nom is None:
			joysticks_.append(joystick[end:])
			dots_.append(dot[end:])
			break
		
		# the new index in ts corresponds to the time of change in nominal direction
		for num, t in enumerate(ts[end:], start=end):
			if t >= next_nom:
				ind = num
				break
	return joysticks_, dots_

def load_fragments(
	agent: str,
	coh: float,
	min_length: int
):
	"""Returns the fragments of the joystick and mean dot direction time series
	after the filtering 'remove_after_tgt'.

	Returns
	-------
	joystick_list : List[List]
		list of the fragments (a fragment is a time series over a time interval)
		of length greater than or equal to `min_length`, of the joystick time series
	dot_list : List[List]
		same as `joystick_list`, but for the mean dot direction time series
	"""
	# connect to the stimulus db
	db = Stimulus_db()
	db.connect()

	# get all unique couples (xp, block)
	xp_blocks = db.cur.execute("""
		SELECT DISTINCT xp, block FROM Main
	""").fetchall()

	# for each block and xp, load the joystick, dot, time steps, nominal angle
	# and target time steps data (pretty much everything)
	# then filter the data before accumulating them
	joystick_list = []; dot_list = []
	for xp, block in xp_blocks:
		row = db.cur.execute("""
			SELECT
				times,
				nominal_angle,
				target_times,
				joystick,
				mean_dot
			FROM
				Main
			WHERE
				agent = ?
				AND coh = ?
				AND xp = ?
				AND block = ?
		""", (agent, coh, xp, block)).fetchone()
		if row is not None:
			# the time steps at which the joystick and dot data are sampled
			ts = json.loads(row[0])

			# the exact times at which a change in nominal direction occurs
			nom_ts, _ = zip(*json.loads(row[1]))

			tgt_ts = json.loads(row[2])

			ecc, joy_angle = json.loads(row[3])
			joystick = np.array(ecc) * np.exp(1j * np.array(joy_angle))

			real_part, imag_part = json.loads(row[4])
			dot = np.array(real_part) + 1j * np.array(imag_part)

			# filter the dot and joystick time series
			joysticks_, dots_ = _filter_remove_after(
				joystick, dot, ts, nom_ts, tgt_ts
			)

			# accumulate the filtered data
			joystick_list.extend(joysticks_)
			dot_list.extend(dots_)

	# think about disconnecting the stimulus db
	db.close()

	# keep only the time series that are long enough
	joystick_list = [el for el in joystick_list if len(el) >= min_length]
	dot_list = [el for el in dot_list if len(el) >= min_length]
	return joystick_list, dot_list

def _load_dataset_remove_after(
	agent: str,
	coh: float
) -> Dataset:
	"""Returns the dataset loaded from the stimulus db
	for the filtering option 'remove_after_tgt'.
	"""
	joystick_list, dot_list = load_fragments(agent, coh, min_length=300)
	return Js_Mat_for_blocks(joystick_list, dot_list, 300)

def load_dataset(
	agent: str,
	coh: float,
	filtering_method: Literal['unfiltered', 'remove_after_tgt']
) -> Dataset:
	"""Returns in this order the dot and joystick directions under a canonical matrix form.

	Notes
	-----
	The parameters correspond to the columns of the `Main` table
	of the kernel database.
	
	Parameters
	----------
	agent : str
		name of the agent
	coh : float
		coherence value of the dot signal
	filtering_method : Literal['unfiltered', 'remove_after_tgt']
		how the dot and joystick direction are filtered ;
		if 'remove_after_tgt', the time steps following the apperance of a target are removed
	"""
	if filtering_method == 'unfiltered':
		db = Stimulus_db()
		db.connect()
		raw_signal = db.cur.execute("""
			SELECT joystick, mean_dot
			FROM Main
			WHERE
				coh = ?
				AND agent = ?
		""", (coh, agent)).fetchall()
		db.close()

		J_list = []
		D_list = []
		for joy_raw, dot_raw in raw_signal:
			tab = np.array(json.loads(joy_raw))
			J_list.append(
				tab[0, :] * np.exp(1j * tab[1, :])
			)

			tab = np.array(json.loads(dot_raw))
			D_list.append(
				tab[0, :] + 1j * tab[1, :]
			)

		return Js_Mat_for_blocks(J_list, D_list, 300)

	elif filtering_method == 'remove_after_tgt':
		return _load_dataset_remove_after(agent, coh)

	else:
		raise ValueError("check the value of 'filtering_method'")

def read_kernel(
	kernel_output,
	kernel_type: Literal['raw', 'param1'],
	kernel_method: Literal[
		'curve_fit', 'nested_sampling',
		'lasso', 'ridge', 'linear_reg', 'elastic'
	]
):
	"""Returns the kernel fct (its 300 complex values) as a 1d numpy.ndarray.
	"""
	if kernel_type == 'raw':
		model = Raw_kernel()
	elif kernel_type == 'param1':
		model = Param1_kernel()
	else:
		raise ValueError("check the value of 'kernel_type'")
	model.out = kernel_output
	model.method = kernel_method
	return model.read_kernel_from_out()

def read_fit_fct(
	fit_output,
	fit_type: Literal['param2', 'param3'],
	fit_method: Literal['curve_fit', 'nested_sampling']
):
	"""Returns the kernel fct (its 300 complex values) as a 1d numpy.ndarray.
	"""
	if fit_type == 'param2':
		model = Fit_param2()
	elif fit_type == 'param3':
		model = Fit_param3()
	else:
		raise ValueError("check the value of 'fit_type'")
	model.out = fit_output
	model.method = fit_method
	return model.read_kernel_from_out()

def evaluate(kernel: np.ndarray, dataset: Dataset) -> float:
	"""Computes the error between a joystick direction time series
	and the convolution of `kernel` with a dot direction time series.
	The two time series are contained in `dataset`.
	"""
	Mat, Js = dataset
	return np.sum(np.abs(Js - np.dot(Mat, kernel)) ** 2) / len(Js)

def fit_kernel(
	kernel: np.ndarray,
	fit_type: Literal['param1', 'param2', 'param3'],
	fit_method: Literal['curve_fit', 'nested_sampling']
):
	"""Approximates the modulus of `kernel` according to a model and fit method.
	This model defines a function on [0, 1] that is supposed to interpolate the kernel
	values.
	
	Parameters
	----------
	kernel : numpy.ndarray
		the kernel values, either in real or complex form
	fit_type : Literal
		the model used to approximate the kernel
	fit_method : Literal
		the method used to fit the model
	"""
	if fit_type == 'param1':
		model = Fit_param1()
	elif fit_type == 'param2':
		model = Fit_param2()
	elif fit_type == 'param3':
		model = Fit_param3()
	else:
		raise ValueError("check the value of 'fit_type'")

	model.fit(kernel, fit_method)
	kernel_fct = model.read_kernel_from_out()
	return model.out, kernel_fct

def splitData(dataset: Dataset, testRatio: float) -> List[Tuple[Dataset, Dataset]]:
	"""Returns a list of pairs (train, test) data.
	
	Parameters
	----------
	testRatio : float
		fraction of the number of samples contained in each test set
	"""
	Mat, Js = dataset
	res = []
	n = len(Js)
	test_size = int(n * testRatio)

	for start in range(0, n, test_size):
		# build full window with wrap-around
		window = [(start + i) % n for i in range(n)]

		train_indices = window[:test_size]
		test_indices = window[test_size:]

		train_set = (Mat[train_indices], Js[train_indices])
		test_set = (Mat[test_indices], Js[test_indices])
		res.append((train_set, test_set))
	return res

def crossVal(
	dataset: Dataset,
	kernel_type: Literal['raw', 'param1'],
	kernel_method: Literal[
		'curve_fit', 'nested_sampling',
		'lasso', 'ridge', 'linear_reg', 'elastic'
	],
	method_param: Union[str, Tuple[float, ...]]='no_param'
):
	"""Finds the kernel among the space defined by `kernel_type`,
	that leads to the minimum error on `train_set`.
	The method used to find the optimal kernel is defined by `kernel_method`.
	
	K-fold cross-validation is used to estimate the test error.
	The model parameters returned are those which minimize the sum of the train and test
	errors across all splits of the dataset.
	"""
	test_ratio = 0.3
	kernel_outputs = []
	train_errors = []
	test_errors = []

	# first define a model to be trained
	if kernel_type == 'raw':
		model = Raw_kernel()
	elif kernel_type == 'param1':
		model = Param1_kernel()
	else:
		raise ValueError("check the value of 'kernel_type'")

	# split the dataset into a training and test sets
	for train_set, test_set in splitData(dataset, test_ratio):
		# train the model
		model.train(train_set, kernel_method, method_param)

		# read the kernel
		kernel = model.read_kernel_from_out()

		# compute the train and test errors
		train_error = evaluate(kernel, train_set)
		test_error = evaluate(kernel, test_set)

		# collect data
		kernel_outputs.append(model.out)
		train_errors.append(train_error)
		test_errors.append(test_error)

	tab = zip(kernel_outputs, [x + y for x, y in zip(train_errors, test_errors)])
	kernel_output = min(tab, key=lambda el: el[1])[0]
	return kernel_output, np.mean(train_error), np.mean(test_error)
