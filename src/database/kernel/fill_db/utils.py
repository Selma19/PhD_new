"""Gathers various helper functions and types.
"""

from typing import List
import numpy as np

class Dataset:
	"""The type returned by `_fill_db.load_dataset`.
	It is actually a Tuple of 2 complex-valued matrices.
	The couple corresponds resp. to the dot and joystick directions.
	"""

def D_m_matrix_single_block(D_m, kernelSize: int):
	mat = np.zeros((len(D_m), kernelSize), dtype=complex)

	# fill the first half
	a, b = np.ogrid[0:kernelSize, 0:-kernelSize:-1]
	mat[:kernelSize, :] = np.tril(D_m[a + b])

	# fill the second half
	a, b = np.ogrid[1:len(D_m)-kernelSize+1, kernelSize-1:-1:-1]
	mat[kernelSize:, :] = D_m[a + b]
	return mat

def Js_Mat_for_blocks(joystick_list, dot_list, kernel_size):
	list_mat = [
		D_m_matrix_single_block(dot, kernel_size)
		for dot in dot_list
	]
	return np.vstack(list_mat), np.hstack(joystick_list)

def _expFredFct(x, tau1, tau2, alpha):
	return np.exp(-x / tau1) * (1 - np.exp(- x / tau2)) ** alpha

def exponential_Fred(x: np.ndarray, tau1: float, tau2: float, alpha: float, d: float, A: float):
	# indices where x > d
	indices = np.asarray(x > d).nonzero()

	res = np.zeros_like(x)
	res[indices] = A * _expFredFct(x[indices] - d, tau1, tau2, alpha)
	return res

def vectorized_exp_Fred(
	x: List[float],
	*args
):
	"""Same as `exponential_Fred` but evaluated for multiple parameters at once.
	
	So returns a matrix of shape (len(tau1), len(x)).
	Note that every array of params (tau1, alpha, etc) should have the same length.

	Parameters
	----------
	tau1 : List[float]
	tau2 : List[float]
	alpha : List[float]
	d : List[float]
	A : List[float]
	"""
	xx = np.tile(
		x.reshape(1, len(x)), (len(args[0]), 1)
	)
	args = [
		np.tile(
			arg.reshape(len(arg), 1), (1, len(x))
		) for arg in args
	]
	# indices where x > d
	indices = np.asarray(xx > args[3]).nonzero()
	res = np.zeros_like(xx)
	
	xx = xx[indices]
	args = [arg[indices] for arg in args]

	res[indices] = args[4] * _expFredFct(xx - args[3], *args[:3])
	return res

def _expFredFctInv(
	x: np.ndarray,
	omega1: float,
	omega2: float,
	alpha: float
):
	return np.exp(-x * omega1) * ( 1 - np.exp(- x * omega2) ) ** alpha

def exp_Fred_inv(x: np.ndarray,
	omega1: float, omega2: float,
	alpha: float, d: float, A: float, B1: float, B2: float
):
	# indices where x > d
	indices = np.asarray(x > d).nonzero()

	res = np.zeros_like(x) + B1
	res[indices] = A * _expFredFctInv(x[indices] - d, omega1, omega2, alpha) + B2
	return res

def vectorized_exp_Fred_inv(
	x: List[float],
	*args
):
	"""Same as `exp_Fred_inv` but evaluated for multiple parameters at once.
	
	So returns a matrix of shape (len(omega1), len(x)).
	Note that every array of params (omega1, alpha, etc) should have the same length.

	Parameters
	----------
	omega1 : List[float]
	omega2 : List[float]
	alpha : List[float]
	d : List[float]
	A : List[float]
	B1 : List[float]
	B2 : List[float]
	"""
	xx = np.tile(
		x.reshape(1, len(x)), (len(args[0]), 1)
	)
	args = [
		np.tile(
			arg.reshape(len(arg), 1), (1, len(x))
		) for arg in args
	]
	# indices where x > d
	indices = np.asarray(xx > args[3]).nonzero()
	res = np.zeros_like(xx) + args[5]
	
	xx = xx[indices]
	args = [arg[indices] for arg in args]

	res[indices] = args[4] * _expFredFctInv(xx - args[3], *args[:3]) + args[6]
	return res

def exp_Fred_jumps(
	x: np.ndarray,
	omega1: float, omega2: float,
	alpha: float, d: float, A: float,
	V0: float, V1: float
):
	# indices where x > d
	indices = np.asarray(x > d).nonzero()

	res = np.zeros_like(x)
	res[indices] = A * _expFredFctInv(x[indices] - d, omega1, omega2, alpha)

	# values at the first and last time steps are overwritten (jumps)
	res[0] = V0
	res[-1] = V1

	return res
