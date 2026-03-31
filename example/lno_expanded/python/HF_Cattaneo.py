# import jax.numpy as np
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit, threading_layer, prange
import subprocess
import scipy
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, diags, issparse #Sparse matrix library
from scipy.sparse.linalg import bicgstab, spilu, LinearOperator, inv, spsolve, gmres # Sparse linear algebra library]
from scipy.interpolate import UnivariateSpline
import warnings
from PIL import Image

try:
	import cupy as cp
	import cupyx.scipy.sparse as cupy_sparse
	import cupyx.scipy.sparse.linalg as cupy_sparse_linalg
	_HAS_CUPY = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
	cp = None
	cupy_sparse = None
	cupy_sparse_linalg = None
	_HAS_CUPY = False

_MATERIAL_CACHE = {}
# Check SciPy version for GMRES API compatibility
_SCIPY_VERSION = tuple(map(int, scipy.__version__.split('.')[:2]))
_USE_RTOL = _SCIPY_VERSION >= (1, 12)  # rtol parameter introduced in SciPy 1.12
_DEFAULT_SOLVER_BACKEND = os.environ.get('HYPERION_FDM_BACKEND', 'auto').strip().lower()
def _load_material(file_id):
    """Load and cache material properties for file_id if not already cached"""
    if file_id not in _MATERIAL_CACHE:
        try:
            # Load heat capacity data
            hc_file = f'./NHC_{int(file_id)}.txt'
            hc_data = np.loadtxt(hc_file, delimiter=',')
            
            # Load thermal conductivity data
            tk_file = f'./NK_{int(file_id)}.txt'
            tk_data = np.loadtxt(tk_file, delimiter=',')
            
            # Create spline functions and pre-evaluate on a fine grid
            hc_spline = UnivariateSpline(hc_data[:, 0], hc_data[:, 1], s=0, k=2, ext=3)
            tk_spline = UnivariateSpline(tk_data[:, 0], tk_data[:, 1], s=0, k=2, ext=3)
            
            # Pre-evaluate on a fine temperature grid for faster lookups
            T_min = min(hc_data[0,0], tk_data[0,0])
            T_max = max(hc_data[-1,0], tk_data[-1,0])
            T_grid = np.linspace(T_min, T_max, 2000)  # Fine grid for accuracy
            hc_values = hc_spline(T_grid)
            tk_values = tk_spline(T_grid)
            
            # Store everything in cache
            _MATERIAL_CACHE[file_id] = {
                'hc_data': hc_data,
                'tk_data': tk_data,
                'hc_temps': hc_data[:, 0],
                'tk_temps': tk_data[:, 0],
                'hc_spline': hc_spline,
                'tk_spline': tk_spline,
                'T_grid': T_grid,      # Temperature grid for interpolation
                'hc_values': hc_values,  # Pre-evaluated heat capacity
                'tk_values': tk_values,  # Pre-evaluated thermal conductivity
                'T_min': T_min,        # Temperature range
                'T_max': T_max
            }
            print(f"Loaded material {file_id} into cache")
        except Exception as e:
            print(f"Error loading material {file_id}: {e}")
            raise
    
    return _MATERIAL_CACHE[file_id]

# Pre-load the most common material files (1, 2)
def preload_materials(material_ids=[1, 2]):
    """Preload all material files into cache"""
    print("Preloading material data...")
    for file_id in material_ids:
        _load_material(file_id)
    print("Material preloading complete")

# Replace heat_capacity_array function
def heat_capacity_array(file=1):
    '''
    Returns heat capacity array from cache
    '''
    material = _load_material(file)
    return material['hc_temps'], material['hc_data']

# Replace heat_capacity function
def heat_capacity(file=1):
	'''
	Calculates the heat capacity based on temperature using cached data.
	OPTIMIZED: Direct indexing instead of np.where() search.
	Returns the cached spline function.
	'''
	material = _load_material(file)
	return material['hc_spline']

def new_heat_cap_new_temp(T, file=1):
	'''
	Calculates the new heat capacity and temperature based on a reference temperature.
	Uses cached data to avoid file I/O.
	'''
	if T < 1e-6:
		return 0.0, 0.0
        
	# Get heat capacity at current temp using cached data
	material = _load_material(file)
	CV = material['hc_spline']
	
	# Get valid temperature range from material data
	T_min = material['hc_temps'][0]
	T_max = material['hc_temps'][-1]
	
	# Clamp TP to valid range
	T_clamped = np.clip(T, T_min, T_max)
	
	CP0 = CV(T_clamped)
	
	# Check if at boundaries
	if T_clamped >= T_max - 1e-6:
		# At upper boundary, use backward difference
		delta = 1e-3
		TPm = T_clamped - delta
		CPm = CV(TPm)
		G = (CP0 - CPm) / delta
		A = CP0
		return G, A
	elif T_clamped <= T_min + 1e-6:
		# At lower boundary, use forward difference
		delta = 1e-3
		TPp = T_clamped + delta
		CPp = CV(TPp)
		G = (CPp - CP0) / delta
		A = CP0
		return G, A
	
	# Normal case: use central difference
	delta = 1e-3
	TPm = T_clamped - delta
	TPp = T_clamped + delta
	CPm = CV(TPm)
	CPp = CV(TPp)

	# Linear interpolation to find G and A
	G = (CPp - CPm) / (2*(delta))
	A = CP0
	return G, A

# Replace thermal_conductivity_GA function
def thermal_conductivity_GA(file, TP, file2=None, TP2=None, boundary=False, B=None, BT=None):
	'''
	Calculate harmonic average thermal conductivity using pre-evaluated data.
	Uses linear interpolation from pre-computed values for speed.
	'''
	if TP < 1e-6:
		return 0.0

	# Get first material's data and interpolate TK1
	material1 = _load_material(file)
	T_grid = material1['T_grid']
	tk_values = material1['tk_values']
	
	# Fast linear interpolation for first conductivity
	slope = (len(T_grid)-1) / (material1['T_max'] - material1['T_min'])
	idx = int((TP - material1['T_min']) * slope)
	idx = max(0, min(len(T_grid)-2, idx))
	t = (TP - T_grid[idx]) / (T_grid[idx+1] - T_grid[idx])
	TK1_value = tk_values[idx] * (1-t) + tk_values[idx+1] * t
	
	if TK1_value == 0:
		return 0.0

	# Get second thermal conductivity based on case
	if boundary:
		if B is not None:
			TK2_value = B
		else:
			# Use first material's values for boundary temp
			if BT < 1e-6:
				return 0.0
			idx = int((BT - material1['T_min']) * slope)
			idx = max(0, min(len(T_grid)-2, idx))
			t = (BT - T_grid[idx]) / (T_grid[idx+1] - T_grid[idx])
			TK2_value = tk_values[idx] * (1-t) + tk_values[idx+1] * t
	else:
		# Handle interfaces
		TP2 = TP if TP2 is None else TP2
		if TP2 < 1e-6:
			return 0.0

		if file2 is None or file2 == file:
			# Same material - reuse first material's values
			idx = int((TP2 - material1['T_min']) * slope)
			idx = max(0, min(len(T_grid)-2, idx))
			t = (TP2 - T_grid[idx]) / (T_grid[idx+1] - T_grid[idx])
			TK2_value = tk_values[idx] * (1-t) + tk_values[idx+1] * t
		else:
			# Different material
			material2 = _load_material(file2)
			T_grid2 = material2['T_grid']
			tk_values2 = material2['tk_values']
			slope2 = (len(T_grid2)-1) / (material2['T_max'] - material2['T_min']) 
			idx = int((TP2 - material2['T_min']) * slope2)
			idx = max(0, min(len(T_grid2)-2, idx))
			t = (TP2 - T_grid2[idx]) / (T_grid2[idx+1] - T_grid2[idx])
			TK2_value = tk_values2[idx] * (1-t) + tk_values2[idx+1] * t

	if TK2_value == 0:
		return 0.0
	
	# Compute harmonic average
	return 2 * (TK1_value * TK2_value) / (TK1_value + TK2_value)
	

def sparse_matrix_1d(grid_size, grid, TP, B1, TB1,dx=1.0, const=False, k_temp_dependent=None):
	"""
	Constructs the sparse matrix for the 1D heat flow equation using a finite difference method.
	OPTIMIZED: Vectorized thermal conductivity calculations for massive speedup.	
	Parameters:
	-----------
	NY, NX : int
		Grid dimensions (rows, columns)
	dx : float
		Grid spacing
	boundary_conditions : dict, optional
		Boundary condition specifications
		
	Returns:
	--------
	scipy.sparse matrix in CSC format
	"""
	NY = grid_size
	NX = 1 
	N_total = NY * NX
	NZ = 1

	# Pre-allocate arrays for sparse matrix construction
	# Each interior node has 3 entries (left, center, right), boundaries have 2
	nnz_estimate = 3 * N_total
	row_indices_A = np.empty(nnz_estimate, dtype=np.int32)
	col_indices_A = np.empty(nnz_estimate, dtype=np.int32)
	values_A = np.empty(nnz_estimate, dtype=np.float64)
	BA = np.zeros(N_total, dtype=np.float64)
	
	coeff = 1.0 / (dx * dx)
	entry_count = 0
	
	# Pre-compute all thermal conductivities in batches
	# Determine whether thermal conductivity should be temperature dependent
	# Backwards compatible: if k_temp_dependent is None, fall back to 'not const'
	if k_temp_dependent is None:
		k_temp_dependent = not const

	# Reference temperatures for const (or disabled-k) case
	if not k_temp_dependent:
		TP_calc = np.full(NY, 200.0)
	else:
		TP_calc = TP.copy()
	
	# Vectorized computation of interface conductivities
	# For interior interfaces between nodes i and i+1
	interior_mask = np.arange(NY-1)
	TP1_interior = TP_calc[interior_mask]
	TP2_interior = TP_calc[interior_mask + 1]
	grid1_interior = grid[interior_mask]
	grid2_interior = grid[interior_mask + 1]
	
	# Batch compute interior conductivities
	K_interior = np.zeros(NY-1)
	unique_pairs = {}
	for i in range(NY-1):
		pair_key = (grid[i], grid[i+1])
		if pair_key not in unique_pairs:
			unique_pairs[pair_key] = []
		unique_pairs[pair_key].append(i)
	
	# Process each unique material pair
	for (file1, file2), indices in unique_pairs.items():
		indices = np.array(indices)
		T1_batch = TP_calc[indices]
		T2_batch = TP_calc[indices + 1]
		
		# Load materials once per pair
		material1 = _load_material(file1)
		if file2 == file1:
			material2 = material1
		else:
			material2 = _load_material(file2)
		
		# Vectorized interpolation for batch
		T_grid1 = material1['T_grid']
		tk_values1 = material1['tk_values']
		slope1 = (len(T_grid1)-1) / (material1['T_max'] - material1['T_min'])
		
		idx1 = ((T1_batch - material1['T_min']) * slope1).astype(int)
		idx1 = np.clip(idx1, 0, len(T_grid1)-2)
		t1 = (T1_batch - T_grid1[idx1]) / (T_grid1[idx1+1] - T_grid1[idx1])
		TK1_batch = tk_values1[idx1] * (1-t1) + tk_values1[idx1+1] * t1
		
		T_grid2 = material2['T_grid']
		tk_values2 = material2['tk_values']
		slope2 = (len(T_grid2)-1) / (material2['T_max'] - material2['T_min'])
		
		idx2 = ((T2_batch - material2['T_min']) * slope2).astype(int)
		idx2 = np.clip(idx2, 0, len(T_grid2)-2)
		t2 = (T2_batch - T_grid2[idx2]) / (T_grid2[idx2+1] - T_grid2[idx2])
		TK2_batch = tk_values2[idx2] * (1-t2) + tk_values2[idx2+1] * t2
		
		# Harmonic average (vectorized)
		valid = (TK1_batch > 0) & (TK2_batch > 0)
		K_batch = np.zeros_like(TK1_batch)
		K_batch[valid] = 2 * (TK1_batch[valid] * TK2_batch[valid]) / (TK1_batch[valid] + TK2_batch[valid])
		
		K_interior[indices] = K_batch
	
	# Compute boundary conductivities
	if not k_temp_dependent:
		K_top = thermal_conductivity_GA(grid[0], TP_calc[0], boundary=True, BT=200)
		K_bottom = thermal_conductivity_GA(grid[NY-1], TP_calc[NY-1], boundary=True, BT=200)
	else:
		K_top = thermal_conductivity_GA(grid[0], TP_calc[0], boundary=True, BT=TB1[0])
		K_bottom = thermal_conductivity_GA(grid[NY-1], TP_calc[NY-1], boundary=True, BT=TB1[1])
	
	# Build sparse matrix using pre-computed conductivities
	for ii in range(NY):
		idx = ii
		diagonal_val = 0.0
		
		# Up neighbor (i-1)
		if ii > 0:
			K_val = K_interior[ii-1] * coeff
			row_indices_A[entry_count] = idx
			col_indices_A[entry_count] = ii - 1
			values_A[entry_count] = K_val
			entry_count += 1
			diagonal_val -= K_val
		else:
			# Top boundary
			K_val = K_top * coeff
			diagonal_val -= K_val
			BA[idx] += K_val * TB1[0]
		
		# Down neighbor (i+1)
		if ii < NY - 1:
			K_val = K_interior[ii] * coeff
			row_indices_A[entry_count] = idx
			col_indices_A[entry_count] = ii + 1
			values_A[entry_count] = K_val
			entry_count += 1
			diagonal_val -= K_val
		else:
			# Bottom boundary
			K_val = K_bottom * coeff
			diagonal_val -= K_val
			BA[idx] += K_val * TB1[1]
		
		# Diagonal entry
		row_indices_A[entry_count] = idx
		col_indices_A[entry_count] = idx
		values_A[entry_count] = diagonal_val
		entry_count += 1
	
	# Trim arrays to actual size
	row_indices_A = row_indices_A[:entry_count]
	col_indices_A = col_indices_A[:entry_count]
	values_A = values_A[:entry_count]
	
	coo_A = coo_matrix((values_A, (row_indices_A, col_indices_A)), shape=(N_total, N_total), dtype=np.float64)
	matrix_A = coo_A.tocsc()

	return matrix_A, BA



def G_A_find(T, grid):
	'''
	Find G and A arrays based on temperature array TP.
	VECTORIZED: Process all temperatures at once for massive speedup.
	Parameters:
	T : ndarray of float, shape (N,)
		Temperature array.
	Returns:
	ndarray of float, shape (N,)
		G array.
	ndarray of float, shape (N,)
		A array.
	'''
	n = len(T)
	array_G = np.zeros(n)
	array_A = np.zeros(n)
	
	# Pre-load all unique materials
	unique_files = np.unique(grid)
	materials = {file: _load_material(file) for file in unique_files}
	
	delta = 1e-3
	
	# Process each unique material file
	for file_id in unique_files:
		# Get mask for all nodes with this material
		mask = grid == file_id
		T_nodes = T[mask]
		
		material = materials[file_id]
		T_min = material['T_min']
		T_max = material['T_max']
		T_grid = material['T_grid']
		hc_values = material['hc_values']
		
		# Clamp temperatures to valid range
		T_clamped = np.clip(T_nodes, T_min, T_max)
		
		# Vectorized linear interpolation for CP0 (A values)
		slope = (len(T_grid)-1) / (T_max - T_min)
		indices = ((T_clamped - T_min) * slope).astype(int)
		indices = np.clip(indices, 0, len(T_grid)-2)
		t = (T_clamped - T_grid[indices]) / (T_grid[indices+1] - T_grid[indices])
		CP0 = hc_values[indices] * (1-t) + hc_values[indices+1] * t
		
		# Handle boundary cases and compute gradients
		at_upper = T_clamped >= (T_max - 1e-6)
		at_lower = T_clamped <= (T_min + 1e-6)
		normal = ~(at_upper | at_lower)
		
		G_vals = np.zeros_like(T_clamped)
		
		# Upper boundary: backward difference
		if np.any(at_upper):
			T_minus = T_clamped[at_upper] - delta
			indices_m = ((T_minus - T_min) * slope).astype(int)
			indices_m = np.clip(indices_m, 0, len(T_grid)-2)
			t_m = (T_minus - T_grid[indices_m]) / (T_grid[indices_m+1] - T_grid[indices_m])
			CPm = hc_values[indices_m] * (1-t_m) + hc_values[indices_m+1] * t_m
			G_vals[at_upper] = (CP0[at_upper] - CPm) / delta
		
		# Lower boundary: forward difference
		if np.any(at_lower):
			T_plus = T_clamped[at_lower] + delta
			indices_p = ((T_plus - T_min) * slope).astype(int)
			indices_p = np.clip(indices_p, 0, len(T_grid)-2)
			t_p = (T_plus - T_grid[indices_p]) / (T_grid[indices_p+1] - T_grid[indices_p])
			CPp = hc_values[indices_p] * (1-t_p) + hc_values[indices_p+1] * t_p
			G_vals[at_lower] = (CPp - CP0[at_lower]) / delta
		
		# Normal case: central difference
		if np.any(normal):
			T_minus = T_clamped[normal] - delta
			T_plus = T_clamped[normal] + delta
			
			# Interpolate for T-delta
			indices_m = ((T_minus - T_min) * slope).astype(int)
			indices_m = np.clip(indices_m, 0, len(T_grid)-2)
			t_m = (T_minus - T_grid[indices_m]) / (T_grid[indices_m+1] - T_grid[indices_m])
			CPm = hc_values[indices_m] * (1-t_m) + hc_values[indices_m+1] * t_m
			
			# Interpolate for T+delta
			indices_p = ((T_plus - T_min) * slope).astype(int)
			indices_p = np.clip(indices_p, 0, len(T_grid)-2)
			t_p = (T_plus - T_grid[indices_p]) / (T_grid[indices_p+1] - T_grid[indices_p])
			CPp = hc_values[indices_p] * (1-t_p) + hc_values[indices_p+1] * t_p
			
			G_vals[normal] = (CPp - CPm) / (2*delta)
		
		# Store results back to output arrays
		array_G[mask] = G_vals
		array_A[mask] = CP0
	
	return array_G, array_A



def Gamma(G, A, TP, TPP, tau, dt):
	"""
	OPTIMIZED: Vectorized computation of Gamma coefficient.
	Parameters:
	- G : ndarray (dC_v/dT)
	- A : ndarray (C_v)
	- TP : T_{t-Δt}
	- TPP : T_{t-2Δt}
	- tau, dt : scalars
	
	Returns the coefficient matrix Γ for the Cattaneo equation.
	"""
	gamma_old = (TP - TPP) / dt  # ∂T/∂t at previous step
	dt_inv = 1.0 / dt
	dt2_inv = 1.0 / (dt * dt)
	
	# Vectorized computation (all operations are element-wise on arrays)
	return A * dt_inv + 2.0 * tau * G * gamma_old * dt_inv + tau * A * dt2_inv


def Omega(G, A, TP, TPP, tau, dt):
	'''
	OPTIMIZED: Vectorized Omega coefficient from Eq. (34)
	G = dC_v/dT
	A = C_v
	
	Returns: Ω 
	'''
	gamma_old = (TP - TPP) / dt
	dt_inv = 1.0 / dt
	dt2_inv = 1.0 / (dt * dt)
	
	# Vectorized computation
	term1 = -A * TP * dt_inv
	term2 = -2.0 * tau * G * gamma_old * TP * dt_inv
	term3 = -G * tau * (gamma_old * gamma_old)
	term4 = tau * A * (TPP - 2.0 * TP) * dt2_inv
	
	return term1 + term2 + term3 + term4

def nl_func_Cattaneo(T, TS, HA, A, G, BA, TP, TPP, dt=1, tau=1):
	'''
	Non-linear function for the heat flow equation.
	'''
	gamma = Gamma(G, A, TP, TPP, tau=tau, dt=dt)
	omega = Omega(G, A, TP, TPP, tau=tau, dt=dt)

	gamma = diags(gamma).tocsc()

	return (gamma-HA).dot(T) - (BA - omega)



def residual_F(T, TP, TPP, grid_size, grid, dx, dt, tau, const,
			   cache_key, const_cache, k_temp_dependent=None, c_temp_dependent=None, BC=(50, 50)):
	"""
	OPTIMIZED: Compute residual F(T) with caching for const=True cases.
	- T: current iterate (1D array)
	- TP, TPP: previous timesteps (1D arrays)
	"""
	NY = grid_size
	# boundary values used in sparse_matrix_1d 
	B1 = [BC[0], BC[1]]
	TB1 = [BC[0], BC[1]]  # Just use same temps for properties if not explicitly specified

	# Map new flags to maintain backwards compatibility with 'const'
	if k_temp_dependent is None:
		k_temp_dependent = not const
	if c_temp_dependent is None:
		c_temp_dependent = not const

	# get HA, BA, G, A depending on const flag (and independent k/c flags)
	if const:
		# cached tuple: (HA, BA) - HA/BA do not depend on heat capacity here
		if cache_key in const_cache:
			HA, BA = const_cache[cache_key]
		else:
			# Build HA/BA once. sparse_matrix_1d accepts k_temp_dependent flag so
			# boundaries/HA can be constant even if c (heat capacity) will be dynamic.
			HA, BA = sparse_matrix_1d(grid_size, grid, TP, B1, TB1, dx=dx, const=True, k_temp_dependent=k_temp_dependent)
			const_cache[cache_key] = (HA, BA)

		# Determine A and G depending on heat-capacity temperature dependence
		if c_temp_dependent:
			# A and G depend on current T
			G, A = G_A_find(T, grid)
		else:
			# Use pre-evaluated heat capacity at 200K (constant across time)
			G = np.zeros(NY, dtype=np.float64)
			A = np.empty(NY, dtype=np.float64)
			for i, file_id in enumerate(np.unique(grid)):
				material = _load_material(file_id)
				mask = grid == file_id
				# Fast interpolation at T=200
				T_ref = 200.0
				T_grid = material['T_grid']
				hc_values = material['hc_values']
				slope = (len(T_grid)-1) / (material['T_max'] - material['T_min'])
				idx = int((T_ref - material['T_min']) * slope)
				idx = max(0, min(len(T_grid)-2, idx))
				t = (T_ref - T_grid[idx]) / (T_grid[idx+1] - T_grid[idx])
				A_val = hc_values[idx] * (1-t) + hc_values[idx+1] * t
				A[mask] = A_val
	else:
		# dynamic: A,G depend on current T, and HA,BA assembled at current T
		# Build HA/BA at current temperatures. If k_temp_dependent is False,
		# sparse_matrix_1d will assemble HA using a reference temperature.
		HA, BA = sparse_matrix_1d(grid_size, grid, T, B1, TB1, dx=dx, k_temp_dependent=k_temp_dependent)
		if c_temp_dependent:
			G, A = G_A_find(T, grid)
		else:
			G = np.zeros(NY, dtype=np.float64)
			A = np.empty(NY, dtype=np.float64)
			for i, file_id in enumerate(np.unique(grid)):
				material = _load_material(file_id)
				mask = grid == file_id
				# Fast interpolation at T=200
				T_ref = 200.0
				T_grid = material['T_grid']
				hc_values = material['hc_values']
				slope = (len(T_grid)-1) / (material['T_max'] - material['T_min'])
				idx = int((T_ref - material['T_min']) * slope)
				idx = max(0, min(len(T_grid)-2, idx))
				t = (T_ref - T_grid[idx]) / (T_grid[idx+1] - T_grid[idx])
				A_val = hc_values[idx] * (1-t) + hc_values[idx+1] * t
				A[mask] = A_val

	# compute gamma and omega (nodewise) - vectorized
	gamma = Gamma(G, A, TP, TPP, tau=tau, dt=dt)
	omega = Omega(G, A, TP, TPP, tau=tau, dt=dt)

	# build residual: (diag(gamma) - HA) @ T - (BA - omega)
	Jdiag = diags(gamma, format='csc')
	R = (Jdiag - HA).dot(T) - (BA - omega)

	return R, gamma, HA, BA, A, G

# Global cache for const=True matrices (never change, compute once!)
_CONST_MATRIX_CACHE = {}
_CONST_PRECOND_CACHE = {}  # Cache for ILU preconditioner
_CONST_HEAT_CAPACITY_CACHE = {}
_LINEAR_SYSTEM_CACHE = {}


def _constant_heat_capacity(grid, T_ref=200.0):
	"""Return constant heat capacity vector evaluated once at T_ref."""
	key = (tuple(grid), float(T_ref))
	if key in _CONST_HEAT_CAPACITY_CACHE:
		return _CONST_HEAT_CAPACITY_CACHE[key]

	A = np.empty(len(grid), dtype=np.float64)
	for file_id in np.unique(grid):
		material = _load_material(file_id)
		mask = grid == file_id
		T_grid = material['T_grid']
		hc_values = material['hc_values']
		slope = (len(T_grid)-1) / (material['T_max'] - material['T_min'])
		idx = int((T_ref - material['T_min']) * slope)
		idx = max(0, min(len(T_grid)-2, idx))
		t = (T_ref - T_grid[idx]) / (T_grid[idx+1] - T_grid[idx])
		A_val = hc_values[idx] * (1-t) + hc_values[idx+1] * t
		A[mask] = A_val

	_CONST_HEAT_CAPACITY_CACHE[key] = A
	return A


def _supports_direct_linear_solve(const, k_temp_dependent, c_temp_dependent):
	return const and (not k_temp_dependent) and (not c_temp_dependent)


def _resolve_solver_backend(solver_backend, grid_size, direct_linear):
	backend = (_DEFAULT_SOLVER_BACKEND if solver_backend is None else solver_backend).lower()
	if backend not in ('auto', 'cpu', 'gpu', 'newton'):
		raise ValueError(f"Unsupported solver_backend='{solver_backend}'")
	if backend == 'newton':
		return 'newton'
	if backend == 'gpu':
		if not _HAS_CUPY:
			warnings.warn("GPU backend requested but CuPy CUDA sparse support is unavailable. Falling back to CPU.")
			return 'cpu'
		if not direct_linear:
			warnings.warn("GPU backend is implemented only for the constant-property direct solve path. Falling back to Newton-Krylov.")
			return 'newton'
		return 'gpu'
	if backend == 'cpu':
		return 'cpu'
	if direct_linear and _HAS_CUPY and grid_size >= 256:
		return 'gpu'
	return 'cpu' if direct_linear else 'newton'


def _normalize_tau(tau, grid_size):
	"""Return tau as a length-grid_size float64 vector and whether it is scalar-valued."""
	tau_arr = np.asarray(tau, dtype=np.float64)
	if tau_arr.ndim == 0:
		return np.full(grid_size, float(tau_arr), dtype=np.float64), True
	tau_arr = tau_arr.reshape(-1)
	if tau_arr.size != grid_size:
		raise ValueError(f"tau must be scalar or length {grid_size}, got shape {tau_arr.shape}")
	is_scalar = np.allclose(tau_arr, tau_arr[0])
	return tau_arr.copy(), bool(is_scalar)


def _solve_tridiagonal_numpy(lower, diag, upper, rhs):
	"""Thomas algorithm for one tridiagonal system on CPU."""
	n = diag.shape[0]
	if n == 1:
		return rhs / diag
	c_prime = np.empty(n - 1, dtype=np.float64)
	d_prime = np.empty(n, dtype=np.float64)
	beta = diag[0]
	c_prime[0] = upper[0] / beta
	d_prime[0] = rhs[0] / beta
	for i in range(1, n - 1):
		beta = diag[i] - lower[i - 1] * c_prime[i - 1]
		c_prime[i] = upper[i] / beta
		d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / beta
	beta = diag[n - 1] - lower[n - 2] * c_prime[n - 2]
	d_prime[n - 1] = (rhs[n - 1] - lower[n - 2] * d_prime[n - 2]) / beta
	x = np.empty(n, dtype=np.float64)
	x[n - 1] = d_prime[n - 1]
	for i in range(n - 2, -1, -1):
		x[i] = d_prime[i] - c_prime[i] * x[i + 1]
	return x


@njit(cache=True)
def _solve_tridiagonal_numpy_jit(lower, diag, upper, rhs, c_prime, d_prime, x):
	"""In-place Thomas algorithm workspace version for long CPU rollouts."""
	n = diag.shape[0]
	if n == 1:
		x[0] = rhs[0] / diag[0]
		return x
	beta = diag[0]
	c_prime[0] = upper[0] / beta
	d_prime[0] = rhs[0] / beta
	for i in range(1, n - 1):
		beta = diag[i] - lower[i - 1] * c_prime[i - 1]
		c_prime[i] = upper[i] / beta
		d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / beta
	beta = diag[n - 1] - lower[n - 2] * c_prime[n - 2]
	d_prime[n - 1] = (rhs[n - 1] - lower[n - 2] * d_prime[n - 2]) / beta
	x[n - 1] = d_prime[n - 1]
	for i in range(n - 2, -1, -1):
		x[i] = d_prime[i] - c_prime[i] * x[i + 1]
	return x


@njit(cache=True)
def _rollout_constant_property_cpu_numba(T0, lower, diag_base, upper, BA, A, tau_vec,
			dt, num_steps, save_every):
	"""JIT rollout for the constant-property CPU benchmark path."""
	grid_size = T0.shape[0]
	inv_dt = 1.0 / dt
	inv_dt2 = 1.0 / (dt * dt)
	n_saves = num_steps // save_every + 1
	history = np.empty((n_saves, grid_size), dtype=np.float64)
	history[0, :] = T0

	T = T0.copy()
	T_prev = T0.copy()
	diag = np.empty(grid_size, dtype=np.float64)
	rhs = np.empty(grid_size, dtype=np.float64)
	c_prime = np.empty(max(grid_size - 1, 1), dtype=np.float64)
	d_prime = np.empty(grid_size, dtype=np.float64)
	x = np.empty(grid_size, dtype=np.float64)
	save_idx = 1

	for step in range(num_steps):
		for i in range(grid_size):
			gamma = A[i] * (inv_dt + tau_vec[i] * inv_dt2)
			diag[i] = diag_base[i] + gamma
			omega = -A[i] * T[i] * inv_dt + tau_vec[i] * A[i] * (T_prev[i] - 2.0 * T[i]) * inv_dt2
			rhs[i] = BA[i] - omega
		T_new = _solve_tridiagonal_numpy_jit(lower, diag, upper, rhs, c_prime, d_prime, x)
		T_prev[:] = T
		T[:] = T_new
		if (step + 1) % save_every == 0:
			history[save_idx, :] = T
			save_idx += 1

	return history


def _solve_tridiagonal_cupy(lower, diag, upper, rhs):
	"""Thomas algorithm for one tridiagonal system on GPU."""
	n = diag.shape[0]
	if n == 1:
		return rhs / diag
	c_prime = cp.empty(n - 1, dtype=diag.dtype)
	d_prime = cp.empty(n, dtype=diag.dtype)
	beta = diag[0]
	c_prime[0] = upper[0] / beta
	d_prime[0] = rhs[0] / beta
	for i in range(1, n - 1):
		beta = diag[i] - lower[i - 1] * c_prime[i - 1]
		c_prime[i] = upper[i] / beta
		d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / beta
	beta = diag[n - 1] - lower[n - 2] * c_prime[n - 2]
	d_prime[n - 1] = (rhs[n - 1] - lower[n - 2] * d_prime[n - 2]) / beta
	x = cp.empty(n, dtype=diag.dtype)
	x[n - 1] = d_prime[n - 1]
	for i in range(n - 2, -1, -1):
		x[i] = d_prime[i] - c_prime[i] * x[i + 1]
	return x


def _build_direct_linear_system(grid_size, grid, dx, BC, backend):
	"""Build and cache constant-property tridiagonal coefficients."""
	key = (tuple(grid), int(grid_size), float(dx), tuple(float(v) for v in BC))
	cache_entry = _LINEAR_SYSTEM_CACHE.get(key)
	if cache_entry is not None:
		if backend != 'gpu' or 'lower_gpu' in cache_entry:
			return cache_entry

	HA, BA = sparse_matrix_1d(
		grid_size, grid, np.full(grid_size, 200.0, dtype=np.float64), BC, BC,
		dx=dx, const=True, k_temp_dependent=False,
	)
	A = _constant_heat_capacity(grid)
	lower = -HA.diagonal(-1).astype(np.float64, copy=True)
	upper = -HA.diagonal(1).astype(np.float64, copy=True)
	diag_base = (-HA.diagonal()).astype(np.float64, copy=True)

	cache_entry = {
		'BA': BA,
		'A': A,
		'lower': lower,
		'upper': upper,
		'diag_base': diag_base,
	}
	if backend == 'gpu':
		cache_entry['lower_gpu'] = cp.asarray(lower)
		cache_entry['upper_gpu'] = cp.asarray(upper)
		cache_entry['diag_base_gpu'] = cp.asarray(diag_base)
		cache_entry['BA_gpu'] = cp.asarray(BA)
		cache_entry['A_gpu'] = cp.asarray(A)
	_LINEAR_SYSTEM_CACHE[key] = cache_entry
	return cache_entry


def _solve_direct_linear_step(TP, TPP, dt, tau, system, backend):
	"""Solve one constant-property step without Newton iteration."""
	tau_vec, _ = _normalize_tau(tau, TP.shape[0])
	A = system['A']
	inv_dt = 1.0 / dt
	inv_dt2 = 1.0 / (dt * dt)
	gamma = A * (inv_dt + tau_vec * inv_dt2)
	diag = system['diag_base'] + gamma
	omega = -A * TP * inv_dt + tau_vec * A * (TPP - 2.0 * TP) * inv_dt2
	rhs = system['BA'] - omega
	if backend == 'gpu':
		tau_gpu = cp.asarray(tau_vec)
		A_gpu = system['A_gpu']
		TP_gpu = cp.asarray(TP)
		TPP_gpu = cp.asarray(TPP)
		diag_gpu = system['diag_base_gpu'] + A_gpu * (inv_dt + tau_gpu * inv_dt2)
		omega_gpu = -A_gpu * TP_gpu * inv_dt + tau_gpu * A_gpu * (TPP_gpu - 2.0 * TP_gpu) * inv_dt2
		rhs_gpu = system['BA_gpu'] - omega_gpu
		T_new = _solve_tridiagonal_cupy(system['lower_gpu'], diag_gpu, system['upper_gpu'], rhs_gpu)
		return cp.asnumpy(T_new)
	return _solve_tridiagonal_numpy(system['lower'], diag, system['upper'], rhs)


def rollout_constant_property_cpu(T0, dt, tau, num_steps, save_every, system):
	"""Fast JIT rollout for long constant-property CPU reference runs."""
	tau_vec, _ = _normalize_tau(tau, T0.shape[0])
	return _rollout_constant_property_cpu_numba(
		np.asarray(T0, dtype=np.float64),
		system['lower'],
		system['diag_base'],
		system['upper'],
		system['BA'],
		system['A'],
		tau_vec,
		float(dt),
		int(num_steps),
		int(save_every),
	)

def nl_solve_HF_1d_Cattaneo(grid_size, grid, TP, TPP,
	dx=1.0, dt=1.0, tau=0.0,
	tol=1e-8, max_newton_iters=500,
	gmres_tol=1e-6, gmres_maxit=1000,
	const=True, relax=1.0, verbose=False,
	row_eps=1e-16, fd_eps_scale=1e-6,
	k_temp_dependent=None, c_temp_dependent=None,
	BC=None, initial_guess=None, solver_backend=None):
	"""
	Newton-Krylov with automatic row-scaling and diagonal preconditioning.
	Returns: T_new, A_last, info
	"""
	NY = grid_size
	n = NY

	global _CONST_MATRIX_CACHE
	const_cache = _CONST_MATRIX_CACHE
	# Map new flags to defaults if not provided (backwards compatible with 'const')
	if k_temp_dependent is None:
		k_temp_dependent = not const
	if c_temp_dependent is None:
		c_temp_dependent = not const

	if BC is None:
		BC = (TP[0], TP[-1])

	direct_linear = _supports_direct_linear_solve(const, k_temp_dependent, c_temp_dependent)
	backend = _resolve_solver_backend(solver_backend, grid_size, direct_linear)
	if backend in ('cpu', 'gpu') and direct_linear:
		system = _build_direct_linear_system(grid_size, grid, dx, BC, backend)
		T_new = _solve_direct_linear_step(TP, TPP, dt, tau, system, backend)
		info = {
			"converged": True,
			"iters": 1,
			"reason": f"direct_{backend}",
		}
		return T_new, system['A'], info

	cache_key = (tuple(grid), dx, grid_size, dt, bool(k_temp_dependent), bool(c_temp_dependent), tuple(BC))

	if initial_guess is not None:
		T = initial_guess.copy()
	else:
		T = TP.copy()
	info = {"converged": False, "iters": 0}

	for newton_iter in range(1, max_newton_iters + 1):
		# compute residual and helpers at current T
		F, gamma, HA, BA, A, G = residual_F(T, TP, TPP, grid_size, grid, dx, dt, tau,
							const, cache_key, const_cache, k_temp_dependent=k_temp_dependent, c_temp_dependent=c_temp_dependent, BC=BC)

		Fnorm = np.linalg.norm(F)
		if verbose:
			print(f"[Newton {newton_iter}] ||F|| = {Fnorm:.6e}")

		if Fnorm < tol:
			info.update({"converged": True, "iters": newton_iter - 1, "reason": "residual_tol"})
			return T, A, info

		# Check for NaN values early
		if np.isnan(F).any() or np.isnan(T).any():
			warnings.warn("NaN detected in solution. Returning last valid state.")
			info.update({"converged": False, "iters": newton_iter, "reason": "nan_detected"})
			return TP.copy(), A, info

		# Build approximate Jacobian matrix P = diag(gamma) - HA  for scaling & preconditioning
		P_mat = diags(gamma).tocsc() - HA  


		# Row scaling vector s_i = max(|P_ii|, row_eps)
		diagP = P_mat.diagonal()
		row_scale = np.maximum(np.abs(diagP), row_eps)   # avoid zeros
		S_vec = 1.0 / row_scale                          # S = diag(1/row_scale)

		# Scaled residual
		F_scaled = S_vec * F   # elementwise scaling of rows
		F_scaled_norm = np.linalg.norm(F_scaled)
		
		if verbose:
			print(f"[Newton {newton_iter}] ||F_scaled|| = {F_scaled_norm:.6e}")

		# Check convergence on SCALED residual
		if F_scaled_norm < tol:
			info.update({"converged": True, "iters": newton_iter - 1, "reason": "residual_tol"})
		
			return T, A, info

		# Preconditioner on scaled matrix: try ILU on S * P_mat if possible
		# Build scaled P diagonal for simple preconditioner
		try:
			# Form scaled sparse P for spilu: multiply rows of P_mat by S_vec
			# Efficient row scaling: left-multiply by diag(S_vec)
			S_mat = diags(S_vec)
			P_scaled = (S_mat.dot(P_mat)).tocsc()
			ilu = spilu(P_scaled, drop_tol=1e-6, fill_factor=20)
			def msolve(x): return ilu.solve(x)
			M = LinearOperator((n, n), matvec=msolve)
			if verbose:
				print("  Preconditioner: spilu on scaled P built.")
		except Exception as e:
			# fallback to diagonal preconditioner (use diagonal of scaled P)
			diagP_scaled = S_vec * diagP
			diag_safe = np.where(np.abs(diagP_scaled) < 1e-30, 1e-30, diagP_scaled)
			invdiag = 1.0 / diag_safe
			M = LinearOperator((n, n), matvec=lambda x: invdiag * x)
			if verbose:
				print(f"  spilu on scaled P failed ({e}). Using diagonal preconditioner.")

			# Build matrix-free Jacobian-vector product for scaled Jacobian S * J
		# This captures the full temperature dependence via finite differences
		def Jv_scaled(v):
			vnorm = np.linalg.norm(v)
			if vnorm < 1e-14:
				return np.zeros_like(v)
			# Use a relative epsilon for better numerical accuracy
			# FIX: divide by vnorm directly so that eps * ||v|| is independent of tiny v
			eps = fd_eps_scale * max(1.0, np.linalg.norm(T)) / vnorm
			eps = max(eps, 1e-10)  # Ensure eps is not too small
			
			Tplus = T + eps * v
			Tminus = T - eps * v
			Fp, *_ = residual_F(Tplus, TP, TPP, grid_size, grid, dx, dt, tau, const, 
			                    cache_key, const_cache, k_temp_dependent=k_temp_dependent, 
			                    c_temp_dependent=c_temp_dependent, BC=BC)
			Fm, *_ = residual_F(Tminus, TP, TPP, grid_size, grid, dx, dt, tau, const, 
			                    cache_key, const_cache, k_temp_dependent=k_temp_dependent, 
			                    c_temp_dependent=c_temp_dependent, BC=BC)
			Jv_raw = (Fp - Fm) / (2.0 * eps)
			return S_vec * Jv_raw

		Jop = LinearOperator((n, n), matvec=Jv_scaled)

		# Solve scaled linear system (S J) delta = -S F using BiCGSTAB
		rhs = -F_scaled
		
		try:
			# Use previous delta as initial guess if available (warm start)
			x0 = np.zeros(n) if newton_iter == 1 else delta.copy() * 0.5
			
			# BiCGSTAB with matrix-free Jacobian
			if _USE_RTOL:
				delta, bicg_info = bicgstab(Jop, rhs, x0=x0, rtol=gmres_tol, 
				                            maxiter=gmres_maxit, M=M)
			else:
				delta, bicg_info = bicgstab(Jop, rhs, x0=x0, tol=gmres_tol, 
				                            maxiter=gmres_maxit, M=M)
			
			if bicg_info != 0:
				if verbose:
					print(f"  BiCGSTAB did not fully converge (info={bicg_info})")
				# Check if the result is still usable
				if np.isnan(delta).any() or np.linalg.norm(delta) > 1e10:
					raise ValueError("BiCGSTAB returned invalid delta")
					
		except Exception as e:
			if verbose:
				print(f"  BiCGSTAB failed: {e}. Trying GMRES.")
			# Try GMRES as fallback (sometimes more robust)
			try:
				if _USE_RTOL:
					delta, gmres_info = gmres(Jop, rhs, rtol=gmres_tol, maxiter=gmres_maxit, M=M)
				else:
					delta, gmres_info = gmres(Jop, rhs, tol=gmres_tol, maxiter=gmres_maxit, M=M)
			except Exception as e2:
				warnings.warn(f"Both BiCGSTAB and GMRES failed: {e2}. Using preconditioned gradient step.")
				# Use preconditioned steepest descent step as last resort
				delta = M.matvec(rhs) * 0.1

		# Check for NaN in delta
		if np.isnan(delta).any():
			warnings.warn("NaN in delta. Using preconditioned gradient step.")
			delta = M.matvec(rhs) * 0.01

		# Note: delta is the solution of (S J) delta = -S F, which is same delta as unscaled system
		# because we scaled rows only; update T
		dnorm = np.linalg.norm(delta)
		if verbose:
			print(f"  ||delta|| = {dnorm:.6e}")

		# backtracking line search on the scaled residual
		alphaP = relax
		T_candidate = T + alphaP * delta
		F_candidate, *_ = residual_F(T_candidate, TP, TPP, grid_size, grid, dx, dt, tau, const, cache_key, const_cache, k_temp_dependent=k_temp_dependent, c_temp_dependent=c_temp_dependent, BC=BC)
		F_candidate_scaled_norm = np.linalg.norm(S_vec * F_candidate)
		
		if F_candidate_scaled_norm < F_scaled_norm:
			T = T_candidate
			if verbose:
				print(f"  Accepted alpha={alphaP:.3g}, new ||F_scaled||={F_candidate_scaled_norm:.6e}")
		else:
			# backtrack until improvement
			accepted = False
			for bt in range(10):
				alphaP *= 0.5
				T_candidate = T + alphaP * delta
				F_candidate, *_ = residual_F(T_candidate, TP, TPP, grid_size, grid, dx, dt, tau, const, cache_key, const_cache, k_temp_dependent=k_temp_dependent, c_temp_dependent=c_temp_dependent, BC=BC)
				F_candidate_scaled_norm = np.linalg.norm(S_vec * F_candidate)
				if F_candidate_scaled_norm < F_scaled_norm:
					T = T_candidate
					accepted = True
					if verbose:
						print(f"  Backtrack: accepted alpha={alphaP:.3g}, new ||F_scaled||={F_candidate_scaled_norm:.6e}")
					break
			if not accepted:
				# if nothing helps, apply a small damped step to avoid stagnation
				T = T + 1e-3 * delta
				if verbose:
					print("  Backtrack failed: applied tiny step to avoid stagnation.")

		# optional termination on small step
		if np.linalg.norm(delta) < 1e-12:
			info.update({"converged": True, "iters": newton_iter, "reason": "delta_small"})
			# Stress = compute_stress(T, strain, grid)
			stress = None
			return T, A, info

	info.update({"converged": False, "iters": max_newton_iters, "reason": "max_iters"})
	return T, A, info

