from copy import copy

import numpy as np
from numba import jit
from scipy.integrate import RK45, DOP853, BDF
from scipy.linalg import lu_factor, lu_solve

import simulator_0d.src.py0D.global_data as gd
import simulator_0d.src.py0D.kinetics.perso_scipy_jac_modified as psjm

EPS = np.finfo(float).eps


class Euler_int():
    def __init__(self, func, t0, y0, tf, max_step = 1e3, func_dt = None, dt_add_val_dic = dict(dT_dtau_max=10, coeff_dt_species=0.1)):
        if func_dt is None:
            func_dt = self.dt_Euler
        self.t      = t0
        self.t0     = t0
        self.t_old  = 0
        self.tf     = tf

        self.y      = np.array(y0)
        self.y0     = np.array(y0)
        self.y_old  = np.array(0)

        self.func   = func
        self.func_dt= func_dt
        self.dt     = 0
        self.max_dt = max_step
        self.dt_add_val_dic = dt_add_val_dic
        self.status = "ongoing"
        # self.y0_shape = None


        

    def dt_Euler(self, *args, **kwargs):
        nb_points = 100
        return (self.tf-self.t0)/nb_points


    def step(self, dt = None, *args, **kwargs):
        if (self.func_dt is None and dt is None) or (self.func_dt is not None and dt is not None):
            raise ValueError("You must specify either a function to determine dt or give a dt")
        if self.func_dt is None:
            self.dt = dt
        else:
            self.dt = self.func_dt(self, dt_add_val_dic = self.dt_add_val_dic, *args, **kwargs)
        self.dt = np.min([self.dt, self.max_dt])
        self.t_old = self.t
        self.t = self.t_old + self.dt
        self.y_old = copy(self.y)
        self.y  = self.y_old + self.dt * self.func(self.t_old, self.y_old)

        if self.t>=self.tf:
            # self.y = self.y_old + (self.tf - self.t_old) * self.func(self.t_old, self.y_old)
            self.status = "finished"


def get_Y_from_y(y):
    Q = y[2:]
    raY = Q[gd.i_Y:]
    Y = (raY.T / Q[gd.i_d]).T
    return Y


def get_y_from_modified_Y(y, Y):
    Q = y[2:]
    raY = Q[gd.i_d] * Y
    y[(gd.i_Y+2):] = raY
    return y


# def adjust_Y_0(density, Y):
#     idx_s_small = np.argwhere(Y<1e-14)
#     C = (density * Y.T).T/gd.MW_gas_arr
#     C_el = np.zeros(gd.len_list_of_element)
#     adjusted_C = np.zeros(C.shape)
#     C_new = np.zeros(C.shape)
#     if len(idx_s_small)!=0:
#         C_el = np.sum(C[idx_s_small[0]] * gd.element_species_array[:, idx_s_small[0]], axis = 1)
#         species_maj_per_el_ratio_sup1_idx = np.empty(gd.len_list_of_element, dtype=int)
#         for i_el in range(gd.len_list_of_element):
#             arg_C_max_tmp = np.argmax(C[gd.el_ratio_sup1_list[i_el]])
#             species_maj_per_el_ratio_sup1_idx[i_el] = gd.el_ratio_sup1_list[i_el][arg_C_max_tmp]

#         el_species_mat = gd.element_species_array[:, species_maj_per_el_ratio_sup1_idx]
#         adjusted_C[species_maj_per_el_ratio_sup1_idx] = C_el @ np.linalg.inv(el_species_mat)
#         adjusted_C[idx_s_small[0]] = - C[idx_s_small[0]]
#     C_new = adjusted_C + C
#     Y_new = C_new * gd.MW_gas_arr / density
#     # Y_new = np.where((Y_new<0) & (Y_new>-1e-20), 0, Y_new)
#     return Y_new

def iterate(self, *args, **kwargs):
    self.t_values = [0]
    self.y_values = [self.y]
    self.vec_carac = [self.foo_vec_carac(self)]
    self.cpt = 0

    if isinstance(self, DOP853) or isinstance(self, RK45) or isinstance(self, BDF):
        while self.status != 'finished':
            self.cpt = self.cpt + 1
            if hasattr(self, "bool_modified"):
                self.step(self, *args, **kwargs)
            else:
                self.step(*args, **kwargs)

            Y           = get_Y_from_y(self.y)
            density     = self.y[gd.i_d+2]/self.y[1]
            Y_adjusted  = gd.adjust_Y_0(density, Y)
            self.y      = get_y_from_modified_Y(self.y, Y_adjusted)

            # if (self.y[(gd.i_Y+2):]<0).any():
            #     raise ValueError("mass fraction < 0 during BDF integration")
            self.get_hist(self)

    elif isinstance(self, Euler_int):
        while self.status != 'finished':
            self.cpt = self.cpt + 1
            self.step(y = self.y, *args, **kwargs)
            self.t_values.append(self.t)
            self.y_values.append(self.y)

            # print(self.cpt, "%.2e" %self.t_values[-1], "%.2e" %(self.t_values[-1] - self.t_values[-2]))
            if (self.y[(gd.i_Y+1):]<0).any():
                raise ValueError("Y<0")


def get_Q(self, y0_shape = None):
    y_values_arr = np.array(self.y_values)
    if hasattr(self, "y0_shape"):
        if self.y0_shape is None:
            Q = y_values_arr[:, 2:]
        else:
            y_unflat = np.reshape(y_values_arr, (y_values_arr.shape[0],) + self.y0_shape)
            Q = y_unflat[:, :, 2:]
    elif y_values_arr.shape[0]==1:

        Q = self.y[2:]
    else:
        Q = y_values_arr[:, 2:]

    return Q

def foo_vec_carac(self):
    y_rel = copy(self.y)
    y_rel[7:] = y_rel[7:] / y_rel[3] # energie
    y_rel[0] = self.y[0]/3000 # temperature
    y_rel[3] = self.y[3]/100 # densité
    y_rel[4] = 0 # QDMx
    y_rel[5] = 0 # QDMy
    y_rel[6] = 0 # energie
    return np.linalg.norm(y_rel)


def get_int_Q(self, *args, **kwargs):
    self.iterate(self, *args, **kwargs)
    return self.t_values, self.get_Q(self)


def _get_hist(self):
    self.t_values.append(self.t)
    self.y_values.append(self.y)
    self.vec_carac.append(self.foo_vec_carac(self))

def _not_get_hist(self):
    pass

def add_method(self):
    self.iterate        = iterate
    self.get_Q          = get_Q
    self.get_int_Q      = get_int_Q
    self.foo_vec_carac  = foo_vec_carac
    if self.bool_get_hist:
        self.get_hist = _get_hist
    else:
        self.get_hist = _not_get_hist


def add_attr(self, kwargs):
    for kwarg in kwargs:
        setattr(self, kwarg, kwargs[kwarg])



@jit
def fun_vectorized(fun, t, y, args):
    f = np.empty_like(y)
    for i, yi in enumerate(y.T):
        f[:, i] = fun(t, yi, *args)
    return f




def init_BDF(int_BDF, y0, dt, t0=0, fun_wo_arg=None, args=tuple()):
############################### init ode simplified #############################################""
    """from integrate._ivp.base"""
    int_BDF.t_old = None
    int_BDF.t = t0
    int_BDF.t_bound = dt
    int_BDF.y = y0
    int_BDF.status = 'running'
    int_BDF.first_step = 1e-15
    int_BDF.nfev = 0
    int_BDF.njev = 0
    int_BDF.nlu = 0
    int_BDF.bool_modified = True

    """
    Il faut remettre à jour la fonction car les arguments (*args)
    sont propre à une fonction donnée. Donc, si on ne met pas à jour
    la fonction pour le calcul suivant, ceux sont les arguments
    précédents qui sont utilisés
    """
        
    
    int_BDF.fargs = args


    if int_BDF.status != 'running':
        raise RuntimeError("Attempt to step on a failed or finished "
                            "solver.")



############################### init BDF simplified #############################################""

    f = fun_wo_arg(int_BDF.t, int_BDF.y, *int_BDF.fargs)
    int_BDF.h_abs = psjm.select_initial_step(fun_wo_arg, int_BDF.t, int_BDF.y, f,
                                        int_BDF.direction, 1,
                                        int_BDF.rtol, int_BDF.atol, int_BDF.fargs)

    int_BDF.h_abs_old = None
    int_BDF.error_norm_old = None

    if "fun_vectorized_nb" not in int_BDF.__dict__.keys():
        int_BDF.fun_vectorized_nb = fun_vectorized
        int_BDF.fun_single_nb = fun_wo_arg
        int_BDF.jac, int_BDF.J = _validate_jac_new(int_BDF, None, None)
        int_BDF._step_impl = psjm._step_impl
        int_BDF.step       = psjm.step
        # sparsity = np.ones(y0.shape)
        # sparsity[gd.i_d]
        # sparsity[gd.i_vx]
        # sparsity[gd.i_vy]
        # sparsity[gd.i_np]
    else:
        int_BDF.J = int_BDF.jac(int_BDF.t, int_BDF.y)

    # int_BDF.jac_factor = None

    int_BDF.jac_factor = np.where(int_BDF.jac_factor>5, EPS ** 0.5, int_BDF.jac_factor)



    def lu(A):
        int_BDF.nlu += 1
        return lu_factor(A, overwrite_a=True)


    def solve_lu(LU, b):
        return lu_solve(LU, b, overwrite_b=True)


    I = np.identity(int_BDF.n, dtype=int_BDF.y.dtype)

    int_BDF.lu = lu
    int_BDF.solve_lu = solve_lu
    int_BDF.I = I

    MAX_ORDER = 5
    kappa = np.array([0, -0.1850, -1/9, -0.0823, -0.0415, 0])
    int_BDF.gamma = np.hstack((0, np.cumsum(1 / np.arange(1, MAX_ORDER + 1))))
    int_BDF.alpha = (1 - kappa) * int_BDF.gamma
    int_BDF.error_const = kappa * int_BDF.gamma + 1 / np.arange(1, MAX_ORDER + 2)

    D = np.empty((MAX_ORDER + 3, int_BDF.n), dtype=int_BDF.y.dtype)
    D[0] = int_BDF.y
    D[1] = f * int_BDF.h_abs * int_BDF.direction
    int_BDF.D = D

    int_BDF.order = 1
    int_BDF.n_equal_steps = 0
    int_BDF.LU = None

    return int_BDF


def _validate_jac_new(int_BDF, jac, sparsity):
    t0 = int_BDF.t
    y0 = int_BDF.y

    if jac is None:
        def jac_wrapped(t, y):
            int_BDF.njev += 1
            f = int_BDF.fun_single_nb(t, y, *int_BDF.fargs)
            J, int_BDF.jac_factor = psjm.num_jac_nb(int_BDF.fun_vectorized_nb,int_BDF.fun_single_nb,
                                            t, y, f,
                                            int_BDF.atol, int_BDF.jac_factor,
                                            sparsity, int_BDF.fargs)
            # J, int_BDF.jac_factor = psjm.num_jac(int_BDF.fun_vectorized,
            #                                 t, y, f,
            #                                 int_BDF.atol, int_BDF.jac_factor,
            #                                 sparsity)
            return J
        J = jac_wrapped(t0, y0)

    return jac_wrapped, J

