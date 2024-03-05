"""Model with one lambda parameters and stickiness. Stickiness is implemented with as (s_out + 1) 
without any transformation on s_out + 1 such as exponentiation or sigmoid.
"""
from typing import Tuple, Union
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np
import pandas as pd
from pandas import DataFrame
from func import *

Parameters = Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]

def model(n_loci:int, diag_start:int):
    """Returns needed functions to fit the model on a hic matrix of a single chromosome.
    """

    def parameter_transformation(parameters:Parameters) -> Parameters:
        def sigmoid(x):
            return 1/(1+np.exp(-x))
        return (parameters[0],
                sigmoid(parameters[1]),
                sigmoid(parameters[2]),
                sigmoid(parameters[3]),
                sigmoid(parameters[4]),
                parameters[5]
                )

    def write_parameters(path:str,
                         parameters:Parameters,
                         bin_size:int,
                         chrom_name:str) -> DataFrame:
        start = np.arange(0, len(parameters[0]) * bin_size, bin_size)
        end = np.arange(bin_size, len(parameters[0]) * bin_size + 1, bin_size)

        dfpar = pd.DataFrame(np.vstack(parameters).T,
                  columns=["u_0", "p_right", "p_left", "lambda", "lambda_background", "stickiness"])
        dfpar.insert(0, "end", end)
        dfpar.insert(0, "start", start)
        dfpar.insert(0, "chrom", chrom_name)

        dfpar.to_csv(path, sep="\t", index=False)
        return dfpar

    @jax.jit
    def calc_contact_l(u_0, p_l, lmbd, h, v_max):
        divisor_l = logsumexp(p_l, lmbd + h - v_max)
        multiplier_l = p_l - jnp.roll(divisor_l, 1)
        c_l = jnp.triu(multiplier_l.reshape(1, -1).repeat(n_loci, axis=0), 1)
        return jnp.cumsum(c_l, axis = 1) + u_0.reshape(-1, 1)

    @jax.jit
    def calc_contact_r(u_0, p_r, lmbd, h, v_max):
        divisor_r = logsumexp(p_r, lmbd + h - v_max)
        multiplier_r = jnp.roll(p_r, 1) - divisor_r
        c_r = jnp.triu(multiplier_r.reshape(1, -1).repeat(n_loci, axis=0), 1)
        return jnp.cumsum(c_r, axis = 1) + u_0.reshape(1, -1)


    @jax.jit
    def calc_contact_combined(u_0, p_l, p_r, lmbd, h, v_max):
        divisor = logsumexp(logsumexp(p_r, p_l), lmbd + h - v_max)
        multiplier_l = p_l - jnp.roll(divisor, 1)
        c_l = jnp.triu(multiplier_l.reshape(1, -1).repeat(n_loci, axis=0), 1)

        multiplier_r = jnp.roll(p_r, 1) - divisor
        c_r = jnp.triu(multiplier_r.reshape(1, -1).repeat(n_loci, axis=0), 1)
        return jnp.cumsum(c_l, axis = 1) + u_0.reshape(-1, 1), jnp.cumsum(c_r, axis = 1) + u_0.reshape(1, -1)

    @jax.jit
    def hic(u_0:ArrayLike,
            p_r:ArrayLike,
            p_l:ArrayLike,
            lmbd:ArrayLike,
            sticky:ArrayLike) -> ArrayLike:
        """
        Constraint p and l
        u_0: initial concentration
        p_r: slowdown of the right leg (exp CTCF)
        p_l: slowdown of the left leg (exp CTCF)
        lmbd_r: sequence specific decay of the right leg
        lmbd_l: sequence specific decay of the left leg
        lmbd_b: background decay    
        sticky: stickiness of the chromatin in BOTH directions
        """
        h = 0.0
        v_max = 0.0
        p_r, p_l = (jax.nn.log_sigmoid(i) for i in (p_r, p_l))
        lmbd = jax.nn.log_sigmoid(lmbd)
        #norms = jax.nn.softplus(norms)

        contact_l, contact_r = calc_contact_combined(u_0/2, p_l, p_r, lmbd, h, v_max)

        s_out = jnp.outer(sticky, sticky)
        return contact_l, contact_r, s_out

    def model_init(mat:ArrayLike, stickiness_carry:Union[ArrayLike, None] = None) -> Parameters:
        width = mat.shape[0]
        u_0 = jnp.diag(mat).mean()
        p_r = jnp.ones(width) * 4.5
        p_l = jnp.ones(width) * 4.5
        lmbd = jnp.ones(width) * -4.0
        #lmbd = jnp.array(-4.0)
        #norms = diag_mean_log(mat)
        sticky = jnp.ones(width) * 1e-4 
        if stickiness_carry is not None:
            sticky = sticky.at[:len(stickiness_carry)].set(stickiness_carry)
        return u_0, p_r, p_l, lmbd, sticky

    def model_init_whole(size):
        return (np.zeros(size),
                np.full(size, -np.inf),
                np.full(size, -np.inf),
                np.full(size, -np.inf),
                np.zeros(size))

    @jax.jit
    def mse_loss(a, b, weight):
        loci_n = a.shape[0] - diag_start
        loci_n = loci_n * (loci_n+1) / 2
        #return jax.scipy.special.logsumexp(weight * jnp.triu(2 * logabssubexp(a, b), diag_start)) - loci_n
        return jnp.sum(weight * jnp.triu(jnp.power(a - b, 2), diag_start)) / loci_n

    @jax.jit
    def total_loss(mat:ArrayLike,
                u_0:ArrayLike,
                p_r:ArrayLike,
                p_l:ArrayLike,
                lmbd:ArrayLike,
                slow_down:ArrayLike,
                params:ArrayLike) -> float:
        contact_l, contact_r, sticky = hic(u_0, p_r, p_l, lmbd, slow_down)
        contact_l, contact_r  = (diag_normalize(i, params) for i in (contact_l, contact_r))
        mat_half = mat - 2
        res = mat - logsumexp(contact_l, contact_r)
        weight = jnp.exp(mat_half) 
        #weight = jnp.ones_like(mat_half) 
        l1 = mse_loss(contact_l, mat_half, weight)
        l2 = mse_loss(contact_r, mat_half, weight)
        l3 = mse_loss(sticky, res, weight)
        return l1 + l2 + l3

#    @jax.jit
#    def res_loss(mat:ArrayLike,
#                u_0:ArrayLike,
#                p_r:ArrayLike,
#                p_l:ArrayLike,
#                lmbd:ArrayLike,
#                slow_down:ArrayLike) -> float:
#        contact_l, contact_r, sticky = hic(u_0, p_r, p_l, lmbd, slow_down)
#        res = mat - logsumexp(contact_l, contact_r)
#        l3 = mse_loss(res, sticky)
#        return l3

    val_grad_contact_loss = jax.jit(jax.value_and_grad(total_loss, argnums = (1, 2, 3, 4, 5)))
#    val_grad_res_loss = jax.jit(jax.value_and_grad(res_loss, argnums = 5))

    def pass_carry(params:Parameters, overlap:int) -> ArrayLike:
        return params[5][-overlap:]

    def crop_params(params:Parameters, crop_sides) -> Parameters:
        return (params[0], *(i[crop_sides:-crop_sides] for i in params[1:]))

    return val_grad_contact_loss, model_init, model_init_whole, \
            parameter_transformation, write_parameters, pass_carry, hic, crop_params, \
            total_loss
