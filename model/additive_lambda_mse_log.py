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

def model(start_diag:int, end_diag:int):
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

    def contactmap(u_0:ArrayLike,
                   p_r:ArrayLike,
                   p_l:ArrayLike,
                   lmbd:ArrayLike):
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
        #n = min(len(u_0), end_diag)-1
        n = len(u_0)-1
        #u_0 = jnp.zeros_like(u_0)
        @jax.jit
        def _inner(carry, x):
            left, right, slowdown_left, slowdown_right, prev_state = carry # ith diagonal
            nom = logsumexp(jnp.roll(prev_state, 1) + jnp.roll(left, 1), prev_state + right)
            right = jnp.roll(right, 1)
            slowdown_right = jnp.roll(slowdown_right, 1)
            denom = logsumexp(logsumexp(left , right), logsumexp(slowdown_left, slowdown_right))
            curr_state = nom - denom
            return (left, right, slowdown_left, slowdown_right, curr_state), curr_state
        #lmbd = jax.nn.log_sigmoid(lmbd)
        mat = jax.lax.scan(_inner,
                           (jax.nn.log_sigmoid(p_l),
                            jax.nn.log_sigmoid(p_r),
                            lmbd,
                            lmbd,
                            u_0),
                           None,
                           length=n)[1]
        return jnp.vstack([u_0, mat])

    def model_init(mat:ArrayLike) -> Parameters:
        width = mat.shape[0]
        u_0 = jnp.diag(mat)
        p_r = jnp.ones(width) * 4.5
        p_l = jnp.ones(width) * 4.5
        lmbd = jnp.ones(width) * -4.0
        return u_0, p_r, p_l, lmbd

    def model_init_whole(size):
        return (np.zeros(size),
                np.full(size, -np.inf),
                np.full(size, -np.inf),
                np.full(size, -np.inf),
                np.zeros(size))

    @jax.jit
    def mse_loss(a, b, weight):
        loci_n = a.shape[0] - start_diag
        no_diag = min(loci_n, end_diag)
        loci_n = (2*loci_n - no_diag + 1) * no_diag / 2
        return jnp.sum(jnp.triu(weight * jnp.power(a - b, 2))[start_diag:]) / loci_n

    @jax.jit
    def total_loss(mat:ArrayLike,
                   u_0:ArrayLike,
                   p_r:ArrayLike,
                   p_l:ArrayLike,
                   lmbd:ArrayLike) -> float:
        mat_pred = contactmap(u_0, p_r, p_l, lmbd)
        #return mse_loss(mat[:end_diag], mat_pred[:end_diag], jnp.exp(mat)[:end_diag])
        return mse_loss(mat[:end_diag], mat_pred[:end_diag], jnp.ones_like(mat)[:end_diag])
        #return mse_loss(mat[:end_diag], mat_pred, jnp.ones_like(mat)[:end_diag])

    val_grad_contact_loss = jax.jit(jax.value_and_grad(total_loss, argnums = (2, 3, 4)))

    def pass_carry(params:Parameters, overlap:int) -> ArrayLike:
        return params[5][-overlap:]

    def crop_params(params:Parameters, crop_sides) -> Parameters:
        return (params[0], *(i[crop_sides:-crop_sides] for i in params[1:]))

    return val_grad_contact_loss, model_init, model_init_whole, \
            parameter_transformation, write_parameters, pass_carry, contactmap, crop_params, \
            total_loss
