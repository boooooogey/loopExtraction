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
from log_func import *

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
        c_l = c_l.at[jnp.diag_indices_from(c_l)].set(u_0)
        return jnp.cumsum(c_l, axis = 1)

    @jax.jit
    def calc_contact_r(u_0, p_r, lmbd, h, v_max):
        divisor_r = logsumexp(p_r, lmbd + h - v_max)
        multiplier_r = jnp.roll(p_r, 1) - divisor_r
        c_r = jnp.triu(multiplier_r.reshape(1, -1).repeat(n_loci, axis=0), 1)
        c_r = c_r.at[jnp.diag_indices_from(c_r)].set(u_0)
        return jnp.cumsum(c_r, axis = 1)

    @jax.jit
    def hic(p_r:ArrayLike,
            p_l:ArrayLike,
            lmbd:ArrayLike,
            sticky:ArrayLike,
            norms:ArrayLike) -> ArrayLike:
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

        hic_r = calc_contact_l(jnp.log(0.5), p_l, lmbd, h, v_max)

        hic_l = calc_contact_r(jnp.log(0.5), p_r, lmbd, h, v_max)

        hic = logsumexp(hic_l, hic_r)

        s_out = jnp.outer(sticky, sticky)
        hic = hic + s_out
        return hic #diag_scale_log(hic, norms)

    @jax.jit
    def hic_debug(u_0:ArrayLike,
                  p_r:ArrayLike,
            p_l:ArrayLike,
            lmbd:ArrayLike,
            sticky:ArrayLike,
            norms:ArrayLike) -> ArrayLike:
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

        hic_r = calc_contact_l(u_0, p_l, lmbd, h, v_max)

        hic_l = calc_contact_r(u_0, p_r, lmbd, h, v_max)

        hic = logsumexp(hic_l, hic_r)

        s_out = jnp.outer(sticky, sticky)
        hic = hic + s_out
        return hic #diag_scale_log(hic, norms)

    def model_init(mat:ArrayLike,
                         stickiness_carry:Union[ArrayLike, None] = None) -> Parameters:
        width = mat.shape[0]
        u_0 = jnp.array(1.0)
        p_r = jnp.ones(width) * 4.5
        p_l = jnp.ones(width) * 4.5
        lmbd = jnp.ones(width) * -4.0
        #lmbd = jnp.array(-4.0)
        norms = diag_mean_log(mat)
        sticky = jnp.ones(width) * 1e-6 
        if stickiness_carry is not None:
            sticky = sticky.at[:len(stickiness_carry)].set(stickiness_carry)
        return p_r, p_l, lmbd, sticky, norms

    def model_init_whole(size):
        return (np.zeros(size),
                np.full(size, -np.inf),
                np.full(size, -np.inf),
                np.full(size, -np.inf),
                np.zeros(size))

    @jax.jit
    def kl_loss(mat:ArrayLike,
                mat_model:ArrayLike) -> float:
        mat_diag_normed = diag_scale_log(mat, -jnp.log(jnp.arange(mat.shape[0])+1)[::-1])
        mat_model_diag_normed = diag_scale_log(mat_model, -jnp.log(jnp.arange(mat.shape[0])+1)[::-1])
        kl = jnp.exp(mat_diag_normed) * (mat_diag_normed - mat_model_diag_normed)
        return jnp.sum(jnp.triu(kl, diag_start))

    @jax.jit
    def normed_loss(mat:ArrayLike,
                    mat_model:ArrayLike) -> float:
        diffp2 = 2 * logabssubexp(mat, mat_model)
        diffp2 = diffp2.at[jnp.tril_indices_from(diffp2, diag_start-1)].set(-jnp.inf)
        total_loci = n_loci - diag_start
        total_loci = total_loci * (total_loci+1) / 2
        return jnp.exp(jax.scipy.special.logsumexp(diffp2) - jnp.log(total_loci))

    @jax.jit
    def factorized_loss(mat:ArrayLike,
                        p_r:ArrayLike,
                        p_l:ArrayLike,
                        lmbd:ArrayLike,
                        sticky:ArrayLike,
                        norms:ArrayLike) -> float:
        mat_model = hic(p_r, p_l, lmbd, sticky, norms)
        return kl_loss(mat, mat_model)# + normed_loss(mat, mat_model)

    val_grad_factorized_loss = jax.jit(jax.value_and_grad(factorized_loss, argnums = (1, 2, 3, 4, 5)))
    #val_grad_l2_loss = jax.jit(jax.value_and_grad(kl_loss, argnums = (1, 2, 3, 4, 5, 6)))

    def pass_carry(params:Parameters, overlap:int) -> ArrayLike:
        return params[5][-overlap:]

    return val_grad_factorized_loss, model_init, model_init_whole, parameter_transformation, \
            write_parameters, pass_carry, hic, kl_loss, normed_loss, hic_debug
