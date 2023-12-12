"""Model with one lambda parameters and stickiness. Stickiness is implemented with as (s_out + 1) 
without any transformation on s_out + 1 such as exponentiation or sigmoid.
"""
from functools import partial
from typing import Tuple, Union
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpy as np
import pandas as pd
from pandas import DataFrame

Parameters = Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike, ArrayLike]

def model():
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
        dfpar["start"] = start
        dfpar["end"] = end
        dfpar["chrom"] = chrom_name

        dfpar = dfpar.iloc[:, [8, 6, 7] + list(range(6))]
        dfpar.to_csv(path, sep="\t", index=False)
        return dfpar

    @partial(jax.jit, static_argnums = 0)
    def hic(n_loci:int,
            u_0:ArrayLike,
            p_r:ArrayLike,
            p_l:ArrayLike,
            lmbd:ArrayLike,
            lmbd_b:ArrayLike,
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
        h = 1
        v_max = 1
        p_r, p_l = (jax.nn.sigmoid(i) for i in (p_r, p_l))
        lmbd, lmbd_b = (jax.nn.softplus(i) for i in (lmbd, lmbd_b))
        #lmbd, lmbd_b = (jax.nn.sigmoid(i) for i in (lmbd, lmbd_b))

        mat_triu = jnp.triu(jnp.ones((n_loci, n_loci)), 1)
        mat_eye = jnp.eye(n_loci) * u_0 / 2
        mat_tril = jnp.tril(jnp.ones((n_loci, n_loci)), -1) + mat_eye

        divisor_r = p_r + (lmbd + lmbd_b) * h / v_max
        multiplier_r = jnp.roll(p_r, 1) / divisor_r
        c_r = mat_triu * multiplier_r.reshape(1, -1)
        hic_r = jnp.cumprod(c_r + mat_tril, axis = 1)

        divisor_l = p_l + (lmbd + lmbd_b) * h / v_max
        multiplier_l = p_l / jnp.roll(divisor_l, 1)
        c_l = mat_triu * multiplier_l.reshape(1, -1)
        hic_l = jnp.cumprod(c_l + mat_tril, axis = 1)

        hic = hic_l + hic_r

        s_out = jnp.outer(sticky, sticky)
        hic = hic * (1 + s_out)

        return hic

    def model_init(mat:ArrayLike,
                   stickiness_carry:Union[ArrayLike, None] = None) -> Parameters:
        width = mat.shape[0]
        u_0 = jnp.diag(mat).mean()
        p_r = jnp.ones(width)
        p_l = jnp.ones(width)
        lmbd = jnp.zeros(width)
        lmbd_b = jnp.array(0.01)
        sticky = jnp.ones(width)*0.01
        if stickiness_carry is not None:
            sticky = sticky * stickiness_carry
        return u_0, p_r, p_l, lmbd, lmbd_b, sticky

    def model_init_whole(size):
        return (np.zeros(size),
                np.full(size, -np.inf),
                np.full(size, -np.inf),
                np.full(size, -np.inf),
                np.full(size, -np.inf),
                np.zeros(size))

    @jax.jit
    def l2_loss(mat:ArrayLike,
                u_0:ArrayLike,
                p_r:ArrayLike,
                p_l:ArrayLike,
                lmbd:ArrayLike,
                lmbd_b:ArrayLike,
                slow_down:ArrayLike) -> float:
        mat_model = hic(mat.shape[0], u_0, p_r, p_l, lmbd, lmbd_b, slow_down)
        return jnp.mean(jnp.power(jnp.triu(mat - mat_model, 1), 2))

    val_grad_l2_loss = jax.jit(jax.value_and_grad(l2_loss, argnums = (1, 2, 3, 4, 5, 6)))

    def pass_carry(params:Parameters, overlap:int) -> ArrayLike:
        return params[5][-overlap:]

    return val_grad_l2_loss, model_init, model_init_whole, parameter_transformation, \
            write_parameters, pass_carry
