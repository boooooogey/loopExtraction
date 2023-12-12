"""Utility functions to fit a model on the patches from the diagonal of a HiC matrix.
"""
import sys
import time
from typing import Tuple, Generator
from pandas import DataFrame
from jax.typing import ArrayLike
import jax.numpy as jnp
import numpy as np
import optax

def progressbar(it, prefix="", size:int = 60, out=sys.stdout):
    """Prints a progressbar for the iterator given.

    Args:
        it (_type_): _description_
        prefix (str, optional): _description_. Defaults to "".
        size (int, optional): _description_. Defaults to 60.
        out (_type_, optional): _description_. Defaults to sys.stdout.

    Yields:
        _type_: _description_
    """
    count = len(it)
    start = time.time()
    def show(j):
        x = int(size*j/count)
        remaining = ((time.time() - start) / j) * (count - j)

        mins, sec = divmod(remaining, 60)
        time_str = f"{int(mins):02}:{sec:05.2f}"

        print(f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j}/{count} Est wait {time_str}",
              end='\r', file=out, flush=True)

    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)

def fit_patch(patch: ArrayLike,
              carry: ArrayLike,
              f_df_dx: callable,
              model_init: callable,
              n_iter: int = 2000,
              learning_rate: float = 1e-1,
              convergence_threshold: float = 1e-5) -> Tuple[Generator[ArrayLike, None, None], bool]:
    """ Fits the model to a patch

    Args:
        patch (ArrayLike): Patch along the diagonal of a HiC matrix.
        carry (ArrayLike): pass message from the previous message.
        f_df_dx (callable): a function that returns value and gradient of a loss function.
        model_init (callable): a function that initiate the model for the patch.
        n_iter (int, optional): the number of iterations before the function terminates. Defaults to
                                2000.
        learning_rate (float, optional): Learning rate for the training. Defaults to 1e-1.
        convergence_threshold (float, optional): If the change in loss drops below this value the
                                                 execution is stopped. Defaults to 1e-5.

    Returns:
        Tuple[Generator[ArrayLike, None, None], bool]: Returns the set of parameters as a generator
                                                       and a boolean to indicate it has converged or
                                                       not.
    """

    patch, carry = (jnp.array(i) for i in (patch, carry))
    params = model_init(patch)

    init_multiplier_stickiness = jnp.ones(patch.shape[0])
    init_multiplier_stickiness.at[:len(carry)].set(jnp.sign(carry))

    params = model_init(patch, init_multiplier_stickiness)
    optimizer = optax.adamw(learning_rate)

    # Initialize parameters of the model + optimizer.
    opt_state = optimizer.init(params)
    loss_array = []

    converged = False

    for i in range(n_iter):
        loss_curr, grads = f_df_dx(patch, *params)
        loss_array.append(loss_curr)
        if loss_curr == min(loss_array):
            params_best = params
        #print(f"{i}: loss = {loss_curr}")
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        if len(loss_array) > 1 and jnp.abs(loss_curr - loss_array[-2]) < convergence_threshold:
            converged = True
            break

    return (np.array(i) for i in params_best), converged 

def broadcast_over_diagonal_chrom(model: callable,
                                  patch_transformation: callable,
                                  mat: ArrayLike,
                                  shape: int,
                                  overlap: int,
                                  path: str,
                                  chrom_name: str,
                                  bin_size: int) -> DataFrame:
    """ Broadcast fit_hic over chromosome

    Args:
        model (callable): Model to be fit to the data.
        transformation (callable): transformation for each patch.
        mat (ArrayLike)): the matrix to be broadcasted on.
        shape (int): sliding window size. The sliding window is a square.
        overlap (int): overlapping parts of the windows. 
        path (str): file path to save the parameters.
        chrom_name (str): name of the chromosome.
        bin_size (int): size of a bin in the hic matrix.

    Returns:
        Returns the fitted parameters for the whole chromosome.
    """
    print("Initiliazing the parameters...")
    f_df_dx, model_init, model_init_whole, parameter_transformation, write_parameters, pass_carry = model()
    if np.any(np.isnan(mat)):
        mat[np.isnan(mat)] = 0
    nonzero_mask = np.where(np.sum(mat, axis=0) != 0)[0]
    #initiliaze the parameters for the whole chromosome
    parameters = model_init_whole(mat.shape[0])
    mat = mat[np.ix_(nonzero_mask, nonzero_mask)]
    carry = np.ones(overlap)
    step_size = shape - overlap
    number_of_iterations = int(np.ceil((mat.shape[0]-shape) / step_size)) + 1
    print("Fitting the model on patches from the diagonal of the HiC matrix")
    any_patch_fail = False
    for i in progressbar(range(number_of_iterations)):
        start, end = i * step_size, min(i * step_size + shape, mat.shape[0])
        curr_param, converged = fit_patch(patch_transformation(mat[start:end, start:end]),
                                          carry, f_df_dx, model_init)
        if not converged:
            any_patch_fail = True
            #print(f"{i}: did not converged!")
        for k, p in enumerate(curr_param):
            parameters[k][nonzero_mask[start:end]] = p
        carry = pass_carry(parameters, overlap)
    if any_patch_fail:
        print("For some patches, the model did not converge!")
    print("Completed.")
    return write_parameters(path, parameter_transformation(parameters), bin_size, chrom_name)
