from colibri.param_initialisation import (
    pdf_initial_parameters as colibri_pdf_initial_parameters,
)
from jax.nn.initializers import glorot_normal, zeros
import jax
import jax.numpy as jnp


def pdf_initial_parameters(pdf_model, param_initialiser_settings, replica_index=-1):

    if "random_seed" in param_initialiser_settings:
        random_seed = jax.random.PRNGKey(
            param_initialiser_settings["random_seed"] + replica_index
        )
    else:
        random_seed = jax.random.PRNGKey(replica_index)

    # Add error if initialiser type is zeros
    if param_initialiser_settings["type"] == "zeros":
        raise ValueError(
            "MC initialiser type 'zeros' is not compatible with the n3fit model."
            " Please choose a different initialiser type (e.g. 'normal', 'uniform', 'glorot_norm')."
        )

    if param_initialiser_settings["type"] == "glorot_norm":
        # Get layer shapes
        shapes = [v.shape for v in pdf_model.n3fit_model.trainable_variables]

        # For biases: zeros
        bias_init_fn = zeros

        # For weights: glorot
        weight_init_fn = glorot_normal()

        subkeys = jax.random.split(random_seed, len(shapes))

        initialized_params = [
            (bias_init_fn if len(shape) == 1 else weight_init_fn)(key, shape).ravel()
            for shape, key in zip(shapes, subkeys)
        ]

        return jnp.concatenate(initialized_params)

    return colibri_pdf_initial_parameters(
        pdf_model, param_initialiser_settings, replica_index
    )
