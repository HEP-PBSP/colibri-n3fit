from colibri.param_initialisation import (
    pdf_initial_parameters as colibri_pdf_initial_parameters,
)
from jax.nn.initializers import glorot_normal, zeros
import jax
import jax.numpy as jnp


def _initial_pdf_parameters(pdf_model, param_initialiser_settings, replica_index=-1):

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


def pdf_initial_parameters(
    forward_map, param_initialiser_settings, replica_index=-1, simu_parameters=None
):
    #TODO: Try to use Simunet's initialisation functions here instead of reimplementing them
    """Initialise PDF weights and append scaled SIMU weights in runcard order."""
    pdf_weights = _initial_pdf_parameters(
        forward_map.pdf_model, dict(param_initialiser_settings), replica_index
    )
    if not simu_parameters: 
        return pdf_weights
    seed = param_initialiser_settings.get("random_seed", 0)
    #TODO: Check this is how we want to select the seed
    key = jax.random.fold_in(jax.random.PRNGKey(seed), replica_index % (2**32)) 
    keys = jax.random.split(jax.random.fold_in(key, 1397312853), len(simu_parameters))
    weights = []
    for parameter, key in zip(simu_parameters, keys):
        ini = parameter["initialisation"]
        if ini["type"] == "constant":
            value = jnp.asarray(ini["value"], dtype=pdf_weights.dtype)
        elif ini["type"] == "uniform":
            value = jax.random.uniform(
                key, (), minval=ini["minval"], maxval=ini["maxval"], dtype=pdf_weights.dtype
            )
        else:
            raise ValueError(f"Unknown SIMU initialisation: {ini}")
        weights.append(value * parameter.get("scale", 1.0))
    return jnp.concatenate((pdf_weights, jnp.stack(weights)))
