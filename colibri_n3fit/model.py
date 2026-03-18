"""
colibri_n3fit.model.py

"""

import jax.numpy as jnp
from colibri.pdf_model import PDFModel
from n3fit.model_gen import _pdfNN_layer_generator, ReplicaSettings
from validphys.n3fit_data import replica_nnseed
import tensorflow as tf
from colibri_n3fit.utils import FLAV_INFO_NNPDF40
import numpy as np
import math
from jax import lax
import keras
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")


class N3FitPDFModel(PDFModel):

    def __init__(
        self,
        replica_index,
        nnseed,
        flav_info: list = FLAV_INFO_NNPDF40,
        replica_range_settings: dict = {"min_replica": 1, "max_replica": 1},
        impose_sumrule: bool = True,
        fitbasis: str = "EVOL",
        nodes: list = [25, 20, 8],
        activations: list = ["tanh", "tanh", "linear"],
        initializer_name: str = "glorot_normal",
        layer_type: str = "dense",
        **kwargs,
    ):
        """
        Args:
            **kwargs: Arguments for n3fit model configuration
                      (nodes, activations, initializer_name, etc.)
        """
        # Store all constructor arguments for serialization
        self._init_args = {
            "flav_info": flav_info,
            "replica_range_settings": replica_range_settings,
            "impose_sumrule": impose_sumrule,
            "fitbasis": fitbasis,
            "nodes": nodes,
            "activations": activations,
            "initializer_name": initializer_name,
            "layer_type": layer_type,
            **kwargs,
        }

        self.model_kwargs = kwargs

        keras.backend.set_floatx("float64")

        nnseed_rep = replica_nnseed(replica_index, nnseed)

        self.n3fit_model = _pdfNN_layer_generator(
            [
                ReplicaSettings(nodes=nodes, activations=activations, seed=nnseed_rep)
                for i in range(
                    replica_range_settings["min_replica"],
                    replica_range_settings["max_replica"] + 1,
                )
            ],
            impose_sumrule=impose_sumrule,
            flav_info=flav_info,
            fitbasis=fitbasis,
            **kwargs,
        )
        self.x_in = self.n3fit_model.x_in
        self.static_weights = self.n3fit_model.get_weights()[
            :16
        ]  # First 16 weights aren't trainable
        self.modifiable_shapes = [w.shape for w in self.n3fit_model.get_weights()[16:]]
        self.total_modifiable_params = sum(np.prod(s) for s in self.modifiable_shapes)

    def __getstate__(self):
        """
        Custom Pickle serialization: save initialization arguments.

        Required to reconstruct the n3fit model upon deserialization.
        """
        state = {"_init_args": self._init_args}
        return state

    def __setstate__(self, state):
        """Custom Pickle deserialization: restore initialization arguments."""
        self.__init__(**state["_init_args"])

    @property
    def n_parameters(self):
        """Return total number of trainable parameters."""
        return sum(jnp.size(w) for w in self.n3fit_model.trainable_weights)

    @property
    def param_names(self):
        return [f"w_{i}" for i in range(self.n_parameters)]

    def grid_values_func(self, xgrid):
        """
        Returns a function that computes the PDF values on xgrid.
        Maintains same input/output structure as n3fit_pdf_grid.
        """

        xgrid_tensor = jnp.array(xgrid)[None, :, None].astype(jnp.float64)
        xgrid_integration_jax = jnp.array(
            tf.convert_to_tensor(self.n3fit_model.x_in["xgrid_integration"])
        )

        input_dict = {
            "pdf_input": xgrid_tensor,
            "xgrid_integration": xgrid_integration_jax,
        }

        def unpack_params(flat_params, shapes):
            """Returns the weight parameters in the format expected by the n3fit model."""
            params_list = []
            pos = 0
            for shape in shapes:
                size = math.prod(shape)
                param = lax.dynamic_slice(flat_params, (pos,), (size,))
                param = param.reshape(shape)
                params_list.append(param)
                pos += size
            return params_list

        def pdf_func(_params):

            # Retrive the shapes of the trainable variables
            shapes = [v.shape for v in self.n3fit_model.trainable_variables]

            weights = unpack_params(_params, shapes)

            nn_layer = self.n3fit_model.layers[
                4
            ]  # Change (if sum rules [4], if not [3])

            # Set the weights of the layer
            nn_layer.set_weights(weights)

            model_output = self.n3fit_model(input_dict)

            pdf_output = jnp.squeeze(model_output, axis=0)

            pdf_array = jnp.transpose(pdf_output, axes=(0, 2, 1))

            pdf_array = jnp.squeeze(pdf_array, axis=0)

            return jnp.array(pdf_array)

        return pdf_func
