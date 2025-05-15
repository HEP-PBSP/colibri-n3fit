"""
example_pdf_model.model.py

"""

import jax.numpy as jnp
from colibri.pdf_model import PDFModel
from validphys import convolution
from n3fit.model_gen import pdfNN_layer_generator
import tensorflow as tf
from example_pdf_model.utils import FLAV_INFO_NNPDF40
import numpy as np

class N3FitPDFModel(PDFModel):

    def __init__(
        self,
        flav_info: list = FLAV_INFO_NNPDF40,
        replica_range_settings: dict = {"min_replica": 1, "max_replica": 1},
        impose_sumrule: bool = True,
        fitbasis: str = "EVOL",
        nodes: list = [25, 20, 8],
        activations: list = ["tanh", "tanh", "linear"],
        initializer_name: str = "glorot_normal",
        layer_type: str = "dense",
        **kwargs
    ):
        """
        Args:
            **kwargs: Arguments for n3fit model configuration
                      (nodes, activations, initializer_name, etc.)
        """
        self.model_kwargs = kwargs
        
        self.n3fit_model = pdfNN_layer_generator(
            nodes=nodes,
            activations=activations,
            initializer_name=initializer_name,
            layer_type=layer_type,
            seed=range(
                replica_range_settings["min_replica"],
                replica_range_settings["max_replica"] + 1,
            ),
            impose_sumrule=impose_sumrule,
            flav_info=flav_info,
            fitbasis=fitbasis,
            num_replicas=(
                replica_range_settings["max_replica"] 
                - replica_range_settings["min_replica"] 
                + 1
            ),
            **kwargs
        )
        self.x_in = self.n3fit_model.x_in

    @property
    def n_parameters(self):
        """Return total number of trainable parameters."""
        return sum(tf.size(w).numpy() for w in self.n3fit_model.trainable_weights)

    @property
    def param_names(self):
        return [f"w_{i}" for i in range(self.n_parameters)]

    def grid_values_func(self, xgrid):
        """
        Returns a function that computes the PDF values on xgrid.
        Maintains same input/output structure as n3fit_pdf_grid.
        """
        print('Nparameters:', self.n_parameters)
        xgrid_tensor = tf.convert_to_tensor(np.array(xgrid)[None, :, None], dtype=tf.float32)

        input_dict = {
            "pdf_input": xgrid_tensor,
            "xgrid_integration": self.n3fit_model.x_in["xgrid_integration"]
        }

        def pdf_func(_params):
         
            pdf_output = tf.squeeze(self.n3fit_model(input_dict), axis=0)  
           
            pdf_array = tf.transpose(pdf_output, perm=[0, 2, 1]) 

            pdf_array = tf.squeeze(pdf_array, axis=0)
    
            return jnp.array(pdf_array)  

        return pdf_func


class ExamplePDFModel(PDFModel):
    """ """

    def __init__(self, fitted_flavours):
        self.fitted_flavours = fitted_flavours

    @property
    def param_names(self):
        """The fitted parameters of the model."""
        return [
            "alpha",
            "beta",
        ]

    @property
    def n_parameters(self):
        """The number of parameters of the model."""
        return len(self.param_names)

    def _pdf(self, x, alpha, beta):
        """ """
        return x ** (1 - alpha) * (1 - x) ** beta

    def grid_values_func(self, xgrid):
        """This function should produce a grid values function, which takes
        in the model parameters, and produces the PDF values on the grid xgrid.
        """
        xgrid = jnp.array(xgrid)

        def pdf_func(params):
            """ """
            alpha = params[0]
            beta = params[1]
            pdf_grid = []
            for fl in convolution.FK_FLAVOURS:
                if fl in self.fitted_flavours:
                    pdf_grid.append(self._pdf(xgrid, alpha, beta))
                else:
                    pdf_grid.append(jnp.zeros_like(xgrid))

            return jnp.array(pdf_grid)

        return pdf_func
