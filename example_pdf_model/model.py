"""
example_pdf_model.model.py

"""
import jax
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
        self.static_weights = self.n3fit_model.get_weights()[:16]  #First 16 weights aren't trainable
        self.modifiable_shapes = [w.shape for w in self.n3fit_model.get_weights()[16:]]
        self.total_modifiable_params = sum(np.prod(s) for s in self.modifiable_shapes)

    @property
    def n_parameters(self):
        """Return total number of trainable parameters."""
        return sum(tf.size(w).numpy() for w in self.n3fit_model.trainable_weights)

    @property
    def param_names(self):
        return [f"w_{i}" for i in range(self.n_parameters)]

    def update_weights(self, params_np):
        """Update TF model weights (must be called outside JIT)"""
        new_weights = []
        param_pos = 0
        for shape in self.modifiable_shapes:
            size = np.prod(shape)
            new_w = params_np[param_pos:param_pos+size].reshape(shape)
            new_weights.append(new_w)
            param_pos += size
        updated_weights = self.static_weights + new_weights
        self.n3fit_model.set_weights(updated_weights)


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

            params_np = jax.device_get(_params)

            self.update_weights(params_np) 

            pdf_output = tf.squeeze(self.n3fit_model(input_dict), axis=0)  
           
            pdf_array = tf.transpose(pdf_output, perm=[0, 2, 1]) 

            pdf_array = tf.squeeze(pdf_array, axis=0)
    
            return jnp.array(pdf_array)  

        return pdf_func
