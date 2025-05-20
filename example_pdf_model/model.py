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
import math
from jax import lax

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

    def update_weights(self, params):
        """Update TF model weights (must be called outside JIT)"""
        if hasattr(params, '_value'):  # Handle traced arrays
            params = params._value
        params_np = np.array(jax.device_get(params))
        new_weights = []
        param_pos = 0
        for shape in self.modifiable_shapes:
            size = np.prod(shape)
            new_w = params_np[param_pos:param_pos+size].reshape(shape)
            new_weights.append(new_w)
            param_pos += size

        updated_weights = self.static_weights + new_weights
        # import IPython ; IPython.embed()
        self.n3fit_model.set_weights(updated_weights)


    def grid_values_func(self, xgrid):
        """
        Returns a function that computes the PDF values on xgrid.
        Maintains same input/output structure as n3fit_pdf_grid.
        """

        xgrid_tensor =  jnp.array(xgrid)[None, :, None].astype(jnp.float32)

        input_dict = {
            "pdf_input": xgrid_tensor,
            "xgrid_integration": self.n3fit_model.x_in["xgrid_integration"]
        }
        def unpack_params(flat_params):
            shapes = [(2, 25), (25,), (25, 20), (20,), (20, 8), (8,)]
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
       
            shapes = [v.shape for v in self.n3fit_model.trainable_variables]
        
            weights = unpack_params(_params)
           
            nn_layer = self.n3fit_model.layers[4]
      
            print(self.n3fit_model.input_shape)
            print(input_dict['pdf_input'].shape) # Check expected input shapes
            print(input_dict['xgrid_integration'].shape)
            nn_layer.set_weights(weights)
           
            model_output = self.n3fit_model(input_dict) #

            pdf_output = jnp.squeeze(model_output, axis=0)
    
            pdf_array = jnp.transpose(pdf_output, axes=(0, 2, 1))
      
            pdf_array = jnp.squeeze(pdf_array, axis=0)
 

            return jnp.array(pdf_array)  

        return pdf_func
