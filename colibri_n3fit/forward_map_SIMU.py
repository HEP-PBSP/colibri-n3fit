"""SIMUnet corrections on Colibri's SM FK predictions."""

import jax.numpy as jnp
from colibri.forward_map import ForwardMap
from simunet.smeft import apply_smeft_correction


class SIMUnet_ForwardMap(ForwardMap):
    """
    Forward map for SIMUnet SMEFT corrections.
    Allows for fitting both PDF parameters and SIMU coefficients simultaneously.
    Parameters:
        pdf_model: The PDF model used for predictions.
        pred_func: Function to compute SM predictions given PDF and FK tables.
        pdf_grid_func: Function to compute the PDF grid given parameters.
        simu_responses: Pre-computed SMEFT responses for each parameter given in simu_parameters in the runcard, defined in simu_data.py.
    """

    def __init__(self, pdf_model, pred_func, pdf_grid_func, simu_responses):
        super().__init__(pdf_model, extra_param_names=list(simu_responses.parameter_names))
        self._pred_func = pred_func
        self._pdf_grid_func = pdf_grid_func
        self._responses = tuple(jnp.asarray(r) for r in simu_responses.values) # These are the SMEFT effects, R. T_SMEFT = T_SM (1 + sum_i c_i * R_i)
        self.scales = jnp.asarray(simu_responses.scales) # The scale applied to each SMEFT coefficiet, defined in the runcard

    def __call__(self, fk_tables, params):
        pdf = self._pdf_grid_func(params[:self.n_pdf_params])
        sm = self._pred_func(pdf, fk_tables)
        return apply_smeft_correction(
            sm, params[self.n_pdf_params:], self._responses
        ), pdf

    def physical_coefficients(self, params):
        """Return physical coefficients in runcard order from a full vector."""
        return params[self.n_pdf_params:] / self.scales

    def parameters_with_coefficients(self, pdf_params, coefficients):
        """Pack physical coefficients supplied by name; omitted names mean zero."""
        unknown = set(coefficients) - set(self.extra_param_names)
        if unknown:
            raise ValueError(f"Unknown SIMU coefficients: {sorted(unknown)}")
        pdf_params = jnp.asarray(pdf_params)
        if pdf_params.shape != (self.n_pdf_params,):
            raise ValueError("Incorrect PDF parameter count")
        physical = jnp.asarray([coefficients.get(n, 0.0) for n in self.extra_param_names])
        return jnp.concatenate((pdf_params, physical * self.scales))


def forward_map(_pred_data, pdf_model, FIT_XGRID, simu_responses):
    """Override Colibri's provider with the SIMUnet forward map."""
    return SIMUnet_ForwardMap(
        pdf_model, _pred_data, pdf_model.grid_values_func(FIT_XGRID), simu_responses
    )
