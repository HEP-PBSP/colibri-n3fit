from colibri.param_initialisation import (
    pdf_initial_parameters as colibri_pdf_initial_parameters,
)


def pdf_initial_parameters(pdf_model, param_initialiser_settings, replica_index=-1):
    # Add error if initialiser type is zeros
    if param_initialiser_settings["type"] == "zeros":
        raise ValueError(
            "MC initialiser type 'zeros' is not compatible with the n3fit model."
            " Please choose a different initialiser type (e.g. 'normal', 'uniform', 'glorot_norm')."
        )
    return colibri_pdf_initial_parameters(
        pdf_model, param_initialiser_settings, replica_index
    )
