"""
example_pdf_model.config.py

"""

import dill
import logging
from example_pdf_model.model import ExamplePDFModel
from example_pdf_model.model import N3FitPDFModel

from colibri.config import Environment, colibriConfig
from colibri.constants import FLAVOUR_TO_ID_MAPPING

from example_pdf_model.utils import FLAV_INFO_NNPDF40

log = logging.getLogger(__name__)


class Environment(Environment):
    pass


class ExampleConfig(colibriConfig):
    """
    GPConfig class Inherits from colibri.config.colibriConfig
    """

    def produce_fitted_flavours(self, flavour_mapping):
        return flavour_mapping

    def produce_pdf_model(self, output_path, fitted_flavours):
        """
        Produce the example PDF model.
        """
        # model = ExamplePDFModel(fitted_flavours)
        model = N3FitPDFModel(
            fitted_flavours=fitted_flavours, 
            flav_info=FLAV_INFO_NNPDF40, 
            replica_range_settings={"min_replica": 1, "max_replica": 1000},
            impose_sumrule=True,
            fitbasis="EVOL",
            nodes=[25, 20, 8],
            activations=["tanh", "tanh", "linear"],
            initializer_name="glorot_normal",
            layer_type="dense",
    )
        # dump model to output_path using dill
        # this is mainly needed by scripts/ns_resampler.py
        with open(output_path / "pdf_model.pkl", "wb") as file:
            dill.dump(model, file)
        return model
