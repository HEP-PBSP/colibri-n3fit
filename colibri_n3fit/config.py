"""
colibri_n3fit.config.py

"""

import dill
import logging
from colibri_n3fit.model import N3FitPDFModel

from colibri.config import Environment, colibriConfig

from colibri_n3fit.utils import FLAV_INFO_NNPDF40

log = logging.getLogger(__name__)


class Environment(Environment):
    pass


class ColibriN3fitConfig(colibriConfig):
    """
    GPConfig class Inherits from colibri.config.colibriConfig
    """

    def produce_pdf_model(
        self,
        output_path,
    ):
        """
        Produce the example PDF model.
        """
        model = N3FitPDFModel(
            flav_info=FLAV_INFO_NNPDF40,
            replica_range_settings={"min_replica": 1, "max_replica": 1},
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
