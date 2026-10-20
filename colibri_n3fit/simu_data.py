"""Prepare SIMU responses in Colibri's prediction order."""

from dataclasses import dataclass

import numpy as np
from simunet.loader import SIMUnetLoader
from simunet.smeft import parameter_responses


@dataclass(frozen=True)
class SIMUResponses:
    values: tuple
    point_index: tuple
    parameter_names: tuple
    scales: tuple


def simu_responses(data, simu_parameters=None):
    """
    Produces the SIMU responses, R, in the order of the data points in the input data. 
    The response for a dataset for a given parameter, p, is defined as:
    R_i = (1/scale_p) * sum_j (c_j * (K-factor_j / K-factor_SM))
    where c_j are the coefficients of the linear combination for parameter p, and K-factor_j 
    is defined in the SIMU factor file for the dataset. The scale_p is defined in the runcard for parameter p.
    """
    names = tuple(p["name"] for p in simu_parameters)
    scales = tuple(float(p.get("scale", 1.0)) for p in simu_parameters)
    if len(set(names)) != len(names):
        raise ValueError("SIMU parameter names must be unique")
    inputs = list(data)
    datasets = list(data.datasets)

    blocks = [[] for _ in simu_parameters]
    point_index = []
    for dataset_input, dataset in zip(inputs, datasets):
        if getattr(dataset, "use_fixed_predictions", False):
            raise NotImplementedError(
                "Colibri's FK prediction pipeline does not yet support use_fixed_predictions"
            )
        ndata = dataset.commondata.ndata
        cuts = dataset.cuts.load()

        point_index.extend((dataset.name, int(i)) for i in cuts)
        simu_fac_order = getattr(dataset_input, "simu_fac", None)
        if simu_parameters and simu_fac_order is not None:
            paths = getattr(dataset, "simu_parameters_names_CF", None)
            if not paths:
                raise ValueError(f"Missing SIMU factor paths for {dataset.name}")
            unique_paths = set(paths.values())
            if len(unique_paths) != 1:
                raise ValueError(f"Expected one SIMU file for {dataset.name}")
            factors = SIMUnetLoader.load_simu_factors(next(iter(unique_paths)))
            raw = parameter_responses(simu_parameters, factors[simu_fac_order])
        else:
            raw = [np.zeros(ndata) for _ in simu_parameters]
        for column, response in zip(blocks, raw):
            response = np.asarray(response, dtype=float)
            column.append(response[cuts])
    values = tuple(np.concatenate(b) if b else np.empty(0) for b in blocks)
    return SIMUResponses(values, tuple(point_index), names, scales)
