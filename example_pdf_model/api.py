"""
api.py

This module contains the `reportengine` programmatic API, initialized with the
colibri providers, Config and Environment.

Example:
--------

Simple Usage:

>> from example_pdf_model.api import API
>> fig = API.plot_pdfs(pdf="NNPDF_nlo_as_0118", Q=100)
>> fig.show()
"""

import logging

from reportengine import api
from colibri.app import colibri_providers
from example_pdf_model.app import ex_pdf_providers
from example_pdf_model.config import ExampleConfig, Environment


log = logging.getLogger(__name__)

# API needed its own module, so that it can be used with any Matplotlib backend
# without breaking validphys.app
API = api.API(ex_pdf_providers + colibri_providers, ExampleConfig, Environment)
