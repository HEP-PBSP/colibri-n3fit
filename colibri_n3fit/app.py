"""
colibri_n3fit.app.py

"""

from colibri.app import colibriApp
from colibri_n3fit.config import ColibriN3fitConfig, Environment


colibri_n3fit_providers = [
    "colibri_n3fit.model",
    "colibri_n3fit.param_initialisation",
    "colibri_n3fit.simu_data",
    "colibri_n3fit.forward_map_SIMU",
]


class N3fitPdfApp(colibriApp):
    config_class = ColibriN3fitConfig
    environment_class = Environment


def main():
    a = N3fitPdfApp(name="colibri_n3fit", providers=colibri_n3fit_providers)
    a.main()


if __name__ == "__main__":
    main()
