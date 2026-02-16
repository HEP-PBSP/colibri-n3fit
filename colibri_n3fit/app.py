"""
colibri_n3fit.app.py

"""

from colibri.app import colibriApp
from colibri_n3fit.config import ColibriN3fitConfig


colibri_n3fit_providers = [
    "colibri_n3fit.model",
]


class N3fitPdfApp(colibriApp):
    config_class = ColibriN3fitConfig


def main():
    a = N3fitPdfApp(name="colibri_n3fit", providers=colibri_n3fit_providers)
    a.main()


if __name__ == "__main__":
    main()
