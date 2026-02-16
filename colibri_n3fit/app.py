"""
colibri_n3fit.app.py

"""

from colibri.app import colibriApp
from colibri_n3fit.config import ColibriN3fitConfig


ex_pdf_providers = [
    "colibri_n3fit.model",
]


class ExPdfApp(colibriApp):
    config_class = ColibriN3fitConfig


def main():
    a = ExPdfApp(name="colibri_n3fit", providers=ex_pdf_providers)
    a.main()


if __name__ == "__main__":
    main()
