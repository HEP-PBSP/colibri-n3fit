"""
colibri-n3fit.app.py

"""

from colibri.app import colibriApp
from colibri-n3fit.config import ExampleConfig


ex_pdf_providers = [
    "colibri-n3fit.model",
]


class ExPdfApp(colibriApp):
    config_class = ExampleConfig


def main():
    a = ExPdfApp(name="gp", providers=ex_pdf_providers)
    a.main()


if __name__ == "__main__":
    main()
