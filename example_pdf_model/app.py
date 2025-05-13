"""
example_pdf_model.app.py

"""

from colibri.app import colibriApp
from example_pdf_model.config import ExampleConfig


ex_pdf_providers = [
    "example_pdf_model.model",
]


class ExPdfApp(colibriApp):
    config_class = ExampleConfig


def main():
    a = ExPdfApp(name="gp", providers=ex_pdf_providers)
    a.main()


if __name__ == "__main__":
    main()
