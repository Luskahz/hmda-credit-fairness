"""Smoke test: o pacote pode ser importado após instalação editável."""

import hmda_credit_fairness


def test_package_importable() -> None:
    assert hmda_credit_fairness.__doc__
