import pytest

@pytest.mark.parametrize("module_name", [
    "adversarial_prototype_decomposition",
    "adversarial_prototype_decomposition.base.apd",
    "adversarial_prototype_decomposition.classifier.classifiers",
    "adversarial_prototype_decomposition.sampler.glvq_sampler",
])
def test_public_modules_importable(module_name):
    """All core modules should import without errors."""
    __import__(module_name)