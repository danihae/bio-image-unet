"""Package-level smoke tests: imports, the public API and the released version.

test_tiling.py and test_predict_shapes.py cover the prediction path in depth, but
nothing in the suite imports data.py, train.py or losses.py, so a bad import or a
syntax error in those would reach PyPI unnoticed -- they are only executed when a
user trains a model. Every subpackage's __init__ re-exports its whole API, so
importing the subpackages exercises those modules for the cost of an import.
"""
import importlib
import pathlib

import pytest

SUBPACKAGES = ('bio_image_unet.unet', 'bio_image_unet.unet3d', 'bio_image_unet.siam_unet',
               'bio_image_unet.multi_output_unet', 'bio_image_unet.multi_output_unet3d',
               'bio_image_unet.utils', 'bio_image_unet.progress')

# The network families deliberately expose the same three entry points; that
# uniformity is what lets callers swap one family for another.
FAMILIES = SUBPACKAGES[:5]
ENTRY_POINTS = ('DataProcess', 'Trainer', 'Predict')

NETWORKS = (('bio_image_unet.unet', 'Unet'),
            ('bio_image_unet.unet', 'AttentionUnet'),
            ('bio_image_unet.unet3d', 'UNet3D'),
            ('bio_image_unet.siam_unet', 'Siam_UNet'),
            ('bio_image_unet.multi_output_unet', 'MultiOutputUnet'),
            ('bio_image_unet.multi_output_unet', 'MultiOutputNestedUNet'),
            ('bio_image_unet.multi_output_unet3d', 'MultiOutputUnet3D'))


@pytest.mark.parametrize('name', SUBPACKAGES)
def test_subpackage_imports(name):
    importlib.import_module(name)


@pytest.mark.parametrize('name', FAMILIES)
def test_family_exposes_entry_points(name):
    module = importlib.import_module(name)
    missing = [attr for attr in ENTRY_POINTS if not hasattr(module, attr)]
    assert not missing, f'{name} no longer exports {missing}'


@pytest.mark.parametrize('name, network', NETWORKS)
def test_network_class_is_exported(name, network):
    assert hasattr(importlib.import_module(name), network)


def test_installed_version_matches_pyproject():
    """The built artefact must carry the version pyproject declares.

    Releases are cut by pushing a v* tag, so a stale or mismatched version here is
    the failure that publishes the wrong release -- and PyPI accepts a version only
    once, leaving no way to correct it in place.
    """
    tomllib = pytest.importorskip('tomllib', reason='needs Python 3.11+ to read pyproject.toml')
    from importlib.metadata import PackageNotFoundError, version

    pyproject = pathlib.Path(__file__).resolve().parents[1] / 'pyproject.toml'
    if not pyproject.exists():
        pytest.skip('not running from a source checkout')
    declared = tomllib.loads(pyproject.read_text())['tool']['poetry']['version']

    try:
        installed = version('bio-image-unet')
    except PackageNotFoundError:
        pytest.skip('package is not installed, nothing to compare against')
    assert installed == declared
