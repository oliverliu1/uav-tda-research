import uav_tda


def test_package_imports_and_has_version():
    assert isinstance(uav_tda.__version__, str)
    assert uav_tda.__version__
