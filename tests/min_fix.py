import pytest
@pytest.fixture
def my_fix(): return 1
def test_fix(my_fix): assert my_fix == 1
