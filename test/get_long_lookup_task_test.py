from pytest import fixture


@fixture
def op():
    from tasks import get_task
    return get_task


def test_get_lookup(op):
    l = op("long_lookup")
    assert l.name == "long_lookup"
