from pytest import fixture


@fixture
def op():
    from tasks import get_task
    return get_task("lookup")


def test_get_lookup(op):
    assert op.name == "lookup"
