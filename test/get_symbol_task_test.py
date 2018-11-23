from pytest import fixture


@fixture
def op():
    from tasks import get_task
    return get_task("symbol_rewriting")


def test_get_symbol_rewriting(op):
    assert op.name == "Symbol Rewriting"
