from LongLookupTables import LongLookupTask
from LookupTables import LookupTask
from SymbolRewriting import SymbolTask
from tasks import Task


def get_task(name,
             is_small=False,
             is_mini=False,
             longer_repeat=5):
    """Return the wanted tasks.

    Args:
        name ({"lookup", "long_lookup", "long_lookup_oneshot",
            "long_lookup_reverse", "noisy_long_lookup_multi", "noisy_long_lookup_single",
            "long_lookup_intermediate_noise", "symbol_rewriting", "SCAN"}) 
            name of the task to get.
        is_small (bool, optional): whether to run a smaller verson of the task.
            Used for getting less statistically significant results.
        is_mini (bool, optional): whether to run a smaller verson of the task.
            Used for testing purposes.
        longer_repeat (int, optional): number of longer test sets.

    Returns:
        task (tasks.tasks.Task): instantiated task.
    """
    name = name.lower()

    # classical lookup table
    if name == "lookup":
        return LookupTask(is_small=is_small, is_mini=is_mini)

    # Long lookup tasks - paser in get_long_lookup_tables can figure out which
    elif "lookup" in name:
        return LongLookupTask(
            name, is_small=is_small, is_mini=is_mini,
            longer_repeat=longer_repeat)

    # classical symbol rewriting task
    elif name == "symbol_rewriting":
        return SymbolTask(is_small=is_small, is_mini=is_mini)

    # classical scan
    elif name == "SCAN":
        raise NotImplementedError(
            "SCAN dataset not yet implemented to be used as sub Task Object")

    else:
        raise ValueError("Unkown name : {}".format(name))
