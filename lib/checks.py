"""Done-when checks for the module notebooks.

    from lib.checks import done_when
    done_when([
        ("You reported train and test numbers for all three", ok_1),
        ("The overfit model's gap is the biggest", ok_2),
    ])

Prints a tick or a cross per criterion and returns True only if all pass. Learner notebooks
just print; solution notebooks wrap it in `assert`, so a broken solution fails the test run.
"""


def done_when(criteria: list[tuple[str, bool]]) -> bool:
    all_ok = True
    for text, ok in criteria:
        ok = bool(ok)
        all_ok &= ok
        print(f"{'✓' if ok else '✗'} {text}")
    print("\nDone. Every criterion passes." if all_ok else "\nNot done yet. Fix the ✗ items above.")
    return all_ok
