

METADATA = {
    'author': 'jt',
    'dataset': 'test'
}


def check(candidate):
    passed = 0
    failed = 0
    try:
        assert candidate([]) == []
        passed += 1
    except (AssertionError, TypeError):
        failed += 1
    try:
        assert candidate([4, {}, [], 23.2, 9, 'adasd']) == [4, 9]
        passed += 1
    except (AssertionError, TypeError):
        failed += 1
    try:
        assert candidate([3, 'c', 3, 3, 'a', 'b']) == [3, 3, 3]
        passed += 1
    except (AssertionError, TypeError):
        failed += 1
    return passed, failed
