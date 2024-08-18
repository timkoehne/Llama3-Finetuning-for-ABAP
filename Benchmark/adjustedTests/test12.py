

METADATA = {
    'author': 'jt',
    'dataset': 'test'
}



def num_tests():
    return 3

def check(candidate):
    passed = 0
    failed = 0
    try:
        assert candidate([]) == None
        passed += 1
    except (AssertionError, TypeError):
        failed += 1
    try:
        assert candidate(['x', 'y', 'z']) == 'x'
        passed += 1
    except (AssertionError, TypeError):
        failed += 1
    try:
        assert candidate(['x', 'yyy', 'zzzz', 'www', 'kkkk', 'abc']) == 'zzzz'
        passed += 1
    except (AssertionError, TypeError):
        failed += 1
    return passed, failed
