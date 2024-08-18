
def num_tests():
    return 7

def check(candidate):
    passed = 0
    failed = 0

    # Check some simple cases
    try:
        assert candidate([1,-2,-3,41,57,76,87,88,99], 3) == -4
        passed += 1
    except (AssertionError, TypeError):
        failed += 1
    try:
        assert candidate([111,121,3,4000,5,6], 2) == 0
        passed += 1
    except (AssertionError, TypeError):
        failed += 1
    try:
        assert candidate([11,21,3,90,5,6,7,8,9], 4) == 125
        passed += 1
    except (AssertionError, TypeError):
        failed += 1
    try:
        assert candidate([111,21,3,4000,5,6,7,8,9], 4) == 24, "This prints if this assert fails 1 (good for debugging!)"
        passed += 1
    except (AssertionError, TypeError):
        failed += 1

    # Check some edge cases that are easy to work out by hand.
    try:
        assert candidate([1], 1) == 1, "This prints if this assert fails 2 (also good for debugging!)"
        passed += 1
    except (AssertionError, TypeError):
        failed += 1

    return passed, failed
