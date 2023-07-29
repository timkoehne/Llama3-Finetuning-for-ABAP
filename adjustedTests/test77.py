def check(candidate):
    passed = 0
    failed = 0

    # Check some simple cases
    try:
        assert candidate(1) == True, "First test error: " + str(candidate(1))
        passed += 1
    except (AssertionError, TypeError):
        failed += 1
    try:
        assert candidate(2) == False, "Second test error: " + str(candidate(2))
        passed += 1
    except (AssertionError, TypeError):
        failed += 1
    try:
        assert candidate(-1) == True, "Third test error: " + str(candidate(-1))
        passed += 1
    except (AssertionError, TypeError):
        failed += 1
    try:
        assert candidate(64) == True, "Fourth test error: " + str(candidate(64))
        passed += 1
    except (AssertionError, TypeError):
        failed += 1
    try:
        assert candidate(180) == False, "Fifth test error: " + str(candidate(180))
        passed += 1
    except (AssertionError, TypeError):
        failed += 1
    try:
        assert candidate(1000) == True, "Sixth test error: " + str(candidate(1000))
        passed += 1
    except (AssertionError, TypeError):
        failed += 1


    # Check some edge cases that are easy to work out by hand.
    try:
        assert candidate(0) == True, "1st edge test error: " + str(candidate(0))
        passed += 1
    except (AssertionError, TypeError):
        failed += 1
    try:
        assert candidate(1729) == False, "2nd edge test error: " + str(candidate(1728))
        passed += 1
    except (AssertionError, TypeError):
        failed += 1

    return passed, failed
