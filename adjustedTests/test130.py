def check(candidate):
    passed = 0
    failed = 0

    # Check some simple cases
    
    try:
        assert candidate(3) == [1, 3, 2.0, 8.0]
        passed += 1
    except (AssertionError, TypeError):
        failed += 1
    try:
        assert candidate(4) == [1, 3, 2.0, 8.0, 3.0]
        passed += 1
    except (AssertionError, TypeError):
        failed += 1
    try:
        assert candidate(5) == [1, 3, 2.0, 8.0, 3.0, 15.0]
        passed += 1
    except (AssertionError, TypeError):
        failed += 1
    try:
        assert candidate(6) == [1, 3, 2.0, 8.0, 3.0, 15.0, 4.0]
        passed += 1
    except (AssertionError, TypeError):
        failed += 1
    try:
        assert candidate(7) == [1, 3, 2.0, 8.0, 3.0, 15.0, 4.0, 24.0]
        passed += 1
    except (AssertionError, TypeError):
        failed += 1
    try:
        assert candidate(8) == [1, 3, 2.0, 8.0, 3.0, 15.0, 4.0, 24.0, 5.0]
        passed += 1
    except (AssertionError, TypeError):
        failed += 1
    try:
        assert candidate(9) == [1, 3, 2.0, 8.0, 3.0, 15.0, 4.0, 24.0, 5.0, 35.0]
        passed += 1
    except (AssertionError, TypeError):
        failed += 1
    try:
        assert candidate(20) == [1, 3, 2.0, 8.0, 3.0, 15.0, 4.0, 24.0, 5.0, 35.0, 6.0, 48.0, 7.0, 63.0, 8.0, 80.0, 9.0, 99.0, 10.0, 120.0, 11.0]
        passed += 1
    except (AssertionError, TypeError):
        failed += 1

    # Check some edge cases that are easy to work out by hand.
    try:
        assert candidate(0) == [1]
        passed += 1
    except (AssertionError, TypeError):
        failed += 1
    try:
        assert candidate(1) == [1, 3]
        passed += 1
    except (AssertionError, TypeError):
        failed += 1
    return passed, failed
