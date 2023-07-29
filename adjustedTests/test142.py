def check(candidate):
    passed = 0
    failed = 0

    # Check some simple cases
    
    try:
        assert candidate([1,2,3]) == 6
        passed += 1
    except (AssertionError, TypeError):
        failed += 1
    try:
        assert candidate([1,4,9]) == 14
        passed += 1
    except (AssertionError, TypeError):
        failed += 1
    try:
        assert candidate([]) == 0
        passed += 1
    except (AssertionError, TypeError):
        failed += 1
    try:
        assert candidate([1,1,1,1,1,1,1,1,1]) == 9
        passed += 1
    except (AssertionError, TypeError):
        failed += 1
    try:
        assert candidate([-1,-1,-1,-1,-1,-1,-1,-1,-1]) == -3
        passed += 1
    except (AssertionError, TypeError):
        failed += 1
    try:
        assert candidate([0]) == 0
        passed += 1
    except (AssertionError, TypeError):
        failed += 1
    try:
        assert candidate([-1,-5,2,-1,-5]) == -126
        passed += 1
    except (AssertionError, TypeError):
        failed += 1
    try:
        assert candidate([-56,-99,1,0,-2]) == 3030
        passed += 1
    except (AssertionError, TypeError):
        failed += 1
    try:
        assert candidate([-1,0,0,0,0,0,0,0,-1]) == 0
        passed += 1
    except (AssertionError, TypeError):
        failed += 1
    try:
        assert candidate([-16, -9, -2, 36, 36, 26, -20, 25, -40, 20, -4, 12, -26, 35, 37]) == -14196
        passed += 1
    except (AssertionError, TypeError):
        failed += 1
    try:
        assert candidate([-1, -3, 17, -1, -15, 13, -1, 14, -14, -12, -5, 14, -14, 6, 13, 11, 16, 16, 4, 10]) == -1448
        passed += 1
    except (AssertionError, TypeError):
        failed += 1
    
    
    # Don't remove this line:
    return passed, failed
