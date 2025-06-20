def cycle_check(connections,  test):
    i, o = test
    if i == o:
        return True

    visited = {o}
    while True:
        num_added = 0
        for conn in connections:
            if conn.inId in visited and conn.outId not in visited:
                if conn.outId == i:
                    return True

                visited.add(conn.outId)
                num_added += 1

        if num_added == 0:
            return False