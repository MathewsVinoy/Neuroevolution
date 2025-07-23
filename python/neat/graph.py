def cycle_check(connections,  test):
    i, o = test
    if i == o:
        return True
    visited = {o}
    while True:
        num_added = 0
        for c in connections:
            if c.inId in visited and c.outId not in visited and c.enable:
                if c.outId == i:
                    return True

                visited.add(c.outId)
                num_added += 1

        if num_added == 0:
            return False




        