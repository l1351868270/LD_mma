
def ci_mnk(m: int, n: int, k: int, cache: int = 1) -> float:
    # cache = 0 no cache
    # cache = 1 cache is could have A[i]
    # cache = 2 cache is large, could have A[i] and B
    floats = 2 * m * n * k
    if cache == 0:
        c_w = m * n
        a_r = m * k * n
        b_r = m * n * k
    elif cache == 1:
        c_w = m * n
        a_r = m * k
        b_r = m * n * k
    else:
        c_w = m * n
        a_r = m * k
        b_r = k * n

    total = c_w + a_r + b_r
    ci = float(floats) / float(total)
    return ci

if __name__ == '__main__':
    print(f'{ci_mnk(200, 200, 200)}')