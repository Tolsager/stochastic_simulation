import numpy as np


def direct_sampling(random_variables: np.ndarray, p: list) -> list:
    dist = []
    for i in range(len(random_variables)):
        for j in range(1, len(p)+1):
            if random_variables[i] <= sum(p[:j]):
                dist.append(j)
                break
    return dist


def rejection_sampling(p: list, n: int) -> list:
    k = len(p)
    c = np.max(p)
    dist = []
    for _ in range(n):
        while True:
            u1 = np.random.random()
            i = np.floor(u1 * k).astype(np.int8)
            if np.random.random() < p[i] / c:
                dist.append(i + 1)
                break
    return dist


def alias_sampling(p: list, n: int) -> list:
    k = len(p)
    l = [i for i in range(k)]
    f = [k * p[i] for i in range(k)]
    g = [i for i in range(k) if f[i] >= 1]
    s = [i for i in range(k) if f[i] < 1]
    while len(s):
        i = g[0]
        j = s[0]
        l[j] = i
        f[i] = f[i] - (1 - f[j])
        if f[i] < 1 - 1e-6:
            g.pop(0)
            s.append(i)
        s.pop(0)
    dist = []
    for _ in range(n):
        u1 = np.random.random()
        i = np.floor(u1 * k).astype(np.int8)
        if np.random.random() < f[i]:
            dist.append(i + 1)
        else:
            dist.append(l[i] + 1)
    return dist
