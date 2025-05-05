import numpy as np
import random as rn
import time


def updatepopulation(x, f):
    row, col = x.shape[0], x.shape[1]
    t1 = np.amin(f)
    tindex1 = np.where(np.amin(f) == t1)
    best = x[tindex1, :]
    t2 = np.amax(f)
    tindex2 = np.where(np.amax(f) == t2)
    worst = x[tindex2, :]
    xnew = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            r = [rn.uniform(0, 1) for e in range(2)]
            xnew[i, j] = x[i, j] + r[0] * (best[0, 0, j] - abs(x[i, j])) - r[1] * (worst[0, 0, j] - abs(x[i, j]))
    return xnew


def trimr(mini, maxi, x):
    row, col = x.shape[0], x.shape[1]
    for i in range(col):
        x[x[:, i]<mini[i], i] = mini[i]
        x[x[:, i]>maxi[i], i] = maxi[i]
    z = x
    return z


def JAYA(x, objfun, lb, ub, max_iter):
    Mini = lb[1, :]
    Maxi = ub[1, :]
    row, col = x.shape[0], x.shape[1]

    bestfitness = np.zeros(max_iter)
    bestso = np.zeros((max_iter, col))
    f = np.zeros(row)
    fnew = np.zeros(row)

    for i in range(row):
        f[i] = objfun(x[i, :])

    gen = 0
    ct = time.time()
    while gen < max_iter:
        # print(gen)
        xnew = updatepopulation(x, f)
        xnew = trimr(Mini, Maxi, xnew)
        for i in range(row):
            fnew[i] = objfun(xnew[i, :])
        for i in range(row):
            a = fnew[i]
            b = f[i]
            if a < b:
                x[i, :] = xnew[i, :]
                f[i] = fnew[i]
        t1 = np.amin(f)
        index = np.where(np.amin(f) == t1)
        bestfitness[gen] = t1
        bestso = x[index, :]
        gen += 1
    ct = time.time() - ct
    bestsol = bestso[bestso.shape[0] - 1, :]
    bestfit = bestfitness[bestfitness.shape[0] - 1]
    return bestfit, bestfitness, bestsol, ct
