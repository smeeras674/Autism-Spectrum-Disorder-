from math import inf
import numpy as np
import numpy.matlib
from scipy.stats import levy
import time
import scipy.special


def MPA(Prey, fobj, lb, ub, Max_iter):  # Marine Predators Algorithm (MPA)
    SearchAgents_no, dim = Prey.shape[0], Prey.shape[1]
    Top_predator_pos = np.zeros((1, dim))
    Top_predator_fit = inf
    Convergence_curve = np.zeros((Max_iter))
    stepsize = np.zeros((SearchAgents_no, dim))
    fitness = np.zeros((SearchAgents_no))
    Xmin = np.matlib.repmat(np.multiply(np.ones((1, dim)), lb), SearchAgents_no, 1)
    Xmax = np.matlib.repmat(np.multiply(np.ones((1, dim)), ub), SearchAgents_no, 1)
    Iter = 0
    FADs = 0.2
    P = 0.5
    ct = time.time()
    while Iter < Max_iter:

        # ------------------- Detecting top predator -----------------
        for i in range(Prey.shape[1 - 1]):
            Flag4ub = Prey[i, :] > ub[i]
            Flag4lb = Prey[i, :] < lb[i]
            Prey[i, :] = (np.multiply(Prey[i, :], (~ (Flag4ub + Flag4lb)))) + np.multiply(ub[i], Flag4ub) + np.multiply(lb[i], Flag4lb)
            fitness[i] = fobj(Prey[i, :])
            if fitness[i] < Top_predator_fit:
                Top_predator_fit = fitness[i]
                Top_predator_pos = Prey[i, :]
        # ------------------- Marine Memory saving -------------------
        if Iter == 0:
            fit_old = fitness
            Prey_old = Prey
        Inx = (fit_old < fitness)
        # Indx = np.matlib.repmat(Inx, 1, dim)
        Indx = np.matlib.repmat(Inx, dim, 1)
        Indx = np.transpose(Indx, (1, 0))
        Prey = np.multiply(Indx, Prey_old) + np.multiply(~ Indx, Prey)
        fitness = np.multiply(Inx, fit_old) + np.multiply(~ Inx, fitness)
        fit_old = fitness
        Prey_old = Prey
        # ------------------------------------------------------------
        Elite = np.matlib.repmat(Top_predator_pos, SearchAgents_no, 1)
        CF = (1 - Iter / Max_iter) ** (2 * Iter / Max_iter)
        RL = 0.05 * levy(SearchAgents_no, dim, 1.5)
        RB = np.random.randn(SearchAgents_no, dim)
        for i in range(Prey.shape[1 - 1]):
            for j in range(Prey.shape[2 - 1]):
                R = np.random.rand()
                # ------------------ Phase 1 (Eq.12) -------------------
                if Iter < Max_iter / 3:
                    stepsize[i, j] = RB[i, j] * (Elite[i, j] - RB[i, j] * Prey[i, j])
                    Prey[i, j] = Prey[i, j] + P * R * stepsize[i, j]
                    # --------------- Phase 2 (Eqs. 13 & 14)----------------
                else:
                    if Iter > Max_iter / 3 and Iter < 2 * Max_iter / 3:
                        if i > Prey.shape[1 - 1] / 2:
                            stepsize[i, j] = RB[i, j] * (RB[i, j] * Elite[i, j] - Prey[i, j])
                            Prey[i, j] = Elite[i, j] + P * CF * stepsize[i, j]
                        else:
                            stepsize[i, j] = RL[i, j] * (Elite[i, j] - RL[i, j] * Prey[i, j])
                            Prey[i, j] = Prey[i, j] + P * R * stepsize[i, j]
                        # ----------------- Phase 3 (Eq. 15)-------------------
                    else:
                        stepsize[i, j] = RL[i, j] * (RL[i, j] * Elite[i, j] - Prey[i, j])
                        Prey[i, j] = Elite[i, j] + P * CF * stepsize[i, j]
        # ------------------ Detecting top predator ------------------
        for i in range(Prey.shape[1 - 1]):
            Flag4ub = Prey[i, :] > ub
            Flag4lb = Prey[i, :] < lb
            Prey[i, :] = (np.multiply(Prey[i, :], (~ (Flag4ub[i] + Flag4lb[i])))) + np.multiply(ub[i], Flag4ub[i]) + np.multiply(lb[i], Flag4lb[i])
            fitness[i] = fobj(Prey[i, :])
            if fitness[i] < Top_predator_fit:
                Top_predator_fit = fitness[i]
                Top_predator_pos = Prey[i, :]
        # ---------------------- Marine Memory saving ----------------
        if Iter == 0:
            fit_old = fitness
            Prey_old = Prey
        Inx = (fit_old < fitness)
        # Indx = np.matlib.repmat(Inx, 1, dim)
        Indx = np.matlib.repmat(Inx, dim, 1)
        Indx = np.transpose(Indx, (1, 0))
        Prey = np.multiply(Indx, Prey_old) + np.multiply(~ Indx, Prey)
        fitness = np.multiply(Inx, fit_old) + np.multiply(~ Inx, fitness)
        fit_old = fitness
        Prey_old = Prey
        # ---------- Eddy formation and FADs  effect (Eq 16) -----------
        if np.random.rand() < FADs:
            U = np.random.rand(SearchAgents_no, dim) < FADs
            Prey = Prey + CF * (
                np.multiply((Xmin[:10,:] + np.multiply(np.random.rand(SearchAgents_no, dim), (Xmax[:SearchAgents_no, :] - Xmin[:SearchAgents_no, :]))), U))
        else:
            r = np.random.rand()
            Rs = Prey.shape[1 - 1]
            stepsize_ = (FADs * (1 - r) + r) * (Prey[np.random.randint(Rs), :] - Prey[np.random.randint(Rs), :])
            Prey[i, :] = Prey[i, :] + stepsize_

        Convergence_curve[Iter] = Top_predator_fit
        Iter = Iter + 1
    ct -= time.time()
    return Top_predator_fit, Convergence_curve, Top_predator_pos, ct


def levy(n, m, beta):
    num = scipy.special.gamma(1 + beta) * np.sin(np.pi * beta / 2)

    den = scipy.special.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)

    sigma_u = (num / den) ** (1 / beta)

    u = np.random.randint(0, np.round(sigma_u), (n, m))
    v = np.random.randint(0, 1, (n, m))
    z = u / (np.abs(v) ** (1 / beta))
    return z
