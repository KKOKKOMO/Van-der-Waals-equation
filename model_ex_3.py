import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit, prange
from model_ex_2 import GasSimulation3DWithCollisions


# Функция шага с LJ-взаимодействием
@njit(parallel=True)
def _numba_step_with_collisions_and_LJ(
    positions, velocities,
    dt, L, m, radius,
    epsilon, sigma_LJ, r_cutoff
):
    N = positions.shape[0]
    R2 = (2*radius)**2
    rc2 = r_cutoff**2

    total_impulse = 0.0
    total_collisions = 0

    # 1) Стенки
    for i in prange(N):
        imp_i = 0.0
        col_i = 0
        for ax in range(3):
            p = positions[i,ax] + velocities[i,ax] * dt
            v = velocities[i,ax]
            if p < 0.0:
                imp_i += 2*m*abs(v); v = -v; p = 0.0; col_i += 1
            elif p > L:
                imp_i += 2*m*abs(v); v = -v; p = L;   col_i += 1
            positions[i,ax]  = p
            velocities[i,ax] = v
        total_impulse    += imp_i
        total_collisions += col_i

    # 2) Собираем силы LJ
    forces = np.zeros((N,3), dtype=np.float64)
    for i in range(N):
        xi = positions[i]
        for j in range(i+1, N):
            dx = positions[j,0] - xi[0]
            dy = positions[j,1] - xi[1]
            dz = positions[j,2] - xi[2]
            dist2 = dx*dx + dy*dy + dz*dz

            # 2.1 жёсткое столкновение
            if dist2 < R2:
                dvx = velocities[j,0] - velocities[i,0]
                dvy = velocities[j,1] - velocities[i,1]
                dvz = velocities[j,2] - velocities[i,2]
                dot = dvx*dx + dvy*dy + dvz*dz
                if dot < 0.0:
                    dist = math.sqrt(dist2)
                    nx = dx/dist; ny = dy/dist; nz = dz/dist
                    p_imp = dot
                    dpx = p_imp*nx; dpy = p_imp*ny; dpz = p_imp*nz
                    velocities[i,0] += dpx/m
                    velocities[i,1] += dpy/m
                    velocities[i,2] += dpz/m
                    velocities[j,0] -= dpx/m
                    velocities[j,1] -= dpy/m
                    velocities[j,2] -= dpz/m
                    total_collisions += 1

            # 2.2 LJ-сила
            elif dist2 < rc2:
                inv_r2 = 1.0 / dist2
                inv_r6 = inv_r2**3
                inv_r12 = inv_r6**2
                F = 24*epsilon * inv_r2 * (2*(sigma_LJ**12)*inv_r12 - (sigma_LJ**6)*inv_r6)
                fx = F * dx
                fy = F * dy
                fz = F * dz
                forces[i,0] += fx; forces[i,1] += fy; forces[i,2] += fz
                forces[j,0] -= fx; forces[j,1] -= fy; forces[j,2] -= fz

    # 3) Коррекция скоростей по силам
    for i in prange(N):
        velocities[i,0] += (forces[i,0]/m)*dt
        velocities[i,1] += (forces[i,1]/m)*dt
        velocities[i,2] += (forces[i,2]/m)*dt

    return total_impulse, total_collisions

# Обёртка запуска
@njit()
def run_simulation_numba_with_LJ(
    positions, velocities, steps, dt, L, m, radius,
    epsilon, sigma_LJ, r_cutoff
):
    impulse = 0.0
    collisions = 0
    for _ in range(steps):
        imp, col = _numba_step_with_collisions_and_LJ(
            positions, velocities,
            dt, L, m, radius,
            epsilon, sigma_LJ, r_cutoff
        )
        impulse    += imp
        collisions += col
    return impulse, collisions


# Новый подкласс
class GasSimulation3DWithLennardJones(GasSimulation3DWithCollisions):
    def __init__(
        self,
        dt: float,
        epsilon: float,
        sigma_LJ: float,
        r_cutoff: float,
        radius: float = 1e-9,
        **kwargs
    ):
        super().__init__(dt=dt, radius=radius, **kwargs)
        # Параметры LJ
        self.epsilon = epsilon
        self.sigma_LJ = sigma_LJ
        self.r_cutoff = r_cutoff

    def _simulate(self):
        # Вызываем обновлённый симулятор с LJ
        return run_simulation_numba_with_LJ(
            self.positions, self.velocities,
            self.steps, self.dt, self.L,
            self.m, self.radius,
            self.epsilon, self.sigma_LJ, self.r_cutoff
        )
