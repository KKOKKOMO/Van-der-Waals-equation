import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit, prange
from model_ex_1 import GasSimulation3D

# Расширяем функцию шага для учёта столкновений между молекулами
@njit(parallel=True)
def _numba_step_with_collisions(positions, velocities, dt, L, m, radius):
    N = positions.shape[0]
    min_dist_sq = (2*radius)**2

    total_impulse = 0.0
    total_collisions = 0

    # 1) Стенки — как было
    for i in prange(N):
        imp_i = 0.0
        col_i = 0
        for ax in range(3):
            p = positions[i, ax] + velocities[i, ax]*dt
            v = velocities[i, ax]
            if p < 0.0:
                imp_i += 2.0*m*abs(v); v = -v; p = 0.0; col_i += 1
            elif p > L:
                imp_i += 2.0*m*abs(v); v = -v; p = L;   col_i += 1
            positions[i, ax] = p
            velocities[i, ax] = v
        total_impulse    += imp_i
        total_collisions += col_i

    # 2) Между молекулами — только меняем скорости правильно
    for i in range(N):
        xi = positions[i]
        vi = velocities[i]
        for j in range(i+1, N):
            dx = positions[j,0] - xi[0]
            dy = positions[j,1] - xi[1]
            dz = positions[j,2] - xi[2]
            dist_sq = dx*dx + dy*dy + dz*dz

            if dist_sq < min_dist_sq:
                dist = math.sqrt(dist_sq)
                nx = dx / dist
                ny = dy / dist
                nz = dz / dist

                # относительная скорость вдоль нормали
                v_rel_x = vi[0] - velocities[j,0]
                v_rel_y = vi[1] - velocities[j,1]
                v_rel_z = vi[2] - velocities[j,2]
                v_n = v_rel_x*nx + v_rel_y*ny + v_rel_z*nz

                if v_n > 0.0:
                    # корректируем
                    velocities[i,0] -= v_n * nx
                    velocities[i,1] -= v_n * ny
                    velocities[i,2] -= v_n * nz
                    velocities[j,0] += v_n * nx
                    velocities[j,1] += v_n * ny
                    velocities[j,2] += v_n * nz
                    total_collisions += 1

    return total_impulse, total_collisions


@njit()
def run_simulation_numba_with_collisions(positions, velocities, steps, dt, L, m, radius):
    impulse = 0.0
    collisions = 0
    for _ in range(steps):
        imp_step, col_step = _numba_step_with_collisions(
            positions, velocities, dt, L, m, radius
        )
        impulse    += imp_step
        collisions += col_step
    return impulse, collisions


class GasSimulation3DWithCollisions(GasSimulation3D):
    def __init__(self, radius=1e-9, **kwargs):
        super().__init__(**kwargs)
        self.radius = radius

    def _simulate(self):
        return run_simulation_numba_with_collisions(
            self.positions, self.velocities, self.steps, self.dt, self.L, self.m, self.radius
        )
    
    def _initialize_arrays_no_overlap(self):
        np.random.seed(1)
        particles_per_axis = int(np.ceil(self.N ** (1/3)))
        spacing = self.radius * 2.1  # чуть больше диаметра
        positions = []
        for x in range(particles_per_axis):
            for y in range(particles_per_axis):
                for z in range(particles_per_axis):
                    if len(positions) < self.N:
                        pos = np.array([x, y, z]) * spacing + self.radius
                        positions.append(pos)
        self.positions = np.array(positions)
        
        # Центрируем внутри куба L
        self.positions -= self.positions.mean(axis=0)
        self.positions += self.L / 2
    
        sigma = np.sqrt(self.k_B * self.T / self.m)
        self.velocities = np.random.normal(0.0, sigma, size=(self.N, 3))