import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import njit, prange


np.random.seed(8)
@njit(parallel=True)
def _numba_step(positions, velocities, dt, L, m):
    """
    Один шаг: двигаем молекулы, считаем отражения от стенок и
    накапливаем импульс и число столкновений.
    """
    N = positions.shape[0]
    total_impulse = 0.0
    total_collisions = 0

    # Распараллеленный по i цикл: Numba сам сведёт редукцию total_impulse и total_collisions
    for i in prange(N):
        # локальные кванты для этой молекулы
        imp_i = 0.0
        col_i = 0
        for ax in range(3):
            pos = positions[i, ax] + velocities[i, ax] * dt
            vel = velocities[i, ax]

            if pos < 0.0:
                imp_i += 2.0 * m * abs(vel)
                vel = -vel
                pos = 0.0
                col_i += 1
            elif pos > L:
                imp_i += 2.0 * m * abs(vel)
                vel = -vel
                pos = L
                col_i += 1

            positions[i, ax] = pos
            velocities[i, ax] = vel

        # Наращиваем глобальные редукции
        total_impulse    += imp_i
        total_collisions += col_i

    return total_impulse, total_collisions

@njit
def run_simulation_numba(positions, velocities, steps, dt, L, m):
    impulse = 0.0
    collisions = 0
    for _ in range(steps):
        imp_step, col_step = _numba_step(  # ← без лишних аргументов
            positions, velocities, dt, L, m
        )
        impulse    += imp_step
        collisions += col_step
    return impulse, collisions

class GasSimulation3D:
    """
    Класс для моделирования идеального газа в кубической камере.
    """
    def __init__(
        self,
        N: int = 10000,
        T: float = 300.0,
        L: float = 1.0,
        dt: float = 1e-5,
        steps: int = 10000,
        m: float = 1.67e-27,
        k_B: float = 1.38e-23,
    ):
        self.N = N
        self.T = T
        self.L = L
        self.dt = dt
        self.steps = steps
        self.m = m
        self.k_B = k_B

        self.positions: np.ndarray = np.empty((0, 3))
        self.velocities: np.ndarray = np.empty((0, 3))
        self.P_sim: float = 0.0

    def _initialize_arrays(self) -> None:
        """
        Генерирует начальные позиции и скорости по нормальному распределению.
        """
        np.random.seed(1)
        self.positions = np.random.rand(self.N, 3) * self.L  # V(rms)^2 = SUM(V_x^2 + V_y^2 + V_z^2)  E_k = 0.5*m*V(rms)^2 = 3/2* K * T
        #v_rms = np.sqrt(3 * self.k_B * self.T / self.m)
        sigma = np.sqrt(self.k_B * self.T / self.m)
        self.velocities = np.random.normal(0.0, sigma, size=(self.N, 3))

    def _simulate(self) -> float:
        """
        Запускает numba-ускоренную симуляцию и возвращает суммарный импульс.
        """
        return run_simulation_numba(
            self.positions, self.velocities, self.steps, self.dt, self.L, self.m
        )

    def run(self) -> float:
        """
        Выполняет один запуск симуляции: инициализация + расчет давления.
        """
        self._initialize_arrays()
        impulse, collisions = self._simulate()
        total_time = self.steps * self.dt
        surface_area = 6 * self.L**2
        self.P_sim = impulse / (total_time * surface_area)
        return self.P_sim, collisions

    def ideal_pressure(self) -> float:
        return self.N * self.k_B * self.T / (self.L**3)

    def actual_temperature(self) -> float:
        KE = 0.5 * self.m * np.sum(self.velocities**2) # E_K = 0.5*m*sum(V**2) E_k = 3/2*N*K*T
        return (2.0 / 3.0) * KE / (self.N * self.k_B)

    def _sweep(
        self,
        values: np.ndarray,
        setter: callable,
        qty_name: str
    ) -> tuple[np.ndarray, list[float], list[float], list[float], list[float]]:
        """
        Общий метод для параметрического обхода (sweep) по списку значений.
        Сохраняет исходное значение атрибута qty_name и восстанавливает его после выполнения.
        Возвращает: значения, P_sim, P_ideal, T_act, относительную ошибку P_err.
        """
        records = []
        original_value = getattr(self, qty_name)
        try:
            for val in tqdm(values, desc=f"Sweep {qty_name}"):
                setter(val)
                P_sim, collisions = self.run()
                P_ideal = self.ideal_pressure()
                T_act   = self.actual_temperature()
                P_err   = abs(P_sim - P_ideal) / P_ideal * 100.0
                total_time = self.steps * self.dt
                collision_rate = collisions / total_time
                V = self.L**3

                records.append({
                qty_name:    val,
                'N (particles)': self.N,
                'P_sim (Pa)': P_sim,
                'P_ideal (Pa)': P_ideal,
                'T_actual (K)': T_act,
                'T_ideal (K)': self.T,
                'Volume (m^3)': V,
                'Number of collisions': collisions,
                'Collision rate (1/s)': collision_rate,
                'P_error (%)': P_err,
                'Total time (s)': self.dt * self.steps
                })
        finally:
            setattr(self, qty_name, original_value)

        return pd.DataFrame(records)

    def sweep_T(self, T_list: list[float]) -> tuple:
        return self._sweep(T_list, lambda T: setattr(self, 'T', T), 'T')

    def sweep_N(self, N_list: list[int]) -> tuple:
        return self._sweep(N_list, lambda N: setattr(self, 'N', int(N)), 'N')

    def sweep_V(self, L_list: list[int]) -> tuple:
        return self._sweep(L_list, lambda L: setattr(self, 'L', float(L)), 'L')

    def plot_with_error(
        self,
        x: np.ndarray,
        y1: list[float],
        y2: list[float],
        err: list[float],
        *,
        xlabel: str,
        ylabel: str,
        errlabel: str,
        title: str,
        suptitle_title: str,
        error_title: str = 'Approximation error to ideal gas',
        marker_size: float = 5.0,
        figsize: tuple[float, float] = (10, 4)
        ) -> None:

        # Первый график: Симуляция vs Идеальный газ
        fig, (ax1, ax2) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=figsize
        )
        fig.suptitle(suptitle_title)
        # Левый: Симуляция vs Идеальный газ
        ax1.plot(x, y1, 'o-', label='Simulation', markersize=marker_size)
        ax1.plot(x, y2, 'x--', label='ideal gas', markersize=marker_size)
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.legend(loc='best')
        ax1.grid(True)
        ax1.set_title(f'Our model: {title}')

        # Правый: Ошибка аппроксимации
        ax2.plot(x, err, 's:', label=errlabel, markersize=marker_size)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(errlabel)
        ax2.legend(loc='best')
        ax2.grid(True)
        ax2.set_title(error_title)

        fig.tight_layout()
        plt.show()