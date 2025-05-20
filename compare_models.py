import numpy as np
import matplotlib.pyplot as plt


def compare_models_T(
    models: dict[str, object],
    microparams: dict[str, dict],
    common_params: dict,
    T_list: np.ndarray,
    seed: int = 12345
) -> dict[str, dict]:
    """
    Run temperature sweep for multiple gas models and plot results.

    Parameters
    ----------
    models : dict
        Mapping from model name to simulation class (uninitialized).
        e.g. {
            'Ideal Gas': GasSimulation3D,
            'Hard Spheres': GasSimulation3DWithCollisions,
            'Lennard-Jones': GasSimulation3DWithLennardJones
        }
    microparams : dict
        Per-model initialization kwargs, e.g.
        {
            'Ideal Gas': {'dt': dt1},
            'Hard Spheres': {'dt': dt1, 'radius': radius},
            'Lennard-Jones': {'dt': dt3, 'radius': radius,
                              'epsilon': epsilon, 'sigma_LJ': sigma_LJ,
                              'r_cutoff': r_cutoff}
        }
    common_params : dict
        Shared initialization kwargs: N, L, T, steps, m.
    T_list : np.ndarray
        Array of temperatures to sweep.
    seed : int
        Random seed for reproducible initial conditions.

    Returns
    -------
    results : dict
        Mapping from model name to its sweep_T result dict.
    """
    # 1) Generate common initial positions & velocities
    np.random.seed(seed)
    N = common_params['N']
    L = common_params['L']
    T0 = common_params['T']
    m = common_params['m']
    radius = common_params.get('radius', None)

    # create grid positions and velocities once
    ppa = int(np.ceil(N ** (1/3)))
    coords = (np.arange(ppa) + 0.5) * (L / ppa)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
    pos0 = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T[:N]
    k_B = 1.380649e-23
    sigma_v = np.sqrt(k_B * T0 / m)
    vel0 = np.random.normal(0.0, sigma_v, size=(N, 3))

    results = {}
    # 2) Loop over each model
    for name, ModelClass in models.items():
        # instantiate with common + micro parameters
        init_kwargs = {**common_params, **microparams.get(name, {})}
        sim = ModelClass(**init_kwargs)
        # assign identical IC
        sim.positions = pos0.copy()
        sim.velocities = vel0.copy()

        # perform sweep_T
        sweep_res = sim.sweep_T(T_list)
        # Plot
        sim.plot_with_error(
            sweep_res['T_ideal (K)'],
            sweep_res['P_sim (Pa)'],
            sweep_res['P_ideal (Pa)'],
            sweep_res['P_error (%)'],
            xlabel='T, K',
            ylabel='P, Pa',
            suptitle_title = f'{name}\nN = {N}, L = {L} m^3',
            errlabel='ΔP, Pa',
            title=f'Isothermal process'
        )
        results[name] = sweep_res

    return results

def compare_models_N(
    models: dict[str, type],
    microparams: dict[str, dict],
    common_params: dict,
    N_list: np.ndarray,
    seed: int = 12345
) -> dict[str, dict]:
    """
    Run particle-number sweep for multiple gas models and plot results.

    Parameters
    ----------
    models : dict
        Названия моделей → их классы.
    microparams : dict
        Инициализационные параметры для каждой модели.
    common_params : dict
        Общие параметры: L, T, steps, dt, m, radius.
        (Поле 'N' здесь не используется — оно задаётся из N_list.)
    N_list : np.ndarray
        Массив значений числа частиц для прогона.
    seed : int
        Random seed.

    Returns
    -------
    results : dict
        Для каждой модели её результат sweep_N.
    """
    L = common_params['L']
    T = common_params['T']
    m = common_params['m']
    radius = common_params.get('radius', None)

    results = {}
    for name, ModelClass in models.items():
        # 1) каждый раз создаём симулятор с заглушкой N
        #    реальный N будет подставлен внутри sweep_N
        init_kwargs = {**common_params, **microparams.get(name, {})}
        sim = ModelClass(**init_kwargs)

        # 2) единые начальные условия на «максимальное» N
        #    (текстурно из первого элемента списка)
        N0 = int(N_list[0])
        np.random.seed(seed)
        ppa = int(np.ceil(N0 ** (1/3)))
        coords = (np.arange(ppa) + 0.5) * (L / ppa)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
        pos0 = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T[:N0]
        k_B = 1.380649e-23
        sigma_v = np.sqrt(k_B * T / m)
        vel0 = np.random.normal(0.0, sigma_v, size=(N0, 3))

        sim.positions = pos0.copy()
        sim.velocities = vel0.copy()

        # 3) прогоняем sweep_N
        sweep_res = sim.sweep_N(list(N_list))

        # 4) рисуем
        sim.plot_with_error(
            x=N_list,
            y1=sweep_res['P_sim (Pa)'],
            y2=sweep_res['P_ideal (Pa)'],
            err=sweep_res['P_error (%)'],
            xlabel='N, particles',
            ylabel='P, Pa',
            suptitle_title = f'{name}\nT = {T} K, L = {L} m^3',
            errlabel='ΔP, %',
            title=f'(P vs N at T=const, L=const)'
        )
        results[name] = sweep_res

    return results

def compare_models_V(
    models: dict[str, type],
    microparams: dict[str, dict],
    common_params: dict,
    L_list: np.ndarray,
    seed: int = 12345
    ) -> dict[str, dict]:
    """
    Run volume (cube edge length) sweep for multiple gas models and plot results.

    Parameters
    ----------
    models : dict
        Названия моделей → их классы.
    microparams : dict
        Инициализационные параметры для каждой модели.
    common_params : dict
        Общие параметры: N, T, steps, dt, m, radius.
        (Поле 'L' здесь не используется — оно задаётся из L_list.)
    L_list : np.ndarray
        Массив значений длины стороны куба.
    seed : int
        Random seed.

    Returns
    -------
    results : dict
        Для каждой модели её результат sweep_V.
    """
    N = common_params['N']
    T = common_params['T']
    m = common_params['m']
    radius = common_params.get('radius', None)

    results = {}
    for name, ModelClass in models.items():
        init_kwargs = {**common_params, **microparams.get(name, {})}
        sim = ModelClass(**init_kwargs)

        # единые IC на «минимальное» L
        L0 = float(L_list[0])
        np.random.seed(seed)
        ppa = int(np.ceil(N ** (1/3)))
        coords = (np.arange(ppa) + 0.5) * (L0 / ppa)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')
        pos0 = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T[:N]
        k_B = 1.380649e-23
        sigma_v = np.sqrt(k_B * T / m)
        vel0 = np.random.normal(0.0, sigma_v, size=(N, 3))

        sim.positions = pos0.copy()
        sim.velocities = vel0.copy()

        sweep_res = sim.sweep_V(list(L_list))

        sim.plot_with_error(
            x=L_list,
            y1=sweep_res['P_sim (Pa)'],
            y2=sweep_res['P_ideal (Pa)'],
            err=sweep_res['P_error (%)'],
            xlabel='L, m',
            ylabel='P, Pa',
            suptitle_title = f'{name}\nT = {T} K, N = {N}',
            errlabel='ΔP, %',
            title=f'Isochoric process'
        )
        results[name] = sweep_res

    return results
