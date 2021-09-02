import numpy as np
import pyswarms as ps
from scipy.integrate import odeint
from covid19dh import covid19
import matplotlib.pyplot as plt
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
from pso import PSO


df_ukr = covid19("UKR", raw=False, verbose=False, start="2020-03-03", end="2021-03-01")[0].reset_index()
df_gbr = covid19("GBR", raw=False, verbose=False, start="2020-03-03", end="2021-03-01")[0].reset_index()


def mean_absolute_deviation(params, model, y0, data):
    y = swarm_ode_solver(model, y0, params, len(data))
    res = 0
    for t in range(0, len(data)):
        res += np.abs(np.add(y[:, t, 2], y[:, t, 1]) - data[t]) / len(data)
    return res


def mean_square_deviation(params, model, y0, data):
    y = swarm_ode_solver(model, y0, params, len(data))
    res = 0
    for t in range(0, len(data)):
        res += np.square(np.add(y[:, t, 2], y[:, t, 1]) - data[t]) / len(data)
    return res


def seir(y, t, params):
    """ SEIR model

        params:
        a - absolute population increase (daily)
        b - relative spread
        d - exposed recovery rate
        e1- exposed mortality rate + rate of transfer to infectious
        e2- infectious mortality + rate of recovery
        g - exposed rate of becoming infectious
        l - infectious recovery rate
        m - normal mortality rate
    """
    a, b, d, e1, e2, g, l, m = params
    S, E, I, R = y
    return [
        a - b * S / (S + E + I + R) * I - m * S,
        b * S / (S + E + I + R) * I - e1 * E,
        g * E - e2 * I,
        d * E + l * I - m * R
    ]


def seis(y, t, params):
    """ SEIR model

            params:
            a - absolute population increase (daily)
            b - relative spread
            e - infectious rate of recovery
            q - exposed mortality
            r - infectious recovery rate
            m - normal mortality rate
        """
    a, b, m, r, q, e = params
    S, E, I = y
    return [
        a - b * S / (S + E + I) * I - m * S + r * I,
        b * S / (S + E + I) * I - (q + m) * E,
        e * E - (r + m) * I
    ]

def seirs(y, t, params):
    """ SEIRS model, not used
    """
    a, b, e, g, m, r, q = params
    S, E, I, R = y
    return [
        a - b * S / (S + E + I + R) * I - m * S + g * R,
        b * S / (S + E + I + R) * I - (q + m) * E,
        e * E - (r + m) * I,
        r * I - (g + m) * R
    ]


def swarm_ode_solver(model, y0, params, n):
    """
    Solves ODE for the search-space
    :param model: ODE
    :param params: ODE parameters
    :param n: dimensions
    :return: numeric solution to given ODE
    """
    res = np.empty((0, n, len(y0)))
    for particle in params:
        res = np.append(res, [odeint(model, y0, np.arange(n), args=(particle,))], axis=0)
    return res


def display(df, model, deviation, bounds):
    """ Solves parameter optimization problem for given model and data,
        plots data via plt. Uses PSO by PySwarms
    """
    options = {'c1': 2, 'c2': 2, 'w': 0.9}
    y0 = (df.at[0, 'population'] - df.at[0, 'confirmed'], 0,  df.at[0, 'confirmed'], 0)
    data = np.subtract(df['confirmed'].to_numpy(), np.add(df['recovered'].to_numpy(), df['deaths'].to_numpy()))
    args = {'model': model, 'y0': y0, 'data': data}

    optimizer = ps.single.GlobalBestPSO(n_particles=80, dimensions=len(bounds[0]), options=options, bounds=bounds)
    cost, pos = optimizer.optimize(deviation, iters=512, **args)

    t = np.arange(len(data))
    t5 = np.arange(len(data)*5)
    sol = odeint(model, y0, t, args=(pos,))
    plt.plot(t, np.add(sol[:, 2], sol[:, 1]), 'b', label='model data')
    plt.plot(t, data, 'g', label='real data')
    plt.legend(loc='best')
    plt.show()

    print(cost)
    print(pos)

    plot_cost_history(cost_history=optimizer.cost_history)
    plt.show()

    sol = odeint(model, y0, t5, args=(pos,))
    plt.plot(t5, np.add(sol[:, 2], sol[:, 1]), 'b', label='model data')
    plt.plot(t, data, 'g', label='real data')
    plt.legend(loc='best')
    plt.show()


def display_custom(df, model, deviation, bounds):
    """ Solves parameter optimization problem for given model and data,
            plots data via plt. Uses PSO from pso.py
        """
    y0 = (df.at[0, 'population'] - df.at[0, 'confirmed'], 0,  df.at[0, 'confirmed'])
    data = np.subtract(df['confirmed'].to_numpy(), np.add(df['recovered'].to_numpy(), df['deaths'].to_numpy()))

    optimizer = PSO(40, 6, bounds, 0.9, 2, 3)
    cost, pos = optimizer.optimize(deviation, 10, model=model, y0=y0, data=data)

    t = np.arange(len(data))
    t5 = np.arange(len(data)*5)
    sol = odeint(model, y0, t, args=(pos,))
    plt.plot(t, np.add(sol[:, 2], sol[:, 1]), 'b', label='model data')
    plt.plot(t, data, 'g', label='real data')
    plt.legend(loc='best')
    plt.show()

    print(cost)
    print(pos)

    sol = odeint(model, y0, t5, args=(pos,))
    plt.plot(t5, np.add(sol[:, 2], sol[:, 1]), 'b', label='model data')
    plt.plot(t, data, 'g', label='real data')
    plt.legend(loc='best')
    plt.show()


# display_custom(df_ukr, seis, mean_absolute_deviation, (np.zeros(6), np.array([1024, 1, 1, 1, 1, 1])))
display(df_ukr, seir, mean_absolute_deviation, (np.zeros(8), np.array([1024, 1, 1, 1, 1, 1, 1, 1])))
