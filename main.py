import numpy as np
import pyswarms as ps
from scipy.integrate import odeint
from covid19dh import covid19
import matplotlib.pyplot as plt
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)

df, _ = covid19("UKR", raw=False, start="2020-03-03", end="2021-02-03")
df = df.reset_index()
I_target = np.subtract(df['confirmed'].to_numpy(), np.add(df['recovered'].to_numpy(), df['deaths'].to_numpy()))
N = df.at[0, 'population']


def gof(params):
    y = ode_solver(params)
    res = 0
    for t in range(0, len(df)):
        res += np.abs(np.add(y[:, t, 1], y[:, t, 2]) - I_target[t]) / len(df)
    return res


def seis(y, t, params):
    a, b, m, r, q, e = params
    S, E, I = y
    return [a - b*S/N*I - m*S + r*I, b*S/N*I - (q + m) * E, e*E - (r + m) * I]


def ode_solver(params):
    res = np.empty((0, len(df), 3))
    for particle in params:
        res = np.append(res, [odeint(seid, (N-1, 1, 0), np.arange(len(df)), args=(particle,))], axis=0)
    return res


options = {'c1': 0.3, 'c2': 0.5, 'w': 0.9}

# Call instance of PSO
optimizer = ps.single.GlobalBestPSO(n_particles=150, dimensions=6, options=options, bounds=(np.zeros(6), np.ones(6)))

# Perform optimization
cost, pos = optimizer.optimize(gof, iters=200)

t = np.arange(len(df))

print(pos)

sol = odeint(seird, (N-1, 1, 0), np.arange(len(I_target)), args=(pos,))

plt.plot(t, np.add(sol[:, 2], sol[:, 1]), 'b', label='model')
plt.plot(t, I_target, 'g', label='target')
plt.legend(loc='best')
plt.show()
plot_cost_history(cost_history=optimizer.cost_history)
plt.show()

