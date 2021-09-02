import numpy as np


# Attempt to implement SPOS algorithm
class PSO:
    def __init__(self, particles, dimensions, bounds, w, c, K):
        self.w = w                                      # inertial weight
        self.c = c                                      # acceleration
        self.K = K                                      # neighborhood size
        self.S = particles                              # population
        self.D = dimensions                             # dimensions number
        self.E = np.array((bounds[0], bounds[1])).T     # search space (boundaries)
        self.x = np.empty((0, self.D))                  # particle positions
        self.N = np.empty((self.S, self.K))             # neighborhood graph
        for i in range(0, self.S):
            self.x = np.append(self.x, [np.random.uniform(bounds[0], bounds[1])], axis=0)
        self.v = np.empty((0, self.D))                  # particle velocities
        for i in range(0, self.S):
            self.v = np.append(self.v, [np.random.uniform(bounds[0] - self.x[i], bounds[1] - self.x[i])], axis=0)
        self.p = np.copy(self.x)                        # local best position

    def reset_neighborhood(self):
        # create new neighborhood topography
        self.N = np.random.randint(0, self.S, (self.S, self.K))
        self.l = np.copy(self.p)                        # neighborhood best position
        self.l_val = np.copy(self.p_val)

        # each particle gives info to all neighbors
        for i in range(0, self.S):
            for j in range(0, self.K):
                if self.p_val[i] < self.l_val[self.N[i][j]]:
                    self.l[self.N[i][j]] = np.copy(self.p[i])

    def optimize(self, f, iterations, **kwargs):
        self.x_val = f(self.x, **kwargs)
        self.p_val = np.copy(self.x_val)
        self.g_best, self.g_best_val = self.p[np.argmin(self.p_val)], np.min(self.p_val)
        self.reset_neighborhood()
        for t in range(0, iterations):
            updated = False

            # calculate center of gravity
            G = np.add(self.x, self.c * np.add(self.p, np.subtract(self.l, 2 * self.x)) / 3)
            for i in range(0, self.S):
                if np.all(np.equal(self.p[i], self.l[i])):
                    # for those which store pest neighborhood result
                    G[i] = np.add(self.x[i], self.c * np.subtract(self.p[i], self.x[i]) / 2)

            # constructing hypersphere of this radius
            radius = np.sqrt(np.sum(np.square(np.subtract(G, self.x))))

            # x_r is random point inside hypersphere
            x_r = np.empty((0, self.D))
            for i in range(0, self.S):
                temp = np.random.uniform(-1, 1, self.D)
                r = np.sqrt(np.sum(np.square(temp)))
                temp = temp / r * np.random.uniform(0, radius)
                x_r = np.append(x_r, [temp], axis=0)

            # update velocity and position
            self.v, self.x = np.subtract(np.add(self.w * self.v, x_r), self.x), np.add(self.w * self.v, x_r)

            # check for out-of-bounds particles
            for i in range(0, self.S):
                for d in range(0, self.D):
                    if self.x[i][d] < self.E[d][0]:
                        self.x[i][d] = self.E[d][0]
                        self.v[i][d] = -0.5 * self.v[i][d]
                    elif self.x[i][d] > self.E[d][1]:
                        self.x[i][d] = self.E[d][1]
                        self.v[i][d] = -0.5 * self.v[i][d]

            # calculate values for new positions
            self.x_val = f(self.x, **kwargs)
            for i in range(0, self.S):
                if self.x_val[i] < self.p_val[i]:
                    # new local best found, write it
                    self.p[i] = np.copy(self.x[i])
                    self.p_val[i] = self.x_val[i]
                    if self.g_best_val < self.p_val[i]:
                        # new global best found, write it
                        self.g_best = np.copy(self.p[i])
                        self.g_best_val = self.p_val[i]
                        updated = True
                    if self.p_val[i] < self.l_val[i]:
                        # inform yourself
                        self.l[i] = np.copy(self.p[i])
                        self.l_val[i] = self.p_val[i]
            for i in range(0, self.S):
                # inform neighbors
                for j in range(0, self.K):
                    if self.p_val[i] < self.l_val[self.N[i][j]]:
                        self.l[self.N[i][j]] = np.copy(self.p[i])
                        self.l_val[self.N[i][j]] = self.p_val[i]
            print(t, '/', iterations, ' cost: ', self.g_best_val)

            # no progress, update neighborhood topology
            if not updated:
                self.reset_neighborhood()
        return self.g_best_val, self.g_best
