from functools import reduce

import numpy as np


class Distribution:
    def __init__(self, arr: [float]):
        assert reduce(lambda a, b: a+b, arr) == 1
        self._distribution = arr

    def __len__(self):
        return len(self._distribution)

    def avg(self):
        return reduce(lambda value, element: value + element[0] * element[1], enumerate(self._distribution), 0)

    def variance(self):
        return reduce(
            lambda value, element: value + (element[0] ** 2) * element[1],
            enumerate(self._distribution),
            0
        ) - self.avg() ** 2

    def get_i(self, i):
        assert i > 0
        return self._distribution[i - 1]


class Equation:
    def __init__(self,
                 num_servers: int,
                 tasks_distribution: Distribution,
                 _lambda: float,
                 _mu: float,
                 problems_max_number: int = 100,
                 ):
        assert num_servers > 0
        assert len(tasks_distribution) > num_servers
        assert _lambda > 0
        assert _mu > 0

        self.matrices = []

        self.matrix = np.array([])
        self.answer = np.array([])

        self.num_servers = num_servers
        self._distribution = tasks_distribution
        self._mu = _mu
        self._lambda = _lambda
        self.problems_max_number = problems_max_number

    def build(self, last=False):
        self.matrices = []
        self.matrices.append(self.__fill_first_part())
        self.matrices.append(self.__fill_second_part())
        self.matrices.append(self.__fill_third_part())

        matrix = np.vstack(self.matrices)

        if not last:
            self.matrix = np.vstack((matrix[0:matrix.shape[0] - 1], np.ones(matrix.shape[1])))
            self.answer = np.zeros(matrix.shape[1])
            self.answer[self.answer.__len__() - 1] = 1
        else:
            self.matrix = np.vstack((np.ones(matrix.shape[1]), matrix[1:]))
            self.answer = np.zeros(matrix.shape[1])
            self.answer[0] = 1

    def solve(self):
        return np.linalg.solve(self.matrix, self.answer)

    def __fill_first_part(self):
        M = self.num_servers

        local_res = self.__get_initial_zeros_row()
        local_res[0] = -self._lambda
        local_res[1] = self._mu

        res = [local_res]

        for i in range(1, M + 1):
            local_res = self.__get_initial_zeros_row()

            for j in range(i, 0, -1):
                local_res[j - 1] = self._lambda * self._distribution.get_i(i - j + 1)

            local_res[i] = -self._lambda - i * self._mu

            if i == M:
                local_res[i + 1] = self._mu * M
            else:
                local_res[i + 1] = self._mu * (i + 1)

            res.append(local_res)

        return np.array(res)

    def __fill_second_part(self):
        K = len(self._distribution)
        M = self.num_servers
        N = self.problems_max_number

        res = []

        for i in range(M + 1, M+K+1):
            local_res = self.__get_initial_zeros_row()

            if i <= K:
                for j in range(0, M + 1):
                    local_res[j] = self._lambda * self._distribution.get_i(i-j)
            else:
                for j in range(i - K, M + 1):
                    local_res[j] = self._lambda * self._distribution.get_i(i-j)
            local_res[i] = -self._lambda - M * self._mu
            local_res[i+1] = M * self._mu
            local_res[M+K+1] = M * self._mu * self._distribution.get_i(i-M)
            res.append(local_res)

        return np.array(res)

    def __fill_third_part(self):
        """
        States from i = M + K + 1 (n=1, r=1, m=M) to M+NK,
        where N - is max problems number in the system
        :return:
        """
        K = len(self._distribution)
        M = self.num_servers
        N = self.problems_max_number

        res = []

        for i in range(M+K+1, M + N*K + 1):
            local_res = self.__get_initial_zeros_row()
            if i < M + (N-1) * K + 1:
                local_res[i - K] = self._lambda
                local_res[i] = -self._lambda - M * self._mu
                local_res[i + 1] = M * self._mu

                distrib_index = (i - M) % K

                if distrib_index == 0:
                    index = M + ((i - M) // K) * K + 1
                    local_res[index] = M * self._mu * self._distribution.get_i(K)
                else:
                    index = M + ((i - M) // K + 1) * K + 1
                    local_res[index] = M * self._mu * self._distribution.get_i(distrib_index)
            else:
                local_res[i - K] = self._lambda
                local_res[i] = - M * self._mu
                if i < M + N*K:
                    local_res[i + 1] = M * self._mu

            res.append(local_res)

        return np.array(res)

    def __get_initial_zeros_row(self):
        K = len(self._distribution)
        M = self.num_servers
        N = self.problems_max_number

        return np.zeros(M + N*K + 1)
