import numpy as np

from analytics.main import Equation, Distribution


def test_main():
    distrib = Distribution([0.1, 0.01, 0.3, 0.59])
    _lambda = 0.5
    _mu = 1

    K = len(distrib)
    M = 2

    eq = Equation(
        M,
        distrib,
        _lambda,
        _mu,
        50
    )

    eq.build(last=True)

    # assert for first matrix

    first = eq.matrices[0]

    assert first[0][0] == -_lambda
    assert first[0][1] == _mu

    assert first[1][0] == _lambda * distrib.get_i(1)
    assert first[1][1] == -_lambda - _mu
    assert first[1][2] == 2 * _mu

    assert first[2][0] == _lambda * distrib.get_i(2)
    assert first[2][1] == _lambda * distrib.get_i(1)
    assert first[2][2] == -_lambda - 2 * _mu
    assert first[2][3] == 2 * _mu

    assert first.shape[0] == M+1

    # TODO add comparison other blocks to zeros

    # assert for second matrix

    second = eq.matrices[1]

    assert second[0][0] == _lambda * distrib.get_i(M+1)
    assert second[0][1] == _lambda * distrib.get_i(M)
    assert second[0][2] == _lambda * distrib.get_i(M-1)
    assert second[0][3] == -_lambda - M * _mu
    assert second[0][4] == M * _mu
    assert second[0][M + K + 1] == M * _mu * distrib.get_i(1)
    for i in range(M + 3, M+K+1):
        assert second[0][i] == 0
    for i in range(M+K+2, len(second[0])):
        assert second[0][i] == 0

    assert second[1][0] == _lambda * distrib.get_i(M+2)
    assert second[1][1] == _lambda * distrib.get_i(M+1)
    assert second[1][2] == _lambda * distrib.get_i(M)
    assert second[1][3] == 0
    assert second[1][4] == -_lambda - M * _mu
    assert second[1][5] == M * _mu
    assert second[1][M + K + 1] == M * _mu * distrib.get_i(2)
    for i in range(M + 4, M+K+1):
        assert second[1][i] == 0
    for i in range(M + K + 2, len(second[1])):
        assert second[1][i] == 0

    assert second[2][0] == 0
    assert second[2][1] == _lambda * distrib.get_i(K)
    assert second[2][2] == _lambda * distrib.get_i(K - 1)
    assert second[2][3] == 0
    assert second[2][4] == 0
    assert second[2][5] == -_lambda - M * _mu
    assert second[2][6] == M * _mu
    for i in range(M + 5, M+K+1):
        assert second[2][i] == 0
    for i in range(M + K + 2, len(second[2])):
        assert second[2][i] == 0

    assert second[3][0] == 0
    assert second[3][1] == 0
    assert second[3][2] == _lambda * distrib.get_i(K)
    assert second[3][3] == 0
    assert second[3][4] == 0
    assert second[3][5] == 0
    assert second[3][6] == -_lambda - M * _mu
    assert second[3][7] == M * _mu * distrib.get_i(K)
    for i in range(M + K + 2, len(second[3])):
        assert second[3][i] == 0

    assert second.shape[0] == K

    # assert for third matrix

    third = eq.matrices[2]

    assert third[0][M+1] == _lambda
    assert third[0][M+K+1] == -_lambda-M * _mu
    assert third[0][M+K+2] == M * _mu
    assert third[0][M+2*K+1] == M * _mu * distrib.get_i(1)

    assert third[1][M + 2] == _lambda
    assert third[1][M + K + 2] == -_lambda - M * _mu
    assert third[1][M + K + 3] == M * _mu
    assert third[1][M + 2 * K + 1] == M * _mu * distrib.get_i(2)

    assert third[2][M + 3] == _lambda
    assert third[2][M + K + 3] == -_lambda - M * _mu
    assert third[2][M + K + 4] == M * _mu
    assert third[2][M + 2 * K + 1] == M * _mu * distrib.get_i(3)

    assert third[3][M + K] == _lambda
    assert third[3][M + 2*K] == -_lambda - M * _mu
    assert third[3][M + 2*K + 1] == M * _mu * distrib.get_i(K)

    # for i in range(M + K + 1, M + K * (eq.problems_max_number - 2)):
    #     assert third[i][i-K] == _lambda
    for i in range(M+K+1, M + (eq.problems_max_number - 1) * K + 1):
        indexes = []
        row_index = i - (M+K+1)
        indexes.append(i - K)
        assert third[row_index][i - K] == _lambda
        indexes.append(i)
        assert third[row_index][i] == -_lambda - M * _mu
        if (row_index + 1) % K == 0:
            assert third[row_index][M+1+K*(i // K)] == M * _mu * distrib.get_i(K)
            indexes.append(M+1+K*(i // K))
        else:
            assert third[row_index][row_index + M + K + 2] == M * _mu
            indexes.append(row_index + M + K + 2)
            assert third[row_index][M+1+K*((i - M) // K + 1)] == M * _mu * distrib.get_i(row_index % K + 1)
            indexes.append(M+1+K*((i - M) // K + 1))

        for j in range(len(third[row_index])):
            if j not in indexes:
                assert third[row_index][j] == 0

    solution = eq.solve()
