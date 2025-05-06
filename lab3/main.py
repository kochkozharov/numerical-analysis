import math
import matplotlib.pyplot as plt
from nm.ip import *

def interpolation_error(f, P, x_star):
    """
    Оценка реальной погрешности: |f(x_star) - P(x_star)|
    """
    return abs(f(x_star) - P(x_star))


def f(x):
    return math.log(x) + x


def plot_polynomials(X, Y, P1, P2, x_star):
    xs = [X[0] + i*(X[-1]-X[0])/200 for i in range(201)]
    ys_f = [f(x) for x in xs]
    ys1 = [P1(x) for x in xs]
    ys2 = [P2(x) for x in xs]

    plt.figure()
    plt.plot(xs, ys_f, label='f(x)')
    plt.plot(xs, ys1, '--', label='Lagrange')
    plt.plot(xs, ys2, ':', label='Newton')
    plt.scatter(X, Y, c='red')
    plt.scatter([x_star], [f(x_star)], c='green', label='x*')
    plt.legend()
    plt.show()


def main():
    for Xi in [ [0.1, 0.5, 0.9, 1.3], [0.1, 0.5, 1.1, 1.3] ]:
        Yi = [f(x) for x in Xi]
        # Лагранж
        P_L = lagrange_poly(Xi, Yi)
        # Ньютон
        P_N = newton_poly(Xi, Yi)
        # Вычисляем погрешность в x*
        x_star = 0.8
        err_L = interpolation_error(f, P_L, x_star)
        err_N = interpolation_error(f, P_N, x_star)
        print(f"Узлы: {Xi}")
        print(f"Погрешность Лагранжа в {x_star}: {err_L:e}")
        print(f"Погрешность Ньютона в {x_star}: {err_N:e}\n")
        # График
        plot_polynomials(Xi, Yi, P_L, P_N, x_star)

if __name__ == '__main__':
    main()
