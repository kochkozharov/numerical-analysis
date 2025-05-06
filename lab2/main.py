from nm.nls import *
from math import exp, cos, sin
import matplotlib.pyplot as plt
import sys


def main():
    if len(sys.argv) < 2:
        print("Ошибка: Не указано число в аргументах!")
        sys.exit(1)
    
    try:
        number = float(sys.argv[1])  
    except ValueError:
        print("Ошибка: Аргумент должен быть числом!")
        sys.exit(1)

    if number == 1:
        def f(x):
            return x * exp(x) + x**2 - 1

        def df(x):
            return exp(x) * (1 + x) + 2*x

        def phi(x):
            return (1 + x) / (exp(x) + x + 1)
        eps = 1e-8
        x0 = 0.5
        
        root_si, errors_si = simple_iteration(phi, x0, eps)
        root_newton, errors_newton = newton(f, df, x0, eps)
        
        # Вывод результатов
        print(f"Простой итерации: корень ≈ {root_si:.12f}, итераций: {len(errors_si)}")
        print(f"Ньютона:           корень ≈ {root_newton:.12f}, итераций: {len(errors_newton)}")
        
        plt.figure()
        plt.semilogy(range(1, len(errors_si)+1), errors_si, marker='o', label='Простая итерация')
        plt.semilogy(range(1, len(errors_newton)+1), errors_newton, marker='x', label="Ньютона")
        plt.xlabel('Номер итерации')
        plt.ylabel('Погрешность')
        plt.title('Зависимость погрешности от номера итерации')
        plt.legend()
        plt.grid(True)
        plt.show()

    elif number == 2:
        def f2(x):
            x1, x2 = x
            return [2*x1 - cos(x2),
                    2*x2 - exp(x1)]

        def phi2(x):
            x1, x2 = x
            return [0.5 * cos(x2),
                    0.5 * exp(x1)]

        def jac2(x):
            x1, x2 = x
            return [[2, sin(x2)],
                    [-exp(x1), 2]]

        eps = 1e-8
        x0 = [0.5, 0.5]

        print("Метод простой итерации:")
        xs_si, err_si = generalized_simple_iteration(phi2, x0, eps)
        for i, (x, e) in enumerate(zip(xs_si[1:], err_si), 1):
            print(f" итерация {i}: x = {x}, погрешность = {e}")
        print("Решение SI:", xs_si[-1])

        print("\nМетод Ньютона:")
        xs_n, err_n = generalized_newton(f2, jac2, x0, eps)
        for i, (x, e) in enumerate(zip(xs_n[1:], err_n), 1):
            print(f" итерация {i}: x = {x}, погрешность = {e}")
        print("Решение Ньютона:", xs_n[-1])
        plt.figure()
        plt.plot(range(1, len(err_si)+1), err_si, label='Simple Iteration')
        plt.plot(range(1, len(err_n)+1), err_n, label='Newton')
        plt.yscale('log')
        plt.xlabel('Номер итерации')
        plt.ylabel('Погрешность')
        plt.title('Зависимость погрешности от номера итерации')
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        raise RuntimeError("Такого задания нет")


if __name__ == "__main__":
    main()