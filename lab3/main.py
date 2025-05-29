import math
import matplotlib.pyplot as plt
from nm.ip import *
import sys


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

    if len(sys.argv) < 2:
        print("Ошибка: Не указано число в аргументах!")
        sys.exit(1)
    
    try:
        number = float(sys.argv[1])  
    except ValueError:
        print("Ошибка: Аргумент должен быть числом!")
        sys.exit(1)

    if number == 1:
        for Xi in [ [0.1, 0.5, 0.9, 1.3], [0.1, 0.5, 1.1, 1.3] ]:
            Yi = [f(x) for x in Xi]
            # Лагранж
            P_L = lagrange_poly(Xi, Yi)
            # Ньютон
            C_N = newton_coeffs(Xi, Yi)
            P_N = lambda x: poly_eval(x, C_N)
            # Вычисляем погрешность в x*
            x_star = 0.8
            err_L = interpolation_error(f, P_L, x_star)
            err_N = interpolation_error(f, P_N, x_star)
            print(f"Узлы: {Xi}")
            print(f"Погрешность Лагранжа в {x_star}: {err_L:e}")
            print(f"Погрешность Ньютона в {x_star}: {err_N:e}\n")
            # График
            plot_polynomials(Xi, Yi, P_L, P_N, x_star)
    elif number == 2:
        X = [0.1, 0.5, 0.9, 1.3, 1.7]
        Y = [-2.2026, -0.19315, 0.79464, 1.5624, 2.2306]
        x_star = 0.8

        # Строим сплайн
        a, b, c, d = compute_natural_cubic_spline(X, Y)
        
        # Выводим параметры сплайна
        print("Параметры сплайна:")
        for i in range(len(X)-1):
            print(f"Отрезок [{X[i]}, {X[i+1]}]:")
            print(f"  a[{i}] = {a[i]:.6f}")
            print(f"  b[{i}] = {b[i]:.6f}")
            print(f"  c[{i}] = {c[i]:.6f}")
            print(f"  d[{i}] = {d[i]:.6f}")

        # Вычисляем в x*
        y_star = spline_eval(x_star, X, a, b, c, d)
        print(f"Сплайн в x*={x_star}: {y_star:.6f}")

        # Функции для вычисления первой и второй производных сплайна
        def spline_derivative1(x, X, b, c, d):
            # находим i, что x в [X[i], X[i+1]]
            n = len(b)
            for i in range(n):
                x_left, x_right = X[i], X[i+1]
                if (x_left <= x < x_right) or (i == n-1 and x == x_right):
                    dx = x - x_left
                    return b[i] + 2*c[i]*dx + 3*d[i]*dx**2
            # Обработка случая, когда x вне диапазона узлов
            if x < X[0]:
                return b[0]  # Используем значение производной в левой границе
            if x >= X[-1]:
                i = n - 1  # Последний отрезок
                dx = X[-1] - X[-2]  # Используем последний интервал
                return b[i] + 2*c[i]*dx + 3*d[i]*dx**2  # Значение на правой границе
            return 0

        def spline_derivative2(x, X, c, d):
            # находим i, что x в [X[i], X[i+1]]
            n = len(c)
            for i in range(n):
                x_left, x_right = X[i], X[i+1]
                if (x_left <= x < x_right) or (i == n-1 and x == x_right):
                    dx = x - x_left
                    return 2*c[i] + 6*d[i]*dx
            # Обработка случая, когда x вне диапазона узлов
            if x < X[0]:
                return 2*c[0]  # Используем значение второй производной в левой границе
            if x >= X[-1]:
                i = n - 1  # Последний отрезок
                dx = X[-1] - X[-2]  # Используем последний интервал
                return 2*c[i] + 6*d[i]*dx  # Значение на правой границе
            return 0

        # График
        xs = [X[0] + i*(X[-1] - X[0])/200 for i in range(201)]
        ys = [spline_eval(x, X, a, b, c, d) for x in xs]
        ys_d1 = [spline_derivative1(x, X, b, c, d) for x in xs]
        ys_d2 = [spline_derivative2(x, X, c, d) for x in xs]
        
        # График сплайна
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 3, 1)
        plt.plot(xs, ys, label='Сплайн S(x)')
        plt.scatter(X, Y, c='red', label='Узлы')
        plt.scatter([x_star], [y_star], c='green', label='x*')
        plt.legend()
        plt.title('Кубический сплайн')
        
        # График первой производной
        plt.subplot(1, 3, 2)
        plt.plot(xs, ys_d1, label='S\'(x)')
        # Значения первой производной в узлах
        Y_d1 = [spline_derivative1(x, X, b, c, d) for x in X]
        plt.scatter(X, Y_d1, c='red', label='Узлы')
        plt.legend()
        plt.title('Первая производная')
        
        # График второй производной
        plt.subplot(1, 3, 3)
        plt.plot(xs, ys_d2, label='S\'\'(x)')
        # Значения второй производной в узлах
        Y_d2 = [spline_derivative2(x, X, c, d) for x in X]
        plt.scatter(X, Y_d2, c='red', label='Узлы')
        plt.legend()
        plt.title('Вторая производная')
        
        plt.tight_layout()
        plt.show()
    elif number == 3:
        X = [0.1, 0.5, 0.9, 1.3, 1.7, 2.1]
        Y = [-2.2026, -0.19315, 0.79464, 1.5624, 2.2306, 2.8419]

        results = {}
        for deg in [1, 2]:
            coeffs = ls_coeffs(X, Y, deg)
            err = sum_squared_errors(X, Y, coeffs)
            results[deg] = (coeffs, err)
            print(f"Степень {deg}: coeffs = {coeffs}, SSE = {err:.6f}")

        # Построение графиков
        xs = [X[0] + i*(X[-1]-X[0])/300 for i in range(301)]
        plt.figure()
        # исходные точки
        plt.scatter(X, Y, c='red', label='Данные')
        # полиномы
        for deg, (coeffs, _) in results.items():
            ys = [poly_eval(x, coeffs) for x in xs]
            plt.plot(xs, ys, label=f'Полином {deg}-й степени')

        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('МНК аппроксимация')
        plt.show()
    elif number == 4:
        X = [0.0, 1.0, 2.0, 3.0, 5.0]
        Y = [0.0, 2.0, 3.4142, 4.7321, 6.0]
        X_star = 2.0
        cf = newton_coeffs(X, Y)
        
        d1 = differentiate_poly(cf)
        d2 = differentiate_poly(d1)    
        P = lambda x: poly_eval(x, cf)
        dP = lambda x: poly_eval(x, d1)
        ddP = lambda x: poly_eval(x, d2)
        
        print(f"P({X_star})  = {P(X_star)}")
        print(f"P'({X_star}) = {dP(X_star)}")
        print(f"P''({X_star})= {ddP(X_star)}")
        
        # 5) Строим графики
        xs  = [i*0.1 for i in range(0, 61)]
        ys  = [P(x) for x in xs]
        ys1  = [dP(x) for x in xs]
        ys2  = [ddP(x) for x in xs]
        
        plt.figure()
        plt.plot(xs, ys)
        plt.title("Newton Interpolating Polynomial")
        plt.xlabel("x"); plt.ylabel("P(x)")
        
        plt.figure()
        plt.plot(xs, ys1)
        plt.title("First Derivative P'(x)")
        plt.xlabel("x"); plt.ylabel("P'(x)")
        
        plt.figure()
        plt.plot(xs, ys2)
        plt.title("Second Derivative P''(x)")
        plt.xlabel("x"); plt.ylabel("P''(x)")
        
        plt.show()
    elif number == 5:
        def f(x):
            return x**2 / (x**4 + 256)
        a = 0
        b = 2
        h1 = 0.5
        h2 = 0.25
        k = int(h1 / h2)

        # Метод прямоугольников
        I_rect1 = rectangle_method(f, a, b, h1)
        I_rect2 = rectangle_method(f, a, b, h2)
        err_rect = runge_romberg(I_rect1, I_rect2, k, 1)

        # Метод трапеций
        I_trap1 = trapezoid_method(f, a, b, h1)
        I_trap2 = trapezoid_method(f, a, b, h2)
        err_trap = runge_romberg(I_trap1, I_trap2, k, 2)

        # Метод Симпсона
        I_simp1 = simpson_method(f, a, b, h1)
        I_simp2 = simpson_method(f, a, b, h2)
        err_simp = runge_romberg(I_simp1, I_simp2, k, 4)

        print("Метод прямоугольников:")
        print(f"  h = {h1}: {I_rect1}")
        print(f"  h = {h2}: {I_rect2}")
        print(f"  Погрешность: {err_rect}\n")

        print("Метод трапеций:")
        print(f"  h = {h1}: {I_trap1}")
        print(f"  h = {h2}: {I_trap2}")
        print(f"  Погрешность: {err_trap}\n")

        print("Метод Симпсона:")
        print(f"  h = {h1}: {I_simp1}")
        print(f"  h = {h2}: {I_simp2}")
        print(f"  Погрешность: {err_simp}\n")

    else: 
        pass

if __name__ == '__main__':
    main()
