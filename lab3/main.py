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

        # Вычисляем в x*
        y_star = spline_eval(x_star, X, a, b, c, d)
        print(f"Сплайн в x*={x_star}: {y_star:.6f}")

        # График
        xs = [X[0] + i*(X[-1] - X[0])/200 for i in range(201)]
        ys = [spline_eval(x, X, a, b, c, d) for x in xs]
        plt.figure()
        plt.plot(xs, ys, label='Spline')
        plt.scatter(X, Y, c='red', label='Nodes')
        plt.scatter([x_star], [y_star], c='green', label='x*')
        plt.legend()
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

    else: 
        pass

if __name__ == '__main__':
    main()
