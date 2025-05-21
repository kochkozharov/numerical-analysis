from nm.du import *
import sys
import math


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

        # Точное решение и первоначальные условия

        def f(x, y1, y2):
            return (2*x*y2 - 2*y1) / (x**2 - 1)
        def y_exact(x):
            return x**2 + x + 1
        def numerical_derivative(func, x, delta):
            return (func(x + delta) - func(x - delta)) / (2 * delta)

        x0, x_end = 2.0, 3.0
        h = 0.1
        n = int((x_end - x0) / h)
        y0 = [y_exact(x0), numerical_derivative(y_exact, x0, h)]

        # Получаем решения для шагов h и h/2
        methods = [
            ("Euler", euler_system, 1),
            ("RK4", rk4_system, 4),
            ("Adams4", adams_4, 4)
        ]
        results = {}
        for name, method, p in methods:
            xs_h, ys_h = method(lambda x, y: [y[1], f(x, y[0], y[1])], x0, y0, h, n)
            xs_h2, ys_h2 = method(lambda x, y: [y[1], f(x, y[0], y[1])], x0, y0, h/2, int((x_end-x0)/(h/2)))
            # Рунге-Ромберга в конечной точке
            corrections = runge_romberg_grid(ys_h, ys_h2, p)

            rr_err = max(max(abs(d) for d in delta) for delta in corrections)

            # Погрешность сравнения с точным решением
            abs_errs = [abs(ys_h[i][0] - y_exact(xs_h[i])) for i in range(len(xs_h))]
            max_err = max(abs_errs)
            yh = ys_h[-1][0]
            results[name] = {
                "y_end": yh,
                "rr_error": rr_err,
                "max_abs_error": max_err
            }

        # Вывод результатов
        print("Метод   | RR-оценка          | Max abs error vs exact")
        for name in results:
            r = results[name]
            print(f"{name:<7} | {r['rr_error']:.9e}    | {r['max_abs_error']:.9e}")



    elif number == 2:
        a, b = 0, math.pi/6
        ya = 2
        yb = 2.5 - 0.5 * math.log(3)
        N = 20

        # Define ODE y'' = tan(x)*y' - 2*y
        def F(x, y, dy):
            return dy, math.tan(x)*dy - 2*y
        def y_exact(x):
           return math.sin(x) + 2 - math.sin(x) * math.log((1 + math.sin(x)) / (1 - math.sin(x)))

        # Shooting
        xs_s, ys_s = shooting(a, b, ya, yb, N, F)
        xs_s2, ys_s2 = shooting(a, b, ya, yb, 2*N, F)
        err_s = max_error(ys_s, y_exact, xs_s)
        rr_s = runge_romberg(ys_s, ys_s2, p=4)
        print(f"Shooting: max error = {err_s:.2e}, Runge-Romberg = {rr_s:.2e}")

        # Finite difference with p,q,g
        def p(x): return -math.tan(x)
        def q(x): return 2
        def g(x): return 0

        xs_f, ys_f = finite_difference(a, b, ya, yb, N, p, q, g)
        xs_f2, ys_f2 = finite_difference(a, b, ya, yb, 2*N, p, q, g)
        err_f = max_error(ys_f, y_exact, xs_f)
        rr_f = runge_romberg(ys_f, ys_f2, p=2)
        print(f"Finite Difference: max error = {err_f:.2e}, Runge-Romberg = {rr_f:.2e}")
    else: 
        pass

if __name__ == '__main__':
    main()
