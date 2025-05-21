from nm.du import *
from nm.ip import runge_romberg
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

        # Точное решение и первоначальные условия

        def f(x, y1, y2):
            return (2*x*y2 - 2*y1) / (x**2 - 1)
        def y_exact(x):
            return x**2 + x + 1

        def dy_exact(x):
            return 2*x + 1
        x0, x_end = 2.0, 3.0
        h = 0.1
        n = int((x_end - x0) / h)
        y0 = [y_exact(x0), dy_exact(x0)]

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
            yh = ys_h[-1][0]
            yh2 = ys_h2[-2][0]
            rr_err = runge_romberg(yh, yh2, 2, p)
            # Погрешность сравнения с точным решением
            abs_errs = [abs(ys_h[i][0] - y_exact(xs_h[i])) for i in range(len(xs_h))]
            max_err = max(abs_errs)
            results[name] = {
                "y_end": yh,
                "rr_error": rr_err,
                "max_abs_error": max_err
            }

        # Вывод результатов
        print("Метод   |  y(3)    | RR-оценка  | Max abs error vs exact")
        for name in results:
            r = results[name]
            print(f"{name:<7}| {r['y_end']:.9f} | {r['rr_error']:.9e}    | {r['max_abs_error']:.9e}")



    elif number == 2:
        pass
    else: 
        pass

if __name__ == '__main__':
    main()
