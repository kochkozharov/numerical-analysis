from nm.du import *
import sys
import math
import matplotlib.pyplot as plt
import numpy as np

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
            ("Euler", euler, 1),
            ("RK4", rk4, 4),
            ("Adams4", adams_4, 4)
        ]
        
        # Создаем график для сравнения методов
        plt.figure(figsize=(12, 8))
        # Добавляем точное решение (для сравнения)
        x_exact = np.linspace(x0, x_end, 100)
        y_exact_values = [y_exact(x) for x in x_exact]
        plt.plot(x_exact, y_exact_values, 'k-', linewidth=2, label='Точное решение')
        
        results = {}
        for name, method, p in methods:
            xs_h, ys_h = method(lambda x, y: [y[1], f(x, y[0], y[1])], x0, y0, h, n)
            print(f"\n=== Метод: {name} ===")
            print(f"{'i':>2} {'x_i':>8} {' y1_i':>12} {' Δy1_i':>12} {' exact':>12} {' err':>12}")
            print("-"*(2+1+8+1+12+1+12+1+12+1+12))

            # Печатаем узлы и приращения
            prev_y1, prev_y2 = 0.0, 0.0
            for i, (x, y) in enumerate(zip(xs_h, ys_h)):
                y1 = y
                dy1 = y1 - prev_y1 if i > 0 else 0.0
                exact = y_exact(x)
                err = abs(exact - y1)
                print(f"{i:2d} {x:8.4f} {y1:12.6f} {dy1:12.6f} {exact:12.6f} {err:12.9f}")

                prev_y1 = y1
            xs_h2, ys_h2 = method(lambda x, y: [y[1], f(x, y[0], y[1])], x0, y0, h/2, int((x_end-x0)/(h/2)))
            # Рунге-Ромберга в конечной точке
            rr_err = runge_romberg(ys_h, ys_h2, p)

            # Погрешность сравнения с точным решением
            abs_errs = max_error(ys_h, y_exact, xs_h)
            yh = ys_h[-1]
            results[name] = {
                "y_end": yh,
                "rr_error": rr_err,
                "max_abs_error": abs_errs
            }
            
            # Добавляем линию метода на график
            plt.plot(xs_h, ys_h, 'o-', label=f'{name}')

        # Вывод результатов
        print("Метод   | RR-оценка          | Max abs error vs exact")
        for name in results:
            r = results[name]
            print(f"{name:<7} | {r['rr_error']:.9e}    | {r['max_abs_error']:.9e}")
            
        # Настройка графика для первой задачи
        plt.title('Сравнение методов решения ОДУ 2-го порядка')
        plt.xlabel('x')
        plt.ylabel('y(x)')
        plt.grid(True)
        plt.legend()
        plt.show()


    elif number == 2:

        def print_comparison_table(xs, ys, y_exact_func):
            print(f"{'x':>10} | {'y числ.':>15} | {'y точн.':>15} | {'|ошибка|':>12}")
            print("-" * 60)
            for x, y_num in zip(xs, ys):
                y_true = y_exact_func(x)
                error = abs(y_num - y_true)
                print(f"{x:10.5f} | {y_num:15.8f} | {y_true:15.8f} | {error:12.2e}")
        a, b = 0, math.pi/6
        ya = 2
        yb = 2.5 - 0.5 * math.log(3)
        N = 20

        def F(x, y, dy):
            return dy, math.tan(x)*dy - 2*y
        def y_exact(x):
           return math.sin(x) + 2 - math.sin(x) * math.log((1 + math.sin(x)) / (1 - math.sin(x)))

        # Создаем график для сравнения методов
        plt.figure(figsize=(12, 8))
        
        # Добавляем точное решение (для сравнения)
        x_exact = np.linspace(a, b, 100)
        y_exact_values = [y_exact(x) for x in x_exact]
        plt.plot(x_exact, y_exact_values, 'k-', linewidth=2, label='Точное решение')

        # Shooting
        xs_s, ys_s, iters_s = shooting(a, b, ya, yb, N, F)
        xs_s2, ys_s2, iters_s2 = shooting(a, b, ya, yb, 2*N, F)
        err_s = max_error(ys_s, y_exact, xs_s)
        rr_s = runge_romberg(ys_s, ys_s2, p=4)
        print_comparison_table(xs_s, ys_s, y_exact)
        print_comparison_table(xs_s2, ys_s2, y_exact)
        print(f"Shooting: max error = {err_s:.2e}, Runge-Romberg = {rr_s:.2e}")
        print(f"Shooting: iterations {N} = {iters_s}")
        print(f"Shooting: iterations {2*N} = {iters_s2}")
        
        # Добавляем метод стрельбы на график
        plt.plot(xs_s, ys_s, 'o-', label=f'Метод стрельбы (N={N})')
        plt.plot(xs_s2, ys_s2, 's-', label=f'Метод стрельбы (N={2*N})')

        # Finite difference with p,q,g
        def p(x): return -math.tan(x)
        def q(x): return 2
        def g(x): return 0

        xs_f, ys_f = finite_difference(a, b, ya, yb, N, p, q, g)
        xs_f2, ys_f2 = finite_difference(a, b, ya, yb, 2*N, p, q, g)
        err_f = max_error(ys_f, y_exact, xs_f)
        rr_f = runge_romberg(ys_f, ys_f2, p=2)
        print_comparison_table(xs_f, ys_f, y_exact)
        print(f"Finite Difference: max error = {err_f:.2e}, Runge-Romberg = {rr_f:.2e}")
        
        # Добавляем конечно-разностный метод на график
        plt.plot(xs_f, ys_f, 'x-', label=f'Конечно-разностный метод (N={N})')
        plt.plot(xs_f2, ys_f2, 'd-', label=f'Конечно-разностный метод (N={2*N})')
        
        # Настройка графика для второй задачи
        plt.title('Сравнение методов решения краевой задачи')
        plt.xlabel('x')
        plt.ylabel('y(x)')
        plt.grid(True)
        plt.legend()
        plt.show()
        
        # График ошибок
        plt.figure(figsize=(12, 8))
        errors_s = [abs(y - y_exact(x)) for x, y in zip(xs_s, ys_s)]
        errors_f = [abs(y - y_exact(x)) for x, y in zip(xs_f, ys_f)]
        plt.semilogy(xs_s, errors_s, 'o-', label=f'Ошибка метода стрельбы (N={N})')
        plt.semilogy(xs_f, errors_f, 'x-', label=f'Ошибка конечно-разностного метода (N={N})')
        plt.title('Сравнение ошибок методов')
        plt.xlabel('x')
        plt.ylabel('|ошибка|')
        plt.grid(True)
        plt.legend()
        plt.show()
    else: 
        pass

if __name__ == '__main__':
    main()
