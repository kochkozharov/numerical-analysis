from nm import *
import sys

def print_matrix(A):
    for row in A:
        # Форматируем каждый элемент в строке с шириной 5 символов
        print(" ".join(f"{elem:5}" for elem in row))

# Пример использования:
if __name__ == "__main__":

# Проверка наличия аргумента
    if len(sys.argv) < 2:
        print("Ошибка: Не указано число в аргументах!")
        sys.exit(1)

    # Проверка, что аргумент — число
    try:
        number = float(sys.argv[1])  # или int(), если нужно целое
    except ValueError:
        print("Ошибка: Аргумент должен быть числом!")
        sys.exit(1)
    if number == 1:
        n = int(input())
        
        A = []
        for i in range(n):
            row = list(map(float, input().split()))
            A.append(row)
        
        b = list(map(float, input().split()))
        
        # Решение системы
        x = solve_system(A, b)
        print("\nРешение системы A*x = b:")
        print(x)
        
        # LU-разложение для получения определителя
        L, U, P, pivot_sign = lu_decomposition(A)
        det = determinant(U, pivot_sign)
        print("\nОпределитель матрицы A:")
        print(det)
        
        # Вычисление обратной матрицы
        try:
            A_inv = inverse_matrix(A)
            print("\nОбратная матрица A:")
            print_matrix(A_inv)
        except Exception as e:
            print("Не удалось вычислить обратную матрицу:", e)
        print("\nПроверка обратной матрицы A_inv:")
        print_matrix(matrix_multiply(A,A_inv))
        print("\nПроверка решения СЛАУ:")
        print(vector_matrix_multiply(A, x))
    elif number == 2:
        n = int(input())
        a = list(map(float, input().split()))
        b = list(map(float, input().split()))
        c = list(map(float, input().split()))
        d = list(map(float, input().split()))
        x = thomas_algorithm(a, b, c, d)
        print("Решение системы A*x = b:")
        print(x)
        A = construct_tridiagonal_matrix(a, b, c)
        print("Проверка решения СЛАУ:")
        print(vector_matrix_multiply(A, x))
    elif number == 3:
        n = int(input())
        
        # Ввод матрицы
        A = []
        for i in range(n):
            row = list(map(float, input().split()))
            A.append(row)
        
        # Ввод вектора правых частей
        b = list(map(float, input().split()))

        eps = float(input())

        # Метод простых итераций (Якоби)
        x_jacobi, iterations_jacobi = simple_iteration_method(A, b, eps)
        print("Метод простых итераций (Якоби):")
        print("Найденное решение:", x_jacobi)
        print("Количество итераций:", iterations_jacobi)

        print("\nПроверка решения СЛАУ:")
        print(vector_matrix_multiply(A, x_jacobi))

        # Метод Зейделя (Gauss-Seidel)
        x_seidel, iterations_seidel = gauss_seidel_method(A, b, eps)
        print("\nМетод Зейделя (Gauss–Seidel):")
        print("Найденное решение:", x_seidel)
        print("Количество итераций:", iterations_seidel)

        print("\nПроверка решения СЛАУ:")
        print(vector_matrix_multiply(A, x_seidel))
    elif number == 4:
        print("Введите размер квадратной матрицы:")
        n = int(input())
        
        # Ввод матрицы
        print("Введите матрицу (каждая строка через пробел):")
        A = []
        for i in range(n):
            row = list(map(float, input().split()))
            A.append(row)
        
        print("Введите точность:")
        eps = float(input())
        eigenvalues, eigenvectors, error_list = jacobi_rotation_method(A, eps)
        print("Собственные значения:")
        for val in eigenvalues:
            print(f"{val:10.6f}")

        print("\nСобственные векторы:")
        for v in eigenvectors:
            print(v)
        print(f"\nКоличество итераций: {len(error_list)-1}")
        print("\nЗначения максимальной погрешности на каждой итерации:")
        for i, err in enumerate(error_list):
            print(f"Итерация {i:4d}: погрешность = {error_list[i]:.6e}")
        print(f"Финальная погрешность: {error_list[-1]:.6e}")

        print("\nПроверка (A*v_i == S_i*v_i):")
        for eigenvector, eigenvalue in zip(eigenvectors, eigenvalues):
            expected = vector_matrix_multiply(A, eigenvector)
            actual = [eigenvalue * i for i in eigenvector]
            print("Ожидаемый результат: ", expected)
            print("Полученный результат: ", actual)
    else:
        raise RuntimeError("Такого задания нет")