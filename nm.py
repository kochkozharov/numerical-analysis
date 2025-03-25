def lu_decomposition(A):
    """
    Выполняет LU-разложение матрицы A с выбором главного элемента (partial pivoting).
    A – квадратная матрица, представлена как список списков.
    
    Возвращает кортеж (L, U, P, pivot_sign), где:
      L – нижняя треугольная матрица (единичная диагональ),
      U – верхняя треугольная матрица,
      P – вектор перестановок (список индексов),
      pivot_sign – знак перестановок (для определения определителя).
    """
    n = len(A)
    # Копии матрицы A, а также инициализация L как единичной матрицы и вектора перестановок P
    U = [row[:] for row in A]
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        L[i][i] = 1.0
    P = list(range(n))
    pivot_sign = 1

    for k in range(n):
        # Поиск главного элемента в столбце k
        pivot = k
        max_val = abs(U[k][k])
        for i in range(k + 1, n):
            if abs(U[i][k]) > max_val:
                max_val = abs(U[i][k])
                pivot = i
        if max_val == 0:
            raise ValueError("Матрица вырождена (определитель равен 0)")
        # Меняем местами строки k и pivot в U и корректируем L и P
        if pivot != k:
            U[k], U[pivot] = U[pivot], U[k]
            # Меняем только элементы L до столбца k
            for j in range(k):
                L[k][j], L[pivot][j] = L[pivot][j], L[k][j]
            P[k], P[pivot] = P[pivot], P[k]
            pivot_sign = -pivot_sign

        # Вычисление множителей и обнуление элементов ниже главного
        for i in range(k + 1, n):
            factor = U[i][k] / U[k][k]
            L[i][k] = factor
            for j in range(k, n):
                U[i][j] = U[i][j] - factor * U[k][j]
    return L, U, P, pivot_sign


def lu_solve(L, U, P, b):
    """
    Решает СЛАУ A*x = b, используя LU-разложение.
    На вход подаётся:
      L, U, P – матрицы и перестановочный вектор, полученные из lu_decomposition,
      b – свободный вектор (список).
    
    Возвращает вектор x – решение системы.
    """
    n = len(L)
    # Перестановка элементов вектора b согласно P: pb[i] = b[P[i]]
    pb = [b[P[i]] for i in range(n)]
    # Прямой ход: решаем L*y = pb
    y = [0.0] * n
    for i in range(n):
        y[i] = pb[i]
        for j in range(i):
            y[i] -= L[i][j] * y[j]
    # Обратный ход: решаем U*x = y
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, n):
            x[i] -= U[i][j] * x[j]
        x[i] /= U[i][i]
    return x


def determinant(U, pivot_sign):
    """
    Вычисляет определитель матрицы A, используя верхнюю треугольную матрицу U из LU-разложения.
    Определитель равен произведению диагональных элементов U, умноженному на знак перестановок.
    """
    n = len(U)
    det = pivot_sign
    for i in range(n):
        det *= U[i][i]
    return det


def inverse_matrix(A):
    """
    Вычисляет обратную матрицу A_inv для невырожденной квадратной матрицы A,
    используя LU-разложение и решение СЛАУ для каждого столбца единичной матрицы.
    
    Возвращает A_inv как список списков, где каждый вложенный список – строка обратной матрицы.
    """
    n = len(A)
    L, U, P, pivot_sign = lu_decomposition(A)
    # Каждый столбец обратной матрицы получается решением A*x = e_i,
    # где e_i – i-й столбец единичной матрицы.
    inv_cols = []
    for i in range(n):
        # Формируем вектор e (единичный вектор с 1 на позиции i)
        e = [0.0] * n
        e[i] = 1.0
        x = lu_solve(L, U, P, e)
        inv_cols.append(x)
    # Полученные векторы являются столбцами обратной матрицы.
    # Транспонируем их, чтобы получить строки.
    A_inv = [[inv_cols[j][i] for j in range(n)] for i in range(n)]
    return A_inv


def solve_system(A, b):
    """
    Решает систему линейных уравнений A*x = b.
    На вход подаются:
      A – квадратная матрица коэффициентов (список списков),
      b – свободный вектор (список).
    
    Возвращает решение x в виде списка.
    """
    L, U, P, pivot_sign = lu_decomposition(A)
    x = lu_solve(L, U, P, b)
    return x


def matrix_multiply(A, B):
    """
    Перемножает две матрицы A и B.
    Если A имеет размер m x n, а B – n x p, то возвращаемая матрица будет размером m x p.
    """
    m = len(A)
    n = len(A[0])
    if n != len(B):
        raise ValueError("Невозможно перемножить матрицы: число столбцов A не равно числу строк B")
    p = len(B[0])
    C = [[0.0 for _ in range(p)] for _ in range(m)]
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

def vector_matrix_multiply(A, v):
    """
    Умножает матрицу A (размера m x n) на вектор v (размера n).
    Возвращает вектор размера m.
    """
    m = len(A)
    n = len(A[0])
    if n != len(v):
        raise ValueError("Невозможно перемножить: число столбцов A не равно размеру вектора v")
    result = [0.0] * m
    for i in range(m):
        for j in range(n):
            result[i] += A[i][j] * v[j]
    return result

def thomas_algorithm(a, b, c, d):
    """
    Решает СЛАУ с трёхдиагональной матрицей методом прогонки (алгоритмом Томаса).

    Параметры:
      a : список длины n-1, содержащий элементы поддиагонали (нижняя диагональ).
      b : список длины n, содержащий элементы главной диагонали.
      c : список длины n-1, содержащий элементы наддиагонали (верхняя диагональ).
      d : список длины n, содержащий правые части системы.

    Возвращает:
      x : список длины n, содержащий решение системы.
    """
    n = len(d)
    # Создадим копии диагоналей, чтобы не изменять исходные данные
    cp = c[:]         # копия верхней диагонали
    bp = b[:]         # копия главной диагонали
    dp = d[:]         # копия вектора правых частей

    # Прямой ход (прямой проход)
    for i in range(1, n):
        factor = a[i - 1] / bp[i - 1]
        bp[i] = bp[i] - factor * cp[i - 1]
        dp[i] = dp[i] - factor * dp[i - 1]

    # Обратный ход (обратная подстановка)
    x = [0.0] * n
    x[n - 1] = dp[n - 1] / bp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = (dp[i] - cp[i] * x[i + 1]) / bp[i]
    return x


def construct_tridiagonal_matrix(a, b, c):
    """
    Восстанавливает полную матрицу СЛАУ из трёх диагоналей.
    
    Параметры:
      a : список длины n-1, элементы поддиагонали.
      b : список длины n, элементы главной диагонали.
      c : список длины n-1, элементы наддиагонали.
    
    Возвращает:
      A : список списков (полная матрица размера n x n).
    """
    n = len(b)
    A = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        A[i][i] = b[i]
        if i > 0:
            A[i][i - 1] = a[i - 1]
        if i < n - 1:
            A[i][i + 1] = c[i]
    return A