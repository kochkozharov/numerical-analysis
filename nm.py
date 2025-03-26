from math import cos, sin, atan, pi, sqrt

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

def simple_iteration_method(A, b, eps, max_iterations=10000):
    """
    Решает СЛАУ A*x = b методом простых итераций (метод Якоби).
    
    Параметры:
      A : квадратная матрица системы (2D список)
      b : вектор правых частей (список)
      eps : требуемая точность (остановка итераций, когда максимальное изменение меньше eps)
      max_iterations : максимальное число итераций (по умолчанию 10000)
    
    Возвращает:
      x : найденное решение системы (список)
      iterations : количество итераций, потребовавшихся для сходимости
    """
    n = len(A)
    # Начальное приближение (нулевой вектор)
    x_old = [0.0] * n
    x_new = [0.0] * n
    iterations = 0

    while iterations < max_iterations:
        for i in range(n):
            sum_val = 0.0
            for j in range(n):
                if j != i:
                    sum_val += A[i][j] * x_old[j]
            x_new[i] = (b[i] - sum_val) / A[i][i]

        # Определяем максимальное изменение между итерациями
        diff = max(abs(x_new[i] - x_old[i]) for i in range(n))
        iterations += 1
        if diff < eps:
            break
        x_old = x_new[:]  # копирование для следующей итерации

    return x_new, iterations


def gauss_seidel_method(A, b, eps, max_iterations=10000):
    """
    Решает СЛАУ A*x = b методом Зейделя (Gauss–Seidel).
    
    Параметры:
      A : квадратная матрица системы (2D список)
      b : вектор правых частей (список)
      eps : требуемая точность (остановка итераций, когда максимальное изменение меньше eps)
      max_iterations : максимальное число итераций (по умолчанию 10000)
    
    Возвращает:
      x : найденное решение системы (список)
      iterations : количество итераций, потребовавшихся для сходимости
    """
    n = len(A)
    x = [0.0] * n  # начальное приближение
    iterations = 0

    while iterations < max_iterations:
        x_old = x[:]  # сохраняем предыдущие значения для оценки сходимости
        for i in range(n):
            sum1 = 0.0
            for j in range(i):  # используем уже обновленные значения
                sum1 += A[i][j] * x[j]
            sum2 = 0.0
            for j in range(i + 1, n):  # используем значения предыдущей итерации
                sum2 += A[i][j] * x_old[j]
            x[i] = (b[i] - sum1 - sum2) / A[i][i]
        iterations += 1
        diff = max(abs(x[i] - x_old[i]) for i in range(n))
        if diff < eps:
            break

    return x, iterations

def find_max_off_diagonal(A):
    """Находит максимальный по модулю внедиагональный элемент матрицы и возвращает его индексы (i, j)."""
    n = len(A)
    max_val = 0.0
    p, q = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i][j]) > max_val:
                max_val = abs(A[i][j])
                p, q = i, j
    return p, q

def compute_rotation(A, i, j):
    """Вычисляет параметры вращения (cos, sin) для обнуления элемента A[i][j]."""
    if A[i][j] == 0:
        return 1.0, 0.0  # Нет вращения
    
    tau = (A[j][j] - A[i][i]) / (2 * A[i][j])
    if tau >= 0:
        t = 1.0 / (tau + sqrt(1 + tau**2))
    else:
        t = -1.0 / (-tau + sqrt(1 + tau**2))
    
    c = 1.0 / sqrt(1 + t**2)
    s = t * c
    return c, s

def apply_rotation(A, c, s, i, j):
    """Применяет вращение к матрице A для обнуления элементов A[i][j] и A[j][i]."""
    n = len(A)
    # Обновление элементов в строках i и j
    for k in range(n):
        if k != i and k != j:
            # Элементы вне диагонали
            a_ik = c * A[i][k] - s * A[j][k]
            a_jk = s * A[i][k] + c * A[j][k]
            A[i][k], A[j][k] = a_ik, a_jk
            # Симметричные элементы
            A[k][i], A[k][j] = a_ik, a_jk
    
    # Обновление диагональных элементов
    a_ii = c**2 * A[i][i] - 2 * c * s * A[i][j] + s**2 * A[j][j]
    a_jj = s**2 * A[i][i] + 2 * c * s * A[i][j] + c**2 * A[j][j]
    a_ij = 0.0  # Обнуляем элемент
    
    A[i][i], A[j][j] = a_ii, a_jj
    A[i][j] = A[j][i] = 0.0

def update_eigenvectors(V, c, s, i, j):
    """Обновляет матрицу собственных векторов после вращения."""
    n = len(V)
    for k in range(n):
        v_ki = V[k][i]
        v_kj = V[k][j]
        V[k][i] = c * v_ki - s * v_kj
        V[k][j] = s * v_ki + c * v_kj

def jacobi_rotation_method(A, epsilon=1e-10, max_iterations=1000):
    """
    Реализация метода вращений Якоби для симметрических матриц.
    
    Параметры:
        A - исходная симметрическая матрица (2D список)
        epsilon - точность вычислений
        max_iterations - максимальное число итераций
    
    Возвращает:
        eigenvalues - список собственных значений
        eigenvectors - матрица собственных векторов (каждый столбец - вектор)
        errors - история изменения погрешности
    """
    # Проверка симметричности матрицы
    n = len(A)
    for i in range(n):
        for j in range(i, n):
            if abs(A[i][j] - A[j][i]) > 1e-12:
                raise ValueError("Матрица не симметрична")
    
    # Инициализация матрицы собственных векторов
    V = [[0.0]*n for _ in range(n)]
    for i in range(n):
        V[i][i] = 1.0
    
    A = [row[:] for row in A]  # Копия матрицы
    errors = []
    
    for iteration in range(max_iterations):
        # Находим максимальный внедиагональный элемент
        p, q = find_max_off_diagonal(A)
        max_off = abs(A[p][q])
        errors.append(max_off)
        
        if max_off < epsilon:
            break
        
        # Вычисляем параметры вращения
        c, s = compute_rotation(A, p, q)
        
        # Применяем вращение к матрице
        apply_rotation(A, c, s, p, q)
        
        # Обновляем собственные векторы
        update_eigenvectors(V, c, s, p, q)
    
    # Извлекаем собственные значения и векторы
    eigenvalues = [A[i][i] for i in range(n)]
    eigenvectors = [[V[i][j] for i in range(n)] for j in range(n)]
    
    return eigenvalues, eigenvectors, errors