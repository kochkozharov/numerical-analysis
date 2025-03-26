from math import cos, sin, atan, pi, sqrt, copysign
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
    
    U = [row[:] for row in A]
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        L[i][i] = 1.0
    P = list(range(n))
    pivot_sign = 1
    for k in range(n):
        
        pivot = k
        max_val = abs(U[k][k])
        for i in range(k + 1, n):
            if abs(U[i][k]) > max_val:
                max_val = abs(U[i][k])
                pivot = i
        if max_val == 0:
            raise ValueError("Матрица вырождена (определитель равен 0)")
        
        if pivot != k:
            U[k], U[pivot] = U[pivot], U[k]
            
            for j in range(k):
                L[k][j], L[pivot][j] = L[pivot][j], L[k][j]
            P[k], P[pivot] = P[pivot], P[k]
            pivot_sign = -pivot_sign
        
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
    
    pb = [b[P[i]] for i in range(n)]
    
    y = [0.0] * n
    for i in range(n):
        y[i] = pb[i]
        for j in range(i):
            y[i] -= L[i][j] * y[j]
    
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
    
    
    inv_cols = []
    for i in range(n):
        
        e = [0.0] * n
        e[i] = 1.0
        x = lu_solve(L, U, P, e)
        inv_cols.append(x)
    
    
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
    
    cp = c[:]         
    bp = b[:]         
    dp = d[:]         
    
    for i in range(1, n):
        factor = a[i - 1] / bp[i - 1]
        bp[i] = bp[i] - factor * cp[i - 1]
        dp[i] = dp[i] - factor * dp[i - 1]
    
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
        
        diff = max(abs(x_new[i] - x_old[i]) for i in range(n))
        iterations += 1
        if diff < eps:
            break
        x_old = x_new[:]  
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
    x = [0.0] * n  
    iterations = 0
    while iterations < max_iterations:
        x_old = x[:]  
        for i in range(n):
            sum1 = 0.0
            for j in range(i):  
                sum1 += A[i][j] * x[j]
            sum2 = 0.0
            for j in range(i + 1, n):  
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
        return 1.0, 0.0  
    
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
    
    for k in range(n):
        if k != i and k != j:
            
            a_ik = c * A[i][k] - s * A[j][k]
            a_jk = s * A[i][k] + c * A[j][k]
            A[i][k], A[j][k] = a_ik, a_jk
            
            A[k][i], A[k][j] = a_ik, a_jk
    
    
    a_ii = c**2 * A[i][i] - 2 * c * s * A[i][j] + s**2 * A[j][j]
    a_jj = s**2 * A[i][i] + 2 * c * s * A[i][j] + c**2 * A[j][j]
    a_ij = 0.0  
    
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
    
    n = len(A)
    for i in range(n):
        for j in range(i, n):
            if abs(A[i][j] - A[j][i]) > 1e-12:
                raise ValueError("Матрица не симметрична")
    
    
    V = [[0.0]*n for _ in range(n)]
    for i in range(n):
        V[i][i] = 1.0
    
    A = [row[:] for row in A]  
    errors = []
    
    for iteration in range(max_iterations):
        
        p, q = find_max_off_diagonal(A)
        max_off = abs(A[p][q])
        errors.append(max_off)
        
        if max_off < epsilon:
            break
        
        
        c, s = compute_rotation(A, p, q)
        
        
        apply_rotation(A, c, s, p, q)
        
        
        update_eigenvectors(V, c, s, p, q)
    
    
    eigenvalues = [A[i][i] for i in range(n)]
    eigenvectors = [[V[i][j] for i in range(n)] for j in range(n)]
    
    return eigenvalues, eigenvectors, errors
def householder_qr(A):
    """
    Выполняет QR-разложение матрицы A с использованием преобразований Хаусхолдера.
    
    Возвращает:
        Q - ортогональная матрица (список списков)
        R - верхняя треугольная матрица (список списков)
    """
    n = len(A)
    m = len(A[0])
    R = [row[:] for row in A]  
    Q = [[float(i == j) for j in range(n)] for i in range(n)]  
    for k in range(min(n, m)):
        
        x = [R[i][k] for i in range(k, n)]
        if all(abs(v) < 1e-15 for v in x):
            continue  
        
        alpha = -copysign(sqrt(sum(v**2 for v in x)), x[0])
        u = [x[i] - alpha if i == 0 else x[i] for i in range(len(x))]
        norm_u = sqrt(sum(v**2 for v in u))
        
        if norm_u < 1e-15:
            continue
        
        u = [v / norm_u for v in u]
        
        for j in range(k, m):
            
            dot = sum(u[i - k] * R[i][j] for i in range(k, n))
            for i in range(k, n):
                R[i][j] -= 2 * u[i - k] * dot
        
        for j in range(n):
            dot = sum(u[i - k] * Q[i][j] for i in range(k, n))
            for i in range(k, n):
                Q[i][j] -= 2 * u[i - k] * dot
    
    Q = [[Q[i][j] for i in range(n)] for j in range(n)]
    return Q, R
def qr_algorithm(A, epsilon=1e-10, max_iterations=500):
    """
    QR-алгоритм с неявными сдвигами для поиска собственных значений.
    
    Параметры:
        A - исходная квадратная матрица (список списков)
        epsilon - точность определения сходимости
        max_iterations - максимальное число итераций
        
    Возвращает:
        eigenvalues - список собственных значений (возможно комплексные)
    """
    n = len(A)
    H = [row[:] for row in A]  
    
    for _ in range(max_iterations):
        
        converged = True
        for i in range(n-1):
            if abs(H[i+1][i]) > epsilon:
                converged = False
                break
        if converged:
            break
        
        
        mu = H[n-1][n-1]
        for i in range(n):
            H[i][i] -= mu
        
        Q, R = householder_qr(H)
        H = matrix_multiply(R, Q)
        
        for i in range(n):
            H[i][i] += mu
    
    
    eigenvalues = []
    i = 0
    while i < n:
        if i == n-1 or abs(H[i+1][i]) < epsilon:
            eigenvalues.append(H[i][i])
            i += 1
        else:
            
            a = H[i][i]
            b = H[i][i+1]
            c = H[i+1][i]
            d = H[i+1][i+1]
            trace = a + d
            det = a*d - b*c
            discr = trace**2 - 4*det
            if discr < 0:
                real_part = trace / 2
                imag_part = sqrt(-discr) / 2
                eigenvalues.append(complex(real_part, imag_part))
                eigenvalues.append(complex(real_part, -imag_part))
            else:
                eigenvalues.append((trace + sqrt(discr)) / 2)
                eigenvalues.append((trace - sqrt(discr)) / 2)
            i += 2
    
    return eigenvalues
def is_upper_triangular(A, epsilon):
    """Проверяет, является ли матрица верхней треугольной с точностью epsilon."""
    n = len(A)
    for i in range(n):
        for j in range(i):
            if abs(A[i][j]) > epsilon:
                return False
    return True
