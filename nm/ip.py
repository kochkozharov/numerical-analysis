from nm.nm import solve_system

def lagrange_poly(X, Y):
    """
    Возвращает функцию интерполяции Лагранжа на узлах (X, Y)
    """
    def P(x):
        total = 0
        n = len(X)
        for i in range(n):
            term = Y[i]
            for j in range(n):
                if j != i:
                    term *= (x - X[j]) / (X[i] - X[j])
            total += term
        return total
    return P


def newton_poly(X, Y):
    """
    Возвращает функцию по формуле Ньютона с узлами X и коэффициентами coeffs
    """
    n = len(X)
    # Заполняем таблицу разделенных разностей
    table = [ [0]*n for _ in range(n) ]
    for i in range(n):
        table[i][0] = Y[i]
    for j in range(1, n):
        for i in range(n-j):
            table[i][j] = (table[i+1][j-1] - table[i][j-1]) / (X[i+j] - X[i])
    # коэффициенты — первые элементы столбцов
    coeffs =  [ table[0][j] for j in range(n) ]
    def P(x):
        total = coeffs[0]
        prod = 1
        for j in range(1, len(coeffs)):
            prod *= (x - X[j-1])
            total += coeffs[j] * prod
        return total
    return P

def interpolation_error(f, P, x_star):
    """
    Оценка реальной погрешности: |f(x_star) - P(x_star)|
    """
    return abs(f(x_star) - P(x_star))


def compute_natural_cubic_spline(X, Y):
    """
    Строит коэффициенты натурального кубического сплайна на узлах (X,Y)
    Возвращает списки a,b,c,d для каждого интервала [X[i],X[i+1]]
    """
    n = len(X) - 1  # число отрезков
    h = [X[i+1] - X[i] for i in range(n)]
    
    # Формируем систему для вторых производных M
    A = [[0]*(n+1) for _ in range(n+1)]
    b = [0]*(n+1)
    # натуральные условия: M0 = M_n = 0
    A[0][0] = 1
    A[n][n] = 1
    
    # внутренняя часть системы
    for i in range(1, n):
        A[i][i-1] = h[i-1]
        A[i][i] = 2*(h[i-1] + h[i])
        A[i][i+1] = h[i]
        b[i] = 6*((Y[i+1] - Y[i]) / h[i] - (Y[i] - Y[i-1]) / h[i-1])

    # Решаем систему A*M = b
    M = solve_system(A, b)

    # Вычисляем коэффициенты сплайна
    a = [Y[i] for i in range(n)]
    b_coeff = [0]*n
    c = [0]*n
    d = [0]*n
    for i in range(n):
        b_coeff[i] = (Y[i+1] - Y[i]) / h[i] - (2*M[i] + M[i+1]) * h[i] / 6
        c[i] = M[i] / 2
        d[i] = (M[i+1] - M[i]) / (6*h[i])
    return a, b_coeff, c, d


def spline_eval(x, X, a, b, c, d):
    """
    Вычисляет значение натурального кубического сплайна в точке x
    """
    # находим i, что x в [X[i], X[i+1]]
    n = len(a)
    for i in range(n):
        x_left, x_right = X[i], X[i+1]
        # для последнего отрезка включаем правый конец, иначе полузакрытый
        if (x_left <= x < x_right) or (i == n-1 and x == x_right):
            dx = x - x_left
            return a[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3
    # если всё же вне, возвращаем ближайшую границу
    if x < X[0]:
        return a[0]
    if x > X[-1]:
        dx = X[-1] - X[-2]
        return a[-1] + b[-1]*dx + c[-1]*dx**2 + d[-1]*dx**3
    raise ValueError("x вне диапазона узлов")


def ls_coeffs(X, Y, degree):
    m = degree
    n = len(X)
    # Собираем элементы A^T A: элементы S_k = sum x_i^k
    S = [sum(x**k for x in X) for k in range(2*m + 1)]
    # Собираем вектор правой части T_j = sum y_i * x_i^j
    T = [sum(Y[i] * (X[i]**j) for i in range(n)) for j in range(m+1)]

    # Матрица нормальной системы размер (m+1)x(m+1)
    A_mat = [[S[i+j] for j in range(m+1)] for i in range(m+1)]
    B_vec = T
    return solve_system(A_mat, B_vec)


def poly_eval(x, coeffs):
    """
    Вычисляет значение полинома с коэффициентами coeffs в точке x
    coeffs[0] + coeffs[1] x + coeffs[2] x^2 + ...
    """
    return sum(coeffs[j] * (x**j) for j in range(len(coeffs)))


def sum_squared_errors(X, Y, coeffs):
    """
    Вычисляет сумму квадратов отклонений между данными и полиномом
    """
    return sum((Y[i] - poly_eval(X[i], coeffs))**2 for i in range(len(X)))