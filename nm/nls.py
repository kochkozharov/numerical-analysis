from nm.nm import solve_system

def simple_iteration(phi, x0, eps, max_iter=100):
    errors = []
    x = x0
    for i in range(max_iter):
        x_new = phi(x)
        err = abs(x_new - x)
        errors.append(err)
        if err < eps:
            break
        x = x_new
    return x_new, errors

def newton(f, df, x0, eps, max_iter=100):
    errors = []
    x = x0
    for i in range(max_iter):
        x_new = x - f(x)/df(x)
        err = abs(x_new - x)
        errors.append(err)
        if err < eps:
            break
        x = x_new
    return x_new, errors


def generalized_simple_iteration(phi, x0, tol, max_iter=100):
    """
    Общий метод простой итерации для n-мерной системы.

    Параметры:
    phi       -- итерационная функция φ(x), возвращающая список длины n
    x0        -- начальное приближение (список длины n)
    tol       -- требуемая точность по бесконечной норме
    max_iter  -- максимальное число итераций

    Возвращает:
    xs    -- список приближений [x0, x1, ..., x_k]
    errors-- список погрешностей ||x_{k} - x_{k-1}||_∞
    """
    n = len(x0)
    xs = [x0[:]]
    errors = []
    for k in range(1, max_iter+1):
        x_prev = xs[-1]
        x_new = phi(x_prev)
        # вычисляем бесконечную норму разности
        err = max(abs(x_new[i] - x_prev[i]) for i in range(n))
        xs.append(x_new[:])
        errors.append(err)
        if err < tol:
            break
    return xs, errors


def generalized_newton(f, jacobian, x0, tol, max_iter=100):
    """
    Общий метод Ньютона для n-мерной системы.

    Параметры:
    f         -- функция f(x) возвращает список значений длины n
    jacobian  -- функция J(x) возвращает матрицу n×n (список списков)
    x0        -- начальное приближение (список длины n)
    tol       -- требуемая точность по бесконечной норме
    max_iter  -- максимальное число итераций

    Возвращает:
    xs    -- список приближений [x0, x1, ..., x_k]
    errors-- список погрешностей ||x_{k} - x_{k-1}||_∞
    """
    def newton_step(x):
        A = jacobian(x)
        b = [-val for val in f(x)]
        # solve_system решает A * dx = b
        dx = solve_system(A, b)
        return [x[i] + dx[i] for i in range(len(x))]

    n = len(x0)
    xs = [x0[:]]
    errors = []
    for k in range(1, max_iter+1):
        x_prev = xs[-1]
        x_new = newton_step(x_prev)
        err = max(abs(x_new[i] - x_prev[i]) for i in range(n))
        xs.append(x_new[:])
        errors.append(err)
        if err < tol:
            break
    return xs, errors