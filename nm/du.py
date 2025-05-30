# Преобразование задачи второго порядка в систему первого порядка
# Метод Эйлера для системы y' = F(x, y)
def euler(F, x0, y0, h, n):
    x = x0
    y = y0[:]
    xs, ys = [x], [y[:]]
    for _ in range(n):
        dy = F(x, y)
        y = [y[j] + h * dy[j] for j in range(len(y))]
        x += h
        xs.append(x)
        ys.append(y[:])
    ys_scalar = [y[0] for y in ys]
    return xs, ys_scalar

# Метод Рунге-Кутты 4-го порядка для системы y' = F(x, y)
def rk4_system(F, x0, y0, h, n):
    x = x0
    y = y0[:]
    xs, ys = [x], [y[:]]
    for _ in range(n):
        k1 = F(x, y)
        k2 = F(x + h/2, [y[j] + h/2 * k1[j] for j in range(len(y))])
        k3 = F(x + h/2, [y[j] + h/2 * k2[j] for j in range(len(y))])
        k4 = F(x + h, [y[j] + h * k3[j] for j in range(len(y))])
        y = [y[j] + (h/6)*(k1[j] + 2*k2[j] + 2*k3[j] + k4[j]) for j in range(len(y))]
        x += h
        xs.append(x)
        ys.append(y[:])
    return xs, ys

def rk4(F, x0, y0, h, n):
    xs, ys = rk4_system(F, x0, y0, h, n)
    ys_scalar = [y[0] for y in ys]
    return xs, ys_scalar

# Предиктор-Корректор Адамса 4-го порядка (ABM)
def adams_4(F, x0, y0, h, n):
    # старт 3 шагами RK4
    xs, ys = rk4_system(F, x0, y0, h, 3)
    for i in range(3, n):
        x_im3, x_im2, x_im1, x_i = xs[i-3], xs[i-2], xs[i-1], xs[i]
        y_im3, y_im2, y_im1, y_i = ys[i-3], ys[i-2], ys[i-1], ys[i]
        f_vals = [F(x_im3, y_im3), F(x_im2, y_im2), F(x_im1, y_im1), F(x_i, y_i)]
        # предиктор
        y_pred = [y_i[j] + (h/24)*(55*f_vals[3][j] - 59*f_vals[2][j] + 37*f_vals[1][j] - 9*f_vals[0][j]) for j in range(len(y0))]
        # корректор
        f_ip1 = F(x_i + h, y_pred)
        y_next = [y_i[j] + (h/24)*(9*f_ip1[j] + 19*f_vals[3][j] - 5*f_vals[2][j] + f_vals[1][j]) for j in range(len(y0))]
        xs.append(x_i + h)
        ys.append(y_next)
        ys_scalar = [y[0] for y in ys]
    return xs, ys_scalar


def rk4_step(x, y1, y2, h, F):
    """
    Один шаг классического метода Рунге-Кутты 4-го порядка для системы:
      y1' = y2
      y2' = F2(x, y1, y2)
    F возвращает кортеж (y1', y2').
    """
    k1_1, k1_2 = F(x, y1, y2)
    k2_1, k2_2 = F(x + h/2, y1 + h/2 * k1_1, y2 + h/2 * k1_2)
    k3_1, k3_2 = F(x + h/2, y1 + h/2 * k2_1, y2 + h/2 * k2_2)
    k4_1, k4_2 = F(x + h, y1 + h * k3_1, y2 + h * k3_2)
    y1_new = y1 + h/6 * (k1_1 + 2*k2_1 + 2*k3_1 + k4_1)
    y2_new = y2 + h/6 * (k1_2 + 2*k2_2 + 2*k3_2 + k4_2)
    return y1_new, y2_new

# Общий метод стрельбы
def shooting(a, b, ya, yb, N, F, tol=1e-6, max_iter=50):
    """
    Решение краевой задачи y(a)=ya, y(b)=yb методом стрельбы.
    F(x,y,y') задает правую часть системы (y', y'').
    """
    h = (b - a) / N
    def shoot(m):
        x, y1, y2 = a, ya, m
        for _ in range(N):
            y1, y2 = rk4_step(x, y1, y2, h, F)
            x += h
        return y1

    # Секущая для подбора начального y'(a)
    m0, m1 = 0.0, 1.0
    f0, f1 = shoot(m0) - yb, shoot(m1) - yb
    iters = 0
    for _ in range(max_iter):
        iters += 1
        if abs(f1 - f0) < 1e-12: break
        m2 = m1 - f1 * (m1 - m0) / (f1 - f0)
        f2 = shoot(m2) - yb
        if abs(f2) < tol:
            m1 = m2
            break
        m0, f0, m1, f1 = m1, f1, m2, f2

    # Итоговое вычисление на сетке
    xs = [a + i*h for i in range(N+1)]
    ys = []
    x, y1, y2 = a, ya, m1
    ys.append(y1)
    for _ in range(N):
        y1, y2 = rk4_step(x, y1, y2, h, F)
        x += h
        ys.append(y1)
    return xs, ys, iters

# Конечно-разностный метод для y'' + p(x)y' + q(x)y = g(x)
def finite_difference(a, b, ya, yb, N, p, q, g):
    """
    Краевая задача второго порядка:
      y'' + p(x)*y' + q(x)*y = g(x),  y(a)=ya, y(b)=yb
    Функции p(x), q(x), g(x) задают уравнение в каноническом виде.
    """
    h = (b - a) / N
    xs = [a + i*h for i in range(N+1)]
    A = [0]*(N+1); B = [0]*(N+1); C = [0]*(N+1); D = [0]*(N+1)

    # Граничные условия
    B[0], D[0] = 1, ya
    B[N], D[N] = 1, yb

    for i in range(1, N):
        x = xs[i]
        # Центральные разности:
        # y'' ≈ (y[i-1] - 2y[i] + y[i+1]) / h^2
        # y'  ≈ (y[i+1] - y[i-1]) / (2h)
        A[i] = 1/h**2 - p(x)/(2*h)
        B[i] = -2/h**2 + q(x)
        C[i] = 1/h**2 + p(x)/(2*h)
        D[i] = g(x)

    alpha = [0]*(N+1)
    beta = [0]*(N+1)
    alpha[0], beta[0] = 0, ya
    for i in range(1, N+1):
        denom = B[i] + A[i]*alpha[i-1]
        alpha[i] = -C[i]/denom if i < N else 0
        beta[i] = (D[i] - A[i]*beta[i-1]) / denom
    ys = [0]*(N+1)
    ys[N] = yb
    for i in range(N-1, -1, -1):
        ys[i] = alpha[i]*ys[i+1] + beta[i]
    return xs, ys


def max_error(y_num, y_exact, xs):
    return max(abs(y_num[i] - y_exact(xs[i])) for i in range(len(xs)))

def runge_romberg(y_h, y_h2, p):
    return max(abs((y_h2[2*i] - y_h[i]) / (2**p - 1)) for i in range(len(y_h)))
