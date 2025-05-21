# Преобразование задачи второго порядка в систему первого порядка
# Метод Эйлера для системы y' = F(x, y)
def euler_system(F, x0, y0, h, n):
    x = x0
    y = y0[:]
    xs, ys = [x], [y[:]]
    for _ in range(n):
        dy = F(x, y)
        y = [y[j] + h * dy[j] for j in range(len(y))]
        x += h
        xs.append(x)
        ys.append(y[:])
    return xs, ys

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
    return xs, ys
