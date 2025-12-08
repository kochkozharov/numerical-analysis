"""
Лабораторная работа 8: Численные методы решения параболических уравнений (2D)
Вариант 9

Задача: использовать схемы переменных направлений и дробных шагов
для решения двумерной начально-краевой задачи параболического типа.

Уравнение:
    ∂u/∂t = a ∂²u/∂x² + b ∂²u/∂y² + sin x sin y ( μ cos μt + (a + b) sin μt )

Граничные и начальные условия (вариант 9):
    u(0, y, t) = 0
    u(π/2, y, t) = sin y sin(μ t)
    u(x, 0, t) = 0
    ∂u/∂y (x, π, t) = - sin x sin(μ t)
    u(x, y, 0) = 0

Аналитическое решение:
    U(x, y, t) = sin x sin y sin(μ t)

Параметры варианта:
    1) a = 1, b = 1, μ = 1
    2) a = 2, b = 1, μ = 1
    3) a = 1, b = 2, μ = 1
    4) a = 1, b = 1, μ = 2
"""

import os
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# Параметры области
# ============================================================================

X_MIN = 0.0
X_MAX = np.pi / 2.0
Y_MIN = 0.0
Y_MAX = np.pi
T_MAX = 1.0


# ============================================================================
# Аналитическое решение и правые части
# ============================================================================


def analytical_solution(x: np.ndarray, y: np.ndarray, t: float, mu: float) -> np.ndarray:
    """Аналитическое решение U(x, y, t) = sin x sin y sin(μ t)."""
    X, Y = np.meshgrid(x, y, indexing="ij")
    return np.sin(X) * np.sin(Y) * np.sin(mu * t)


def source_term(x: np.ndarray, y: np.ndarray, t: float, a: float, b: float, mu: float) -> np.ndarray:
    """Правая часть f(x, y, t) для заданного аналитического решения.

    Из уравнения и аналитического решения получаем:
        u_t = a u_xx + b u_yy + f
    Для U = sin x sin y sin(μ t):
        U_t   = μ cos(μ t) sin x sin y
        U_xx  = - sin x sin y sin(μ t)
        U_yy  = - sin x sin y sin(μ t)
    Откуда:
        f = U_t - a U_xx - b U_yy
          = sin x sin y [ μ cos(μ t) + (a + b) sin(μ t) ]
    (совпадает с формулой в условии).
    """
    X, Y = np.meshgrid(x, y, indexing="ij")
    return np.sin(X) * np.sin(Y) * (mu * np.cos(mu * t) + (a + b) * np.sin(mu * t))


# ============================================================================
# Граничные и начальные условия
# ============================================================================


def apply_boundary_conditions(u: np.ndarray, x: np.ndarray, y: np.ndarray, t: float, mu: float) -> None:
    """Применяет граничные условия к сетке u для момента времени t."""
    n_x, n_y = u.shape

    # Левая граница: x = 0, u = 0
    u[0, :] = 0.0

    # Правая граница: x = π/2, u = sin y sin(μ t)
    u[-1, :] = np.sin(y) * np.sin(mu * t)

    # Нижняя граница: y = 0, u = 0
    u[:, 0] = 0.0

    # Верхняя граница: y = π, условие Неймана: u_y = - sin x sin(μ t)
    # Аппроксимация первой производной вперед разностью:
    #   (u_{i,Ny} - u_{i,Ny-1}) / h_y = - sin x_i sin(μ t)
    #   => u_{i,Ny} = u_{i,Ny-1} - h_y sin x_i sin(μ t)
    h_y = (Y_MAX - Y_MIN) / (n_y - 1)
    x_inner = x
    u[:, -1] = u[:, -2] - h_y * np.sin(x_inner) * np.sin(mu * t)


def initial_condition(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Начальное условие: u(x, y, 0) = 0."""
    return np.zeros((x.size, y.size))


# ============================================================================
# Вспомогательные функции для ADI / дробных шагов
# ============================================================================


def thomas_algorithm(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Метод прогонки для трехдиагональной системы.

    Решает: a[i] * x[i-1] + b[i] * x[i] + c[i] * x[i+1] = d[i].
    Граничные уравнения предполагаются уже встроенными в коэффициенты.
    """
    n = d.size
    c_prime = np.zeros(n)
    d_prime = np.zeros(n)

    # Прямой ход
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] - a[i] * c_prime[i - 1]
        c_prime[i] = c[i] / denom if i < n - 1 else 0.0
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denom

    # Обратный ход
    x = np.zeros(n)
    x[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]

    return x


# ============================================================================
# Численные схемы по времени
# ============================================================================


def step_adi(
    u: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    tau: float,
    a: float,
    b: float,
    t_n: float,
    mu: float,
) -> np.ndarray:
    """Один шаг схемы переменных направлений (ADI, схема Писмена-Рэчфорда).

    Шаг 1 (полушаг τ/2, неявно по x, явно по y).
    Шаг 2 (полушаг τ/2, неявно по y, явно по x).
    Источник учитываем симметрично по времени.
    """
    n_x, n_y = u.shape
    h_x = (X_MAX - X_MIN) / (n_x - 1)
    h_y = (Y_MAX - Y_MIN) / (n_y - 1)

    # Полушаг: t_{n+1/2}
    t_half = t_n + 0.5 * tau

    # ---------- Шаг 1: неявно по x ----------
    u_half = np.zeros_like(u)
    r_x = a * tau / (2.0 * h_x * h_x)

    # Для каждой фиксированной j решаем систему по x
    for j in range(1, n_y - 1):
        # Правая часть: u^n + (τ/2)*(b*u_yy + f)
        u_col = u[:, j]
        u_yy = (u[:, j + 1] - 2.0 * u[:, j] + u[:, j - 1]) / (h_y * h_y)
        f_n = source_term(x, y[j : j + 1], t_n, a, b, mu).reshape(-1)
        rhs = u_col + 0.5 * tau * (b * u_yy + f_n)

        # Коэффициенты трехдиагональной матрицы (по x)
        a_tr = -r_x * np.ones(n_x)
        b_tr = (1.0 + 2.0 * r_x) * np.ones(n_x)
        c_tr = -r_x * np.ones(n_x)

        # Граничные точки по x берем из условий (они будут наложены после шага),
        # поэтому систему решаем только для внутренних i=1..n_x-2 с Dirichlet данными.
        # Реализуем путем фиксации u[0] и u[-1] и переноса их в правую часть.
        a_tr[0] = 0.0
        c_tr[0] = 0.0
        b_tr[0] = 1.0
        rhs[0] = u[0, j]

        a_tr[-1] = 0.0
        c_tr[-1] = 0.0
        b_tr[-1] = 1.0
        rhs[-1] = u[-1, j]

        u_half[:, j] = thomas_algorithm(a_tr, b_tr, c_tr, rhs)

    # Копируем граничные значения по y из u (они будут переопределены после второго шага)
    u_half[:, 0] = u[:, 0]
    u_half[:, -1] = u[:, -1]

    # После полушага накладываем смешанные граничные условия
    apply_boundary_conditions(u_half, x, y, t_half, mu)

    # ---------- Шаг 2: неявно по y ----------
    u_new = np.zeros_like(u)
    r_y = b * tau / (2.0 * h_y * h_y)

    for i in range(1, n_x - 1):
        u_row = u_half[i, :]
        u_xx = (u_half[i + 1, :] - 2.0 * u_half[i, :] + u_half[i - 1, :]) / (h_x * h_x)
        f_half = source_term(x[i : i + 1], y, t_half, a, b, mu).reshape(-1)
        rhs = u_row + 0.5 * tau * (a * u_xx + f_half)

        a_tr = -r_y * np.ones(n_y)
        b_tr = (1.0 + 2.0 * r_y) * np.ones(n_y)
        c_tr = -r_y * np.ones(n_y)

        # Нижняя граница y=0: Dirichlet u=0
        a_tr[0] = 0.0
        c_tr[0] = 0.0
        b_tr[0] = 1.0
        rhs[0] = 0.0

        # Верхняя граница y=Y_MAX: реализуем условие Неймана через одностороннюю формулу
        # (u_N - u_{N-1})/h_y = - sin x_i sin(μ t_{n+1})
        t_np1 = t_n + tau
        a_tr[-1] = -1.0 / h_y
        b_tr[-1] = 1.0 / h_y
        c_tr[-1] = 0.0
        rhs[-1] = -np.sin(x[i]) * np.sin(mu * t_np1)

        u_new[i, :] = thomas_algorithm(a_tr, b_tr, c_tr, rhs)

    # Левая и правая границы по x из граничных условий
    u_new[0, :] = 0.0
    u_new[-1, :] = np.sin(y) * np.sin(mu * (t_n + tau))

    return u_new


def step_fractional(
    u: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    tau: float,
    a: float,
    b: float,
    t_n: float,
    mu: float,
) -> np.ndarray:
    """Один шаг схемы дробных шагов (поочередное решение задач по x и y).

    На первом полушаге учитываем оператор по x, на втором — по y.
    """
    n_x, n_y = u.shape
    h_x = (X_MAX - X_MIN) / (n_x - 1)
    h_y = (Y_MAX - Y_MIN) / (n_y - 1)

    # Полушаг по x
    u_half = np.zeros_like(u)
    r_x = a * tau / (2.0 * h_x * h_x)
    t_half = t_n + 0.5 * tau

    for j in range(1, n_y - 1):
        u_col = u[:, j]
        # Источник в полушаге учитываем как f(x,y,t_n)
        f_n = source_term(x, y[j : j + 1], t_n, a, b, mu).reshape(-1)
        rhs = u_col + 0.5 * tau * f_n

        a_tr = -r_x * np.ones(n_x)
        b_tr = (1.0 + 2.0 * r_x) * np.ones(n_x)
        c_tr = -r_x * np.ones(n_x)

        a_tr[0] = 0.0
        c_tr[0] = 0.0
        b_tr[0] = 1.0
        rhs[0] = u[0, j]

        a_tr[-1] = 0.0
        c_tr[-1] = 0.0
        b_tr[-1] = 1.0
        rhs[-1] = u[-1, j]

        u_half[:, j] = thomas_algorithm(a_tr, b_tr, c_tr, rhs)

    u_half[:, 0] = u[:, 0]
    u_half[:, -1] = u[:, -1]
    apply_boundary_conditions(u_half, x, y, t_half, mu)

    # Полушаг по y
    u_new = np.zeros_like(u)
    r_y = b * tau / (2.0 * h_y * h_y)

    for i in range(1, n_x - 1):
        u_row = u_half[i, :]
        f_half = source_term(x[i : i + 1], y, t_half, a, b, mu).reshape(-1)
        rhs = u_row + 0.5 * tau * f_half

        a_tr = -r_y * np.ones(n_y)
        b_tr = (1.0 + 2.0 * r_y) * np.ones(n_y)
        c_tr = -r_y * np.ones(n_y)

        # y=0: Dirichlet
        a_tr[0] = 0.0
        c_tr[0] = 0.0
        b_tr[0] = 1.0
        rhs[0] = 0.0

        # y=Y_MAX: Neumann
        t_np1 = t_n + tau
        a_tr[-1] = -1.0 / h_y
        b_tr[-1] = 1.0 / h_y
        c_tr[-1] = 0.0
        rhs[-1] = -np.sin(x[i]) * np.sin(mu * t_np1)

        u_new[i, :] = thomas_algorithm(a_tr, b_tr, c_tr, rhs)

    u_new[0, :] = 0.0
    u_new[-1, :] = np.sin(y) * np.sin(mu * (t_n + tau))

    return u_new


# ============================================================================
# Решение задачи
# ============================================================================


def solve_parabolic_2d(
    step_func: Callable[[np.ndarray, np.ndarray, np.ndarray, float, float, float, float, float], np.ndarray],
    n_x: int,
    n_y: int,
    n_t: int,
    a: float,
    b: float,
    mu: float,
    method_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Решает задачу для выбранного временного шага (ADI / дробные шаги)."""
    x = np.linspace(X_MIN, X_MAX, n_x)
    y = np.linspace(Y_MIN, Y_MAX, n_y)
    t = np.linspace(0.0, T_MAX, n_t + 1)

    tau = T_MAX / n_t

    # Начальное условие
    u = initial_condition(x, y)
    apply_boundary_conditions(u, x, y, 0.0, mu)

    u_numerical = np.zeros((n_t + 1, n_x, n_y))
    u_numerical[0] = u

    for n in range(n_t):
        t_n = t[n]
        u = step_func(u, x, y, tau, a, b, t_n, mu)
        apply_boundary_conditions(u, x, y, t[n + 1], mu)
        u_numerical[n + 1] = u

    # Аналитическое решение
    u_analytical = np.zeros_like(u_numerical)
    for n in range(n_t + 1):
        u_analytical[n] = analytical_solution(x, y, t[n], mu)

    return x, y, t, u_numerical, u_analytical


# ============================================================================
# Ошибки
# ============================================================================


def compute_error(u_numerical: np.ndarray, u_analytical: np.ndarray) -> np.ndarray:
    return np.abs(u_numerical - u_analytical)


def compute_max_error(u_numerical: np.ndarray, u_analytical: np.ndarray) -> float:
    diff = np.abs(u_numerical - u_analytical)
    return float(np.max(diff))


def compute_l2_error(u_numerical: np.ndarray, u_analytical: np.ndarray, h_x: float, h_y: float) -> float:
    diff = u_numerical - u_analytical
    return float(np.sqrt(h_x * h_y * np.sum(diff ** 2)))


# ============================================================================
# Исследование зависимости погрешности от параметров сетки
# ============================================================================


def study_grid_dependence(
    step_func: Callable[
        [np.ndarray, np.ndarray, np.ndarray, float, float, float, float, float],
        np.ndarray,
    ],
    n_x_values: list,
    n_y_values: list,
    n_t_values: list,
    a: float,
    b: float,
    mu: float,
    method_name: str,
):
    """Исследует зависимость погрешности от h_x, h_y и τ."""

    h_x_values = []
    h_y_values = []
    tau_values = []
    max_errors = []
    l2_errors = []

    for n_x in n_x_values:
        for n_y in n_y_values:
            for n_t in n_t_values:
                h_x = (X_MAX - X_MIN) / (n_x - 1)
                h_y = (Y_MAX - Y_MIN) / (n_y - 1)
                tau = T_MAX / n_t

                x, y, t, u_num, u_anal = solve_parabolic_2d(
                    step_func, n_x, n_y, n_t, a, b, mu, method_name
                )

                max_err = compute_max_error(u_num, u_anal)
                l2_err = compute_l2_error(u_num, u_anal, h_x, h_y)

                h_x_values.append(h_x)
                h_y_values.append(h_y)
                tau_values.append(tau)
                max_errors.append(max_err)
                l2_errors.append(l2_err)

    return (
        np.array(h_x_values),
        np.array(h_y_values),
        np.array(tau_values),
        np.array(max_errors),
        np.array(l2_errors),
    )


# ============================================================================
# Визуализация
# ============================================================================


def plot_solution(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    u_numerical: np.ndarray,
    u_analytical: np.ndarray,
    method_name: str,
    a: float,
    b: float,
    mu: float,
    output_dir: str,
) -> None:
    """Строит контурные графики численного и аналитического решения в несколько моментов времени."""
    X, Y = np.meshgrid(x, y, indexing="ij")
    times_idx = [0, len(t) // 3, 2 * len(t) // 3, -1]

    u_min = min(np.min(u_numerical), np.min(u_analytical))
    u_max = max(np.max(u_numerical), np.max(u_analytical))

    fig, axes = plt.subplots(4, 2, figsize=(12, 16))

    for k, n in enumerate(times_idx):
        ax_num = axes[k, 0]
        ax_an = axes[k, 1]

        cn1 = ax_num.contourf(X, Y, u_numerical[n], levels=20, cmap="viridis", vmin=u_min, vmax=u_max)
        ax_num.set_title(f"Численное, t = {t[n]:.3f}")
        ax_num.set_xlabel("x")
        ax_num.set_ylabel("y")
        ax_num.set_aspect("equal", adjustable="box")
        fig.colorbar(cn1, ax=ax_num)

        cn2 = ax_an.contourf(X, Y, u_analytical[n], levels=20, cmap="viridis", vmin=u_min, vmax=u_max)
        ax_an.set_title(f"Аналитическое, t = {t[n]:.3f}")
        ax_an.set_xlabel("x")
        ax_an.set_ylabel("y")
        ax_an.set_aspect("equal", adjustable="box")
        fig.colorbar(cn2, ax=ax_an)

    plt.suptitle(f"Решение: {method_name}, a={a}, b={b}, mu={mu}", fontsize=14)
    plt.tight_layout()
    filename = os.path.join(output_dir, f"solution_{method_name}_a{a}_b{b}_mu{mu}.png")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_error(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    u_numerical: np.ndarray,
    u_analytical: np.ndarray,
    method_name: str,
    a: float,
    b: float,
    mu: float,
    output_dir: str,
) -> None:
    """Строит контурные графики погрешности в несколько моментов времени."""
    X, Y = np.meshgrid(x, y, indexing="ij")
    error = compute_error(u_numerical, u_analytical)
    times_idx = [0, len(t) // 3, 2 * len(t) // 3, -1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, n in zip(axes.flat, times_idx):
        cn = ax.contourf(X, Y, error[n], levels=20, cmap="hot")
        ax.set_title(f"Погрешность, t = {t[n]:.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(cn, ax=ax)

    plt.suptitle(f"Погрешность: {method_name}, a={a}, b={b}, mu={mu}", fontsize=14)
    plt.tight_layout()
    filename = os.path.join(output_dir, f"error_{method_name}_a{a}_b{b}_mu{mu}.png")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_grid_dependence(
    h_x_values: np.ndarray,
    h_y_values: np.ndarray,
    tau_values: np.ndarray,
    max_errors: np.ndarray,
    l2_errors: np.ndarray,
    method_name: str,
    a: float,
    b: float,
    mu: float,
    output_dir: str,
) -> None:
    """Визуализация зависимости погрешности от h_x, h_y и τ (log-log графики)."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Фильтруем валидные значения
    valid = np.isfinite(max_errors) & (max_errors > 0)
    if not np.any(valid):
        plt.close()
        return

    h_x_vals = h_x_values[valid]
    h_y_vals = h_y_values[valid]
    tau_vals = tau_values[valid]
    err_vals = max_errors[valid]

    # Зависимость от h_x при фиксированных h_y и τ (берем первые несколько комбинаций)
    unique_h_y = np.unique(h_y_vals)
    unique_tau = np.unique(tau_vals)
    for hy in unique_h_y[:2]:
        for tau in unique_tau[:2]:
            mask = (np.abs(h_y_vals - hy) < 1e-12) & (np.abs(tau_vals - tau) < 1e-12)
            if np.any(mask):
                hx = h_x_vals[mask]
                er = err_vals[mask]
                idx = np.argsort(hx)
                axes[0].loglog(hx[idx], er[idx], "o-", label=f"h_y={hy:.3f}, τ={tau:.3f}")

    axes[0].set_xlabel("h_x")
    axes[0].set_ylabel("Max error")
    axes[0].set_title("Зависимость от h_x")
    axes[0].grid(True)
    axes[0].legend(fontsize=8)

    # Зависимость от h_y при фиксированных h_x и τ
    unique_h_x = np.unique(h_x_vals)
    for hx in unique_h_x[:2]:
        for tau in unique_tau[:2]:
            mask = (np.abs(h_x_vals - hx) < 1e-12) & (np.abs(tau_vals - tau) < 1e-12)
            if np.any(mask):
                hy = h_y_vals[mask]
                er = err_vals[mask]
                idx = np.argsort(hy)
                axes[1].loglog(hy[idx], er[idx], "s-", label=f"h_x={hx:.3f}, τ={tau:.3f}")

    axes[1].set_xlabel("h_y")
    axes[1].set_ylabel("Max error")
    axes[1].set_title("Зависимость от h_y")
    axes[1].grid(True)
    axes[1].legend(fontsize=8)

    # Зависимость от τ при фиксированных h_x и h_y
    for hx in unique_h_x[:2]:
        for hy in unique_h_y[:2]:
            mask = (np.abs(h_x_vals - hx) < 1e-12) & (np.abs(h_y_vals - hy) < 1e-12)
            if np.any(mask):
                tau = tau_vals[mask]
                er = err_vals[mask]
                idx = np.argsort(tau)
                axes[2].loglog(tau[idx], er[idx], "^-", label=f"h_x={hx:.3f}, h_y={hy:.3f}")

    axes[2].set_xlabel("τ")
    axes[2].set_ylabel("Max error")
    axes[2].set_title("Зависимость от τ")
    axes[2].grid(True)
    axes[2].legend(fontsize=8)

    plt.suptitle(
        f"Зависимость погрешности: {method_name}, a={a}, b={b}, mu={mu}",
        fontsize=14,
    )
    plt.tight_layout()
    filename = os.path.join(output_dir, f"grid_dependence_{method_name}_a{a}_b{b}_mu{mu}.png")
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    """Запуск расчётов для набора параметров и двух схем."""

    output_dir = "lab8/results"
    os.makedirs(output_dir, exist_ok=True)

    # Сетка
    n_x = 40
    n_y = 40
    n_t = 80

    h_x = (X_MAX - X_MIN) / (n_x - 1)
    h_y = (Y_MAX - Y_MIN) / (n_y - 1)
    tau = T_MAX / n_t

    print("=" * 80)
    print("Лабораторная работа 8: Численные методы решения параболических уравнений (2D)")
    print("Вариант 9")
    print("=" * 80)
    print(f"Сетка: n_x = {n_x}, n_y = {n_y}, n_t = {n_t}")
    print(f"Шаги: h_x = {h_x:.6f}, h_y = {h_y:.6f}, τ = {tau:.6f}")
    print("=" * 80)

    methods = [
        (step_adi, "ADI"),
        (step_fractional, "Fractional"),
    ]

    param_sets = [
        (1.0, 1.0, 1.0),
        (2.0, 1.0, 1.0),
        (1.0, 2.0, 1.0),
        (1.0, 1.0, 2.0),
    ]

    results = []

    for a, b, mu in param_sets:
        print(f"\nПараметры: a = {a}, b = {b}, μ = {mu}")
        print("-" * 80)

        for step_func, method_name in methods:
            print(f"\nМетод: {method_name}")
            print("-" * 40)

            x, y, t, u_num, u_anal = solve_parabolic_2d(
                step_func, n_x, n_y, n_t, a, b, mu, method_name
            )

            max_err = compute_max_error(u_num, u_anal)
            l2_err = compute_l2_error(u_num, u_anal, h_x, h_y)

            print(f"Максимальная погрешность: {max_err:.6e}")
            print(f"L2 норма погрешности: {l2_err:.6e}")

            plot_solution(x, y, t, u_num, u_anal, method_name, a, b, mu, output_dir)
            plot_error(x, y, t, u_num, u_anal, method_name, a, b, mu, output_dir)

            results.append(
                {
                    "method": method_name,
                    "a": a,
                    "b": b,
                    "mu": mu,
                    "max_error": max_err,
                    "l2_error": l2_err,
                }
            )

    # Исследование зависимости погрешности от параметров сетки
    print("\n" + "=" * 80)
    print("Исследование зависимости погрешности от параметров сетки h_x, h_y, τ")
    print("=" * 80)

    n_x_values = [20, 30, 40]
    n_y_values = [20, 30, 40]
    n_t_values = [40, 80, 120]

    # Проводим исследование для метода ADI и базового набора параметров
    base_a, base_b, base_mu = 1.0, 1.0, 1.0
    h_x_vals, h_y_vals, tau_vals, max_errs, l2_errs = study_grid_dependence(
        step_adi,
        n_x_values,
        n_y_values,
        n_t_values,
        base_a,
        base_b,
        base_mu,
        "ADI",
    )

    plot_grid_dependence(
        h_x_vals,
        h_y_vals,
        tau_vals,
        max_errs,
        l2_errs,
        "ADI",
        base_a,
        base_b,
        base_mu,
        output_dir,
    )

    print("\nРезультаты исследования (ADI, a=1, b=1, μ=1):")
    print(f"{'h_x':>10} {'h_y':>10} {'τ':>10} {'Max Error':>15} {'L2 Error':>15}")
    print("-" * 70)
    for i in range(min(20, len(h_x_vals))):
        if np.isfinite(max_errs[i]) and max_errs[i] > 0:
            print(
                f"{h_x_vals[i]:>10.5f} {h_y_vals[i]:>10.5f} {tau_vals[i]:>10.5f} "
                f"{max_errs[i]:>15.6e} {l2_errs[i]:>15.6e}"
            )

    print("\n" + "=" * 80)
    print("Сводная таблица результатов")
    print("=" * 80)
    print(f"{'Метод':<12} {'a':>4} {'b':>4} {'μ':>4} {'Max Error':>15} {'L2 Error':>15}")
    print("-" * 60)
    for res in results:
        print(
            f"{res['method']:<12} {res['a']:>4.1f} {res['b']:>4.1f} {res['mu']:>4.1f} "
            f"{res['max_error']:>15.6e} {res['l2_error']:>15.6e}"
        )

    print("\n" + "=" * 80)
    print(f"Все графики сохранены в папку: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
