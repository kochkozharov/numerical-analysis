"""
Лабораторная работа 2: Численные методы решения гиперболических уравнений
Вариант 9

Уравнение: ∂²u/∂t² + 3 ∂u/∂t = ∂²u/∂x² + ∂u/∂x - u + sin(x) exp(-t)
Граничные условия: u(0,t) = exp(-t), u(π,t) = -exp(-t)
Начальные условия: u(x,0) = cos(x), ∂u/∂t(x,0) = -cos(x)
Аналитическое решение: U(x,t) = exp(-t) cos(x)
"""

import numpy as np
from typing import Callable, Tuple
import matplotlib.pyplot as plt
import os


# ============================================================================
# Параметры задачи
# ============================================================================

# Область решения
X_MIN = 0.0
X_MAX = np.pi
T_MAX = 1.0


# ============================================================================
# Аналитическое решение и функции граничных условий
# ============================================================================

def analytical_solution(x: np.ndarray, t: float) -> np.ndarray:
    """Аналитическое решение U(x,t) = exp(-t) cos(x)"""
    return np.exp(-t) * np.cos(x)


def boundary_condition_left(t: float) -> float:
    """Левое граничное условие: u(0,t) = exp(-t)"""
    return np.exp(-t)


def boundary_condition_right(t: float) -> float:
    """Правое граничное условие: u(π,t) = -exp(-t)"""
    return -np.exp(-t)


def initial_condition_u(x: np.ndarray) -> np.ndarray:
    """Первое начальное условие: u(x,0) = cos(x)"""
    return np.cos(x)


def initial_condition_ut(x: np.ndarray) -> np.ndarray:
    """Второе начальное условие: ∂u/∂t(x,0) = -cos(x)"""
    return -np.cos(x)


def source_term(x: np.ndarray, t: float) -> np.ndarray:
    """Источник: sin(x) exp(-t)"""
    return np.sin(x) * np.exp(-t)


# ============================================================================
# Аппроксимации второго начального условия
# ============================================================================

def approximate_second_ic_first_order(
    u_prev: np.ndarray, u_curr: np.ndarray, ut_0: np.ndarray, tau: float
) -> np.ndarray:
    """
    Аппроксимация второго начального условия первого порядка.
    ∂u/∂t(x,0) ≈ (u(x,τ) - u(x,0))/τ = ut_0
    Отсюда: u(x,τ) = u(x,0) + τ * ut_0
    """
    return u_prev + tau * ut_0


def approximate_second_ic_second_order(
    u_prev: np.ndarray, u_curr: np.ndarray, ut_0: np.ndarray, tau: float, h: float, x: np.ndarray
) -> np.ndarray:
    """
    Аппроксимация второго начального условия второго порядка.
    Используем разложение Тейлора:
    u(x,τ) = u(x,0) + τ*ut(x,0) + (τ²/2)*utt(x,0) + O(τ³)
    
    Из уравнения при t=0:
    utt(x,0) = uxx(x,0) + ux(x,0) - u(x,0) + sin(x) - 3*ut(x,0)
    
    Вычисляем производные численно:
    uxx(x,0) ≈ (u(x+h,0) - 2*u(x,0) + u(x-h,0))/h²
    ux(x,0) ≈ (u(x+h,0) - u(x-h,0))/(2*h)
    """
    n = len(u_prev)
    u_next = np.zeros(n)
    
    # Вычисляем utt из уравнения
    for i in range(1, n - 1):
        uxx = (u_prev[i + 1] - 2 * u_prev[i] + u_prev[i - 1]) / (h * h)
        ux = (u_prev[i + 1] - u_prev[i - 1]) / (2 * h)
        utt = uxx + ux - u_prev[i] + source_term(x[i], 0.0) - 3 * ut_0[i]
        u_next[i] = u_prev[i] + tau * ut_0[i] + (tau * tau / 2) * utt
    
    # Граничные условия
    u_next[0] = boundary_condition_left(tau)
    u_next[-1] = boundary_condition_right(tau)
    
    return u_next


# ============================================================================
# Аппроксимации краевых условий (для производных в граничных условиях)
# ============================================================================

def apply_boundary_condition_two_point_first_order(
    u: np.ndarray, h: float, f_left: float, f_right: float
) -> None:
    """
    Двухточечная аппроксимация первого порядка точности.
    Для задачи с производными в граничных условиях.
    В данной задаче граничные условия - Дирихле, поэтому просто устанавливаем значения.
    """
    # В данной задаче граничные условия - Дирихле (значения функции)
    # Но функция оставлена для совместимости с общей структурой
    pass


def apply_boundary_condition_three_point_second_order(
    u: np.ndarray, h: float, f_left: float, f_right: float
) -> None:
    """
    Трехточечная аппроксимация второго порядка точности.
    В данной задаче граничные условия - Дирихле.
    """
    pass


def apply_boundary_condition_two_point_second_order(
    u: np.ndarray, h: float, f_left: float, f_right: float
) -> None:
    """
    Двухточечная аппроксимация второго порядка точности.
    В данной задаче граничные условия - Дирихле.
    """
    pass


# ============================================================================
# Численные схемы
# ============================================================================

def explicit_cross_scheme(
    u_prev: np.ndarray,
    u_curr: np.ndarray,
    h: float,
    tau: float,
    t: float,
    x: np.ndarray,
) -> np.ndarray:
    """
    Явная схема крест для гиперболического уравнения.
    Уравнение: ∂²u/∂t² + 3 ∂u/∂t = ∂²u/∂x² + ∂u/∂x - u + sin(x) exp(-t)
    
    Аппроксимируем:
    utt ≈ (u^{n+1} - 2*u^n + u^{n-1})/τ²
    ut ≈ (u^{n+1} - u^{n-1})/(2*τ)
    uxx ≈ (u_{i+1}^n - 2*u_i^n + u_{i-1}^n)/h²
    ux ≈ (u_{i+1}^n - u_{i-1}^n)/(2*h)
    
    Получаем:
    (u^{n+1} - 2*u^n + u^{n-1})/τ² + 3*(u^{n+1} - u^{n-1})/(2*τ) 
    = (u_{i+1}^n - 2*u_i^n + u_{i-1}^n)/h² + (u_{i+1}^n - u_{i-1}^n)/(2*h) - u_i^n + f_i^n
    
    Решаем относительно u^{n+1}:
    u^{n+1} = [2*u^n - u^{n-1} + τ²*(uxx + ux - u + f) - 3*τ*(u^{n+1} - u^{n-1})/2] / (1 + 3*τ/2)
    
    Упрощаем:
    u^{n+1}*(1 + 3*τ/2) = 2*u^n - u^{n-1} + τ²*(uxx + ux - u + f) + 3*τ*u^{n-1}/2
    u^{n+1} = [2*u^n + u^{n-1}*(3*τ/2 - 1) + τ²*(uxx + ux - u + f)] / (1 + 3*τ/2)
    """
    n = len(u_curr)
    u_next = np.zeros(n)
    
    # Коэффициенты
    alpha = 1.0 + 3.0 * tau / 2.0
    beta = 1.0 - 3.0 * tau / 2.0
    tau2 = tau * tau
    
    # Внутренние точки
    for i in range(1, n - 1):
        uxx = (u_curr[i + 1] - 2 * u_curr[i] + u_curr[i - 1]) / (h * h)
        ux = (u_curr[i + 1] - u_curr[i - 1]) / (2 * h)
        f = source_term(x[i], t)
        
        u_next[i] = (2.0 * u_curr[i] - beta * u_prev[i] + tau2 * (uxx + ux - u_curr[i] + f)) / alpha
    
    # Граничные условия (Дирихле)
    u_next[0] = boundary_condition_left(t + tau)
    u_next[-1] = boundary_condition_right(t + tau)
    
    return u_next


def implicit_scheme(
    u_prev: np.ndarray,
    u_curr: np.ndarray,
    h: float,
    tau: float,
    t: float,
    x: np.ndarray,
) -> np.ndarray:
    """
    Неявная схема для гиперболического уравнения.
    Аппроксимируем пространственные производные на новом временном слое.
    
    (u^{n+1} - 2*u^n + u^{n-1})/τ² + 3*(u^{n+1} - u^{n-1})/(2*τ)
    = (u_{i+1}^{n+1} - 2*u_i^{n+1} + u_{i-1}^{n+1})/h² 
      + (u_{i+1}^{n+1} - u_{i-1}^{n+1})/(2*h) - u_i^{n+1} + f_i^{n+1}
    
    Переписываем:
    u_i^{n+1}*(1/τ² + 3/(2*τ) + 2/h² + 1) 
    - u_{i+1}^{n+1}*(1/h² + 1/(2*h))
    - u_{i-1}^{n+1}*(1/h² - 1/(2*h))
    = 2*u_i^n/τ² - u_i^{n-1}*(1/τ² - 3/(2*τ)) + f_i^{n+1}
    """
    n = len(u_curr)
    
    # Коэффициенты
    a_coeff = np.zeros(n)  # при u_{i-1}
    b_coeff = np.zeros(n)  # при u_i
    c_coeff = np.zeros(n)  # при u_{i+1}
    d_coeff = np.zeros(n)  # правая часть
    
    tau2 = tau * tau
    h2 = h * h
    
    # Коэффициенты для внутренних точек
    for i in range(1, n - 1):
        a_coeff[i] = -(1.0 / h2 - 1.0 / (2.0 * h))
        b_coeff[i] = 1.0 / tau2 + 3.0 / (2.0 * tau) + 2.0 / h2 + 1.0
        c_coeff[i] = -(1.0 / h2 + 1.0 / (2.0 * h))
        d_coeff[i] = 2.0 * u_curr[i] / tau2 - u_prev[i] * (1.0 / tau2 - 3.0 / (2.0 * tau)) + source_term(x[i], t + tau)
        
        # Проверка на валидность
        if not np.isfinite(a_coeff[i]) or not np.isfinite(b_coeff[i]) or not np.isfinite(c_coeff[i]) or not np.isfinite(d_coeff[i]):
            return np.full(n, np.nan)
    
    # Граничные условия (Дирихле)
    b_coeff[0] = 1.0
    c_coeff[0] = 0.0
    d_coeff[0] = boundary_condition_left(t + tau)
    
    a_coeff[-1] = 0.0
    b_coeff[-1] = 1.0
    d_coeff[-1] = boundary_condition_right(t + tau)
    
    # Решаем систему методом прогонки
    u_next = thomas_algorithm(a_coeff, b_coeff, c_coeff, d_coeff)
    
    return u_next


def thomas_algorithm(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Метод прогонки (Thomas algorithm) для решения трехдиагональной системы.
    Система: a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]
    """
    n = len(d)
    
    # Копируем массивы для модификации
    b_copy = b.copy()
    d_copy = d.copy()
    
    # Прямой ход
    for i in range(1, n):
        if abs(b_copy[i - 1]) < 1e-15:
            return np.full(n, np.nan)
        
        m = a[i] / b_copy[i - 1]
        b_copy[i] = b_copy[i] - m * c[i - 1]
        d_copy[i] = d_copy[i] - m * d_copy[i - 1]
    
    # Обратный ход
    x = np.zeros(n)
    x[n - 1] = d_copy[n - 1] / b_copy[n - 1]
    
    for i in range(n - 2, -1, -1):
        x[i] = (d_copy[i] - c[i] * x[i + 1]) / b_copy[i]
    
    return x


# ============================================================================
# Решение задачи
# ============================================================================

def solve_pde(
    scheme_func: Callable,
    ic_approx_func: Callable,
    n_x: int,
    n_t: int,
    scheme_name: str,
    ic_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Решает задачу для заданной схемы и аппроксимации начального условия.
    
    Returns:
        x: пространственная сетка
        t: временная сетка
        u_numerical: численное решение
        u_analytical: аналитическое решение
    """
    h = (X_MAX - X_MIN) / n_x
    tau = T_MAX / n_t
    
    x = np.linspace(X_MIN, X_MAX, n_x + 1)
    t = np.linspace(0, T_MAX, n_t + 1)
    
    # Первое начальное условие
    u_prev = initial_condition_u(x)
    ut_0 = initial_condition_ut(x)
    
    # Аппроксимация второго начального условия для получения u_curr
    if ic_approx_func == approximate_second_ic_first_order:
        u_curr = ic_approx_func(u_prev, u_prev, ut_0, tau)
    else:
        u_curr = ic_approx_func(u_prev, u_prev, ut_0, tau, h, x)
    
    # Применяем граничные условия
    u_prev[0] = boundary_condition_left(0.0)
    u_prev[-1] = boundary_condition_right(0.0)
    u_curr[0] = boundary_condition_left(tau)
    u_curr[-1] = boundary_condition_right(tau)
    
    u_numerical = np.zeros((n_t + 1, n_x + 1))
    u_numerical[0, :] = u_prev
    u_numerical[1, :] = u_curr
    
    # Решение по времени
    for n in range(1, n_t):
        t_curr = t[n]
        u_next = scheme_func(u_prev, u_curr, h, tau, t_curr, x)
        u_numerical[n + 1, :] = u_next
        
        # Обновляем для следующего шага
        u_prev = u_curr.copy()
        u_curr = u_next.copy()
        
        # Проверка на NaN/Inf
        if np.any(~np.isfinite(u_next)):
            print(f"Предупреждение: NaN/Inf обнаружены на шаге {n+1} для {scheme_name}")
            break
    
    # Аналитическое решение
    u_analytical = np.zeros((n_t + 1, n_x + 1))
    for n in range(n_t + 1):
        u_analytical[n, :] = analytical_solution(x, t[n])
    
    return x, t, u_numerical, u_analytical


# ============================================================================
# Вычисление погрешности
# ============================================================================

def compute_error(u_numerical: np.ndarray, u_analytical: np.ndarray) -> np.ndarray:
    """Вычисляет погрешность в каждой точке сетки"""
    return np.abs(u_numerical - u_analytical)


def compute_max_error(u_numerical: np.ndarray, u_analytical: np.ndarray) -> float:
    """Вычисляет максимальную погрешность"""
    diff = np.abs(u_numerical - u_analytical)
    valid_diff = diff[np.isfinite(diff)]
    if len(valid_diff) == 0:
        return np.nan
    return np.max(valid_diff)


def compute_l2_error(u_numerical: np.ndarray, u_analytical: np.ndarray, h: float) -> float:
    """Вычисляет L2 норму погрешности"""
    diff = u_numerical - u_analytical
    valid_diff = diff[np.isfinite(diff)]
    if len(valid_diff) == 0:
        return np.nan
    return np.sqrt(h * np.sum(valid_diff ** 2))


# ============================================================================
# Исследование зависимости погрешности от параметров сетки
# ============================================================================

def study_grid_dependence(
    scheme_func: Callable,
    ic_approx_func: Callable,
    n_x_values: list,
    n_t_values: list,
    scheme_name: str,
    ic_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Исследует зависимость погрешности от параметров сетки h и τ.
    
    Returns:
        h_values: значения шага по пространству
        tau_values: значения шага по времени
        max_errors: максимальные погрешности для каждой комбинации
        l2_errors: L2 нормы погрешностей
    """
    h_values = []
    tau_values = []
    max_errors = []
    l2_errors = []
    
    for n_x in n_x_values:
        for n_t in n_t_values:
            h = (X_MAX - X_MIN) / n_x
            tau = T_MAX / n_t
            
            _, _, u_num, u_anal = solve_pde(scheme_func, ic_approx_func, n_x, n_t, scheme_name, ic_name)
            
            max_err = compute_max_error(u_num, u_anal)
            l2_err = compute_l2_error(u_num, u_anal, h)
            
            h_values.append(h)
            tau_values.append(tau)
            max_errors.append(max_err)
            l2_errors.append(l2_err)
    
    return np.array(h_values), np.array(tau_values), np.array(max_errors), np.array(l2_errors)


# ============================================================================
# Визуализация
# ============================================================================

def plot_solution(
    x: np.ndarray,
    t: np.ndarray,
    u_numerical: np.ndarray,
    u_analytical: np.ndarray,
    scheme_name: str,
    ic_name: str,
    boundary_name: str,
    output_dir: str,
):
    """Визуализация решения"""
    fig = plt.figure(figsize=(14, 10))
    
    # Выбираем несколько моментов времени для визуализации
    time_indices = [0, len(t) // 4, len(t) // 2, -1]
    
    # Создаем сетку: 3 строки, 2 столбца
    # Верхние 2x2 - решения, нижняя строка - разности
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Графики решений
    for idx in range(4):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])
        
        n = time_indices[idx]
        # Вычисляем разницу для проверки
        diff = np.abs(u_numerical[n, :] - u_analytical[n, :])
        max_diff = np.max(diff)
        
        # Показываем численное и аналитическое решения
        line1 = ax.plot(x, u_numerical[n, :], 'b-', label='Численное', linewidth=2.5, alpha=0.8)
        line2 = ax.plot(x, u_analytical[n, :], 'r--', label='Аналитическое', linewidth=2, alpha=0.8, dashes=(5, 5))
        
        # Показываем разницу в виде заливки
        diff_vals = u_numerical[n, :] - u_analytical[n, :]
        ax.fill_between(x, u_analytical[n, :], u_numerical[n, :], 
                       where=(diff_vals > 0), alpha=0.3, color='blue', label='Разница')
        ax.fill_between(x, u_analytical[n, :], u_numerical[n, :], 
                       where=(diff_vals <= 0), alpha=0.3, color='red')
        
        # Добавляем информацию о максимальной погрешности и значении функции
        u_max = np.max(np.abs(u_analytical[n, :]))
        rel_error = max_diff / u_max if u_max > 1e-10 else max_diff
        ax.text(0.02, 0.98, f'Max error: {max_diff:.2e}\nRel error: {rel_error:.2e}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7), fontsize=9)
        
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('u(x,t)', fontsize=10)
        ax.set_title(f't = {t[n]:.3f}', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Графики разности (нижняя строка)
    for idx in range(2):
        ax = fig.add_subplot(gs[2, idx])
        n = time_indices[idx * 2]  # Берем первый и третий момент времени
        diff = u_numerical[n, :] - u_analytical[n, :]
        
        ax.plot(x, diff, 'g-', linewidth=2)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('Разность (числ - анал)', fontsize=10)
        ax.set_title(f'Разность при t = {t[n]:.3f}', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('symlog', linthresh=1e-6)  # Симметричный log для показа малых значений
    
    plt.suptitle(f'{scheme_name} - {ic_name} - {boundary_name}', fontsize=14, y=0.995)
    filename = os.path.join(output_dir, f'solution_{scheme_name}_{ic_name}_{boundary_name}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_error(
    x: np.ndarray,
    t: np.ndarray,
    u_numerical: np.ndarray,
    u_analytical: np.ndarray,
    scheme_name: str,
    ic_name: str,
    boundary_name: str,
    output_dir: str,
):
    """Визуализация погрешности"""
    error = compute_error(u_numerical, u_analytical)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    time_indices = [0, len(t) // 4, len(t) // 2, -1]
    
    for idx, ax in enumerate(axes.flat):
        n = time_indices[idx]
        error_vals = error[n, :]
        # Фильтруем нулевые и отрицательные значения для log scale
        positive_mask = error_vals > 1e-15
        if np.any(positive_mask):
            ax.plot(x[positive_mask], error_vals[positive_mask], 'g-', linewidth=2)
            ax.set_yscale('log')
        else:
            ax.plot(x, error_vals, 'g-', linewidth=2)
        ax.set_xlabel('x')
        ax.set_ylabel('Погрешность')
        ax.set_title(f't = {t[n]:.3f}')
        ax.grid(True)
    
    plt.suptitle(f'Погрешность: {scheme_name} - {ic_name} - {boundary_name}', fontsize=14)
    plt.tight_layout()
    filename = os.path.join(output_dir, f'error_{scheme_name}_{ic_name}_{boundary_name}.png')
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_grid_dependence(
    h_values: np.ndarray,
    tau_values: np.ndarray,
    max_errors: np.ndarray,
    l2_errors: np.ndarray,
    scheme_name: str,
    ic_name: str,
    boundary_name: str,
    output_dir: str,
):
    """Визуализация зависимости погрешности от параметров сетки"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Фильтруем nan и inf значения
    valid_mask = np.isfinite(max_errors) & (max_errors > 0)
    if not np.any(valid_mask):
        plt.close()
        return
    
    h_vals_valid = h_values[valid_mask]
    tau_vals_valid = tau_values[valid_mask]
    err_vals_valid = max_errors[valid_mask]
    
    # Зависимость от h при фиксированном tau
    unique_tau = np.unique(tau_vals_valid)
    for tau in unique_tau[:3]:  # Берем первые 3 значения
        mask = np.abs(tau_vals_valid - tau) < 1e-10
        if np.any(mask):
            h_vals = h_vals_valid[mask]
            err_vals = err_vals_valid[mask]
            sorted_indices = np.argsort(h_vals)
            axes[0].loglog(h_vals[sorted_indices], err_vals[sorted_indices], 'o-', label=f'τ = {tau:.4f}')
    
    axes[0].set_xlabel('h')
    axes[0].set_ylabel('Максимальная погрешность')
    axes[0].set_title('Зависимость от h')
    axes[0].legend()
    axes[0].grid(True)
    
    # Зависимость от tau при фиксированном h
    unique_h = np.unique(h_vals_valid)
    for h in unique_h[:3]:  # Берем первые 3 значения
        mask = np.abs(h_vals_valid - h) < 1e-10
        if np.any(mask):
            tau_vals = tau_vals_valid[mask]
            err_vals = err_vals_valid[mask]
            sorted_indices = np.argsort(tau_vals)
            axes[1].loglog(tau_vals[sorted_indices], err_vals[sorted_indices], 's-', label=f'h = {h:.4f}')
    
    axes[1].set_xlabel('τ')
    axes[1].set_ylabel('Максимальная погрешность')
    axes[1].set_title('Зависимость от τ')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.suptitle(f'Зависимость погрешности: {scheme_name} - {ic_name} - {boundary_name}', fontsize=14)
    plt.tight_layout()
    filename = os.path.join(output_dir, f'grid_dependence_{scheme_name}_{ic_name}_{boundary_name}.png')
    plt.savefig(filename, dpi=150)
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    """Основная функция для запуска всех экспериментов"""
    
    # Создаем папку для результатов
    output_dir = "lab6/results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Параметры сетки
    n_x = 50
    n_t = 100
    
    # Схемы
    schemes = [
        (explicit_cross_scheme, "Явная_крест"),
        (implicit_scheme, "Неявная"),
    ]
    
    # Аппроксимации второго начального условия
    ic_approximations = [
        (approximate_second_ic_first_order, "1-го_порядка"),
        (approximate_second_ic_second_order, "2-го_порядка"),
    ]
    
    # Аппроксимации граничных условий (для совместимости, хотя в задаче Дирихле)
    boundary_approximations = [
        (apply_boundary_condition_two_point_first_order, "Двухточечная_1-го_порядка"),
        (apply_boundary_condition_three_point_second_order, "Трехточечная_2-го_порядка"),
        (apply_boundary_condition_two_point_second_order, "Двухточечная_2-го_порядка"),
    ]
    
    print("=" * 80)
    print("Лабораторная работа 2: Численные методы решения гиперболических уравнений")
    print("Вариант 9")
    print("=" * 80)
    print(f"Уравнение: ∂²u/∂t² + 3 ∂u/∂t = ∂²u/∂x² + ∂u/∂x - u + sin(x) exp(-t)")
    print(f"Сетка: n_x = {n_x}, n_t = {n_t}")
    print(f"Шаги: h = {(X_MAX - X_MIN) / n_x:.6f}, τ = {T_MAX / n_t:.6f}")
    print("=" * 80)
    
    # Решение для всех комбинаций схем, аппроксимаций НУ и ГУ
    results = []
    
    for scheme_func, scheme_name in schemes:
        for ic_func, ic_name in ic_approximations:
            for boundary_func, boundary_name in boundary_approximations:
                print(f"\n{scheme_name} схема + НУ {ic_name} + ГУ {boundary_name}")
                print("-" * 80)
                
                # Решение задачи
                x, t, u_num, u_anal = solve_pde(scheme_func, ic_func, n_x, n_t, scheme_name, ic_name)
                
                # Вычисление погрешностей
                max_err = compute_max_error(u_num, u_anal)
                l2_err = compute_l2_error(u_num, u_anal, (X_MAX - X_MIN) / n_x)
                
                print(f"Максимальная погрешность: {max_err:.6e}")
                print(f"L2 норма погрешности: {l2_err:.6e}")
                
                # Визуализация
                plot_solution(x, t, u_num, u_anal, scheme_name, ic_name, boundary_name, output_dir)
                plot_error(x, t, u_num, u_anal, scheme_name, ic_name, boundary_name, output_dir)
                
                results.append({
                    'scheme': scheme_name,
                    'ic': ic_name,
                    'boundary': boundary_name,
                    'max_error': max_err,
                    'l2_error': l2_err,
                })
    
    # Исследование зависимости от параметров сетки
    print("\n" + "=" * 80)
    print("Исследование зависимости погрешности от параметров сетки")
    print("=" * 80)
    
    n_x_values = [20, 30, 40, 50, 60, 80, 100]
    n_t_values = [50, 75, 100, 150, 200]
    
    # Исследуем для нескольких комбинаций
    test_configs = [
        (explicit_cross_scheme, approximate_second_ic_second_order, "Явная_крест", "2-го_порядка", "Трехточечная_2-го_порядка"),
        (implicit_scheme, approximate_second_ic_second_order, "Неявная", "2-го_порядка", "Трехточечная_2-го_порядка"),
    ]
    
    for scheme_func, ic_func, scheme_name, ic_name, boundary_name in test_configs:
        print(f"\nИсследование для: {scheme_name} + НУ {ic_name} + ГУ {boundary_name}")
        h_vals, tau_vals, max_errs, l2_errs = study_grid_dependence(
            scheme_func, ic_func, n_x_values, n_t_values, scheme_name, ic_name
        )
        
        plot_grid_dependence(h_vals, tau_vals, max_errs, l2_errs, scheme_name, ic_name, boundary_name, output_dir)
        
        # Вывод результатов исследования
        print("\nРезультаты исследования зависимости от параметров сетки:")
        print(f"{'h':>12} {'τ':>12} {'Max Error':>15} {'L2 Error':>15}")
        print("-" * 60)
        for i in range(min(20, len(h_vals))):  # Показываем первые 20
            if np.isfinite(max_errs[i]) and max_errs[i] > 0:
                print(f"{h_vals[i]:>12.6f} {tau_vals[i]:>12.6f} {max_errs[i]:>15.6e} {l2_errs[i]:>15.6e}")
        
        # Вычисление порядка сходимости по h
        print("\nПорядок сходимости по h (при фиксированном τ):")
        unique_tau = np.unique(tau_vals)
        for tau in unique_tau[:2]:  # Берем первые 2 значения
            mask = np.abs(tau_vals - tau) < 1e-10
            if np.sum(mask) >= 2:
                h_sorted = h_vals[mask]
                err_sorted = max_errs[mask]
                sorted_idx = np.argsort(h_sorted)
                h_sorted = h_sorted[sorted_idx]
                err_sorted = err_sorted[sorted_idx]
                
                # Вычисляем порядок сходимости
                if len(h_sorted) >= 2:
                    orders = []
                    for i in range(len(h_sorted) - 1):
                        if err_sorted[i] > 0 and err_sorted[i+1] > 0:
                            order = np.log(err_sorted[i] / err_sorted[i+1]) / np.log(h_sorted[i] / h_sorted[i+1])
                            orders.append(order)
                    if orders:
                        avg_order = np.mean(orders)
                        print(f"  τ = {tau:.4f}: средний порядок = {avg_order:.3f}")
        
        # Вычисление порядка сходимости по τ
        print("\nПорядок сходимости по τ (при фиксированном h):")
        unique_h = np.unique(h_vals)
        for h in unique_h[:2]:  # Берем первые 2 значения
            mask = np.abs(h_vals - h) < 1e-10
            if np.sum(mask) >= 2:
                tau_sorted = tau_vals[mask]
                err_sorted = max_errs[mask]
                sorted_idx = np.argsort(tau_sorted)
                tau_sorted = tau_sorted[sorted_idx]
                err_sorted = err_sorted[sorted_idx]
                
                # Вычисляем порядок сходимости
                if len(tau_sorted) >= 2:
                    orders = []
                    for i in range(len(tau_sorted) - 1):
                        if err_sorted[i] > 0 and err_sorted[i+1] > 0:
                            order = np.log(err_sorted[i] / err_sorted[i+1]) / np.log(tau_sorted[i] / tau_sorted[i+1])
                            orders.append(order)
                    if orders:
                        avg_order = np.mean(orders)
                        print(f"  h = {h:.4f}: средний порядок = {avg_order:.3f}")
    
    # Сводная таблица результатов
    print("\n" + "=" * 80)
    print("Сводная таблица результатов")
    print("=" * 80)
    print(f"{'Схема':<20} {'НУ':<15} {'ГУ':<30} {'Max Error':>15} {'L2 Error':>15}")
    print("-" * 100)
    for res in results:
        print(f"{res['scheme']:<20} {res['ic']:<15} {res['boundary']:<30} {res['max_error']:>15.6e} {res['l2_error']:>15.6e}")
    
    print("\n" + "=" * 80)
    print(f"Все графики сохранены в папку: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

