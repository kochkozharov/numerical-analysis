"""
Лабораторная работа 5: Численные методы решения параболических уравнений
Вариант 9

Уравнение: ∂u/∂t = a·∂²u/∂x² + b·∂u/∂x
Граничные условия: u_x(0,t) - u(0,t) = -exp(-at)(cos(bt) + sin(bt)), u_x(π,t) - u(π,t) = exp(-at)(cos(bt) + sin(bt))
Начальное условие: u(x,0) = cos(x)
Аналитическое решение: U(x,t) = exp(-at) cos(x + bt)
"""

import numpy as np
from typing import Callable, Tuple
import matplotlib.pyplot as plt
import os


# ============================================================================
# Параметры задачи
# ============================================================================

# Параметры уравнения (вариант 9)
A = 1.0  # a > 0
B = 1.0  # b > 0

# Область решения
X_MIN = 0.0
X_MAX = np.pi
T_MAX = 1.0


# ============================================================================
# Аналитическое решение и функции граничных условий
# ============================================================================

def analytical_solution(x: np.ndarray, t: float) -> np.ndarray:
    """Аналитическое решение U(x,t) = exp(-at) cos(x + bt)"""
    return np.exp(-A * t) * np.cos(x + B * t)


def boundary_condition_left(t: float) -> float:
    """Левое граничное условие: u_x(0,t) - u(0,t) = -exp(-at)(cos(bt) + sin(bt))"""
    return -np.exp(-A * t) * (np.cos(B * t) + np.sin(B * t))


def boundary_condition_right(t: float) -> float:
    """Правое граничное условие: u_x(π,t) - u(π,t) = exp(-at)(cos(bt) + sin(bt))"""
    return np.exp(-A * t) * (np.cos(B * t) + np.sin(B * t))


def initial_condition(x: np.ndarray) -> np.ndarray:
    """Начальное условие: u(x,0) = cos(x)"""
    return np.cos(x)


# ============================================================================
# Аппроксимации краевых условий
# ============================================================================

def apply_boundary_condition_two_point_first_order(
    u: np.ndarray, h: float, f_left: float, f_right: float
) -> None:
    """
    Двухточечная аппроксимация первого порядка точности.
    u_x(0) ≈ (u_1 - u_0)/h, поэтому u_x(0) - u_0 = f => (u_1 - u_0)/h - u_0 = f
    Отсюда: u_0 = (u_1 - h*f) / (1 + h)
    """
    # Проверка на валидность данных
    if not np.isfinite(u[1]) or not np.isfinite(f_left):
        u[0] = np.nan
    else:
        denominator = 1 + h
        if abs(denominator) < 1e-15:
            u[0] = np.nan
        else:
            numerator = u[1] - h * f_left
            u[0] = numerator / denominator
            if not np.isfinite(u[0]) or abs(u[0]) > 1e10:
                u[0] = np.nan
    
    if not np.isfinite(u[-2]) or not np.isfinite(f_right):
        u[-1] = np.nan
    else:
        denominator = 1 - h
        if abs(denominator) < 1e-15:
            u[-1] = np.nan
        else:
            numerator = u[-2] + h * f_right
            u[-1] = numerator / denominator
            if not np.isfinite(u[-1]) or abs(u[-1]) > 1e10:
                u[-1] = np.nan


def apply_boundary_condition_three_point_second_order(
    u: np.ndarray, h: float, f_left: float, f_right: float
) -> None:
    """
    Трехточечная аппроксимация второго порядка точности.
    Левая граница: u_x(0) ≈ (-3*u_0 + 4*u_1 - u_2)/(2*h)
    u_x(0) - u_0 = f => (-3*u_0 + 4*u_1 - u_2)/(2*h) - u_0 = f
    Отсюда: u_0 = (4*u_1 - u_2 - 2*h*f) / (3 + 2*h)
    
    Правая граница: u_x(π) ≈ (3*u_n - 4*u_{n-1} + u_{n-2})/(2*h)
    u_x(π) - u_n = f => (3*u_n - 4*u_{n-1} + u_{n-2})/(2*h) - u_n = f
    Отсюда: u_n = (4*u_{n-1} - u_{n-2} + 2*h*f) / (3 - 2*h)
    """
    # Проверка на валидность данных
    if not np.isfinite(u[1]) or not np.isfinite(u[2]) or not np.isfinite(f_left):
        u[0] = np.nan
    else:
        denominator = 3 + 2 * h
        if abs(denominator) < 1e-15:
            u[0] = np.nan
        else:
            numerator = 4 * u[1] - u[2] - 2 * h * f_left
            u[0] = numerator / denominator
            if not np.isfinite(u[0]) or abs(u[0]) > 1e10:
                u[0] = np.nan
    
    if not np.isfinite(u[-2]) or not np.isfinite(u[-3]) or not np.isfinite(f_right):
        u[-1] = np.nan
    else:
        denominator = 3 - 2 * h
        if abs(denominator) < 1e-15:
            u[-1] = np.nan
        else:
            numerator = 4 * u[-2] - u[-3] + 2 * h * f_right
            u[-1] = numerator / denominator
            if not np.isfinite(u[-1]) or abs(u[-1]) > 1e10:
                u[-1] = np.nan


def apply_boundary_condition_two_point_second_order(
    u: np.ndarray, h: float, f_left: float, f_right: float
) -> None:
    """
    Двухточечная аппроксимация второго порядка точности через ghost point (фиктивную точку).
    
    Метод: используем симметричную формулу для u_x(0) второго порядка:
    u_x(0) ≈ (u_1 - u_{-1})/(2*h)
    
    Из граничного условия: u_x(0) - u_0 = f
    (u_1 - u_{-1})/(2*h) - u_0 = f
    u_1 - u_{-1} - 2*h*u_0 = 2*h*f
    u_{-1} = u_1 - 2*h*u_0 - 2*h*f
    
    Используем стандартную формулу для u_xx(0):
    u_xx(0) ≈ (u_1 - 2*u_0 + u_{-1})/h²
    
    Подставляя u_{-1}:
    u_xx(0) ≈ (u_1 - 2*u_0 + u_1 - 2*h*u_0 - 2*h*f)/h²
    u_xx(0) ≈ 2*(u_1 - u_0 - h*u_0 - h*f)/h²
    
    Используем формулу второго порядка с поправкой:
    u_x(0) ≈ (u_1 - u_0)/h - (h/2)*u_xx(0)
    
    Из граничного условия: u_x(0) - u_0 = f
    (u_1 - u_0)/h - (h/2)*u_xx(0) - u_0 = f
    
    Подставляя u_xx(0) и решая, получаем:
    u_0 = (u_1 - h*f) / (1 - h + h²/2)
    """
    # Проверка на валидность данных
    if not np.isfinite(u[1]) or not np.isfinite(f_left):
        u[0] = np.nan
    else:
        # Используем формулу второго порядка через ghost point
        # u_0 = (u_1 - h*f) / (1 - h + h²/2)
        denominator = 1.0 - h + (h * h) / 2.0
        if abs(denominator) < 1e-15:
            u[0] = np.nan
        else:
            numerator = u[1] - h * f_left
            u0 = numerator / denominator
            
            # Проверка на валидность
            if not np.isfinite(u0) or abs(u0) > 1e10:
                u[0] = np.nan
            else:
                u[0] = u0
    
    # Аналогично для правой границы
    if not np.isfinite(u[-2]) or not np.isfinite(f_right):
        u[-1] = np.nan
    else:
        # Для правой границы: u_x(π) ≈ (u_{n+1} - u_{n-1})/(2*h)
        # Из граничного условия: u_x(π) - u_n = f
        # u_n = (u_{n-1} + h*f) / (1 - h + h²/2)
        denominator = 1.0 - h + (h * h) / 2.0
        if abs(denominator) < 1e-15:
            u[-1] = np.nan
        else:
            numerator = u[-2] + h * f_right
            un = numerator / denominator
            
            # Проверка на валидность
            if not np.isfinite(un) or abs(un) > 1e10:
                u[-1] = np.nan
            else:
                u[-1] = un


# ============================================================================
# Численные схемы
# ============================================================================

def explicit_scheme(
    u: np.ndarray,
    h: float,
    tau: float,
    boundary_func: Callable[[np.ndarray, float, float, float], None],
    f_left: float,
    f_right: float,
) -> np.ndarray:
    """
    Явная конечно-разностная схема.
    u_i^{n+1} = u_i^n + τ*(a*(u_{i+1}^n - 2*u_i^n + u_{i-1}^n)/h² 
                      + b*(u_{i+1}^n - u_{i-1}^n)/(2*h))
    
    Условие устойчивости: τ ≤ min(h²/(2a), h/|b|) для конвекции-диффузии
    """
    u_new = u.copy()
    n = len(u)
    
    # Проверка условия устойчивости
    max_tau_diff = h * h / (2 * A)
    max_tau_conv = h / B if B > 0 else np.inf
    max_tau = min(max_tau_diff, max_tau_conv)
    
    # Проверка на валидность входных данных
    if np.any(~np.isfinite(u)):
        return np.full(n, np.nan)
    
    if tau > max_tau:
        # Если шаг слишком большой, используем подшаги
        num_substeps = int(np.ceil(tau / max_tau)) + 1
        tau_sub = tau / num_substeps
        for _ in range(num_substeps):
            u_prev = u_new.copy()
            u_tmp = u_new.copy()
            # Внутренние точки
            for i in range(1, n - 1):
                # Проверка на валидность данных
                if not np.isfinite(u_prev[i + 1]) or not np.isfinite(u_prev[i]) or not np.isfinite(u_prev[i - 1]):
                    return np.full(n, np.nan)
                
                u_xx = (u_prev[i + 1] - 2 * u_prev[i] + u_prev[i - 1]) / (h * h)
                u_x = (u_prev[i + 1] - u_prev[i - 1]) / (2 * h)
                
                # Проверка на overflow
                if not np.isfinite(u_xx) or not np.isfinite(u_x):
                    return np.full(n, np.nan)
                
                u_tmp[i] = u_prev[i] + tau_sub * (A * u_xx + B * u_x)
                
                # Проверка на overflow после вычисления
                if not np.isfinite(u_tmp[i]) or abs(u_tmp[i]) > 1e10:
                    return np.full(n, np.nan)

            u_new = u_tmp
            
            # Применяем граничные условия
            if boundary_func is apply_boundary_condition_two_point_second_order:
                coef_l = 1.0 + h + (h * h / (2.0 * A)) * (1.0 / tau_sub - B)
                const_l = h * f_left + (h * h / (2.0 * A)) * (-(u_prev[0] / tau_sub) - B * f_left)
                if abs(coef_l) < 1e-15:
                    return np.full(n, np.nan)
                u_new[0] = (u_new[1] - const_l) / coef_l

                coef_r = 1.0 - h + (h * h / (2.0 * A)) * (1.0 / tau_sub - B)
                const_r = -h * f_right + (h * h / (2.0 * A)) * (-(u_prev[-1] / tau_sub) - B * f_right)
                if abs(coef_r) < 1e-15:
                    return np.full(n, np.nan)
                u_new[-1] = (u_new[-2] - const_r) / coef_r
            else:
                boundary_func(u_new, h, f_left, f_right)
            
            # Проверка после граничных условий
            if np.any(~np.isfinite(u_new)):
                return np.full(n, np.nan)
    else:
        # Внутренние точки
        for i in range(1, n - 1):
            # Проверка на валидность данных
            if not np.isfinite(u[i + 1]) or not np.isfinite(u[i]) or not np.isfinite(u[i - 1]):
                return np.full(n, np.nan)
            
            u_xx = (u[i + 1] - 2 * u[i] + u[i - 1]) / (h * h)
            u_x = (u[i + 1] - u[i - 1]) / (2 * h)
            
            # Проверка на overflow
            if not np.isfinite(u_xx) or not np.isfinite(u_x):
                return np.full(n, np.nan)
            
            u_new[i] = u[i] + tau * (A * u_xx + B * u_x)
            
            # Проверка на overflow после вычисления
            if not np.isfinite(u_new[i]) or abs(u_new[i]) > 1e10:
                return np.full(n, np.nan)

        # Применяем граничные условия
        if boundary_func is apply_boundary_condition_two_point_second_order:
            coef_l = 1.0 + h + (h * h / (2.0 * A)) * (1.0 / tau - B)
            const_l = h * f_left + (h * h / (2.0 * A)) * (-(u[0] / tau) - B * f_left)
            if abs(coef_l) < 1e-15:
                return np.full(n, np.nan)
            u_new[0] = (u_new[1] - const_l) / coef_l

            coef_r = 1.0 - h + (h * h / (2.0 * A)) * (1.0 / tau - B)
            const_r = -h * f_right + (h * h / (2.0 * A)) * (-(u[-1] / tau) - B * f_right)
            if abs(coef_r) < 1e-15:
                return np.full(n, np.nan)
            u_new[-1] = (u_new[-2] - const_r) / coef_r
        else:
            boundary_func(u_new, h, f_left, f_right)
        
        # Проверка после граничных условий
        if np.any(~np.isfinite(u_new)):
            return np.full(n, np.nan)
    
    return u_new


def thomas_algorithm(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Метод прогонки (Thomas algorithm) для решения трехдиагональной системы.
    Система: a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]
    """
    n = len(d)
    
    # Проверка на валидность входных данных
    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(b)) or np.any(~np.isfinite(c)) or np.any(~np.isfinite(d)):
        return np.full(n, np.nan)
    
    # Копируем массивы для модификации
    b_copy = b.copy()
    d_copy = d.copy()
    
    # Прямой ход
    for i in range(1, n):
        if abs(b_copy[i - 1]) < 1e-15:
            return np.full(n, np.nan)
        
        m = a[i] / b_copy[i - 1]
        
        # Проверка на overflow
        if not np.isfinite(m) or abs(m) > 1e10:
            return np.full(n, np.nan)
        
        b_copy[i] = b_copy[i] - m * c[i - 1]
        d_copy[i] = d_copy[i] - m * d_copy[i - 1]
        
        # Проверка на overflow после вычислений
        if not np.isfinite(b_copy[i]) or not np.isfinite(d_copy[i]):
            return np.full(n, np.nan)
    
    # Обратный ход
    x = np.zeros(n)
    
    if abs(b_copy[n - 1]) < 1e-15:
        return np.full(n, np.nan)
    
    x[n - 1] = d_copy[n - 1] / b_copy[n - 1]
    
    if not np.isfinite(x[n - 1]):
        return np.full(n, np.nan)
    
    for i in range(n - 2, -1, -1):
        if abs(b_copy[i]) < 1e-15:
            return np.full(n, np.nan)
        
        x[i] = (d_copy[i] - c[i] * x[i + 1]) / b_copy[i]
        
        # Проверка на overflow
        if not np.isfinite(x[i]):
            return np.full(n, np.nan)
    
    return x


def _boundary_relations(
    boundary_func: Callable[[np.ndarray, float, float, float], None],
    h: float,
    f_left: float,
    f_right: float,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    if boundary_func is apply_boundary_condition_two_point_first_order:
        denom_l = 1.0 + h
        denom_r = 1.0 - h
        return (
            (1.0 / denom_l, 0.0, (-h * f_left) / denom_l),
            (1.0 / denom_r, 0.0, (h * f_right) / denom_r),
        )
    if boundary_func is apply_boundary_condition_three_point_second_order:
        denom_l = 3.0 + 2.0 * h
        denom_r = 3.0 - 2.0 * h
        return (
            (4.0 / denom_l, -1.0 / denom_l, (-2.0 * h * f_left) / denom_l),
            (4.0 / denom_r, -1.0 / denom_r, (2.0 * h * f_right) / denom_r),
        )
    if boundary_func is apply_boundary_condition_two_point_second_order:
        denom_l = 1.0 - h + (h * h) / 2.0
        denom_r = 1.0 - h + (h * h) / 2.0
        return (
            (1.0 / denom_l, 0.0, (-h * f_left) / denom_l),
            (1.0 / denom_r, 0.0, (h * f_right) / denom_r),
        )
    raise ValueError("Unknown boundary approximation")


def implicit_scheme(
    u: np.ndarray,
    h: float,
    tau: float,
    boundary_func: Callable[[np.ndarray, float, float, float], None],
    f_left: float,
    f_right: float,
) -> np.ndarray:
    """
    Неявная конечно-разностная схема.
    Решаем систему: u_i^{n+1} - τ*(a*u_xx^{n+1} + b*u_x^{n+1}) = u_i^n
    Используем метод прогонки для решения системы линейных уравнений.
    Граничные условия включаются в систему через аппроксимацию производной.
    """
    n = len(u)

    if n < 3:
        return u.copy()
    
    # Коэффициенты разностной схемы
    alpha = tau * A / (h * h)
    beta = tau * B / (2 * h)
    gamma = 1.0  # коэффициент при u_i^{n+1}
    
    if np.any(~np.isfinite(u)):
        return np.full(n, np.nan)

    (p1, p2, q), (r1, r2, s) = _boundary_relations(boundary_func, h, f_left, f_right)
    if boundary_func is apply_boundary_condition_two_point_second_order:
        k_l = 1.0 - (h * B) / (2.0 * A)
        denom_l = k_l + (1.0 / h) + h / (2.0 * A * tau)
        k_r = 1.0 + (h * B) / (2.0 * A)
        denom_r = (1.0 / h) + h / (2.0 * A * tau) - k_r
        if abs(denom_l) < 1e-15 or abs(denom_r) < 1e-15:
            return np.full(n, np.nan)

        p1 = (1.0 / h) / denom_l
        p2 = 0.0
        q = ((h / (2.0 * A * tau)) * u[0] - k_l * f_left) / denom_l

        r1 = (1.0 / h) / denom_r
        r2 = 0.0
        s = ((h / (2.0 * A * tau)) * u[-1] + k_r * f_right) / denom_r

    m = n - 2
    a_red = np.zeros(m)
    b_red = np.zeros(m)
    c_red = np.zeros(m)
    d_red = np.zeros(m)

    a0 = -alpha + beta
    b0 = gamma + 2.0 * alpha
    c0 = -alpha - beta

    i = 1
    j = 0
    b_red[j] = b0 + a0 * p1
    c_red[j] = c0 + a0 * p2
    d_red[j] = u[i] - a0 * q

    for i in range(2, n - 2):
        j = i - 1
        a_red[j] = a0
        b_red[j] = b0
        c_red[j] = c0
        d_red[j] = u[i]

    i = n - 2
    j = m - 1
    a_red[j] = a0 + c0 * r2
    b_red[j] = b0 + c0 * r1
    c_red[j] = 0.0
    d_red[j] = u[i] - c0 * s

    u_inner = thomas_algorithm(a_red, b_red, c_red, d_red)
    if np.any(~np.isfinite(u_inner)):
        return np.full(n, np.nan)

    u_new = u.copy()
    u_new[1:-1] = u_inner
    u_new[0] = p1 * u_new[1] + p2 * u_new[2] + q
    u_new[-1] = r1 * u_new[-2] + r2 * u_new[-3] + s

    if np.any(~np.isfinite(u_new)):
        return np.full(n, np.nan)

    return u_new


def crank_nicolson_scheme(
    u: np.ndarray,
    h: float,
    tau: float,
    boundary_func: Callable[[np.ndarray, float, float, float], None],
    f_left: float,
    f_right: float,
) -> np.ndarray:
    """
    Схема Кранка-Николсона (полусумма явной и неявной схем).
    u_i^{n+1} - u_i^n = (τ/2)*[L(u^n) + L(u^{n+1})]
    где L(u) = a*u_xx + b*u_x
    
    Переписываем в виде:
    u_i^{n+1} - (τ/2)*L(u^{n+1}) = u_i^n + (τ/2)*L(u^n)
    """
    n = len(u)

    if n < 3:
        return u.copy()
    
    # Коэффициенты разностной схемы
    alpha = tau * A / (2 * h * h)  # для неявной части
    beta = tau * B / (4 * h)  # для неявной части
    gamma = 1.0  # коэффициент при u_i^{n+1}
    
    # Проверка на валидность входных данных
    if np.any(~np.isfinite(u)):
        return np.full(n, np.nan)
    
    # Вычисляем явную часть (правая часть уравнения)
    explicit_rhs = np.zeros(n)
    for i in range(1, n - 1):
        # Проверка на валидность данных перед вычислениями
        if not np.isfinite(u[i + 1]) or not np.isfinite(u[i]) or not np.isfinite(u[i - 1]):
            return np.full(n, np.nan)
        
        u_xx_n = (u[i + 1] - 2 * u[i] + u[i - 1]) / (h * h)
        u_x_n = (u[i + 1] - u[i - 1]) / (2 * h)
        
        # Проверка на overflow
        if not np.isfinite(u_xx_n) or not np.isfinite(u_x_n):
            return np.full(n, np.nan)
        
        explicit_rhs[i] = u[i] + (tau / 2) * (A * u_xx_n + B * u_x_n)
        
        # Проверка на overflow после вычисления
        if not np.isfinite(explicit_rhs[i]) or abs(explicit_rhs[i]) > 1e10:
            return np.full(n, np.nan)
    
    (p1, p2, q), (r1, r2, s) = _boundary_relations(boundary_func, h, f_left, f_right)
    if boundary_func is apply_boundary_condition_two_point_second_order:
        k_l = 1.0 - (h * B) / (2.0 * A)
        denom_l = k_l + (1.0 / h) + h / (2.0 * A * tau)
        k_r = 1.0 + (h * B) / (2.0 * A)
        denom_r = (1.0 / h) + h / (2.0 * A * tau) - k_r
        if abs(denom_l) < 1e-15 or abs(denom_r) < 1e-15:
            return np.full(n, np.nan)

        p1 = (1.0 / h) / denom_l
        p2 = 0.0
        q = ((h / (2.0 * A * tau)) * u[0] - k_l * f_left) / denom_l

        r1 = (1.0 / h) / denom_r
        r2 = 0.0
        s = ((h / (2.0 * A * tau)) * u[-1] + k_r * f_right) / denom_r

    m = n - 2
    a_red = np.zeros(m)
    b_red = np.zeros(m)
    c_red = np.zeros(m)
    d_red = np.zeros(m)

    a0 = -alpha + beta
    b0 = gamma + 2.0 * alpha
    c0 = -alpha - beta

    i = 1
    j = 0
    b_red[j] = b0 + a0 * p1
    c_red[j] = c0 + a0 * p2
    d_red[j] = explicit_rhs[i] - a0 * q

    for i in range(2, n - 2):
        j = i - 1
        a_red[j] = a0
        b_red[j] = b0
        c_red[j] = c0
        d_red[j] = explicit_rhs[i]

    i = n - 2
    j = m - 1
    a_red[j] = a0 + c0 * r2
    b_red[j] = b0 + c0 * r1
    c_red[j] = 0.0
    d_red[j] = explicit_rhs[i] - c0 * s

    u_inner = thomas_algorithm(a_red, b_red, c_red, d_red)
    if np.any(~np.isfinite(u_inner)):
        return np.full(n, np.nan)

    u_new = u.copy()
    u_new[1:-1] = u_inner
    u_new[0] = p1 * u_new[1] + p2 * u_new[2] + q
    u_new[-1] = r1 * u_new[-2] + r2 * u_new[-3] + s

    if np.any(~np.isfinite(u_new)):
        return np.full(n, np.nan)

    return u_new


# ============================================================================
# Решение задачи
# ============================================================================

def solve_pde(
    scheme_func: Callable,
    boundary_func: Callable,
    n_x: int,
    n_t: int,
    scheme_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Решает задачу для заданной схемы и аппроксимации граничных условий.
    
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
    
    # Начальное условие
    u = initial_condition(x)
    u_numerical = np.zeros((n_t + 1, n_x + 1))
    u_numerical[0, :] = u
    
    # Решение по времени
    for n in range(n_t):
        # Используем время на следующем шаге для граничных условий (неявные схемы)
        # или текущее время (явные схемы) - зависит от схемы
        t_bc = t[n + 1]
        f_left = boundary_condition_left(t_bc)
        f_right = boundary_condition_right(t_bc)
        
        u = scheme_func(u, h, tau, boundary_func, f_left, f_right)
        u_numerical[n + 1, :] = u
        
        # Проверка на NaN/Inf
        if np.any(~np.isfinite(u)):
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
    boundary_func: Callable,
    n_x_values: list,
    n_t_values: list,
    scheme_name: str,
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
            
            _, _, u_num, u_anal = solve_pde(scheme_func, boundary_func, n_x, n_t, scheme_name)
            
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
    boundary_name: str,
    output_dir: str,
):
    """Визуализация решения"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Выбираем несколько моментов времени для визуализации
    time_indices = [0, len(t) // 4, len(t) // 2, -1]
    
    for idx, ax in enumerate(axes.flat):
        n = time_indices[idx]
        ax.plot(x, u_numerical[n, :], 'b-', label='Численное', linewidth=2)
        ax.plot(x, u_analytical[n, :], 'r--', label='Аналитическое', linewidth=2)
        ax.set_xlabel('x')
        ax.set_ylabel('u(x,t)')
        ax.set_title(f't = {t[n]:.3f}')
        ax.legend()
        ax.grid(True)
    
    plt.suptitle(f'{scheme_name} - {boundary_name}', fontsize=14)
    plt.tight_layout()
    filename = os.path.join(output_dir, f'solution_{scheme_name}_{boundary_name}.png')
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_error(
    x: np.ndarray,
    t: np.ndarray,
    u_numerical: np.ndarray,
    u_analytical: np.ndarray,
    scheme_name: str,
    boundary_name: str,
    output_dir: str,
):
    """Визуализация погрешности"""
    error = compute_error(u_numerical, u_analytical)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    time_indices = [0, len(t) // 4, len(t) // 2, -1]
    
    for idx, ax in enumerate(axes.flat):
        n = time_indices[idx]
        ax.plot(x, error[n, :], 'g-', linewidth=2)
        ax.set_xlabel('x')
        ax.set_ylabel('Погрешность')
        ax.set_title(f't = {t[n]:.3f}')
        ax.grid(True)
    
    plt.suptitle(f'Погрешность: {scheme_name} - {boundary_name}', fontsize=14)
    plt.tight_layout()
    filename = os.path.join(output_dir, f'error_{scheme_name}_{boundary_name}.png')
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_grid_dependence(
    h_values: np.ndarray,
    tau_values: np.ndarray,
    max_errors: np.ndarray,
    l2_errors: np.ndarray,
    scheme_name: str,
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
    
    plt.suptitle(f'Зависимость погрешности: {scheme_name} - {boundary_name}', fontsize=14)
    plt.tight_layout()
    filename = os.path.join(output_dir, f'grid_dependence_{scheme_name}_{boundary_name}.png')
    plt.savefig(filename, dpi=150)
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    """Основная функция для запуска всех экспериментов"""
    
    # Создаем папку для результатов
    output_dir = "lab5/results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Параметры сетки
    n_x = 50
    n_t = 100
    
    # Схемы
    schemes = [
        (explicit_scheme, "Явная"),
        (implicit_scheme, "Неявная"),
        (crank_nicolson_scheme, "Кранк-Николсон"),
    ]
    
    # Аппроксимации граничных условий
    boundary_approximations = [
        (apply_boundary_condition_two_point_first_order, "Двухточечная 1-го порядка"),
        (apply_boundary_condition_three_point_second_order, "Трехточечная 2-го порядка"),
        (apply_boundary_condition_two_point_second_order, "Двухточечная 2-го порядка"),
    ]
    
    print("=" * 80)
    print("Лабораторная работа 5: Численные методы решения параболических уравнений")
    print("Вариант 9")
    print("=" * 80)
    print(f"Параметры: a = {A}, b = {B}")
    print(f"Сетка: n_x = {n_x}, n_t = {n_t}")
    print(f"Шаги: h = {(X_MAX - X_MIN) / n_x:.6f}, τ = {T_MAX / n_t:.6f}")
    print("=" * 80)
    
    # Решение для всех комбинаций схем и граничных условий
    results = []
    
    for scheme_func, scheme_name in schemes:
        for boundary_func, boundary_name in boundary_approximations:
            print(f"\n{scheme_name} схема + {boundary_name}")
            print("-" * 80)
            
            # Решение задачи
            x, t, u_num, u_anal = solve_pde(scheme_func, boundary_func, n_x, n_t, scheme_name)
            
            # Вычисление погрешностей
            max_err = compute_max_error(u_num, u_anal)
            l2_err = compute_l2_error(u_num, u_anal, (X_MAX - X_MIN) / n_x)
            
            print(f"Максимальная погрешность: {max_err:.6e}")
            print(f"L2 норма погрешности: {l2_err:.6e}")
            
            # Визуализация
            boundary_name_safe = boundary_name.replace(" ", "_")
            plot_solution(x, t, u_num, u_anal, scheme_name, boundary_name_safe, output_dir)
            plot_error(x, t, u_num, u_anal, scheme_name, boundary_name_safe, output_dir)
            
            results.append({
                'scheme': scheme_name,
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

    # Строим графики зависимости для всех комбинаций схем и аппроксимаций ГУ
    for scheme_func, scheme_name in schemes:
        for boundary_func, boundary_name in boundary_approximations:
            boundary_name_safe = boundary_name.replace(" ", "_")
            print(f"\nИсследование для: {scheme_name} + {boundary_name}")

            h_vals, tau_vals, max_errs, l2_errs = study_grid_dependence(
                scheme_func, boundary_func, n_x_values, n_t_values, scheme_name
            )

            plot_grid_dependence(
                h_vals,
                tau_vals,
                max_errs,
                l2_errs,
                scheme_name,
                boundary_name_safe,
                output_dir,
            )

            # Выводим краткий срез (первые 5 точек), чтобы не засорять лог
            print("Результаты (первые 5):")
            print(f"{'h':>12} {'τ':>12} {'Max Error':>15} {'L2 Error':>15}")
            print("-" * 60)
            for i in range(min(5, len(h_vals))):
                print(f"{h_vals[i]:>12.6f} {tau_vals[i]:>12.6f} {max_errs[i]:>15.6e} {l2_errs[i]:>15.6e}")
    
    # Сводная таблица результатов
    print("\n" + "=" * 80)
    print("Сводная таблица результатов")
    print("=" * 80)
    print(f"{'Схема':<20} {'Граничные условия':<30} {'Max Error':>15} {'L2 Error':>15}")
    print("-" * 80)
    for res in results:
        print(f"{res['scheme']:<20} {res['boundary']:<30} {res['max_error']:>15.6e} {res['l2_error']:>15.6e}")
    
    print("\n" + "=" * 80)
    print(f"Все графики сохранены в папку: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

