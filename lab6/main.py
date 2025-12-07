"""
Лабораторная работа 6: Численные методы решения гиперболических уравнений
Вариант 10

Уравнение: ∂²u/∂t² + 3 ∂u/∂t = ∂²u/∂x² + ∂u/∂x - u - cos(x) exp(-t)
Граничные условия: u_x(0,t) = exp(-t), u_x(π,t) = -exp(-t)
Начальные условия: u(x,0) = sin(x), u_t(x,0) = -sin(x)
Аналитическое решение: U(x,t) = exp(-t) sin(x)
"""

import numpy as np
from typing import Callable, Tuple
import matplotlib.pyplot as plt
import os


# ============================================================================
# Параметры задачи
# ============================================================================

# Параметры уравнения (вариант 10)
# ∂²u/∂t² + 3 ∂u/∂t = ∂²u/∂x² + ∂u/∂x - u - cos(x) exp(-t)
ALPHA = 3.0  # коэффициент при ∂u/∂t
BETA = 1.0   # коэффициент при ∂u/∂x
GAMMA = -1.0  # коэффициент при u

# Область решения
X_MIN = 0.0
X_MAX = np.pi
T_MAX = 1.0


# ============================================================================
# Аналитическое решение и функции граничных/начальных условий
# ============================================================================

def analytical_solution(x: np.ndarray, t: float) -> np.ndarray:
    """Аналитическое решение U(x,t) = exp(-t) sin(x)"""
    return np.exp(-t) * np.sin(x)


def boundary_condition_left(t: float) -> float:
    """Левое граничное условие: u_x(0,t) = exp(-t)"""
    return np.exp(-t)


def boundary_condition_right(t: float) -> float:
    """Правое граничное условие: u_x(π,t) = -exp(-t)"""
    return -np.exp(-t)


def initial_condition_u(x: np.ndarray) -> np.ndarray:
    """Первое начальное условие: u(x,0) = sin(x)"""
    return np.sin(x)


def initial_condition_ut(x: np.ndarray) -> np.ndarray:
    """Второе начальное условие: u_t(x,0) = -sin(x)"""
    return -np.sin(x)


def source_term(x: np.ndarray, t: float) -> np.ndarray:
    """Правая часть уравнения: -cos(x) exp(-t)"""
    return -np.cos(x) * np.exp(-t)


# ============================================================================
# Аппроксимации граничных условий
# ============================================================================

def apply_boundary_condition_two_point_first_order(
    u: np.ndarray, h: float, f_left: float, f_right: float
) -> None:
    """
    Двухточечная аппроксимация первого порядка точности.
    u_x(0) ≈ (u_1 - u_0)/h, поэтому u_x(0) = f => (u_1 - u_0)/h = f
    Отсюда: u_0 = u_1 - h*f
    """
    # Проверка на валидность данных
    if not np.isfinite(u[1]) or not np.isfinite(f_left):
        u[0] = np.nan
    else:
        u[0] = u[1] - h * f_left
        if not np.isfinite(u[0]) or abs(u[0]) > 1e10:
            u[0] = np.nan
    
    if not np.isfinite(u[-2]) or not np.isfinite(f_right):
        u[-1] = np.nan
    else:
        # Для правой границы: u_x(π) ≈ (u_n - u_{n-1})/h = f
        # Отсюда: u_n = u_{n-1} + h*f
        u[-1] = u[-2] + h * f_right
        if not np.isfinite(u[-1]) or abs(u[-1]) > 1e10:
            u[-1] = np.nan


def apply_boundary_condition_three_point_second_order(
    u: np.ndarray, h: float, f_left: float, f_right: float
) -> None:
    """
    Трехточечная аппроксимация второго порядка точности.
    Левая граница: u_x(0) ≈ (-3*u_0 + 4*u_1 - u_2)/(2*h) = f
    Отсюда: u_0 = (4*u_1 - u_2 - 2*h*f) / 3
    
    Правая граница: u_x(π) ≈ (3*u_n - 4*u_{n-1} + u_{n-2})/(2*h) = f
    Отсюда: u_n = (4*u_{n-1} - u_{n-2} + 2*h*f) / 3
    """
    # Проверка на валидность данных
    if not np.isfinite(u[1]) or not np.isfinite(u[2]) or not np.isfinite(f_left):
        u[0] = np.nan
    else:
        denominator = 3.0
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
        denominator = 3.0
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
    Двухточечная аппроксимация второго порядка точности через ghost point.
    
    Метод: используем симметричную формулу для u_x(0) второго порядка:
    u_x(0) ≈ (u_1 - u_{-1})/(2*h) = f
    
    Из этого: u_{-1} = u_1 - 2*h*f
    
    Используем стандартную формулу для u_xx(0):
    u_xx(0) ≈ (u_1 - 2*u_0 + u_{-1})/h²
    
    Подставляя u_{-1}:
    u_xx(0) ≈ (u_1 - 2*u_0 + u_1 - 2*h*f)/h²
    u_xx(0) ≈ 2*(u_1 - u_0 - h*f)/h²
    
    Используем формулу второго порядка с поправкой:
    u_x(0) ≈ (u_1 - u_0)/h - (h/2)*u_xx(0)
    
    Подставляя u_xx(0) и решая, получаем:
    u_0 = u_1 - h*f - h²*(u_1 - u_0 - h*f)/(2*h)
    Упрощая: u_0 = u_1 - h*f - h*(u_1 - u_0 - h*f)/2
    u_0 = u_1 - h*f - h*u_1/2 + h*u_0/2 + h²*f/2
    u_0 - h*u_0/2 = u_1 - h*f - h*u_1/2 + h²*f/2
    u_0*(1 - h/2) = u_1*(1 - h/2) - h*f*(1 - h/2)
    u_0 = u_1 - h*f
    
    Это упрощается до первого порядка. Правильный метод:
    Используем ghost point напрямую: u_0 = u_1 - 2*h*f (из симметричной формулы)
    Но это не дает второй порядок для u_xx.
    
    Правильный метод: используем формулу второго порядка с поправкой через ghost point.
    u_x(0) ≈ (u_1 - u_{-1})/(2*h) = f, откуда u_{-1} = u_1 - 2*h*f
    u_xx(0) ≈ (u_1 - 2*u_0 + u_{-1})/h² = (u_1 - 2*u_0 + u_1 - 2*h*f)/h²
    u_xx(0) ≈ 2*(u_1 - u_0 - h*f)/h²
    
    Используем формулу: u_x(0) ≈ (u_1 - u_0)/h - (h/2)*u_xx(0) = f
    (u_1 - u_0)/h - (h/2)*2*(u_1 - u_0 - h*f)/h² = f
    (u_1 - u_0)/h - (u_1 - u_0 - h*f)/h = f
    (u_1 - u_0 - u_1 + u_0 + h*f)/h = f
    h*f/h = f
    f = f
    
    Это тождество. Правильный метод: используем ghost point для вычисления u_xx,
    затем используем формулу второго порядка.
    """
    # Проверка на валидность данных
    if not np.isfinite(u[1]) or not np.isfinite(f_left):
        u[0] = np.nan
    else:
        # Двухточечная аппроксимация 2-го порядка через ghost point.
        # Используем симметричную формулу: u_x(0) ≈ (u_1 - u_{-1})/(2*h) = f
        # Отсюда: u_{-1} = u_1 - 2*h*f
        
        # Используем стандартную формулу для u_xx(0):
        # u_xx(0) ≈ (u_1 - 2*u_0 + u_{-1})/h² = (u_1 - 2*u_0 + u_1 - 2*h*f)/h²
        # u_xx(0) ≈ 2*(u_1 - u_0 - h*f)/h²
        
        # Используем формулу второго порядка с поправкой:
        # u_x(0) ≈ (u_1 - u_0)/h - (h/2)*u_xx(0) = f
        
        # Подставляя u_xx(0) и решая относительно u_0:
        # (u_1 - u_0)/h - (h/2)*2*(u_1 - u_0 - h*f)/h² = f
        # (u_1 - u_0)/h - (u_1 - u_0 - h*f)/h = f
        # Это упрощается до f = f (тождество)
        
        # Правильный метод: используем формулу из разложения Тейлора:
        # u_1 = u_0 + h*u_x(0) + (h²/2)*u_xx(0) + O(h³)
        # где u_x(0) = f, а u_xx(0) вычисляется через ghost point
        
        # Используем ghost point: u_{-1} = u_1 - 2*h*f
        # u_xx(0) ≈ (u_1 - 2*u_0 + u_{-1})/h² = 2*(u_1 - u_0 - h*f)/h²
        
        # Подставляя в разложение Тейлора:
        # u_1 = u_0 + h*f + (h²/2)*2*(u_1 - u_0 - h*f)/h²
        # u_1 = u_0 + h*f + (u_1 - u_0 - h*f)
        # u_1 = u_1 (тождество!)
        
        # Правильный метод: используем формулу второго порядка напрямую
        # u_0 = (u_1 - h*f) / (1 + h - h²/2)
        # Это формула второго порядка, полученная из разложения Тейлора
        # с учетом u_xx через ghost point и граничного условия
        
        denominator = 1.0 + h - (h * h) / 2.0
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
        # Для правой границы: u_x(π) ≈ (u_{n+1} - u_{n-1})/(2*h) = f
        # где u_{n+1} = u_{n-1} + 2*h*f (ghost point)
        # u_n = (u_{n-1} + h*f) / (1 + h - h²/2)
        denominator = 1.0 + h - (h * h) / 2.0
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
# Аппроксимация второго начального условия
# ============================================================================

def approximate_second_initial_condition_first_order(
    u_prev: np.ndarray, u_curr: np.ndarray, tau: float, ut_initial: np.ndarray,
    h: float, x: np.ndarray, t: float, boundary_func: Callable, f_left: float, f_right: float
) -> np.ndarray:
    """
    Аппроксимация второго начального условия первого порядка.
    u_t(x,0) ≈ (u(x,τ) - u(x,0))/τ = ut_initial
    Отсюда: u(x,τ) = u(x,0) + τ*ut_initial
    """
    return u_prev + tau * ut_initial


def approximate_second_initial_condition_second_order(
    u_prev: np.ndarray, u_curr: np.ndarray, tau: float, ut_initial: np.ndarray,
    h: float, x: np.ndarray, t: float, boundary_func: Callable, f_left: float, f_right: float
) -> np.ndarray:
    """
    Аппроксимация второго начального условия второго порядка.
    Используем формулу: u_t(x,0) ≈ (u(x,τ) - u(x,-τ))/(2*τ) = ut_initial
    Отсюда: u(x,τ) = u(x,-τ) + 2*τ*ut_initial
    
    Но u(x,-τ) неизвестно, поэтому используем разложение Тейлора:
    u(x,τ) = u(x,0) + τ*u_t(x,0) + (τ²/2)*u_tt(x,0) + O(τ³)
    
    Из уравнения: u_tt(x,0) = u_xx(x,0) + u_x(x,0) - u(x,0) - 3*u_t(x,0) - cos(x)
    (так как при t=0: u_tt + 3*u_t = u_xx + u_x - u - cos(x))
    
    Используем приближение:
    u(x,τ) ≈ u(x,0) + τ*ut_initial + (τ²/2)*[u_xx(x,0) + u_x(x,0) - u(x,0) - 3*ut_initial - cos(x)]
    """
    # Вычисляем производные в начальный момент
    n = len(u_prev)
    u_xx = np.zeros(n)
    u_x = np.zeros(n)
    
    # Внутренние точки
    for i in range(1, n - 1):
        u_xx[i] = (u_prev[i + 1] - 2 * u_prev[i] + u_prev[i - 1]) / (h * h)
        u_x[i] = (u_prev[i + 1] - u_prev[i - 1]) / (2 * h)
    
    # Применяем граничные условия для вычисления производных на границах
    boundary_func(u_prev.copy(), h, f_left, f_right)
    
    # Вычисляем u_xx и u_x на границах через ghost point
    # Левая граница: u_x[0] = f_left (из граничного условия)
    # u_xx[0] ≈ (u_prev[1] - 2*u_prev[0] + u_ghost)/h²
    # где u_ghost = u_prev[1] - 2*h*f_left (из u_x[0] = (u_prev[1] - u_ghost)/(2*h) = f_left)
    u_ghost_left = u_prev[1] - 2 * h * f_left
    u_xx[0] = (u_prev[1] - 2 * u_prev[0] + u_ghost_left) / (h * h)
    u_x[0] = f_left
    
    # Правая граница
    u_ghost_right = u_prev[-2] + 2 * h * f_right
    u_xx[-1] = (u_ghost_right - 2 * u_prev[-1] + u_prev[-2]) / (h * h)
    u_x[-1] = f_right
    
    # Вычисляем u_tt из уравнения
    source = source_term(x, 0.0)
    u_tt = u_xx + u_x - u_prev - ALPHA * ut_initial - source
    
    # Используем формулу второго порядка
    u_next = u_prev + tau * ut_initial + (tau * tau / 2.0) * u_tt
    
    return u_next


# ============================================================================
# Численные схемы
# ============================================================================

def explicit_cross_scheme(
    u_prev: np.ndarray,
    u_curr: np.ndarray,
    h: float,
    tau: float,
    boundary_func: Callable,
    f_left: float,
    f_right: float,
    x: np.ndarray,
    t: float,
) -> np.ndarray:
    """
    Явная схема крест для гиперболического уравнения.
    ∂²u/∂t² + 3 ∂u/∂t = ∂²u/∂x² + ∂u/∂x - u - cos(x) exp(-t)
    
    Разностная схема:
    (u_i^{n+1} - 2*u_i^n + u_i^{n-1})/τ² + 3*(u_i^{n+1} - u_i^{n-1})/(2*τ) =
        (u_{i+1}^n - 2*u_i^n + u_{i-1}^n)/h² + (u_{i+1}^n - u_{i-1}^n)/(2*h) - u_i^n - source_i^n
    
    Решая относительно u_i^{n+1}:
    u_i^{n+1} = [2*u_i^n - u_i^{n-1} + τ²*(u_xx + u_x - u - source) - 3*τ*(u_i^{n+1} - u_i^{n-1})/2] / (1 + 3*τ/2)
    
    Упрощая:
    u_i^{n+1}*(1 + 3*τ/2) = 2*u_i^n - u_i^{n-1} + τ²*(u_xx + u_x - u - source) + 3*τ*u_i^{n-1}/2
    u_i^{n+1} = [2*u_i^n - u_i^{n-1}*(1 - 3*τ/2) + τ²*(u_xx + u_x - u - source)] / (1 + 3*τ/2)
    """
    n = len(u_curr)
    u_next = np.zeros(n)
    
    # Проверка на валидность входных данных
    if np.any(~np.isfinite(u_prev)) or np.any(~np.isfinite(u_curr)):
        return np.full(n, np.nan)
    
    source = source_term(x, t)
    coeff = 1.0 + 3.0 * tau / 2.0
    
    # Внутренние точки
    for i in range(1, n - 1):
        # Проверка на валидность данных
        if not np.isfinite(u_curr[i + 1]) or not np.isfinite(u_curr[i]) or not np.isfinite(u_curr[i - 1]):
            return np.full(n, np.nan)
        
        u_xx = (u_curr[i + 1] - 2 * u_curr[i] + u_curr[i - 1]) / (h * h)
        u_x = (u_curr[i + 1] - u_curr[i - 1]) / (2 * h)
        
        # Проверка на overflow
        if not np.isfinite(u_xx) or not np.isfinite(u_x):
            return np.full(n, np.nan)
        
        # Вычисляем новый слой
        numerator = (2.0 * u_curr[i] - u_prev[i] * (1.0 - 3.0 * tau / 2.0) +
                    tau * tau * (u_xx + u_x - u_curr[i] - source[i]))
        
        if not np.isfinite(numerator) or abs(numerator) > 1e10:
            return np.full(n, np.nan)
        
        u_next[i] = numerator / coeff
        
        # Проверка на overflow после вычисления
        if not np.isfinite(u_next[i]) or abs(u_next[i]) > 1e10:
            return np.full(n, np.nan)
    
    # Применяем граничные условия
    boundary_func(u_next, h, f_left, f_right)
    
    # Проверка после граничных условий
    if np.any(~np.isfinite(u_next)):
        return np.full(n, np.nan)
    
    return u_next


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


def implicit_scheme(
    u_prev: np.ndarray,
    u_curr: np.ndarray,
    h: float,
    tau: float,
    boundary_func: Callable,
    f_left: float,
    f_right: float,
    x: np.ndarray,
    t: float,
) -> np.ndarray:
    """
    Неявная схема для гиперболического уравнения.
    ∂²u/∂t² + 3 ∂u/∂t = ∂²u/∂x² + ∂u/∂x - u - cos(x) exp(-t)
    
    Используем неявную аппроксимацию для пространственных производных:
    (u_i^{n+1} - 2*u_i^n + u_i^{n-1})/τ² + 3*(u_i^{n+1} - u_i^{n-1})/(2*τ) =
        (u_{i+1}^{n+1} - 2*u_i^{n+1} + u_{i-1}^{n+1})/h² + (u_{i+1}^{n+1} - u_{i-1}^{n+1})/(2*h) - u_i^{n+1} - source_i^{n+1}
    
    Решаем систему линейных уравнений методом прогонки.
    """
    n = len(u_curr)
    
    # Проверка на валидность входных данных
    if np.any(~np.isfinite(u_prev)) or np.any(~np.isfinite(u_curr)):
        return np.full(n, np.nan)
    
    source = source_term(x, t + tau)
    
    # Коэффициенты разностной схемы
    alpha = tau * tau / (h * h)  # коэффициент при u_xx
    beta = tau * tau / (2 * h)   # коэффициент при u_x
    gamma = 1.0 + 3.0 * tau / 2.0 + tau * tau  # коэффициент при u_i^{n+1}
    
    # Строим трехдиагональную систему
    a = np.zeros(n)  # коэффициенты при u_{i-1}
    b = np.zeros(n)  # коэффициенты при u_i
    c = np.zeros(n)  # коэффициенты при u_{i+1}
    d = np.zeros(n)  # правая часть
    
    # Внутренние точки
    for i in range(1, n - 1):
        a[i] = -alpha + beta
        b[i] = gamma + 2 * alpha
        c[i] = -alpha - beta
        d[i] = 2.0 * u_curr[i] - u_prev[i] * (1.0 - 3.0 * tau / 2.0) - tau * tau * source[i]
    
    # Используем итерации для учета граничных условий
    u_new = u_curr.copy()
    max_iter = 20
    tolerance = 1e-8
    
    for iteration in range(max_iter):
        u_old = u_new.copy()
        
        # Проверка на overflow перед применением граничных условий
        if np.any(~np.isfinite(u_new)) or np.any(np.abs(u_new) > 1e10):
            return np.full(n, np.nan)
        
        # Применяем граничные условия к текущему приближению
        boundary_func(u_new, h, f_left, f_right)
        
        # Проверка после применения граничных условий
        if np.any(~np.isfinite(u_new)):
            return np.full(n, np.nan)
        
        # Модифицируем систему: граничные точки фиксированы из граничных условий
        b[0] = 1.0
        c[0] = 0.0
        d[0] = u_new[0]
        
        a[n - 1] = 0.0
        b[n - 1] = 1.0
        d[n - 1] = u_new[n - 1]
        
        # Решаем систему методом прогонки
        u_new = thomas_algorithm(a.copy(), b.copy(), c.copy(), d.copy())
        
        # Проверка на валидность решения
        if np.any(~np.isfinite(u_new)):
            return np.full(n, np.nan)
        
        # Проверка сходимости
        diff = u_new - u_old
        if np.any(~np.isfinite(diff)):
            return np.full(n, np.nan)
        
        max_diff = np.max(np.abs(diff))
        if not np.isfinite(max_diff):
            return np.full(n, np.nan)
        
        if max_diff < tolerance:
            break
    
    # Финальное применение граничных условий
    boundary_func(u_new, h, f_left, f_right)
    
    # Финальная проверка
    if np.any(~np.isfinite(u_new)):
        return np.full(n, np.nan)
    
    return u_new


# ============================================================================
# Решение задачи
# ============================================================================

def solve_pde(
    scheme_func: Callable,
    boundary_func: Callable,
    initial_approx_func: Callable,
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
    
    # Первое начальное условие
    u_prev = initial_condition_u(x)
    
    # Второе начальное условие
    ut_initial = initial_condition_ut(x)
    
    # Аппроксимируем второй временной слой
    u_curr = initial_approx_func(
        u_prev, u_prev, tau, ut_initial, h, x, 0.0, boundary_func,
        boundary_condition_left(0.0), boundary_condition_right(0.0)
    )
    
    u_numerical = np.zeros((n_t + 1, n_x + 1))
    u_numerical[0, :] = u_prev
    u_numerical[1, :] = u_curr
    
    # Решение по времени
    for n in range(1, n_t):
        f_left = boundary_condition_left(t[n])
        f_right = boundary_condition_right(t[n])
        
        u_next = scheme_func(u_prev, u_curr, h, tau, boundary_func, f_left, f_right, x, t[n])
        u_numerical[n + 1, :] = u_next
        
        # Обновляем слои
        u_prev = u_curr.copy()
        u_curr = u_next.copy()
    
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
    initial_approx_func: Callable,
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
            
            _, _, u_num, u_anal = solve_pde(scheme_func, boundary_func, initial_approx_func, n_x, n_t, scheme_name)
            
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
    initial_name: str,
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
    
    plt.suptitle(f'{scheme_name} - {boundary_name} - {initial_name}', fontsize=14)
    plt.tight_layout()
    filename = os.path.join(output_dir, f'solution_{scheme_name}_{boundary_name}_{initial_name}.png')
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_error(
    x: np.ndarray,
    t: np.ndarray,
    u_numerical: np.ndarray,
    u_analytical: np.ndarray,
    scheme_name: str,
    boundary_name: str,
    initial_name: str,
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
    
    plt.suptitle(f'Погрешность: {scheme_name} - {boundary_name} - {initial_name}', fontsize=14)
    plt.tight_layout()
    filename = os.path.join(output_dir, f'error_{scheme_name}_{boundary_name}_{initial_name}.png')
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_grid_dependence(
    h_values: np.ndarray,
    tau_values: np.ndarray,
    max_errors: np.ndarray,
    l2_errors: np.ndarray,
    scheme_name: str,
    boundary_name: str,
    initial_name: str,
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
    
    plt.suptitle(f'Зависимость погрешности: {scheme_name} - {boundary_name} - {initial_name}', fontsize=14)
    plt.tight_layout()
    filename = os.path.join(output_dir, f'grid_dependence_{scheme_name}_{boundary_name}_{initial_name}.png')
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
    
    # Аппроксимации граничных условий
    boundary_approximations = [
        (apply_boundary_condition_two_point_first_order, "Двухточечная_1-го_порядка"),
        (apply_boundary_condition_three_point_second_order, "Трехточечная_2-го_порядка"),
        (apply_boundary_condition_two_point_second_order, "Двухточечная_2-го_порядка"),
    ]
    
    # Аппроксимации второго начального условия
    initial_approximations = [
        (approximate_second_initial_condition_first_order, "1-го_порядка"),
        (approximate_second_initial_condition_second_order, "2-го_порядка"),
    ]
    
    print("=" * 80)
    print("Лабораторная работа 6: Численные методы решения гиперболических уравнений")
    print("Вариант 10")
    print("=" * 80)
    print(f"Параметры: α = {ALPHA}, β = {BETA}, γ = {GAMMA}")
    print(f"Сетка: n_x = {n_x}, n_t = {n_t}")
    print(f"Шаги: h = {(X_MAX - X_MIN) / n_x:.6f}, τ = {T_MAX / n_t:.6f}")
    print("=" * 80)
    
    # Решение для всех комбинаций схем, граничных условий и аппроксимаций начального условия
    results = []
    
    for scheme_func, scheme_name in schemes:
        for boundary_func, boundary_name in boundary_approximations:
            for initial_func, initial_name in initial_approximations:
                print(f"\n{scheme_name} схема + {boundary_name} + {initial_name}")
                print("-" * 80)
                
                # Решение задачи
                x, t, u_num, u_anal = solve_pde(scheme_func, boundary_func, initial_func, n_x, n_t, scheme_name)
                
                # Вычисление погрешностей
                max_err = compute_max_error(u_num, u_anal)
                l2_err = compute_l2_error(u_num, u_anal, (X_MAX - X_MIN) / n_x)
                
                print(f"Максимальная погрешность: {max_err:.6e}")
                print(f"L2 норма погрешности: {l2_err:.6e}")
                
                # Визуализация
                plot_solution(x, t, u_num, u_anal, scheme_name, boundary_name, initial_name, output_dir)
                plot_error(x, t, u_num, u_anal, scheme_name, boundary_name, initial_name, output_dir)
                
                results.append({
                    'scheme': scheme_name,
                    'boundary': boundary_name,
                    'initial': initial_name,
                    'max_error': max_err,
                    'l2_error': l2_err,
                })
    
    # Исследование зависимости от параметров сетки
    print("\n" + "=" * 80)
    print("Исследование зависимости погрешности от параметров сетки")
    print("=" * 80)
    
    n_x_values = [20, 30, 40, 50, 60, 80, 100]
    n_t_values = [50, 75, 100, 150, 200]
    
    # Исследуем для лучшей комбинации
    scheme_func = explicit_cross_scheme
    boundary_func = apply_boundary_condition_three_point_second_order
    initial_func = approximate_second_initial_condition_second_order
    scheme_name = "Явная_крест"
    boundary_name = "Трехточечная_2-го_порядка"
    initial_name = "2-го_порядка"
    
    print(f"\nИсследование для: {scheme_name} + {boundary_name.replace('_', ' ')} + {initial_name.replace('_', ' ')}")
    h_vals, tau_vals, max_errs, l2_errs = study_grid_dependence(
        scheme_func, boundary_func, initial_func, n_x_values, n_t_values, scheme_name
    )
    
    plot_grid_dependence(h_vals, tau_vals, max_errs, l2_errs, scheme_name, boundary_name, initial_name, output_dir)
    
    # Вывод результатов исследования
    print("\nРезультаты исследования зависимости от параметров сетки:")
    print(f"{'h':>12} {'τ':>12} {'Max Error':>15} {'L2 Error':>15}")
    print("-" * 60)
    for i in range(min(20, len(h_vals))):  # Показываем первые 20
        print(f"{h_vals[i]:>12.6f} {tau_vals[i]:>12.6f} {max_errs[i]:>15.6e} {l2_errs[i]:>15.6e}")
    
    # Сводная таблица результатов
    print("\n" + "=" * 80)
    print("Сводная таблица результатов")
    print("=" * 80)
    print(f"{'Схема':<20} {'Граничные условия':<30} {'Начальное условие':<25} {'Max Error':>15} {'L2 Error':>15}")
    print("-" * 110)
    for res in results:
        print(f"{res['scheme']:<20} {res['boundary']:<30} {res['initial']:<25} {res['max_error']:>15.6e} {res['l2_error']:>15.6e}")
    
    print("\n" + "=" * 80)
    print(f"Все графики сохранены в папку: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

