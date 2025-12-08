"""
Лабораторная работа 3: Численные методы решения эллиптических уравнений
Вариант 9

Уравнение: ∂²u/∂x² + ∂²u/∂y² = -2(∂u/∂y) - 3u
Или: ∂²u/∂x² + ∂²u/∂y² + 2(∂u/∂y) + 3u = 0

Граничные условия (Дирихле):
- u(0, y) = exp(-y) cos y
- u(π/2, y) = 0
- u(x, 0) = cos x
- u(x, π/2) = 0

Аналитическое решение: U(x, y) = exp(-y) cos x cos y

Область: [0, π/2] x [0, π/2]
"""

import numpy as np
from typing import Callable, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


# ============================================================================
# Параметры задачи
# ============================================================================

# Область решения
X_MIN = 0.0
X_MAX = np.pi / 2.0
Y_MIN = 0.0
Y_MAX = np.pi / 2.0


# ============================================================================
# Аналитическое решение и функции граничных условий
# ============================================================================

def analytical_solution(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Аналитическое решение U(x, y) = exp(-y) cos x cos y"""
    X, Y = np.meshgrid(x, y, indexing='ij')
    return np.exp(-Y) * np.cos(X) * np.cos(Y)


def boundary_condition_left(y: np.ndarray) -> np.ndarray:
    """Левое граничное условие: u(0, y) = exp(-y) cos y"""
    return np.exp(-y) * np.cos(y)


def boundary_condition_right(y: np.ndarray) -> np.ndarray:
    """Правое граничное условие: u(π/2, y) = 0"""
    return np.zeros_like(y)


def boundary_condition_bottom(x: np.ndarray) -> np.ndarray:
    """Нижнее граничное условие: u(x, 0) = cos x"""
    return np.cos(x)


def boundary_condition_top(x: np.ndarray) -> np.ndarray:
    """Верхнее граничное условие: u(x, π/2) = 0"""
    return np.zeros_like(x)


# ============================================================================
# Центрально-разностная схема
# ============================================================================

def build_system_matrix(n_x: int, n_y: int, h_x: float, h_y: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Строит систему линейных уравнений для эллиптического уравнения.
    
    Уравнение: ∂²u/∂x² + ∂²u/∂y² + 2(∂u/∂y) + 3u = 0
    
    Центрально-разностная аппроксимация:
    - u_xx ≈ (u_{i+1,j} - 2*u_{i,j} + u_{i-1,j})/h_x²
    - u_yy ≈ (u_{i,j+1} - 2*u_{i,j} + u_{i,j-1})/h_y²
    - u_y ≈ (u_{i,j+1} - u_{i,j-1})/(2*h_y)
    
    Для внутренней точки (i, j):
    (u_{i+1,j} - 2*u_{i,j} + u_{i-1,j})/h_x² + 
    (u_{i,j+1} - 2*u_{i,j} + u_{i,j-1})/h_y² + 
    2*(u_{i,j+1} - u_{i,j-1})/(2*h_y) + 3*u_{i,j} = 0
    
    Переписываем:
    u_{i-1,j}/h_x² + u_{i+1,j}/h_x² + 
    u_{i,j-1}*(1/h_y² - 1/h_y) + u_{i,j+1}*(1/h_y² + 1/h_y) + 
    u_{i,j}*(-2/h_x² - 2/h_y² + 3) = 0
    
    Возвращает:
        - коэффициенты для итерационных методов (в виде функций)
        - правую часть (нулевая для однородного уравнения)
    """
    # Внутренние точки (без границ)
    # Для каждой внутренней точки (i, j) коэффициенты:
    # a_left = 1/h_x² (при u_{i-1,j})
    # a_right = 1/h_x² (при u_{i+1,j})
    # a_bottom = 1/h_y² - 1/h_y (при u_{i,j-1})
    # a_top = 1/h_y² + 1/h_y (при u_{i,j+1})
    # a_center = -2/h_x² - 2/h_y² + 3 (при u_{i,j})
    
    a_left = 1.0 / (h_x * h_x)
    a_right = 1.0 / (h_x * h_x)
    a_bottom = 1.0 / (h_y * h_y) - 1.0 / h_y
    a_top = 1.0 / (h_y * h_y) + 1.0 / h_y
    a_center = -2.0 / (h_x * h_x) - 2.0 / (h_y * h_y) + 3.0
    
    return {
        'a_left': a_left,
        'a_right': a_right,
        'a_bottom': a_bottom,
        'a_top': a_top,
        'a_center': a_center,
    }


def apply_boundary_conditions(u: np.ndarray, x: np.ndarray, y: np.ndarray) -> None:
    """Применяет граничные условия Дирихле к сетке u"""
    n_x, n_y = u.shape
    
    # Левая граница: x = 0
    u[0, :] = boundary_condition_left(y)
    
    # Правая граница: x = π/2
    u[-1, :] = boundary_condition_right(y)
    
    # Нижняя граница: y = 0
    u[:, 0] = boundary_condition_bottom(x)
    
    # Верхняя граница: y = π/2
    u[:, -1] = boundary_condition_top(x)


# ============================================================================
# Итерационные методы
# ============================================================================

def simple_iteration(
    u: np.ndarray,
    coeffs: dict,
    h_x: float,
    h_y: float,
    x: np.ndarray,
    y: np.ndarray,
    max_iter: int = 10000,
    tolerance: float = 1e-6,
) -> Tuple[np.ndarray, int]:
    """
    Метод простых итераций (метод Либмана).
    
    Итерационная формула:
    u_{i,j}^{k+1} = -[a_left*u_{i-1,j}^k + a_right*u_{i+1,j}^k + 
                     a_bottom*u_{i,j-1}^k + a_top*u_{i,j+1}^k] / a_center
    """
    u_new = u.copy()
    n_x, n_y = u.shape
    
    a_left = coeffs['a_left']
    a_right = coeffs['a_right']
    a_bottom = coeffs['a_bottom']
    a_top = coeffs['a_top']
    a_center = coeffs['a_center']
    
    if abs(a_center) < 1e-10:
        raise ValueError(f"a_center слишком мал: {a_center}")
    
    for iteration in range(max_iter):
        u_old = u_new.copy()
        
        # Обновляем внутренние точки
        for i in range(1, n_x - 1):
            for j in range(1, n_y - 1):
                u_new[i, j] = -(
                    a_left * u_old[i - 1, j] +
                    a_right * u_old[i + 1, j] +
                    a_bottom * u_old[i, j - 1] +
                    a_top * u_old[i, j + 1]
                ) / a_center
        
        # Применяем граничные условия
        apply_boundary_conditions(u_new, x, y)
        
        # Проверка на NaN/Inf
        if np.any(~np.isfinite(u_new)):
            print(f"Предупреждение: NaN/Inf на итерации {iteration + 1}")
            break
        
        # Проверка сходимости
        diff = np.abs(u_new - u_old)
        max_diff = np.max(diff)
        
        if (iteration + 1) % 1000 == 0:
            print(f"  Итерация {iteration + 1}: max_diff = {max_diff:.6e}")
        
        if max_diff < tolerance:
            return u_new, iteration + 1
    
    return u_new, max_iter


def gauss_seidel(
    u: np.ndarray,
    coeffs: dict,
    h_x: float,
    h_y: float,
    x: np.ndarray,
    y: np.ndarray,
    max_iter: int = 10000,
    tolerance: float = 1e-6,
) -> Tuple[np.ndarray, int]:
    """
    Метод Зейделя.
    
    Использует уже обновленные значения на текущей итерации:
    u_{i,j}^{k+1} = -[a_left*u_{i-1,j}^{k+1} + a_right*u_{i+1,j}^k + 
                     a_bottom*u_{i,j-1}^{k+1} + a_top*u_{i,j+1}^k] / a_center
    """
    u_new = u.copy()
    n_x, n_y = u.shape
    
    a_left = coeffs['a_left']
    a_right = coeffs['a_right']
    a_bottom = coeffs['a_bottom']
    a_top = coeffs['a_top']
    a_center = coeffs['a_center']
    
    if abs(a_center) < 1e-10:
        raise ValueError(f"a_center слишком мал: {a_center}")
    
    for iteration in range(max_iter):
        u_old = u_new.copy()
        
        # Обновляем внутренние точки (используем уже обновленные значения)
        for i in range(1, n_x - 1):
            for j in range(1, n_y - 1):
                u_new[i, j] = -(
                    a_left * u_new[i - 1, j] +      # уже обновлено
                    a_right * u_old[i + 1, j] +     # еще старое
                    a_bottom * u_new[i, j - 1] +     # уже обновлено
                    a_top * u_old[i, j + 1]          # еще старое
                ) / a_center
        
        # Применяем граничные условия
        apply_boundary_conditions(u_new, x, y)
        
        # Проверка на NaN/Inf
        if np.any(~np.isfinite(u_new)):
            print(f"Предупреждение: NaN/Inf на итерации {iteration + 1}")
            break
        
        # Проверка сходимости
        diff = np.abs(u_new - u_old)
        max_diff = np.max(diff)
        
        if (iteration + 1) % 1000 == 0:
            print(f"  Итерация {iteration + 1}: max_diff = {max_diff:.6e}")
        
        if max_diff < tolerance:
            return u_new, iteration + 1
    
    return u_new, max_iter


def sor(
    u: np.ndarray,
    coeffs: dict,
    h_x: float,
    h_y: float,
    x: np.ndarray,
    y: np.ndarray,
    omega: float = 1.5,
    max_iter: int = 10000,
    tolerance: float = 1e-6,
) -> Tuple[np.ndarray, int]:
    """
    Метод простых итераций с верхней релаксацией (SOR - Successive Over-Relaxation).
    
    u_{i,j}^{k+1} = (1 - ω)*u_{i,j}^k + ω*u_{i,j}^{GS}
    где u_{i,j}^{GS} - значение из метода Зейделя
    """
    u_new = u.copy()
    n_x, n_y = u.shape
    
    a_left = coeffs['a_left']
    a_right = coeffs['a_right']
    a_bottom = coeffs['a_bottom']
    a_top = coeffs['a_top']
    a_center = coeffs['a_center']
    
    if abs(a_center) < 1e-10:
        raise ValueError(f"a_center слишком мал: {a_center}")
    
    for iteration in range(max_iter):
        u_old = u_new.copy()
        
        # Обновляем внутренние точки
        for i in range(1, n_x - 1):
            for j in range(1, n_y - 1):
                # Значение по методу Зейделя
                u_gs = -(
                    a_left * u_new[i - 1, j] +      # уже обновлено
                    a_right * u_old[i + 1, j] +     # еще старое
                    a_bottom * u_new[i, j - 1] +     # уже обновлено
                    a_top * u_old[i, j + 1]          # еще старое
                ) / a_center
                
                # Релаксация
                u_new[i, j] = (1 - omega) * u_old[i, j] + omega * u_gs
        
        # Применяем граничные условия
        apply_boundary_conditions(u_new, x, y)
        
        # Проверка на NaN/Inf
        if np.any(~np.isfinite(u_new)):
            print(f"Предупреждение: NaN/Inf на итерации {iteration + 1}")
            break
        
        # Проверка сходимости
        diff = np.abs(u_new - u_old)
        max_diff = np.max(diff)
        
        if (iteration + 1) % 1000 == 0:
            print(f"  Итерация {iteration + 1}: max_diff = {max_diff:.6e}")
        
        if max_diff < tolerance:
            return u_new, iteration + 1
    
    return u_new, max_iter


# ============================================================================
# Решение задачи
# ============================================================================

def solve_elliptic(
    method_func: Callable,
    n_x: int,
    n_y: int,
    method_name: str,
    omega: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Решает эллиптическую задачу для заданного итерационного метода.
    
    Returns:
        x: сетка по x
        y: сетка по y
        u_numerical: численное решение
        u_analytical: аналитическое решение
        iterations: количество итераций
    """
    h_x = (X_MAX - X_MIN) / n_x
    h_y = (Y_MAX - Y_MIN) / n_y
    
    x = np.linspace(X_MIN, X_MAX, n_x + 1)
    y = np.linspace(Y_MIN, Y_MAX, n_y + 1)
    
    # Начальное приближение (нули для внутренних точек)
    u = np.zeros((n_x + 1, n_y + 1))
    apply_boundary_conditions(u, x, y)
    
    # Строим коэффициенты системы
    coeffs = build_system_matrix(n_x, n_y, h_x, h_y)
    
    # Решаем итерационным методом
    if method_func == sor:
        u_numerical, iterations = method_func(u, coeffs, h_x, h_y, x, y, omega=omega)
    else:
        u_numerical, iterations = method_func(u, coeffs, h_x, h_y, x, y)
    
    # Аналитическое решение
    u_analytical = analytical_solution(x, y)
    
    return x, y, u_numerical, u_analytical, iterations


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


def compute_l2_error(u_numerical: np.ndarray, u_analytical: np.ndarray, h_x: float, h_y: float) -> float:
    """Вычисляет L2 норму погрешности"""
    diff = u_numerical - u_analytical
    valid_diff = diff[np.isfinite(diff)]
    if len(valid_diff) == 0:
        return np.nan
    return np.sqrt(h_x * h_y * np.sum(valid_diff ** 2))


# ============================================================================
# Исследование зависимости погрешности от параметров сетки
# ============================================================================

def study_grid_dependence(
    method_func: Callable,
    n_x_values: list,
    n_y_values: list,
    method_name: str,
    omega: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Исследует зависимость погрешности от параметров сетки h_x и h_y.
    
    Returns:
        h_x_values: значения шага по x
        h_y_values: значения шага по y
        max_errors: максимальные погрешности для каждой комбинации
        l2_errors: L2 нормы погрешностей
    """
    h_x_values = []
    h_y_values = []
    max_errors = []
    l2_errors = []
    
    for n_x in n_x_values:
        for n_y in n_y_values:
            h_x = (X_MAX - X_MIN) / n_x
            h_y = (Y_MAX - Y_MIN) / n_y
            
            _, _, u_num, u_anal, _ = solve_elliptic(method_func, n_x, n_y, method_name, omega)
            
            max_err = compute_max_error(u_num, u_anal)
            l2_err = compute_l2_error(u_num, u_anal, h_x, h_y)
            
            h_x_values.append(h_x)
            h_y_values.append(h_y)
            max_errors.append(max_err)
            l2_errors.append(l2_err)
    
    return np.array(h_x_values), np.array(h_y_values), np.array(max_errors), np.array(l2_errors)


# ============================================================================
# Визуализация
# ============================================================================

def plot_solution(
    x: np.ndarray,
    y: np.ndarray,
    u_numerical: np.ndarray,
    u_analytical: np.ndarray,
    method_name: str,
    output_dir: str,
):
    """Визуализация решения"""
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Общий диапазон значений для численного и аналитического решений
    u_min = min(np.min(u_numerical), np.min(u_analytical))
    u_max = max(np.max(u_numerical), np.max(u_analytical))
    
    fig = plt.figure(figsize=(16, 12))
    
    # 3D графики
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, u_numerical, cmap='viridis', alpha=0.8, vmin=u_min, vmax=u_max)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u(x,y)')
    ax1.set_title('Численное решение')
    ax1.set_zlim(u_min, u_max)
    plt.colorbar(surf1, ax=ax1, shrink=0.5)
    
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(X, Y, u_analytical, cmap='viridis', alpha=0.8, vmin=u_min, vmax=u_max)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('u(x,y)')
    ax2.set_title('Аналитическое решение')
    ax2.set_zlim(u_min, u_max)
    plt.colorbar(surf2, ax=ax2, shrink=0.5)
    
    # 2D контурные графики
    ax3 = fig.add_subplot(2, 2, 3)
    contour1 = ax3.contourf(X, Y, u_numerical, levels=20, cmap='viridis', vmin=u_min, vmax=u_max)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Численное решение (контуры)')
    ax3.set_aspect('equal', adjustable='box')
    plt.colorbar(contour1, ax=ax3)
    
    ax4 = fig.add_subplot(2, 2, 4)
    contour2 = ax4.contourf(X, Y, u_analytical, levels=20, cmap='viridis', vmin=u_min, vmax=u_max)
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_title('Аналитическое решение (контуры)')
    ax4.set_aspect('equal', adjustable='box')
    plt.colorbar(contour2, ax=ax4)
    
    plt.suptitle(f'Решение: {method_name}', fontsize=14)
    plt.tight_layout()
    filename = os.path.join(output_dir, f'solution_{method_name}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_error(
    x: np.ndarray,
    y: np.ndarray,
    u_numerical: np.ndarray,
    u_analytical: np.ndarray,
    method_name: str,
    output_dir: str,
):
    """Визуализация погрешности"""
    error = compute_error(u_numerical, u_analytical)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    fig = plt.figure(figsize=(14, 5))
    
    # 3D график погрешности
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax1.plot_surface(X, Y, error, cmap='hot', alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('Погрешность')
    ax1.set_title('Погрешность (3D)')
    plt.colorbar(surf, ax=ax1, shrink=0.5)
    
    # 2D контурный график погрешности
    ax2 = fig.add_subplot(1, 2, 2)
    contour = ax2.contourf(X, Y, error, levels=20, cmap='hot')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Погрешность (контуры)')
    plt.colorbar(contour, ax=ax2)
    
    plt.suptitle(f'Погрешность: {method_name}', fontsize=14)
    plt.tight_layout()
    filename = os.path.join(output_dir, f'error_{method_name}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_grid_dependence(
    h_x_values: np.ndarray,
    h_y_values: np.ndarray,
    max_errors: np.ndarray,
    l2_errors: np.ndarray,
    method_name: str,
    output_dir: str,
):
    """Визуализация зависимости погрешности от параметров сетки"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Фильтруем nan и inf значения
    valid_mask = np.isfinite(max_errors) & (max_errors > 0)
    if not np.any(valid_mask):
        plt.close()
        return
    
    h_x_vals_valid = h_x_values[valid_mask]
    h_y_vals_valid = h_y_values[valid_mask]
    err_vals_valid = max_errors[valid_mask]
    
    # Зависимость от h_x при фиксированном h_y
    unique_h_y = np.unique(h_y_vals_valid)
    for h_y in unique_h_y[:3]:  # Берем первые 3 значения
        mask = np.abs(h_y_vals_valid - h_y) < 1e-10
        if np.any(mask):
            h_x_vals = h_x_vals_valid[mask]
            err_vals = err_vals_valid[mask]
            sorted_indices = np.argsort(h_x_vals)
            axes[0].loglog(h_x_vals[sorted_indices], err_vals[sorted_indices], 'o-', label=f'h_y = {h_y:.4f}')
    
    axes[0].set_xlabel('h_x')
    axes[0].set_ylabel('Максимальная погрешность')
    axes[0].set_title('Зависимость от h_x')
    axes[0].legend()
    axes[0].grid(True)
    
    # Зависимость от h_y при фиксированном h_x
    unique_h_x = np.unique(h_x_vals_valid)
    for h_x in unique_h_x[:3]:  # Берем первые 3 значения
        mask = np.abs(h_x_vals_valid - h_x) < 1e-10
        if np.any(mask):
            h_y_vals = h_y_vals_valid[mask]
            err_vals = err_vals_valid[mask]
            sorted_indices = np.argsort(h_y_vals)
            axes[1].loglog(h_y_vals[sorted_indices], err_vals[sorted_indices], 's-', label=f'h_x = {h_x:.4f}')
    
    axes[1].set_xlabel('h_y')
    axes[1].set_ylabel('Максимальная погрешность')
    axes[1].set_title('Зависимость от h_y')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.suptitle(f'Зависимость погрешности: {method_name}', fontsize=14)
    plt.tight_layout()
    filename = os.path.join(output_dir, f'grid_dependence_{method_name}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    """Основная функция для запуска всех экспериментов"""
    
    # Создаем папку для результатов
    output_dir = "lab7/results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Параметры сетки
    n_x = 50
    n_y = 50
    
    # Итерационные методы
    methods = [
        (simple_iteration, "Простые_итерации"),
        (gauss_seidel, "Зейдель"),
        (sor, "SOR"),
    ]
    
    print("=" * 80)
    print("Лабораторная работа 3: Численные методы решения эллиптических уравнений")
    print("Вариант 9")
    print("=" * 80)
    print(f"Уравнение: ∂²u/∂x² + ∂²u/∂y² + 2(∂u/∂y) + 3u = 0")
    print(f"Сетка: n_x = {n_x}, n_y = {n_y}")
    print(f"Шаги: h_x = {(X_MAX - X_MIN) / n_x:.6f}, h_y = {(Y_MAX - Y_MIN) / n_y:.6f}")
    print("=" * 80)
    
    # Решение для всех методов
    results = []
    
    for method_func, method_name in methods:
        print(f"\n{method_name}")
        print("-" * 80)
        
        # Решение задачи
        omega = 1.5 if method_func == sor else 1.0
        x, y, u_num, u_anal, iterations = solve_elliptic(method_func, n_x, n_y, method_name, omega)
        
        # Вычисление погрешностей
        max_err = compute_max_error(u_num, u_anal)
        l2_err = compute_l2_error(u_num, u_anal, (X_MAX - X_MIN) / n_x, (Y_MAX - Y_MIN) / n_y)
        
        print(f"Количество итераций: {iterations}")
        print(f"Критерий остановки: max_diff < 1.0e-6")
        print(f"Максимальная погрешность: {max_err:.6e}")
        print(f"L2 норма погрешности: {l2_err:.6e}")
        
        # Визуализация
        plot_solution(x, y, u_num, u_anal, method_name, output_dir)
        plot_error(x, y, u_num, u_anal, method_name, output_dir)
        
        results.append({
            'method': method_name,
            'iterations': iterations,
            'max_error': max_err,
            'l2_error': l2_err,
        })
    
    # Исследование зависимости от параметров сетки
    print("\n" + "=" * 80)
    print("Исследование зависимости погрешности от параметров сетки")
    print("=" * 80)
    
    n_x_values = [15, 20, 25, 30, 40, 50]
    n_y_values = [15, 20, 25, 30, 40, 50]
    
    # Исследуем для всех методов
    for method_func, method_name in methods:
        print(f"\nИсследование для: {method_name}")
        omega = 1.5 if method_func == sor else 1.0
        h_x_vals, h_y_vals, max_errs, l2_errs = study_grid_dependence(
            method_func, n_x_values, n_y_values, method_name, omega
        )
        
        plot_grid_dependence(h_x_vals, h_y_vals, max_errs, l2_errs, method_name, output_dir)
        
        # Вывод результатов исследования
        print("\nРезультаты исследования зависимости от параметров сетки:")
        print(f"{'h_x':>12} {'h_y':>12} {'Max Error':>15} {'L2 Error':>15}")
        print("-" * 60)
        for i in range(min(20, len(h_x_vals))):  # Показываем первые 20
            if np.isfinite(max_errs[i]) and max_errs[i] > 0:
                print(f"{h_x_vals[i]:>12.6f} {h_y_vals[i]:>12.6f} {max_errs[i]:>15.6e} {l2_errs[i]:>15.6e}")
        
        # Вычисление порядка сходимости по h_x
        print("\nПорядок сходимости по h_x (при фиксированном h_y):")
        unique_h_y = np.unique(h_y_vals)
        for h_y in unique_h_y[:2]:  # Берем первые 2 значения
            mask = np.abs(h_y_vals - h_y) < 1e-10
            if np.sum(mask) >= 3:  # Нужно минимум 3 точки для надежной оценки
                h_x_sorted = h_x_vals[mask]
                err_sorted = max_errs[mask]
                sorted_idx = np.argsort(h_x_sorted)
                h_x_sorted = h_x_sorted[sorted_idx]
                err_sorted = err_sorted[sorted_idx]
                
                # Фильтруем валидные значения
                valid = (err_sorted > 0) & (h_x_sorted > 0)
                h_x_sorted = h_x_sorted[valid]
                err_sorted = err_sorted[valid]
                
                # Вычисляем порядок сходимости (используем последние пары для более точной оценки)
                if len(h_x_sorted) >= 2:
                    orders = []
                    # Используем только последние пары (где шаг меньше)
                    start_idx = max(0, len(h_x_sorted) - 3)
                    for i in range(start_idx, len(h_x_sorted) - 1):
                        if err_sorted[i] > 0 and err_sorted[i+1] > 0 and h_x_sorted[i] > h_x_sorted[i+1]:
                            order = np.log(err_sorted[i] / err_sorted[i+1]) / np.log(h_x_sorted[i] / h_x_sorted[i+1])
                            if np.isfinite(order) and order > -5 and order < 5:  # Фильтруем выбросы
                                orders.append(order)
                    if orders:
                        avg_order = np.mean(orders)
                        std_order = np.std(orders)
                        print(f"  h_y = {h_y:.4f}: средний порядок = {avg_order:.3f} ± {std_order:.3f} (из {len(orders)} измерений)")
        
        # Вычисление порядка сходимости по h_y
        print("\nПорядок сходимости по h_y (при фиксированном h_x):")
        unique_h_x = np.unique(h_x_vals)
        for h_x in unique_h_x[:2]:  # Берем первые 2 значения
            mask = np.abs(h_x_vals - h_x) < 1e-10
            if np.sum(mask) >= 3:  # Нужно минимум 3 точки
                h_y_sorted = h_y_vals[mask]
                err_sorted = max_errs[mask]
                sorted_idx = np.argsort(h_y_sorted)
                h_y_sorted = h_y_sorted[sorted_idx]
                err_sorted = err_sorted[sorted_idx]
                
                # Фильтруем валидные значения
                valid = (err_sorted > 0) & (h_y_sorted > 0)
                h_y_sorted = h_y_sorted[valid]
                err_sorted = err_sorted[valid]
                
                # Вычисляем порядок сходимости
                if len(h_y_sorted) >= 2:
                    orders = []
                    # Используем только последние пары (где шаг меньше)
                    start_idx = max(0, len(h_y_sorted) - 3)
                    for i in range(start_idx, len(h_y_sorted) - 1):
                        if err_sorted[i] > 0 and err_sorted[i+1] > 0 and h_y_sorted[i] > h_y_sorted[i+1]:
                            order = np.log(err_sorted[i] / err_sorted[i+1]) / np.log(h_y_sorted[i] / h_y_sorted[i+1])
                            if np.isfinite(order) and order > -5 and order < 5:  # Фильтруем выбросы
                                orders.append(order)
                    if orders:
                        avg_order = np.mean(orders)
                        std_order = np.std(orders)
                        print(f"  h_x = {h_x:.4f}: средний порядок = {avg_order:.3f} ± {std_order:.3f} (из {len(orders)} измерений)")
    
    # Сводная таблица результатов
    print("\n" + "=" * 80)
    print("Сводная таблица результатов")
    print("=" * 80)
    print(f"{'Метод':<25} {'Итерации':>10} {'Max Error':>15} {'L2 Error':>15}")
    print("-" * 70)
    for res in results:
        print(f"{res['method']:<25} {res['iterations']:>10} {res['max_error']:>15.6e} {res['l2_error']:>15.6e}")
    
    print("\n" + "=" * 80)
    print(f"Все графики сохранены в папку: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
