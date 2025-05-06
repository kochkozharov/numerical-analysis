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
