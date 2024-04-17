import numpy as np
from scipy.interpolate import CubicHermiteSpline, CubicSpline
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

############################## lab 3  ####################################

# Экспериментальные данные
t_values = np.array([0, 1, 4, 5])
x_values = np.array([1, 7, 5, 1])
y_values = np.array([4, 4, 1, 8])
x_accel_values = np.array([0, None, None, 0])
y_accel_values = np.array([0, None, None, 0])

# Заменяем пропущенные значения ускорений
for i in range(len(x_accel_values)):
    if x_accel_values[i] is None:
        x_accel_values[i] = np.polyval(np.polyfit(t_values[i-1:i+2], x_values[i-1:i+2], 2), t_values[i])
    if y_accel_values[i] is None:
        y_accel_values[i] = np.polyval(np.polyfit(t_values[i-1:i+2], y_values[i-1:i+2], 2), t_values[i])

# Интерполяция данных
t_interp = np.linspace(0, 5, 100)  # Увеличиваем количество точек для плавного графика

# Многочлен Ньютона
coeffs_x_newton = np.polyfit(t_values, x_values, deg=len(t_values)-1)
coeffs_y_newton = np.polyfit(t_values, y_values, deg=len(t_values)-1)
x_interp_newton = np.polyval(coeffs_x_newton, t_interp)
y_interp_newton = np.polyval(coeffs_y_newton, t_interp)

# Многочлен Эрмита
x_interp_hermite = CubicHermiteSpline(t_values, x_values, x_accel_values)(t_interp)
y_interp_hermite = CubicHermiteSpline(t_values, y_values, y_accel_values)(t_interp)

# Кубические сплайны
cs_x = CubicSpline(t_values, x_values)
cs_y = CubicSpline(t_values, y_values)
x_interp_cs = cs_x(t_interp)
y_interp_cs = cs_y(t_interp)

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(x_interp_newton, y_interp_newton, label="Многочлен Ньютона")
plt.plot(x_interp_hermite, y_interp_hermite, label="Многочлен Эрмита")
plt.plot(x_interp_cs, y_interp_cs, label="Кубические сплайны")
plt.scatter(x_values, y_values, color='red', label="Экспериментальные данные")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Траектории движения тела')
plt.legend()
plt.grid(True)
plt.show()

# Определение длины траектории для каждого метода интерполяции
def path_length(x_vals, y_vals):
    return np.sum(np.sqrt(np.diff(x_vals)**2 + np.diff(y_vals)**2))

length_newton = path_length(x_interp_newton, y_interp_newton)
length_hermite = path_length(x_interp_hermite, y_interp_hermite)
length_cs = path_length(x_interp_cs, y_interp_cs)

print("Длина траектории (Многочлен Ньютона):", length_newton)
print("Длина траектории (Многочлен Эрмита):", length_hermite)
print("Длина траектории (Кубические сплайны):", length_cs)

############################## lab 4 ####################################

def F(x):
	return 0.091 + 1000 * x * np.log(1000 * x) - 30


# Создание массива значений x для построения графика
x_values = np.linspace(0.01, 0.1, 100)
y_values = F(x_values)

# Построение графика
plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values, label='$F(x)$')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.xlabel('$x$')
plt.ylabel('$F(x)$')
plt.title('График функции $F(x)$')
plt.legend()
plt.grid(True)
plt.show()


def bisection_method(f, a, b, tol=1e-4, max_iter=100):
	if f(a) * f(b) > 0:
		raise ValueError("Функция не меняет знак на заданном интервале")

	iter_count = 0
	while (b - a) / 2 > tol:
		c = (a + b) / 2
		if f(c) == 0:
			return c, iter_count
		elif f(a) * f(c) < 0:
			b = c
		else:
			a = c
		iter_count += 1
		if iter_count >= max_iter:
			break

	return (a + b) / 2, iter_count


# Применение метода бисекции к функции F(x)
root_bisection, iterations_bisection = bisection_method(F, 0.01, 0.1)
print("Метод бисекции:")
print("Корень:", root_bisection)
print("Число итераций:", iterations_bisection)


def newton_method(f, df, x0, tol=1e-4, max_iter=100):
	iter_count = 0
	while True:
		x1 = x0 - f(x0) / df(x0)
		iter_count += 1
		if abs(x1 - x0) < tol or iter_count >= max_iter:
			break
		x0 = x1
	return x1, iter_count


# Производная функции F(x)
def dF(x):
	return 1000 * (1 + np.log(1000 * x))


# Применение метода Ньютона к функции F(x)
root_newton, iterations_newton = newton_method(F, dF, 0.01)
print("\nМетод Ньютона:")
print("Корень:", root_newton)
print("Число итераций:", iterations_newton)


def simple_iteration_method(phi, x0, tol=1e-4, max_iter=100):
	iter_count = 0
	while True:
		x1 = phi(x0)
		iter_count += 1
		if abs(x1 - x0) < tol or iter_count >= max_iter:
			break
		x0 = x1
	return x1, iter_count


# Функция для метода простых итераций
def phi(x):
	return (30 - 0.091) / (1000 * np.log(1000 * x))


# Применение метода простых итераций к функции F(x)
root_simple_iteration, iterations_simple_iteration = simple_iteration_method(phi, 0.01)
print("\nМетод простых итераций:")
print("Корень:", root_simple_iteration)
print("Число итераций:", iterations_simple_iteration)

############################## lab 5 ####################################

# заданные данные
x0 = 0
y0 = 2
xa = 5
ya = 3
xb = 6
yb = 4
x1 = 6
y1 = 0
m = 3
k = 0.1
vb = -1.5
g = 9.81


def f(t, y):
    # функция правых частей системы ОДУ
    vx, vy = y
    F = -k / m * np.sqrt((vx - vb) ** 2 + vy ** 2) * (vx - vb)

    return [F, -g - k / m * np.sqrt((vx - vb) ** 2 + vy ** 2) * vy]


# решение системы ОДУ методом Эйлера при шаге t=0.1
dt = 0.1
N = 50
x_euler = np.zeros(N)
y_euler = np.zeros(N)
x_euler[0], y_euler[0] = x0, y0

for i in range(1, N):
    vx, vy = f(0, [x_euler[i - 1], y_euler[i - 1]])
    x_euler[i] = x_euler[i - 1] + vx * dt
    y_euler[i] = y_euler[i - 1] + vy * dt

print("Решение системы ОДУ (метод Эйлера, шаг 0.1):", [x_euler[-1], y_euler[-1]])

# решение системы ОДУ методом Эйлера при шаге t=0.2
dt = 0.2
N = 25
x_euler = np.zeros(N)
y_euler = np.zeros(N)
x_euler[0], y_euler[0] = x0, y0

for i in range(1, N):
    vx, vy = f(0, [x_euler[i - 1], y_euler[i - 1]])
    x_euler[i] = x_euler[i - 1] + vx * dt
    y_euler[i] = y_euler[i - 1] + vy * dt

print("Решение системы ОДУ (метод Эйлера, шаг 0.2):", [x_euler[-1], y_euler[-1]])

# оценка погрешности методом Рунге
p_euler_1 = 1
dt_2 = dt / 2
x_euler_2 = np.zeros(N * 2)
y_euler_2 = np.zeros(N * 2)
x_euler_2[0], y_euler_2[0] = x0, y0

for i in range(1, N * 2):
    vx, vy = f(0, [x_euler_2[i - 1], y_euler_2[i - 1]])
    x_euler_2[i] = x_euler_2[i - 1] + vx * dt_2
    y_euler_2[i] = y_euler_2[i - 1] + vy * dt_2

if (y_euler[::2][-1] - y_euler[-1]) == 0 or (y_euler[::2][-1] - y_euler_2[-1]) == 0:
    p_euler_1 = float('inf')
else:
    p_euler_1 = np.log(abs((y_euler[::2][-1] - y_euler[-1]) / (y_euler[::2][-1] - y_euler_2[-1]))) / np.log(2)

print("Оценка погрешности метода Эйлера для шага 0.2:", p_euler_1)

p_euler_2 = 1
dt_2 = dt / 2
x_euler_2 = np.zeros(N * 2)
y_euler_2 = np.zeros(N * 2)
x_euler_2[0], y_euler_2[0] = x0, y0

for i in range(1, N * 2):
    vx, vy = f(0, [x_euler_2[i - 1], y_euler_2[i - 1]])
    x_euler_2[i] = x_euler_2[i - 1] + vx * dt_2
    y_euler_2[i] = y_euler_2[i - 1] + vy * dt_2

if (y_euler[::4][-1] - y_euler[-1]) == 0 or (y_euler[::4][-1] - y_euler_2[-1]) == 0:
    p_euler_2 = float('inf')
else:
    p_euler_2 = np.log(abs((y_euler[::4][-1] - y_euler[-1]) / (y_euler[::4][-1] - y_euler_2[-1]))) / np.log(2)

print("Оценка погрешности метода Эйлера для шага 0.1:", p_euler_2)

# решение системы ОДУ методом Рунге-Кутта 4 порядка при шаге t=0.1
sol_1 = solve_ivp(f, [0, 5], [0, 2], t_eval=np.linspace(0, 5, 50))

print("Решение системы ОДУ (метод Рунге-Кутта 4 порядка, шаг 0.1):", [sol_1.y[0][-1], sol_1.y[1][-1]])

# оценка погрешности методом Рунге
p_rk_1 = 4
dt = 0.1
dt_2 = dt / 2
sol_2 = solve_ivp(f, [0, 5], [0, 2], t_eval=np.linspace(0, 5, 100))

err_x = abs(sol_1.y[0][::2][-1] - sol_2.y[0][-1]) / (2 ** p_rk_1 - 1)
err_y = abs(sol_1.y[1][::2][-1] - sol_2.y[1][-1]) / (2 ** p_rk_1 - 1)

print("Оценка погрешности метода Рунге-Кутта 4 порядка для шага 0.2:", [err_x, err_y])

p_rk_2 = 4
dt = 0.2
dt_2 = dt / 2
sol_3 = solve_ivp(f, [0, 5], [0, 2], t_eval=np.linspace(0, 5, 51))

err_x = abs(sol_2.y[0][::2][-1] - sol_3.y[0][-1]) / (2 ** p_rk_2 - 1)
err_y = abs(sol_2.y[1][::2][-1] - sol_3.y[1][-1]) / (2 ** p_rk_2 - 1)

print("Оценка погрешности метода Рунге-Кутта 4 порядка для шага 0.2:", [err_x, err_y])

# подбор угла и скорости методом стрельбы
def hit_target(v):
    def f(t, y):
        vx, vy = y
        F = -k / m * np.sqrt((vx - vb) ** 2 + vy ** 2) * (vx - vb) + v[0] * v[1]
        return [F, -g - k / m * np.sqrt((vx - vb) ** 2 + vy ** 2) * vy]

    sol = solve_ivp(f, [0, 10], [0, 2], t_eval=np.linspace(0, 10, 50))
    return sol.y[0][-1] - x1, sol.y[1][-1] - y1


vx0 = 100
vy0 = np.sqrt((g * (x1 - x0) ** 2) / (
            2 * (vx0 - vb) ** 2 * (np.cos(ya)) ** 2 + 2 * (yb - y0 - (x1 - x0) * np.tan(ya)) / (x1 - xa) - g))

v = [vx0, vy0]
v_min, v_max = [-500, 500], [0, 500]

while v_max[1] - v_min[1] > 0.01:
    v_mid = [(v_min[i] + v_max[i]) / 2 for i in range(2)]

    if hit_target(v_mid)[1] > 0:
        v_max = v_mid
    else:
        v_min = v_mid

v = [(v_min[i] + v_max[i]) / 2 for i in range(2)]
print("Начальная скорость ядра:", np.linalg.norm(v))
print("Угол а:", np.degrees(np.arctan(vy0 / vx0)))

# наибольшая высота ядра в полете
print("Наибольшая высота ядра в полете:", max(sol_1.y[1]))

# время полета ядра до попадания в цель
t_max = 0
for i in range(1, len(sol_1.t)):
    if sol_1.y[0][i] > xa and sol_1.y[0][i] < xb and sol_1.y[1][i] > ya and sol_1.y[1][i] < yb:
        t_max = sol_1.t[i]
        break

if t_max == 0:
    t_max = (y1 - y0 + (x1 - x0) * np.tan(ya)) / ((v[0] * np.cos(ya) - vb) * np.cos(ya) * g)

print("Время полета ядра до попадания в цель:", t_max)

# траектория полета ядра
plt.plot(sol_1.y[0], sol_1.y[1], label='solution (RK4), h=0.1')
plt.plot(sol_2.y[0], sol_2.y[1], label='solution (RK4), h=0.2')
plt.plot(x_euler, y_euler, '--', label=f'solution (Euler), h=0.1 (p={p_euler_2:.2f})')
plt.plot(x_euler_2[::2], y_euler_2[::2], '--', label=f'solution (Euler), h=0.2 (p={p_euler_1:.2f})')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Траектория полета ядра')
plt.legend()
plt.show()
