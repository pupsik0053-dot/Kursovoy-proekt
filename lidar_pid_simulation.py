import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

dt = 0.004          # шаг интегрирования, с (250 Гц)
T  = 5.0            # время симуляции, с
N  = int(T / dt)    # число шагов

Kp      = 0.3
Ki      = 0.00005
Kd      = 1.0
I_max   = 400.0
k_u     = 0.001     # коэффициент передачи тяги → ускорение

h_setpoint = 1000.0  # уставка, мм
h0         = 1200.0  # начальная высота, мм

t_arr = np.arange(N) * dt
h     = np.zeros(N)
v     = np.zeros(N)    # вертикальная скорость, мм/с
u     = np.zeros(N)    # выход ПИД

P_arr = np.zeros(N)
I_arr = np.zeros(N)
D_arr = np.zeros(N)

# Начальные условия
h[0]   = h0
v[0]   = 0.0
e_prev = 0.0
I_mem  = 0.0

WIND_TIME = 2.0  
WIND_MM   = 200.0 

wind_applied = False

for k in range(N - 1):
    # Порыв ветра
    if t_arr[k] >= WIND_TIME and not wind_applied:
        h[k] += WIND_MM
        wind_applied = True

    e = h_setpoint - h[k]

    I_mem  = np.clip(I_mem + Ki * e, -I_max, I_max)
    P_term = Kp * e
    I_term = I_mem
    D_term = Kd * (e - e_prev)
    e_prev = e

    u_k = P_term + I_term + D_term
    u[k] = u_k

    P_arr[k] = P_term
    I_arr[k] = I_term
    D_arr[k] = D_term
    v[k+1] = v[k] + k_u * u_k * dt * 1000   # мм/с
    h[k+1] = h[k] + v[k+1] * dt

P_arr[-1] = P_arr[-2]; I_arr[-1] = I_arr[-2]; D_arr[-1] = D_arr[-2]

plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.4})
fig = plt.figure(figsize=(13, 8))
gs  = gridspec.GridSpec(2, 1, hspace=0.45)

# График 1 — высота
ax1 = fig.add_subplot(gs[0])
ax1.plot(t_arr, h, color='steelblue', lw=1.8, label='Высота h(t)')
ax1.axhline(h_setpoint, color='red', ls='--', lw=1.5, label='Уставка 1000 мм')
ax1.axvline(WIND_TIME, color='orange', ls=':', lw=1.4, label='Порыв ветра (t=2 с)')
ax1.set_xlabel('Время, с')
ax1.set_ylabel('Высота, мм')
ax1.set_title('ПИД высоты — отработка уставки и внешнего возмущения')
ax1.legend(loc='upper right')
ax1.set_xlim(0, T)

# График 2 — составляющие ПИД
ax2 = fig.add_subplot(gs[1])
ax2.plot(t_arr, P_arr, color='forestgreen', lw=1.4, label='P-составляющая')
ax2.plot(t_arr, I_arr, color='purple',      lw=1.4, label='I-составляющая')
ax2.plot(t_arr, D_arr, color='darkorange',  lw=1.4, label='D-составляющая')
ax2.axvline(WIND_TIME, color='orange', ls=':', lw=1.4)
ax2.set_xlabel('Время, с')
ax2.set_ylabel('Выход регулятора, у.е.')
ax2.set_title('Три составляющие ПИД-регулятора высоты')
ax2.legend(loc='upper right')
ax2.set_xlim(0, T)

fig.savefig('pid_altitude_simulation.png', dpi=150, bbox_inches='tight')
plt.close()
print("График сохранён: pid_altitude_simulation.png")

initial_error = abs(h0 - h_setpoint)
threshold = 0.05 * initial_error
settle_idx = next((i for i in range(N) if abs(h[i] - h_setpoint) < threshold and t_arr[i] > 0.1), None)
settle_time = t_arr[settle_idx] if settle_idx else ">5 с"
print(f"Время установления (±5%): {settle_time} с")
print("\nОтветы на вопросы:")
print(f"  Начальная ошибка: {initial_error:.0f} мм")
print(f"  Kp={Kp}, Ki={Ki}, Kd={Kd}")
print("  ПИД крена Kd≈20, ПИД высоты Kd=1.0 — разница в 20 раз,")
print("  обусловленная медленной вертикальной динамикой по сравнению с угловой.")