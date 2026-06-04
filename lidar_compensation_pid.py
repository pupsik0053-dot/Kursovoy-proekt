

import numpy as np
import matplotlib.pyplot as plt

dt = 0.004
T  = 10.0
N  = int(T / dt)
t  = np.arange(N) * dt

Kp = 0.3; Ki = 0.00005; Kd = 1.0; I_max = 400.0
h_setpoint_mm = 1000.0

h_true_m = 1.0  # истинная высота, м

roll_deg  = np.where(t < 3, 20 * np.sin(2 * np.pi * t / 3), 0.0)
pitch_deg = np.zeros(N)

# Шум
rng   = np.random.default_rng(0)
noise = rng.normal(0, 0.002, N)

h_lidar_raw_m = h_true_m / np.cos(np.radians(roll_deg))
h_lidar_raw_mm = h_lidar_raw_m * 1000 + noise * 1000

# Компенсированное расстояние 
h_lidar_comp_mm = h_lidar_raw_mm * np.cos(np.radians(roll_deg))

def run_pid(h_measured_arr):
    u_out = np.zeros(N)
    e_prev = 0.0; I_mem = 0.0
    for k in range(N):
        e    = h_setpoint_mm - h_measured_arr[k]
        I_mem = np.clip(I_mem + Ki * e, -I_max, I_max)
        u_out[k] = Kp * e + I_mem + Kd * (e - e_prev)
        e_prev = e
    return u_out

u_raw  = run_pid(h_lidar_raw_mm)   # без компенсации
u_comp = run_pid(h_lidar_comp_mm)  # с компенсацией

plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.4})
fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
fig.suptitle('Задание 3 — влияние компенсации угла наклона на ПИД высоты', fontsize=13)

axes[0].plot(t, roll_deg, color='darkorange', lw=1.5)
axes[0].set_ylabel('Крен, °'); axes[0].set_title('Угол крена')

axes[1].plot(t, h_lidar_raw_mm,  color='red',       lw=1.2, label='Без компенсации', alpha=0.8)
axes[1].plot(t, h_lidar_comp_mm, color='steelblue',  lw=1.2, label='С компенсацией',  alpha=0.8)
axes[1].axhline(h_setpoint_mm, color='green', ls='--', lw=1.3, label='Уставка')
axes[1].set_ylabel('Высота, мм'); axes[1].set_title('Показания лидара')
axes[1].legend()

axes[2].plot(t, u_raw,  color='red',      lw=1.2, label='u_pid без компенсации', alpha=0.8)
axes[2].plot(t, u_comp, color='steelblue', lw=1.2, label='u_pid с компенсацией',  alpha=0.8)
axes[2].axhline(0, color='black', lw=0.8, ls=':')
axes[2].set_xlabel('t, с'); axes[2].set_ylabel('u_pid, у.е.')
axes[2].set_title('Управляющий сигнал ПИД')
axes[2].legend()

plt.tight_layout()
fig.savefig('lidar_compensation_pid.png', dpi=150, bbox_inches='tight')
plt.close()
print("График сохранён: lidar_compensation_pid.png")
print(f"\nМакс. колебание u_pid БЕЗ компенсации: {u_raw[:750].ptp():.1f}")
print(f"Макс. колебание u_pid С  компенсацией: {u_comp[:750].ptp():.1f}")