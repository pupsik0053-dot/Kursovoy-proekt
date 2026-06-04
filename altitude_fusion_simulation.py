

import numpy as np
import matplotlib.pyplot as plt

dt = 0.004       # 250 Гц
T  = 40.0        # время симуляции, с
N  = int(T / dt)
t  = np.arange(N) * dt

LIDAR_TO_BARO_M = 3.8   
BARO_TO_LIDAR_M = 3.5   
BLEND_STEP      = 0.002  

LIDAR_MAX_M = 4.0        # предел измерения лидара
LIDAR_NOISE = 0.0015     
BARO_NOISE  = 0.05     
BARO_OFFSET = 0.12       

h_true = np.zeros(N)
phase1 = int(15 / dt)   # подъём
phase2 = int(20 / dt)   # начало зависания
phase3 = int(35 / dt)   # начало снижения

for k in range(N):
    if k < phase1:
        h_true[k] = 6.0 * k / phase1
    elif k < phase2:
        h_true[k] = 6.0
    elif k < phase3:
        h_true[k] = 6.0 - 6.0 * (k - phase2) / (phase3 - phase2)
    else:
        h_true[k] = 0.0

rng = np.random.default_rng(7)

# Лидар: действителен только до 4 м
lidar_raw = h_true + rng.normal(0, LIDAR_NOISE, N)
lidar_valid = (lidar_raw < LIDAR_MAX_M) & (lidar_raw > 0.05)
lidar_reading = np.where(lidar_valid, lidar_raw, np.nan)

baro_reading = h_true + BARO_OFFSET + rng.normal(0, BARO_NOISE, N)

blend_alpha   = np.zeros(N)
fused         = np.zeros(N)
alpha         = 0.0
offset_est    = BARO_OFFSET       

for k in range(N):
    if lidar_valid[k] and h_true[k] < LIDAR_TO_BARO_M:
        offset_est = 0.99 * offset_est + 0.01 * (baro_reading[k] - lidar_raw[k])

    if lidar_valid[k] and lidar_raw[k] < LIDAR_TO_BARO_M:
        alpha = max(0.0, alpha - BLEND_STEP)   # тянемся к лидару
    else:
        alpha = min(1.0, alpha + BLEND_STEP)   # тянемся к барометру

    blend_alpha[k] = alpha

    # Смешанная высота
    h_l = lidar_raw[k] if lidar_valid[k] else (fused[k-1] if k > 0 else h_true[k])
    h_b = baro_reading[k] - offset_est
    fused[k] = (1.0 - alpha) * h_l + alpha * h_b

plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.4})
fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
fig.suptitle('Задание 4★★ — плавное переключение лидар / барометр', fontsize=13)

# График 1 — высоты
ax = axes[0]
ax.plot(t, h_true,       color='black',     lw=2.0, label='Истинная высота', zorder=5)
ax.plot(t, lidar_reading, color='steelblue', lw=1.2, label='Лидар', alpha=0.7)
ax.plot(t, baro_reading,  color='salmon',    lw=1.2, label='Барометр (сырой)', alpha=0.7)
ax.plot(t, fused,         color='darkgreen', lw=1.8, label='fused_altitude', zorder=4)
ax.axhline(LIDAR_TO_BARO_M, color='orange', ls='--', lw=1.2, label=f'H_high={LIDAR_TO_BARO_M} м')
ax.axhline(BARO_TO_LIDAR_M, color='purple',  ls='--', lw=1.2, label=f'H_low={BARO_TO_LIDAR_M} м')
ax.set_ylabel('Высота, м')
ax.set_title('Показания датчиков и итоговая высота fused_altitude')
ax.legend(ncol=3, fontsize=9)

# График 2 — blend_alpha
ax = axes[1]
ax.plot(t, blend_alpha, color='darkorange', lw=1.6)
ax.axhline(0.0, color='steelblue', ls=':', lw=1.0, label='0 = только лидар')
ax.axhline(1.0, color='red',       ls=':', lw=1.0, label='1 = только барометр')
ax.set_ylabel('blend_alpha'); ax.set_ylim(-0.05, 1.05)
ax.set_title('Коэффициент смешения α (0 = лидар, 1 = барометр)')
ax.legend()

# График 3 — ошибка слияния
ax = axes[2]
error = fused - h_true
ax.plot(t, error * 1000, color='crimson', lw=1.2, label='Ошибка fused, мм')
ax.axhline(0, color='black', lw=0.8)
ax.axhline( 50, color='orange', ls='--', lw=0.9, label='±50 мм')
ax.axhline(-50, color='orange', ls='--', lw=0.9)
ax.set_xlabel('Время, с'); ax.set_ylabel('Ошибка, мм')
ax.set_title('Ошибка итоговой высоты относительно истинной')
ax.legend()

plt.tight_layout()
fig.savefig('altitude_fusion_simulation.png', dpi=150, bbox_inches='tight')
plt.close()

rms = np.sqrt(np.mean(error**2)) * 1000
print(f"График сохранён: altitude_fusion_simulation.png")
print(f"RMS-ошибка fused_altitude: {rms:.1f} мм")
print(f"Макс. ошибка при переключении: {np.max(np.abs(error))*1000:.1f} мм")