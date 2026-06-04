
import numpy as np
import matplotlib.pyplot as plt

dt = 0.004          # 250 Гц
t  = np.arange(0, 10, dt)

h_true = 1.0        # истинная высота, м

roll  = np.where(t < 3, 20 * np.sin(2 * np.pi * t / 3), 0.0)   # градусы
pitch = np.zeros_like(t)

# Показание лидара БЕЗ компенсации 
h_lidar_raw  = h_true / np.cos(np.radians(roll)) / np.cos(np.radians(pitch))

# Показание лидара С компенсацией
h_lidar_comp = h_lidar_raw * np.cos(np.radians(roll)) * np.cos(np.radians(pitch))

rng   = np.random.default_rng(42)
noise = rng.normal(0, 0.002, len(t))
h_lidar_raw  = h_lidar_raw  + noise
h_lidar_comp = h_lidar_comp + noise

phi_threshold = np.degrees(np.arccos(1 / 1.05))
print(f"Ошибка высоты превысит 5 см при крене > {phi_threshold:.1f}°")
print(f"(при истинной высоте 1 м)")

plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.4})
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
fig.suptitle('Влияние угла наклона на показания лидара (h_true = 1 м)', fontsize=13)

# Левый — без компенсации
ax = axes[0]
ax.plot(t, h_lidar_raw * 1000, color='red', lw=1.2, label='Без компенсации', alpha=0.85)
ax.axhline(h_true * 1000, color='green', ls='--', lw=1.5, label='Истинная высота')
ax.fill_between(t, (h_true - 0.05) * 1000, (h_true + 0.05) * 1000,
                alpha=0.15, color='green', label='Допуск ±50 мм')
ax.set_xlabel('t, с'); ax.set_ylabel('Высота, мм')
ax.set_title('Без компенсации')
ax.legend(); ax.set_xlim(0, 10)

# Правый — с компенсацией
ax = axes[1]
ax.plot(t, h_lidar_comp * 1000, color='steelblue', lw=1.2, label='С компенсацией', alpha=0.85)
ax.axhline(h_true * 1000, color='green', ls='--', lw=1.5, label='Истинная высота')
ax.fill_between(t, (h_true - 0.05) * 1000, (h_true + 0.05) * 1000,
                alpha=0.15, color='green', label='Допуск ±50 мм')
ax.set_xlabel('t, с')
ax.set_title('С компенсацией')
ax.legend(); ax.set_xlim(0, 10)

plt.tight_layout()
fig.savefig('tilt_compensation.png', dpi=150, bbox_inches='tight')
plt.close()
print("График сохранён: tilt_compensation.png")
print(f"\nОтвет: При крене {phi_threshold:.1f}° ошибка высоты достигает 5 см.")
print("Отключение ПИД при >30° целесообразно: при таком наклоне дрон активно")
print("маневрирует, лидар смотрит сбоку, и коррекция высоты только мешает.")