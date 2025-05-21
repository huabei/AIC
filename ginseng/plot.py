data_file = 'ginseng.mat'
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
data = scipy.io.loadmat(data_file)

fig = plt.figure(figsize=(16, 10))  # Increased figure size
ax = fig.add_subplot(111, projection='3d')

# Increase spacing between spectra and add transparency
spacing_factor = 1  # Increase spacing between spectrum lines
# Use distinct colors instead of gradient
distinct_colors = plt.cm.tab20(np.arange(32) % 20)

# 绘制三维曲线
for i in range(data['X'].shape[1]):
    ax.plot(data['t'].flatten(), 
            [i * spacing_factor] * len(data['t']), 
            data['X'][:, i], 
            color=distinct_colors[i],
            alpha=0.7,
            label=f'Spectrum {i+1}' if i % 5 == 0 else None)

ax.set_xlabel('Time', labelpad=10)
ax.set_ylabel('Spectrum Index', labelpad=10)
# ax.set_yticks([])
ax.set_zlabel('Intensity', labelpad=5)  # Increased label padding and rotated label
# ax.set_title('3D View of Ginseng Spectra', fontsize=14)


# Adjust view angle for better visualization
ax.view_init(elev=20, azim=50)  # Adjusted view angle

# 将图例放在图外
ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=1)


plt.tight_layout()
plt.show()
