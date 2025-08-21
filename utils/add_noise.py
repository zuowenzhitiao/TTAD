import os
import numpy as np

def add_different_noise(pts, noise_level, noise_class):
    bbdiag = float(np.linalg.norm(pts.max(0) - pts.min(0), 2)) / 2
    if noise_class == 'pulse':
        pulse_intensity = bbdiag * noise_level * 3
        p = 0.2
        x = np.random.rand(pts.shape[0], pts.shape[1])
        f = np.zeros(shape=pts.shape)
        f[x < p / 2] = -pulse_intensity
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if p / 2 < x[i, j] < p:
                    f[i, j] = pulse_intensity
        return pts + f

    elif noise_class == 'uniform':
        b = bbdiag * noise_level * 3
        a = -bbdiag * noise_level * 3
        s_noise = a + (b - a) * np.random.rand(pts.shape[0], pts.shape[1])
        return pts + s_noise

    elif noise_class == 'exp':
        a = bbdiag * noise_level
        e_noise = np.random.exponential(a, size=(pts.shape[0], pts.shape[1]))
        return pts + e_noise

    elif noise_class == 'gaussian':
        noise = np.random.randn(pts.shape[0], pts.shape[1]) * bbdiag * noise_level
        return pts + noise

    else:
        print('No such noise')
        return None

input_dir = '../data/PUNet/pointclouds/test/50000_poisson_tta'
noise_level = 0.01
output_dirs = {
    'pulse': '../data/examples/PUNet_50000_poisson_pulse_tta_' + str(noise_level),
    'uniform': '../data/examples/PUNet_50000_poisson_uniform_tta_' + str(noise_level),
    'exp': '../data/examples/PUNet_50000_poisson_exp_tta_' + str(noise_level),
    'gaussian': '../data/examples/PUNet_50000_poisson_gaussian_tta_' + str(noise_level)

}

for noise_type, output_dir in output_dirs.items():
    os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith('.xyz'):
        filepath = os.path.join(input_dir, filename)
        pts = np.loadtxt(filepath, dtype=np.float32)
        for noise_type in output_dirs.keys():
            noisy_pts = add_different_noise(pts, noise_level, noise_type)
            output_filepath = os.path.join(output_dirs[noise_type], filename)
            np.savetxt(output_filepath, noisy_pts, fmt='%.8f')