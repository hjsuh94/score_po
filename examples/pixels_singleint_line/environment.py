import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter

class Environment:
    """
    Generate a 2D environment with obstacles.
    """

    def __init__(self):
        self.shape_lst = []

    def add_ellipse(self, mu, a, b):
        self.shape_lst.append(["ellipse", mu, a, b])

    def add_rectangle(self, mu, a, b):
        self.shape_lst.append(["rectangle", mu, a, b])

    def sample_ellipse(self, mu, a, b, samples):
        return ((samples[:, 0] - mu[0]) / a) ** 2 + ((samples[:, 1] - mu[1]) / b) ** 2 < 1.0

    def sample_rectangle(self, mu, a, b, samples):
        return torch.logical_and(
            (torch.abs(samples[:, 0] - mu[0]) < a / 2),
            (torch.abs(samples[:, 1] - mu[1]) < b / 2)
        )

    def sample_image(self, num_points, dim, samples=None, filter_obs=True, xlim=1., ell_size=0.05, N_EXP=10, buf=0.2):
        if samples is None:
            samples = 2.0 * xlim * torch.rand(num_points, 2) - xlim
            valid = torch.zeros(num_points)
        else:
            valid = torch.zeros(samples.shape[0])
        robot_size = ell_size*dim*N_EXP

        samples_img = []
        for i in range(samples.shape[0]):
            x = samples[i, 0]
            y = samples[i, 1]
            shape = [(x + (1+buf)*xlim)/(2*(1+buf)*xlim)*dim*N_EXP - robot_size,
                     (y + (1+buf)*xlim)/(2*(1+buf)*xlim)*dim*N_EXP - robot_size,
                     (x + (1+buf)*xlim)/(2*(1+buf)*xlim)*dim*N_EXP + robot_size,
                     (y + (1+buf)*xlim)/(2*(1+buf)*xlim)*dim*N_EXP + robot_size]

            # creating new Image object
            img = Image.new("RGB", (dim*N_EXP, dim*N_EXP))
            # create circle robot
            img1 = ImageDraw.Draw(img)
            img1.ellipse(shape, fill="red", outline="red")
            img = np.array(img)
            img = Image.fromarray(img).filter(ImageFilter.BoxBlur(N_EXP))
            img = np.array(img.resize((int(dim), int(dim)), resample=Image.BICUBIC))

            samples_img.append(img[:, :, 0] / max(img[:, :, 0].max(), 0.001))
        if filter_obs == True:
            for shape in self.shape_lst:
                if shape[0] == "ellipse":
                    valid = torch.logical_or(valid,
                                             self.sample_ellipse(shape[1], shape[2], shape[3], samples)
                                             )
                if shape[0] == "rectangle":
                    valid = torch.logical_or(valid,
                                             self.sample_rectangle(shape[1], shape[2], shape[3], samples)
                                             )
        samples_img = torch.tensor(samples_img)
        return samples_img[~valid.bool(), :], samples[~valid.bool(), :]

    def sample_control_image(self, dim, u_batch, xlim=0.1, ell_size=0.05, N_EXP=10, buf=0.1):
        robot_size = ell_size * dim * N_EXP

        samples_img = []
        for i in range(u_batch.shape[0]):
            x = u_batch[i, 0]
            y = u_batch[i, 1]
            shape = [(x + (1 + buf) * xlim) / (2 * (1 + buf) * xlim) * dim * N_EXP - robot_size,
                     (y + (1 + buf) * xlim) / (2 * (1 + buf) * xlim) * dim * N_EXP - robot_size,
                     (x + (1 + buf) * xlim) / (2 * (1 + buf) * xlim) * dim * N_EXP + robot_size,
                     (y + (1 + buf) * xlim) / (2 * (1 + buf) * xlim) * dim * N_EXP + robot_size]

            # creating new Image object
            img = Image.new("RGB", (dim * N_EXP, dim * N_EXP))
            # create circle robot
            img1 = ImageDraw.Draw(img)
            img1.ellipse(shape, fill="red", outline="red")
            img = np.array(img)
            img = Image.fromarray(img).filter(ImageFilter.BoxBlur(N_EXP))
            img = np.array(img.resize((int(dim), int(dim)), resample=Image.BICUBIC))

            samples_img.append(img[:, :, 0] / max(img[:, :, 0].max(), 0.001))

        samples_img = torch.tensor(samples_img)
        return samples_img

def test():
    env = Environment()
    env.add_ellipse([0, 0], 0.4, 0.4)
    pts = env.sample_image(100000)
    plt.figure()
    plt.plot(pts[:, 0], pts[:, 1], 'ro')
    plt.savefig("env.png")
    plt.close()
