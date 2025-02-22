from pathlib import Path
from typing import *

import cv2
import numpy as np
import matplotlib.pyplot as plt

# --------------------- Set Variables --------------------- #
DATA_PATH = './data/task1and2_hybrid_pyramid'
OUTPUT_PATH = './output/task2'

GAUSSIAN_KERNEL_SIZE = 5
NUMBER_OF_LAYERS = 6
# --------------------------------------------------------- #

class ImagePyramid:
    def __init__(self, kernel_size: int = 5, num_layers: int = 6):
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.kernel = self._gauss_kernel(kernel_size)

    @staticmethod
    def _gauss_kernel(ksize: int, sigma: Optional[float] = None) -> np.ndarray:
        assert ksize & 1
        """ 
            Reference:
            https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#gac05a120c1ae92a6060dd0db190a61afa
        """
        # Calculate sigma if not provided
        if sigma is None or sigma <= 0:
            sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
        
        # Create 2D Gaussian kernel
        x, y = np.ogrid[-ksize//2 + 1:ksize//2 + 1, -ksize//2 + 1:ksize//2 + 1]
        kernel = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
        return kernel / kernel.sum()
    
    @staticmethod
    def _fft(image: np.ndarray) -> np.ndarray:
        # Perform 2D FFT and shift the zero-frequency component to the center
        assert image.ndim == 2
        return np.fft.fftshift(np.fft.fft2(image))

    def _conv2d(self, image: np.ndarray, mode: str = 'replicate') -> np.ndarray:
        assert image.ndim == 2
        assert self.kernel.ndim == 2
        assert self.kernel.shape[0] & 1
        assert self.kernel.shape[1] & 1
        assert mode in ['zero', 'replicate']

        # Prepare padded image
        rows, cols = image.shape
        krows, kcols = self.kernel.shape
        rb_size, cb_size = krows // 2, kcols // 2
        padding = np.zeros((rows + rb_size * 2, cols + cb_size * 2), dtype=float)
        padding[rb_size:-rb_size, cb_size:-cb_size] = image

        # Handle padding modes
        if mode == 'replicate':
            # Replicate border pixels
            padding[rb_size:-rb_size, :cb_size] = image[:, cb_size - 1::-1]
            padding[rb_size:-rb_size:, -cb_size:] = image[:, :-cb_size - 1:-1]
            padding[:rb_size, cb_size:-cb_size] = image[rb_size - 1::-1, :]
            padding[-rb_size:, cb_size:-cb_size] = image[:-rb_size - 1:-1, :]
        elif mode != 'zero':
            raise ValueError("unknown mode")

        # Perform convolution
        res = np.zeros(image.shape)
        for i in range(krows):
            for j in range(kcols):
                res += self.kernel[i, j] * padding[i:i + rows, j:j + cols]
        return res

    def build_pyramids(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        gaussian_pyramid = [image]
        gaussian_spectrum = [self._fft(image)]
        laplacian_pyramid = []
        laplacian_spectrum = []

        for _ in range(self.num_layers - 1):
            # Blur the image
            blurred = self._conv2d(gaussian_pyramid[-1])
            # Compute Laplacian as difference between original and blurred
            laplacian_pyramid.append(gaussian_pyramid[-1] - blurred)
            # Downsample blurred image for next level
            gaussian_pyramid.append(blurred[::2, ::2])
            
            # Compute spectra
            laplacian_spectrum.append(self._fft(laplacian_pyramid[-1]))
            gaussian_spectrum.append(self._fft(gaussian_pyramid[-1]))

        # Add the final Gaussian level to Laplacian pyramid
        laplacian_pyramid.append(gaussian_pyramid[-1])
        laplacian_spectrum.append(gaussian_spectrum[-1])

        return gaussian_pyramid, laplacian_pyramid, gaussian_spectrum, laplacian_spectrum

    @staticmethod
    def visualize(gaussian_pyramid: List[np.ndarray], laplacian_pyramid: List[np.ndarray],
                  gaussian_spectrum: List[np.ndarray], laplacian_spectrum: List[np.ndarray],
                  figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        assert len(gaussian_pyramid) == len(laplacian_pyramid) == len(gaussian_spectrum) == len(laplacian_spectrum)
        fig, axs = plt.subplots(4, len(gaussian_pyramid), figsize=figsize)
        for idx in range(len(gaussian_pyramid)):
            # Display Gaussian and Laplacian pyramids
            axs[0, idx].imshow(gaussian_pyramid[idx], cmap='gray', vmin=0, vmax=255)
            axs[1, idx].imshow(laplacian_pyramid[idx], cmap='gray')
            
            # Normalize and display Gaussian spectrum
            gaussian_spec_norm = np.log(np.abs(gaussian_spectrum[idx]) + 1)
            gaussian_spec_norm = np.interp(gaussian_spec_norm, (gaussian_spec_norm.min(), gaussian_spec_norm.max()), (0, 255))
            axs[2, idx].imshow(gaussian_spec_norm, cmap='winter', vmin=0, vmax=255)
            
            # Normalize and display Laplacian spectrum
            laplacian_spec_norm = np.log(np.abs(laplacian_spectrum[idx]) + 1)
            laplacian_spec_norm = np.interp(laplacian_spec_norm, (laplacian_spec_norm.min(), laplacian_spec_norm.max()), (0, 255))
            axs[3, idx].imshow(laplacian_spec_norm, cmap='winter', vmin=0, vmax=255)

            # Remove axis ticks
            for ax in axs[:, idx]:
                ax.set_xticks([])
                ax.set_yticks([])
        plt.tight_layout()
        return fig
    
def process_images(data_path: Path, output_path: Path):
    assert data_path.exists(), f"Data path {data_path} does not exist"

    pyramid = ImagePyramid(num_layers=NUMBER_OF_LAYERS)
    output_path.mkdir(parents=True, exist_ok=True)

    for file in data_path.glob('*'):
        if file.suffix.lower() not in ['.bmp', '.jpg', '.jpeg', '.png']:
            continue
        
        # Read image in grayscale
        image = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
        # Build pyramids and compute spectra
        gaussian, laplacian, gaussian_spec, laplacian_spec = pyramid.build_pyramids(image)
        
        # Visualize results
        fig = pyramid.visualize(gaussian, laplacian, gaussian_spec, laplacian_spec)
        
        # Save output
        output_file = output_path / f"{file.stem}.png"
        print(f"Saving to {Path(output_file).resolve()}")
        fig.savefig(output_file)
        plt.close(fig)

if __name__ == '__main__':
    process_images(Path(DATA_PATH), Path(OUTPUT_PATH))
