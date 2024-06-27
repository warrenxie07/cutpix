import io
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from scipy.io import loadmat
from scipy.fftpack import fftn, ifftn, fftshift, ifftshift
from scipy.ndimage import gaussian_filter, zoom, map_coordinates


def load_mat(path: Path) -> None:
    """
    Load the mat file from the given path.

    Parameters:
    - path: Path, the path to the mat file.

    Returns:
    - data: np.ndarray, the data from the mat file.
    """
    matData = loadmat(path)
    data = matData["data"]
    return data


def norm(volume) -> np.ndarray:
    """
    Normalize the volume to the range [0, 1].

    Parameters:
    - volume: np.ndarray, the input volume.

    Returns:
    - normalized_x: np.ndarray, the normalized volume.
    """
    MAX = 1.455
    MIN = 1.3370
    normalized_x = (volume.clip(MIN, MAX) - MIN) / (MAX - MIN)
    return normalized_x


def brightness(volume, mag) -> np.ndarray:
    """
    Adjust the brightness of the volume.

    Parameters:
    - volume: np.ndarray, the input volume.
    - mag: float, the magnitude of the brightness adjustment.

    Returns:
    - manipulated_volume: np.ndarray, the volume after brightness adjustment.
    """
    manipulated_volume = (volume * mag).clip(0, volume.max())
    return manipulated_volume


def constrast(volume, mag) -> np.ndarray:
    """
    Adjust the constrast of the volume.

    Parameters:
    - volume: np.ndarray, the input volume.
    - mag: float, the magnitude of the constrast adjustment.

    Returns:
    - manipulated_volume: np.ndarray, the volume after constrast adjustment.
    """
    manipulated_volume = volume ** mag
    return manipulated_volume


def sharpness(volume, mag) -> np.ndarray:
    """
    Adjust the sharpness of the volume.

    Parameters:
    - volume: np.ndarray, the input volume.
    - mag: float, the magnitude of the sharpness adjustment.

    Returns:
    - sharpened_volume: np.ndarray, the volume after sharpness adjustment.
    """
    blur = gaussian_filter(volume, 0.5)
    blur_2 = gaussian_filter(blur, 3)
    sharpened_volume = volume + mag * (blur - blur_2)
    sharpened_volume = sharpened_volume.clip(0, 1)
    return sharpened_volume


def gaussian_noise(volume, mag) -> np.ndarray:
    """
    Simulate Gaussian noise for the microscipic volume.
    x = x + np.random.randn(*x.shape) * mag

    Parameters:
    - volume: np.ndarray, the input volume.
    - mag: float, the magnitude of the Gaussian noise.

    Returns:
    - noised_volume: np.ndarray, the volume after adding Gaussian noise.
    """
    noised_volume = (volume + np.random.randn(*volume.shape)
                     * mag).clip(0, volume.max())
    return noised_volume


def poisson_noise(volume, mag) -> np.ndarray:
    """
    Simulate Poisson noise for the microscipic volume.
    x = np.random.poisson(x * mag) / mag

    Parameters:
    - volume: np.ndarray, the input volume.
    - mag: float, the magnitude of the Poisson noise.

    Returns:
    - noised_volume: np.ndarray, the volume after adding Poisson noise.
    """
    noised_volume = (np.random.poisson(volume * mag) /
                     mag).clip(0, volume.max())
    return noised_volume


def speckle_noise(volume, mag) -> np.ndarray:
    """
    Simulate Speckle noise for the microscipic volume.
    x = x + x * np.random.randn(*x.shape) * mag

    Parameters:
    - volume: np.ndarray, the input volume.
    - mag: float, the magnitude of the Speckle noise.

    Returns:
    - noised_volume: np.ndarray, the volume after adding Speckle noise.
    """
    noised_volume = (volume + volume * np.random.randn(*
                     volume.shape) * mag).clip(0, volume.max())
    return noised_volume


def defocus(volume, mag) -> np.ndarray:
    """
    Simulate defocus for the microscipic volume.
    To simulate defocus, we can apply a Gaussian filter to the volume.

    Parameters:
    - volume: np.ndarray, the input volume.
    - mag: float, the magnitude of the defocus.

    Returns:
    - defocused_volume: np.ndarray, the volume after defocus.
    """
    defocused_volume = gaussian_filter(volume, mag).clip(0, 1)
    return defocused_volume


def spherical_aberration(volume, mag) -> np.ndarray:
    """
    Simulate spherical aberration for the microscipic volume.
    Spherical aberration is a type of lens aberration that occurs
    when light rays passing through the lens periphery are focused 
    at a different point than those passing through the center of the lens.
    To simulate spherical aberration, we can apply different filters 
    to the volume in the frequency domainaccording to the distance 
    from the center of the volume.

    Parameters:
    - volume: np.ndarray, the input volume.
    - mag: float, the magnitude of the spherical aberration.

    Returns:
    - aberrated_volume: np.ndarray, the volume after spherical aberration.
    """
    depth, height, width = volume.shape
    center_z, center_y, center_x = depth // 2, height // 2, width // 2

    # Create the coordinate grid
    z = np.arange(-center_z, depth - center_z)
    y = np.arange(-center_y, height - center_y)
    x = np.arange(-center_x, width - center_x)
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')

    # Calculate the distance from the center in the frequency domain
    distance = np.sqrt(X**2 + Y**2 + Z**2)
    max_distance = np.max(distance)

    # Normalize distances and create the spherical aberration filter
    normalized_distance = distance / max_distance
    spherical_aberration_filter = 1 - (mag * normalized_distance**2)
    spherical_aberration_filter = np.clip(spherical_aberration_filter, 0, 1)

    # Apply Fourier transform to the volume
    volume_fft = fftn(volume)
    volume_fft_shifted = fftshift(volume_fft)

    # Apply the spherical aberration filter in the frequency domain
    volume_fft_aberrated = volume_fft_shifted * spherical_aberration_filter

    # Inverse Fourier transform to get the aberrated volume
    volume_fft_aberrated_shifted = ifftshift(volume_fft_aberrated)
    aberrated_volume = ifftn(volume_fft_aberrated_shifted).real

    return aberrated_volume


def zoom_blur(volume, mag=10) -> np.ndarray:
    """
    Simulate zoom blur for the microscipic volume.
    Zoom blur is a type of motion blur that occurs when the camera
    zooms in or out while taking a photo. It creates a radial blur
    effect that simulates the feeling of motion.
    To simulate zoom blur, we can apply a zoom effect to the volume
    and blend the zoomed images together to create the blur effect.

    Parameters:
    - volume: np.ndarray, the input volume.
    - mag: int, the magnitude of the zoom blur.

    Returns:
    - zoom_blurred_volume: np.ndarray, the volume after zoom blur.
    """
    depth, height, width = volume.shape
    zoom_blurred_volume = np.zeros_like(volume, dtype=np.float32)

    # Apply the zoom blur effect to each slice
    for z in range(depth):
        img = volume[z, :, :]
        zoom_blurred = np.zeros_like(img, dtype=np.float32)

        for i in range(1, mag + 1):
            # Zoom the image
            zoom_factor = 1 + i / (mag * 10.0)
            scaled_img = zoom(img, zoom_factor, order=1)

            # Calculate the coordinates to crop the zoomed image back to original size
            zh, zw = scaled_img.shape
            start_y = (zh - height) // 2
            start_x = (zw - width) // 2
            cropped_img = scaled_img[start_y:start_y +
                                     height, start_x:start_x + width]

            # Blend the zoomed image with the zoom blurred image
            zoom_blurred += cropped_img / mag

        # Assign the zoom blurred slice back to the volume
        zoom_blurred_volume[z, :, :] = zoom_blurred

    # Clip the values to be within the valid range and convert back to the original data type
    zoom_blurred_volume = np.clip(
        zoom_blurred_volume, 0, 1).astype(volume.dtype)

    return zoom_blurred_volume


def low_numerical_aperture(volume, mag) -> np.ndarray:
    """
    Simulate low numerical aperture for the microscipic volume.
    Numerical aperture (NA) is a measure of the ability of a lens to gather light.
    High NA lenses can resolve finer details than low NA lenses.
    To simulate low NA, we can apply a low-pass filter to the volume.

    Parameters:
    - volume: np.ndarray, the input volume.
    - mag: float, the magnitude of the low NA.

    Returns:
    - low_na_volume: np.ndarray, the volume after low NA.
    """
    # Get the dimensions of the volume
    depth, height, width = volume.shape
    center_z, center_y, center_x = depth // 2, height // 2, width // 2

    # Create the coordinate grid
    z = np.arange(-center_z, depth - center_z)
    y = np.arange(-center_y, height - center_y)
    x = np.arange(-center_x, width - center_x)
    Z, Y, X = np.meshgrid(z, y, x, indexing='ij')

    # Calculate the distance from the center in the frequency domain
    distance = np.sqrt(X**2 + Y**2 + Z**2)

    # Apply Fourier transform to the volume
    volume_fft = fftn(volume)
    volume_fft_shifted = fftshift(volume_fft)

    # Create a frequency mask to keep only the desired frequencies
    frequency_mask = distance <= mag

    # Apply the frequency mask
    filtered_volume_fft_shifted = volume_fft_shifted * frequency_mask

    # Inverse Fourier transform to get the filtered volume
    filtered_volume_fft = ifftshift(filtered_volume_fft_shifted)
    noisy_volume = ifftn(filtered_volume_fft).real

    return noisy_volume


def pixelate(volume, mag) -> np.ndarray:
    """
    Simulate zoom artifact for the microscipic volume.
    To simulate pixelation, we can resize the volume to 
    a smaller size then resize it back to the original size.

    Parameters:
    - volume: np.ndarray, the input volume.
    - mag: int, the magnitude of the pixelation.

    Returns:
    - pixelated_volume: np.ndarray, the volume after pixelation.
    """
    depth, height, width = volume.shape
    new_d, new_h, new_w = [int(dim * mag) for dim in volume.shape]

    img_reduced = zoom(volume, (new_d / depth, new_h /
                       height, new_w / width), order=0)
    img_blurred = zoom(img_reduced, (depth / new_d,
                       height / new_h, width / new_w), order=0)
    return img_blurred


def reduce_bits(volume, mag) -> np.ndarray:
    """
    Reduce the number of bits used to represent the volume.

    Parameters:
    - volume: np.ndarray, the input volume.
    - mag: int, the number of bits to keep.

    Returns:
    - reduced_bits_volume: np.ndarray, the volume after reducing the bits.
    """
    mag = 2 ** mag - 1
    reduced_bits_volume = np.floor(volume * mag) / mag
    return reduced_bits_volume


def _get_edge(image, dilation_size, blur_radius) -> np.ndarray:
    """
    Apply edge detection to the image using Canny edge detector.
    Then dilate the edges to create a halo effect.
    Finally, apply Gaussian blur to the dilated edges.

    Parameters:
    - image: np.ndarray, the input image.
    - dilation_size: int, the size of the dilation kernel.
    - blur_radius: int, the radius of the Gaussian blur.

    Returns:
    - blurred_edges: np.ndarray, the blurred edges of the image.
    """
    # Scale image to 0-255 for Canny edge detection
    image_scaled = (image * 255).astype(np.uint8)

    # Detect edges using Canny edge detector
    edges = cv2.Canny(image_scaled, 100, 200)

    # Dilate the edges to make the halo effect more pronounced
    kernel = np.ones((dilation_size, dilation_size), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Apply Gaussian blur to the dilated edges
    blurred_edges = cv2.GaussianBlur(
        dilated_edges, (blur_radius, blur_radius), 0)

    # Normalize the blurred edges to 0-1 range
    blurred_edges = blurred_edges.astype(np.float32) / 255.0
    return blurred_edges


def halo_noise(volume, mag) -> np.array:
    """
    Simulate halo noise for the microscipic volume.
    Halo noise is a common artifact in microscopy images where
    the edges of objects appear brighter than the center due to the 
    nature of scattering effect.
    To simulate halo noise, we can apply edge detection to the volume
    and create a halo effect by blending the original image with the
    blurred edges.

    Parameters:
    - volume: np.ndarray, the input volume.
    - mag: float, the magnitude of the halo noise.

    Returns:
    - halo_volume: np.ndarray, the volume after halo noise.
    """
    blur_radius = 25
    depth, height, width = volume.shape
    halo_volume = np.zeros_like(volume, dtype=np.float32)

    for z in range(depth):
        image = volume[z, :, :]
        blurred_edges = _get_edge(image, 5, blur_radius)

        # Create halo effect by blending the original image with the blurred edges
        halo_slice = image + mag * blurred_edges * (1 - image)
        halo_slice = np.clip(halo_slice, 0, 1)

        halo_volume[z, :, :] = halo_slice

    return halo_volume


def _add_diagonal_noise(image, frequency=0.1, amplitude=0.2) -> np.ndarray:
    """
    Add diagonal noise to the image.
    Diagonal noise is a type of noise that appears as diagonal lines
    in the image. It is often caused by interference patterns in the
    imaging system.
    To simulate diagonal noise, we can add a sinusoidal pattern to the image.

    Parameters:
    - image: np.ndarray, the input image.
    - frequency: float, the frequency of the diagonal noise.
    - amplitude: float, the amplitude of the diagonal noise.

    Returns:
    - noisy_image: np.ndarray, the image after adding diagonal noise.
    """
    height, width = image.shape
    y, x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    diagonal_noise = amplitude * np.sin(2 * np.pi * frequency * (x + y))

    noisy_image = image + diagonal_noise
    noisy_image = np.clip(noisy_image, 0, 1)

    return noisy_image


def coherent_noise(volume, mag) -> np.ndarray:
    """
    Simulate coherent noise for the microscipic volume.
    Coherent noise is a type of noise that appears as patterns or
    structures in the image. It is often caused by interference or
    diffraction effects in the imaging system.
    To simulate coherent noise, we can add diagonal noise to each
    image in the volume. Then, we can blend the noisy image with
    the original image using edge detection since coherent noise
    tends to appear less in the inner regions of objects.

    Parameters:
    - volume: np.ndarray, the input volume.
    - mag: float, the magnitude of the coherent noise.

    Returns:
    - noisy_volume: np.ndarray, the volume after coherent noise.
    """
    frequency = 0.10
    depth, height, width = volume.shape
    noisy_volume = np.zeros_like(volume, dtype=np.float32)

    for z in range(depth):
        image = volume[z, :, :]
        noisy_image = _add_diagonal_noise(image, frequency, mag)
        blurred_edges = _get_edge(image, 5, 35)

        noisy_volume[z, :, :] = noisy_image * \
            (1-blurred_edges) + image * blurred_edges

    return noisy_volume


def waterdrop_noise(volume, mag=0.5) -> np.ndarray:
    """
    Simulate waterdrop noise for the microscipic volume.
    Waterdrop noise is a type of noise that appears as circular
    artifacts in the image. It is often caused by dust particles
    or water droplets on the lens or sensor of the imaging system.
    To simulate waterdrop noise, we can create circular masks at
    random positions in the volume and add intensity to the pixels
    inside the mask.

    Parameters:
    - volume: np.ndarray, the input volume.
    - mag: float, the magnitude of the waterdrop noise.

    Returns:
    - noisy_volume: np.ndarray, the volume after waterdrop noise.
    """
    offset = 30
    depth, height, width = volume.shape
    drop_count = 1
    drop_radius = 15
    noisy_volume = volume.copy()

    for _ in range(drop_count):
        center_x = np.random.randint(offset, width - offset)
        center_y = np.random.randint(offset, height - offset)
        for z in range(depth):
            image = volume[z, :, :]

            # Create circular mask
            mask = np.zeros((height, width), dtype=np.float32)
            cv2.circle(mask, (center_x, center_y), drop_radius, 1, -1)

            # Apply Gaussian blur to the mask several times
            for _ in range(3):
                mask = cv2.GaussianBlur(
                    mask, (drop_radius*2+1, drop_radius*2+1), 0)

            # Add noise
            noisy_volume[z, :, :] += mag * mask
            noisy_volume[z, :, :] = np.clip(noisy_volume[z, :, :], 0, 1)

    return noisy_volume


def ripple_noise(volume, mag=1) -> np.ndarray:
    """
    Simulate ripple noise for the microscipic volume.
    Ripple noise is a type of noise that appears as circular
    patterns in the image. It is often caused by vibrations or
    mechanical disturbances in the imaging system.
    To simulate ripple noise, we can apply a sinusoidal pattern
    to each x-y plane of the volume with random centers.

    Parameters:
    - volume: np.ndarray, the input volume.
    - mag: float, the magnitude of the ripple noise.

    Returns:
    - noisy_volume: np.ndarray, the volume after ripple noise.
    """
    volume = volume.copy()
    ripple_frequency = 50
    depth, height, width = volume.shape
    x = np.linspace(-2, 2, width)
    y = np.linspace(-2, 2, height)
    x, y = np.meshgrid(x, y)

    # Generate random centers for each plane
    centers_x = np.random.uniform(-1, 1)
    centers_y = np.random.uniform(-1, 1)

    # Apply the ripple pattern to each x-y plane with random centers
    for z in range(depth):
        r = np.sqrt((x - centers_x)**2 + (y - centers_y)**2)
        ripple = mag * (np.sin(ripple_frequency * r) / r)
        volume[z, :, :] += ripple

    return volume


def elastic_deform(volume, mag) -> np.ndarray:
    """
    Simulate elastic deformation for the microscipic volume.
    Elastic deformation is a type of distortion that occurs when
    the sample is compressed or stretched during imaging.
    To simulate elastic deformation, we can apply a random displacement
    field to the volume.

    Parameters:
    - volume: np.ndarray, the input volume.
    - mag: float, the magnitude of the elastic deformation.

    Returns:
    - ret_x: np.ndarray, the volume after elastic deformation.
    """
    alpha = mag
    sigma = 10

    shape = volume.shape
    random_state = np.random.RandomState(None)

    dx = (
        gaussian_filter(
            (random_state.rand(shape[1], shape[2]) * 2 - 1),
            sigma,
            mode="constant",
            cval=0,
        )
        * alpha
    )
    dy = (
        gaussian_filter(
            (random_state.rand(shape[1], shape[2]) * 2 - 1),
            sigma,
            mode="constant",
            cval=0,
        )
        * alpha
    )

    _x, _y = np.meshgrid(
        np.arange(shape[1]), np.arange(shape[2]), indexing="ij")
    indices = np.reshape(_x + dx, (-1, 1)), np.reshape(_y + dy, (-1, 1))

    ret_x = np.zeros(shape)
    for i in range(shape[0]):
        ret_x[i] = map_coordinates(volume[i], indices, order=1, mode="reflect").reshape(
            shape[1:]
        )
    return ret_x


def jpeg_compression(volume, quality=75) -> np.ndarray:
    """
    Apply JPEG compression to each x-y plane in the 3D volume.

    Parameters:
    - volume: 3D numpy array, the input volume to be compressed.
    - quality: int, the quality of the JPEG compression (1-95).

    Returns:
    - compressed_volume: 3D numpy array, the volume after JPEG compression.
    """
    depth, height, width = volume.shape
    compressed_volume = np.zeros_like(volume, dtype=np.float32)

    for z in range(depth):
        # Convert the slice to an image
        img = Image.fromarray((volume[z, :, :] * 255).astype(np.uint8))

        # Compress and decompress the image using JPEG
        with io.BytesIO() as buffer:
            img.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            compressed_img = Image.open(buffer)
            # Normalize back to [0, 1]
            compressed_img = np.array(compressed_img) / 255.0

        # Store the compressed slice back in the volume
        compressed_volume[z, :, :] = compressed_img

    return compressed_volume


def get_noise_dict():
    """
    Get the dictionary of noise functions.

    Returns:
    - noise_dict: dict, the dictionary of noise functions.
    """
    noise_dict = {
        "gaussian_noise": gaussian_noise,
        "poisson_noise": poisson_noise,
        "speckle_noise": speckle_noise,
        "defocus": defocus,
        "spherical_aberration": spherical_aberration,
        # "zoom_blur": zoom_blur,
        "low_numerical_aperture": low_numerical_aperture,
        "pixelate": pixelate,
        "reduce_bits": reduce_bits,
        "halo_noise": halo_noise,
        "coherent_noise": coherent_noise,
        "waterdrop_noise": waterdrop_noise,
        "ripple_noise": ripple_noise,
        "brightness": brightness,
        "constrast": constrast,
        "elastic_deform": elastic_deform,
        "jpeg_compression": jpeg_compression,
    }
    return noise_dict


def make_noise(volume, noise_type, severity):
    """
    Apply the specified noise type to the volume.

    Parameters:
    - volume: np.ndarray, the input volume.
    - noise_type: str, the type of noise to apply.
    - severity: int, the severity of the noise. (0-4)

    Returns:
    - noised_volume: np.ndarray, the volume after applying the noise.
    """
    magitude_dict = {
        "gaussian_noise": [0.002, 0.003, 0.004, 0.006, 0.02],
        "poisson_noise": [1024, 512, 128, 64, 32],
        "speckle_noise": [0.05, 0.3, 0.4, 0.5, 0.6],
        "defocus": [0.2, 0.4, 0.45, 0.5, 0.7],
        "spherical_aberration": [1.08, 1.5, 2, 3.5, 14],
        # "zoom_blur": [2, 20, 40, 80, 100],
        "low_numerical_aperture": [60, 50, 40, 30, 20],
        "pixelate": [0.9, 0.8, 0.7, 0.6, 0.5],
        "reduce_bits": [12, 11, 10, 9, 8],
        "halo_noise": [0.01, 0.04, 0.06, 0.1, 0.3],
        "coherent_noise": [0.01, 0.04, 0.07, 0.1, 0.15],
        "waterdrop_noise": [0.05, 0.1, 0.2, 0.3, 0.4],
        "ripple_noise": [0.001, 0.003, 0.005, 0.007, 0.01],
        "brightness": [1.1, 0.9, 1.2, 0.8, 0.7],
        "constrast": [1.02, 0.98, 1.05, 0.95, 0.75],
        "elastic_deform": [5, 10, 40, 120, 200],
        "jpeg_compression": [100, 70, 50, 30, 10],
    }

    noise_fn = get_noise_dict()[noise_type]
    mag = magitude_dict[noise_type][severity]
    noised_volume = noise_fn(volume, mag)
    return noised_volume
