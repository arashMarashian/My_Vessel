import numpy as np
import matplotlib.pyplot as plt

def generate_environment_grid(size=(50, 50), seed=42):
    """Generate a synthetic environmental grid."""
    np.random.seed(seed)
    sea_temp = 15 + 5 * np.random.rand(*size)
    wind_speed = 10 + 2 * np.random.rand(*size)
    wind_angle = 360 * np.random.rand(*size)
    humidity = 50 + 30 * np.random.rand(*size)
    wave_height = 0.5 + 1.5 * np.random.rand(*size)
    return {
        "sea_temp": sea_temp,
        "wind_speed": wind_speed,
        "wind_angle": wind_angle,
        "humidity": humidity,
        "wave_height": wave_height,
    }

def plot_field(field, title="Field"):
    """Plot a single field using matplotlib."""
    plt.imshow(field, origin='lower', cmap='viridis')
    plt.colorbar(label=title)
    plt.title(title)
    plt.show()
