import numpy as np


def create_load_bc(start_position, l, w, m, P):
    x_values = np.linspace(start_position[0], start_position[0] + l, m + 1)
    y_value = start_position[1] - w
    boundary_conditions = {(x, y_value): ((np.nan, np.nan), (0, P * l / (m + 1))) for x in x_values}
    return boundary_conditions


def apply_bc(stiffness, displacement, forces):
    # Make copies of inputs to avoid modifying original arrays
    d, f, k = np.copy(displacement), np.copy(forces), np.copy(stiffness)

    # Replace NaN values in displacement with zeros
    d[np.isnan(d)] = 0

    # Loop over displacement vector
    for i, value in enumerate(d):
        if value != 0:
            # Apply boundary condition by zeroing out corresponding row and column
            k[i, :] = 0
            k[:, i] = 0
            k[i, i] = 1
            # Update f vector with fixed displacement value
            f[i] = value
            f[d == 0] -= stiffness[d == 0, i] * value

    # Zero out rows and columns corresponding to NaN values in f
    new_k = np.copy(k)
    new_k[np.isnan(f)] = 0
    new_k[:, np.isnan(f)] = 0
    new_k[np.isnan(f), np.isnan(f)] = 1
    f[np.isnan(f)] = 0
    S = np.linalg.inv(new_k)
    # Solve for unknown displacements using matrix inversion
    return np.linalg.inv(new_k) @ f
