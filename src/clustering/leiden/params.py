from sklearn.model_selection import ParameterGrid

def _make_unique_int_values(values):
    values = [int(round(v)) for v in values]
    return sorted(set(values))

def make_umap_leiden_grid_params(
        k_neighbors_values: list[float],
        resolution_values: list[float],
):
    k_neighbors_values = _make_unique_int_values(k_neighbors_values)
    resolution_values = _make_unique_int_values(resolution_values)

    param_grid = {
        "k_neighbors": k_neighbors_values,
        "resolution": resolution_values,
    }

    grid = list(ParameterGrid(param_grid))

    return grid