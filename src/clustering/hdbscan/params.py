from sklearn.model_selection import ParameterGrid

def _make_unique_int_values(values):
    values = [int(round(v)) for v in values]
    return sorted(set(values))

def make_umap_hdbscan_grid_params(
        n_neighbors_values: list[int],
        min_samples_values: list[int]
):
    n_neighbors_values = _make_unique_int_values(n_neighbors_values)
    min_samples_values = _make_unique_int_values(min_samples_values)

    param_grid = {
        "n_neighbors": n_neighbors_values,
        "min_samples": min_samples_values
    }

    grid = list(ParameterGrid(param_grid))

    return grid