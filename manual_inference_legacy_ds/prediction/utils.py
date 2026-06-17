from __future__ import annotations


def extract_filtered_input_from_output(
    input_weather_states,
    input_name_to_index,
    output_name_to_index,
):
    """Project the input weather-state tensor onto the output channel ordering.

    For each output channel:
      - if an input channel has the same name, copy its values to the output slot;
      - otherwise (output-only variable, e.g. direct-prediction `tp` when tp is
        absent from input), leave the output slot at zero.

    The returned tensor always has its last axis sized to ``len(output_name_to_index)``,
    matching the output ordering, so downstream `arr[..., selected_indices]` calls
    keyed off the output index space stay in-bounds even when the input/output
    variable sets do not overlap.
    """
    import numpy as np

    n_out = len(output_name_to_index)
    out_shape = input_weather_states.shape[:-1] + (n_out,)
    filtered = np.zeros(out_shape, dtype=input_weather_states.dtype)
    for name, out_idx in output_name_to_index.items():
        if name in input_name_to_index:
            filtered[..., out_idx] = input_weather_states[..., input_name_to_index[name]]
    return filtered, dict(output_name_to_index)
