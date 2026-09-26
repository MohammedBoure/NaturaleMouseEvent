# Mouse Dataset Directory

This directory contains preprocessed trajectory datasets used for training the AI Natural Human Mouse models.

## Files

- **`mouse_dataset_fixed_N256_full.npz`**: Compressed NumPy archive containing 1,216 recorded human mouse movement episodes (175,722 valid steps, padded to a maximum sequence length of $N = 256$).
  - `start_positions`: `(1216, 2)` float32 — Normalized screen coordinates $(x_0, y_0)$ at trajectory start.
  - `target_positions`: `(1216, 2)` float32 — Normalized screen coordinates $(x_{\text{tgt}}, y_{\text{tgt}})$ at target destination.
  - `previous_contexts`: `(1216, 4)` float32 — 4D momentum context vector $[v_x, v_y, \text{click\_density}, \text{avg\_dt\_scaled}]$.
  - `intents`: `(1216, 1)` float32 — Binary flag (1.0 = click target, 0.0 = idle movement/wandering).
  - `seq_tensors`: `(1216, 256, 6)` float32 — Step features $[\Delta x, \Delta y, \Delta t, rem_x, rem_y, \text{action\_code}]$.
  - `padding_masks`: `(1216, 256)` float32 — Binary mask (1.0 = valid recorded step, 0.0 = padded).
