FITS model — explanation and what we plot
=======================================

This note explains what the FITS model implementation in `FITS/models/FITS.py` does, what the notebook is plotting (the variables `full_np`, `xy_out`, `y_pred`, `low_np`, `res_np`) and why the model's tail/prediction may look different from the input signal at the boundary.

Summary (short)
----------------
- FITS performs a frequency-domain low-pass + frequency upsampling pipeline to produce a length-expanded reconstruction of the input time series (it currently only returns the low-frequency reconstruction; the higher-frequency "dominant" branch is commented out in the provided code).
- The notebook visualizes: the original full series (`full_np`), the model's full reconstructed output `xy_out`, the tail prediction `y_pred` (the last pred_len samples of `xy_out`), the low-frequency reconstruction on the input window (`low_np`) and the residual (`res_np`) defined as input minus low-frequency reconstruction.
- If the model is configured to keep only a subset of frequency bins (a low cut-off), the residual will contain the high-frequency information the model didn't model; the predicted tail therefore represents a low-frequency extrapolation and may not include high-frequency detail present in the true future.

Step-by-step: what FITS.Model does
----------------------------------
The forward pass in `FITS/models/FITS.py` follows these main steps (variable names refer to the code):

1. RIN normalization
   - x_mean = mean(x, dim=1, keepdim=True)
   - x_var = var(x, dim=1, keepdim=True) + 1e-5
   - x_norm = (x - x_mean) / sqrt(x_var)
   - This puts each example on a mean-zero, unit-variance scale (per-example, per-channel). The model works in this normalized domain and the final output is denormalized back with sqrt(x_var) and x_mean.

2. Short-time frequency transform (real FFT)
   - low_specx = rfft(x_norm, dim=1)
   - The code then zeroes all bins above `dominance_freq`:
       low_specx[:, dominance_freq:] = 0
     and keeps only the first `dominance_freq` frequency bins:
       low_specx = low_specx[:, 0:dominance_freq, :]
   - This is a low-pass in the frequency domain (keeps only a few low-frequency bins).

3. Frequency-domain upsampling / interpolation
   - The remaining (low) frequency bins are mapped to a larger number of frequency bins (to correspond to the longer output length seq_len + pred_len). This is done by a linear layer defined as `freq_upsampler`. The implementation supports an "individual" mode (per-channel) or a shared linear mapping.
   - After this linear mapping the code zero-pads the frequency array to the length required by an inverse rFFT for the target output length and calls irfft to obtain a time-domain signal of length seq_len + pred_len.

4. Energy compensation and denormalization
   - low_xy = irfft(padded_freqs, dim=1)
   - low_xy = low_xy * length_ratio   # compensate for ener	gy when stretching length
   - xy = low_xy * sqrt(x_var) + x_mean   # reverse the RIN normalization

5. Output
   - The forward method returns two tensors:
       xy: the full length-expanded reconstruction (shape: [B, seq_len + pred_len, channels])
       low_component: the low-frequency part (same as low_xy after denorm; also length-expanded)
   - Note: the implementation has a commented-out path for a separate DLinear/dom component (the residual/high-frequency predictor). In the current code that component is not used; the returned `xy` comes solely from the low-frequency reconstruction.

Shapes and indexing
-------------------
- Input `x` has shape [B, seq_len, C].
- `xy` and `low_component` have shape [B, seq_len + pred_len, C].
- In the notebook, we typically extract the prediction tail as
    y_pred = xy[:, seq_len:, :]
  which is the last `pred_len` samples produced by the model.

What the notebook plots (variable definitions)
---------------------------------------------
- `full_np` — the original full time series you created for the demo (shape [seq_len + pred_len]). This is the ground-truth series (input window + true future).
- `xy_np` — the model's full output `xy_out` converted to a numpy 1-D array of length seq_len + pred_len. This is the low-frequency reconstruction extrapolated into the future.
- `pred_np` — the prediction tail `y_pred` (length pred_len), plotted on the future portion of the timeline.
- `low_np` — the low-frequency reconstruction prefix that corresponds to the input window; in the notebook it is computed as the prefix of `low_component` (length seq_len):
    low_np = low_component[:, :seq_len, channel]
  This is the model's reconstruction of the part of the input that comes from the kept low-frequency bins. If `dominance_freq` is set high enough (close to the full FFT length), `low_np` may be very close to the original `input_x`.
- `res_np` — the residual computed as
    res_np = input_x - low_component_input
  This is the part of the input that the low-pass discarded — i.e., the higher-frequency content (sharp fluctuations) that the FITS low-frequency branch did not model.

Why `low_np` + `res_np` reconstruct the input
----------------------------------------------
Because we split the input spectrum into "kept" (low_specx) and "discarded" (everything above dominance_freq) bins, the discarded portion is the residual. Concretely,

  input_x ≈ low_component_input + residual

If the model's low-frequency cutoff keeps most of the spectrum, the residual will be small and low_component_input will closely match input_x. If the cutoff is low, residual will be large and capture the fine-scale details.

Why the prediction tail or boundary may not "match" the input
-----------------------------------------------------------
Several effects make the model's full output (and especially its predicted tail) differ from the original input near the boundary:

1. The model predicts only the low-frequency branch
   - In this implementation `xy` is produced from the low-frequency branch only; the high-frequency/dominant branch is commented out. So the model cannot reproduce or extrapolate high-frequency detail (the residual). The predicted tail therefore contains only the smoothed, low-frequency continuation of the signal (good for trends and slow seasonalities, bad for sharp spikes).

2. Cutoff / frequency coverage
   - If `dominance_freq` is set smaller than the full spectrum, the model discards higher FFT bins. Those discarded bins contain the high-frequency part of the input. The prediction will therefore miss those components and look smoother than the true future.

3. FFT edge / phase effects and upsampling interpolation
   - The model creates future samples by linearly mapping a small set of frequency bins to a larger set (frequency upsampling). This mapping and the zero-padding used before irfft can alter phases and amplitudes slightly. Edge artifacts are common when moving between time and frequency domains without special windowing.

4. Energy compensation (length_ratio)
   - The code multiplies the time-domain `low_xy` by a `length_ratio` to compensate energy when changing length. That scaling is a crude correction and can change the amplitude shape near the boundary relative to the original.

5. RIN normalization — statistics computed on the input only
   - The per-example mean and variance are computed on the input `x` window and used to denormalize the full length-expanded output. If the true future has a different variance/mean behaviour, the denormalized extrapolation can still differ from the real future.

6. Missing high-frequency modeling and continuity
   - Even if the low-frequency reconstruction matches the input window well, the model's tail is only the low-frequency continuation; the high-frequency residual that made the input's final few samples look a certain way is not extended. That creates an apparent mismatch at the boundary: the predicted tail continues the smooth underlying shape, but the actual future contains small-scale deviations not captured by the low-frequency-only model.

How to validate and debug this in the notebook
----------------------------------------------
- Check the maximum absolute difference between `low_component_input` and `input_x` (reconstruction error for the input window). If it's near zero, the low-frequency branch captured almost all the energy in the input (dominance_freq is near full spectrum).
- Inspect the residual `res_np` (plot it). Large amplitude residuals near the end of the input explain why the low-frequency extrapolation fails to reproduce the true future's fine structure.
- Plot the Fourier magnitudes of the input and of `low_component_input` (or `low_specx`) to see which frequencies were preserved and which were discarded. This makes the cutoff effect explicit.
- Try larger `dominance_freq` values and re-run: more bins preserved → smaller residuals → predicted tail will better match the true future (but at the cost of more expensive frequency mapping and perhaps overfitting).
- The original implementation hints at a second branch (commented `dom_xy = self.Dlinear(dom_x)`) where a separate module would predict the dominant/high-frequency residuals; integrating that path would improve tail fidelity.

Practical recommendations
-------------------------
- If you want to see the low-frequency behaviour only (trend/seasonality), the current setup is fine: the tail will be a smooth extrapolation of those components.
- If you want the model to match the true future more closely (including high-frequency detail):
  - increase `cfg.cut_freq` (dominance_freq) so the low branch keeps more bins, or
  - enable / implement the commented dominant/residual branch (a DLinear or similar predictor on the residual), or
  - combine both: preserve more frequencies and add a residual predictor.

Closing note
------------
The FITS implementation you have is a concise, frequency-domain way to extrapolate the smooth (low-frequency) part of a time series. The notebook's `low_np` is exactly that low-frequency reconstruction on the input window; `res_np` is the part the model discarded. The prediction `y_pred` is a forecast of the low-frequency continuation, so mismatches at the boundary are expected when the high-frequency residual matters for the short-term structure.

If you'd like, I can:
- add a short notebook cell that plots the magnitude spectra of the input and low-frequency reconstruction, or
- add a small DLinear residual predictor into the pipeline (uncomment and wire the `DLinear` component) and retrain to produce a full reconstruction that includes high-frequency detail.

--
Generated explanation for `FITS/models/FITS.py` on request.
