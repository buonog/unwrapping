#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 22:24:33 2025

@author: xap
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

from src.unwrapping_algo import UnwrappingAlgo


class PLLFFTAlgo(UnwrappingAlgo):
    """Piece‑wise Linear Phase‑Locked‑Loop unwrapping via FFT (PLL‑FFT).

    The algorithm breaks the wrapped series into overlapping windows, estimates the
    dominant linear trend (slope) for each window via the FFT of *exp(i·2π·w)*, and
    progressively unwraps the signal by fitting piece‑wise lines.

    Parameters expected in *unwrap_param*:
    --------------------------------------
    fft_len   : int   – window length *ℓ* for the FFT (mandatory)
    step      : int   – hop size *d* (defaults to *fft_len*)
    pad_mode  : str   – how to pad the last window ("edge" | "zero", default "edge")

    Returns
    -------
    dict with keys
        'm' : np.ndarray – piece‑wise linear model evaluated on *timeline*
        'u' : np.ndarray – unwrapped signal
    """

    # ---------------------------------------------------------------------
    # public API
    # ---------------------------------------------------------------------
    def unwrap(self, w: np.ndarray, unwrap_param: Dict) -> Dict:
        fft_len: int = unwrap_param.get("fft_len")
        if fft_len is None:
            raise ValueError("unwrap_param must contain the key 'fft_len'.")
        step: int = unwrap_param.get("step")
        pad_mode: str = unwrap_param.get("pad_mode", "edge")

        N = len(w)
        self.ts_length = N  # keep consistent with the base class behaviour
        if self.timeline is None:
            self.timeline = np.arange(N)

        # output containers
        u_out = np.full(N, np.nan, dtype=float)
        m_out = np.zeros(N, dtype=float)
        kd_out = np.zeros(N, dtype=int)  # optional – useful for debugging/plotting

        # ---- main loop ---------------------------------------------------
        ptr = 0
        while ptr < N:
            # 1) grab window (with padding if needed)
            win_end = min(ptr + fft_len, N)
            segment = w[ptr:win_end]
            seg_len = len(segment)
            if seg_len < fft_len:
                if pad_mode == "edge" and seg_len > 0:
                    pad_value = segment[-1]
                    segment = np.concatenate([segment, np.full(fft_len - seg_len, pad_value)])
                else:  # "zero" (or fallback)
                    segment = np.pad(segment, (0, fft_len - seg_len), mode="constant")

            # 2) estimate slope via FFT
            b_slope = self._estimate_slope_fft(segment, fft_len)

            # 3) set intercept so that the new line passes through the
            #    previously unwrapped sample (ptr‑1) – or origin for the first window
            if ptr == 0:
                intercept = 0.0
            else:
                intercept = u_out[ptr - 1] - b_slope * (ptr - 1)

            # 4) build the reference line for the *unpadded* portion
            n_global = np.arange(ptr, win_end)
            line_vals = intercept + b_slope * n_global

            # 5) unwrap current segment
            kd_local = np.round(w[ptr:win_end] - line_vals).astype(int)
            u_local = w[ptr:win_end] - kd_local

            # 6) decide how many samples to commit to the global output
            write_len = min(step, win_end - ptr)
            sl = slice(ptr, ptr + write_len)
            u_out[sl] = u_local[:write_len]
            m_out[sl] = line_vals[:write_len]
            kd_out[sl] = kd_local[:write_len]

            # 7) advance cursor
            ptr += write_len

        return {"m": m_out, "u": u_out}

    # ------------------------------------------------------------------
    # utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _estimate_slope_fft(segment: np.ndarray, fft_len: int) -> float:
        """Estimate the fractional slope *b* (cycles per sample) that
        best fits *segment* by locating the dominant frequency bin in the FFT of
        exp(i·2π·segment).

        The returned value is wrapped to the interval (‑0.5, 0.5]."""
        y = np.exp(1j * 2 * np.pi * segment)
        fft_vals = np.fft.fft(y, fft_len)
        mags = np.abs(fft_vals)
        k_max: int = int(np.argmax(mags))
        # map to signed fractional frequency
        if k_max > fft_len // 2:
            k_max -= fft_len
        return k_max / fft_len

    # ------------------------------------------------------------------
    # plotting helper (optional)
    # ------------------------------------------------------------------
    def unwrap_plot(self, w: np.ndarray, u: np.ndarray, m: np.ndarray, kd: np.ndarray):
        fig, axs = plt.subplots(2, 1, figsize=(6, 5))
        axs[0].plot(self.timeline, w, "r.", alpha=0.6, label="Wrapped phase")
        axs[0].plot(self.timeline, u, "b.", alpha=0.9, label="Unwrapped phase")
        axs[0].plot(self.timeline, m, "-", linewidth=2.5, color="orange", label="Piecewise model")
        axs[1].plot(self.timeline, kd, "g.", alpha=0.9, label="kd")
        for ax in axs:
            ax.grid(True)
            ax.legend()
        axs[1].set_xlabel("Time index")
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------
# example usage (stand‑alone test) – remove in production
# ---------------------------------------------------------------------
if __name__ == "__main__":
    from datetime import datetime
    from pathlib import Path
    from src.ts_collection import TSCollection, TSSubset
    from src.ts_packets import Unwrapping
    
    
    starttime = datetime.now()
    # generaldata_folder = "/mnt/DATI_PC/AA1_PROGETTI/PS_DATA/Real/"
    # Go three level up respect to current level
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    generaldata_folder = str(base_dir / "PS_DATA" / "Real") + '/'    
    collection_folder = "Toscana_2/"
    collection_file = "TOSCANA_ps.tsc"

    tscollection = TSCollection()
    tscollection.load(generaldata_folder + collection_folder + collection_file)

    # ts_number_list = [22303]
    ts_number_list = [16729]

    
    # ts_number_list = [77, 11453, 26071, 17560, 34348, 35205, 35623, 38047, 56539, 33646, 16476, 16729]

    collection_subset = TSSubset(tscollection.get_collection_dict(), ts_number_list)
    starttime1 = datetime.now()


    # # ############ GENERATE NEW CPLEXmean_UNWRAPPING ++++++++++++++++++++++++++++
    # # # # 1. Define the unwrapping object 2. Create new Unwrapping data
    # unwrap_param = {'min_t_index': 0, 'max_t_index': 200, 'max_slope': 16, "n_cpu": 4}
    unwrap_param = {"fft_len": 64, "step": 2, "pad_mode": "edge", "n_cpu": 4}
    cplex_unwrapping = Unwrapping(collection_subset) # parameter: ts collection linked to this unwrapping 
    cplex_unwrapping.new(unwrapping_name = 'cplexmean_test_subset',
                             unwrapping_algo = 'PLLFFT_unwrap', unwrap_param = unwrap_param,
                             unwrapping_note = "used an integer slope")

    print("Time PLLFFT_unwrap : ", datetime.now()- starttime)
    
    
    # # cplex_unwrapping.save()

    # # # # ############ PLOT UNWRAPPING ++++++++++++++++++++++++++++
    plt.figure(figsize=(6, 3))
    for ts in ts_number_list:
    # for ts in [56829]:
    # for ts in acc1[acc1 < 70].index:  
        wt = collection_subset.get_data('w').loc[ts]
        ut = collection_subset.get_data('u').loc[ts]
        kd_ref = collection_subset.get_data('kd').loc[ts]
        
        wc = cplex_unwrapping.get_data('w').loc[ts]   
        uc = cplex_unwrapping.get_data('u').loc[ts]
        kd_calc = (np.round(wc - uc)).astype(int)
        

        plt.plot(collection_subset.absolute_timeline, wt, '.', label="Original Series")
        plt.plot(collection_subset.absolute_timeline, ut, 'g.', label="Original Series")
        plt.plot(collection_subset.absolute_timeline, -kd_ref, 'r.', label="Original Series")

        plt.title(f"Reference: {ts}")
        # plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.show()
        
        plt.plot(collection_subset.absolute_timeline, wc, '.', label="Original Series")
        plt.plot(collection_subset.absolute_timeline, uc, '.', label="Original Series")
        # plt.plot(collection_subset.absolute_timeline, cplex_unwrapping.get_data('m').loc[ts], '.', label="Original Series")
        plt.plot(collection_subset.absolute_timeline, -kd_calc, '.', label="Original Series")

        plt.title(f"Unwrapping: {ts}")
        # plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.show()