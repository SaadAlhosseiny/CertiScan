# CertiScan: Digital Image Forensics Tool

This project focuses on detecting digital forgeries in documents and images using advanced signal processing and error analysis.

## Key Features:
* **FFT (Fast Fourier Transform):** Detects periodic noise and high-frequency discrepancies often left by digital editing tools.
* **ELA (Error Level Analysis):** Highlights areas with different compression levels, indicating potential tampering.
* **Noise Analysis:** Identifies inconsistencies in the image's natural noise pattern.
* **Masking Module:** Allows for targeted forensic analysis on specific image regions.

## How it works:
1. Run `RUN.py` and provide the image path.
2. The system generates analysis maps (FFT, ELA, Noise).
3. A final verdict is provided based on the calculated forensic scores.
