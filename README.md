# Separating Fetal Heartbeats from Maternal ECG

## Project Overview
This project focuses on the development of a robust signal processing approach to effectively isolate fetal ECG (fECG) signals from maternal ECG (mECG) signals, obtained from publicly available datasets. The methodology involves multiple stages, including data acquisition, preprocessing, maternal ECG identification, fetal ECG extraction, time-frequency analysis, and fetal ECG enhancement.

## Abstract
The separation of fetal heartbeats from maternal ECG signals presents a critical challenge in prenatal monitoring and diagnostics. This project utilizes various digital signal processing techniques to isolate fetal ECG from the combined maternal-fetal ECG signals. Key stages of the methodology include:
- Data acquisition
- Preprocessing (filtering)
- Maternal ECG identification using R-peak detection
- Fetal ECG extraction through adaptive filtering (LMS algorithm)
- Time-frequency analysis
- Fetal ECG enhancement
- Validation through heart rate calculations

## Objective
The main objectives of this project are:
1. Acquire high-resolution ECG signals (200-1000 Hz).
2. Preprocess the signals to remove noise.
3. Identify maternal ECG using R-peak detection.
4. Extract fetal ECG through adaptive filtering (LMS algorithm).
5. Conduct time-frequency analysis (FFT, DFT, STFT).
6. Handle signal overlap with windowing and overlap-add methods.
7. Enhance fetal ECG clarity using wavelet transforms or averaging.
8. Validate the results via heart rate calculations.

## Literature Survey
This section reviews techniques essential for processing ECG signals in maternal-fetal monitoring. Key methods include:
- High-resolution data acquisition
- Preprocessing techniques (low-pass, high-pass, notch filtering)
- R-peak detection for maternal ECG identification
- Adaptive filtering for fetal ECG extraction
- Time-frequency analysis methods (FFT, STFT)

## Methodology
1. **Data Acquisition**: Obtain ECG signals from publicly available online datasets.
2. **Preprocessing**: Apply filtering techniques (low-pass, high-pass, notch filters) to clean the signals.
3. **Maternal ECG Identification**: Detect R-peaks using `scipy.signal.find_peaks` to identify maternal heart rate.
4. **Fetal ECG Extraction**: Use the LMS algorithm to subtract maternal ECG from the composite signal.
5. **Time-Frequency Analysis**: Analyze the frequency components using FFT and STFT for fetal ECG.
6. **Post-Processing**: Calculate heart rates and generate visualizations of the results.

## Expected Outcomes
- Accurate estimation of maternal and fetal heart rates.
- Clear visualizations of raw and processed ECG signals.
- Successful separation of fetal ECG from maternal signals.
- Generation of audio files representing the ECG signals.
- Robust code handling potential issues in ECG data.

## Software/Hardware Requirements
### Software:
- Python 3.x
- Libraries: `numpy`, `scipy`, `PyQt5`, `pyqtgraph`, `pyedflib`, `sounddevice`, `fpdf`
- Access to PhysioNet Database
  Download Database here:
  https://physionet.org/content/nifecgdb/1.0.0/

### Hardware:
- A computer with adequate processing power to run signal processing algorithms.

## References
- "A Review on Maternal and Fetal ECG Signal Processing" by Sushmitha Vinod, Swarnalatha R [Link](https://www.researchgate.net/publication/335500003_A_Review_of_Signal_Processing_Techniques_for_Non-Invasive_Fetal_Electrocardiography)
- Non-Invasive Fetal ECG Database [Link](https://physionet.org/content/nifecgdb/1.0.0/)
- Fetal ECG Extraction Using Adaptive Filters [Link](https://www.researchgate.net/publication/277565024_Fetal_ECG_Extraction_Using_Adaptive_Filters)
- Adaptive filtering in ECG monitoring of the fetal heart rate [Link](https://pubmed.ncbi.nlm.nih.gov/3694088/)

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For questions or feedback, please reach out to:
- Ezhilkirthik M - 2nd Year ECE
Amrita Vishwa Vidyappeetham, Coimbatore.
- Email: ezhilkirthikm@gmail.com

---

This README provides a clear overview of your project, making it accessible for other developers and users who may want to understand, use, or contribute to your work. Feel free to modify any section according to your needs or add more specific details as necessary!
