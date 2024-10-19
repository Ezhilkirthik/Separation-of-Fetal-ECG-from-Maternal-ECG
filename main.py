import sys
import numpy as np
import pyaudio
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QSlider, QComboBox, QTextEdit, QGridLayout,
    QGroupBox, QFormLayout, QLineEdit, QCheckBox, QMessageBox, QAction, QToolBar,
    QStatusBar, QInputDialog, QDialog, QDateEdit
)
from scipy import signal
from scipy.signal import find_peaks, butter, filtfilt, iirnotch, stft
from scipy.fft import fft, fftfreq
from PyQt5.QtWidgets import QToolButton, QColorDialog
from PyQt5.QtCore import Qt, QSize, QTimer, QPropertyAnimation, QDate
from PyQt5.QtGui import QIcon, QFont, QLinearGradient,QBrush, QColor, QPalette
from PyQt5.QtMultimedia import QSound
import pyqtgraph as pg
from pyqtgraph.exporters import ImageExporter
import pyedflib
from scipy.io.wavfile import write
from scipy.io import wavfile
from scipy.signal import resample
import sounddevice as sd
import os
from fpdf import FPDF
from datetime import datetime
import pdf_make

global file_path
def load_ecg_data(file_path):
    with pyedflib.EdfReader(file_path) as f:
        n = f.signals_in_file
        signal_labels = f.getSignalLabels()
        global ecg_signal
        ecg_signal = f.readSignal(0)
        sample_rate = f.getSampleFrequency(0)
        total_samples = len(ecg_signal)
        time_axis = np.linspace(0, total_samples / sample_rate, total_samples)
    return time_axis, ecg_signal, sample_rate, signal_labels

def run_adaptive(file_path):
    # Heart Rate Calculation
    def calculate_heart_rate(r_peaks, fs):
        if len(r_peaks) < 2:  # Check if there are enough peaks to calculate heart rate
            return 0
        rr_intervals = np.diff(r_peaks) / fs  # in seconds
        heart_rate = 60 / np.mean(rr_intervals)  # in beats per minute
        return heart_rate

    # Define filter parameters
    def butter_filter(data, cutoff, fs, btype='low', order=4):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype=btype, analog=False)
        y = filtfilt(b, a, data)
        return y

    def notch_filter(data, notch_freq, fs, quality_factor=30.0):
        nyquist = 0.5 * fs
        normal_notch = notch_freq / nyquist
        b, a = iirnotch(normal_notch, quality_factor)
        y = filtfilt(b, a, data)
        return y

    def compute_fft(data_notch, fs, chunk_size):
        # Prepare to store combined frequency and FFT values
        f_combined = []
        fft_values_combined = []

        # Process data in chunks
        for start in range(0, len(data_notch), chunk_size):
            end = min(start + chunk_size, len(data_notch))  # Handle the last chunk
            data_chunk = data_notch[start:end]
            
            N = len(data_chunk)
            if N == 0:  # Skip empty chunks
                continue

            # Compute frequency array and FFT values for the current chunk
            f = fftfreq(N, 1/fs)[:N//2]  # Positive frequency components
            fft_values = np.abs(fft(data_chunk))[:N//2]  # Magnitude of FFT

            # Store the results
            f_combined.append(f)
            fft_values_combined.append(fft_values)

        # Concatenate results from all chunks
        f_combined = np.concatenate(f_combined)
        fft_values_combined = np.concatenate(fft_values_combined)

        return f_combined, fft_values_combined

    # Adaptive Filtering (LMS Algorithm)
    def adaptive_filter(x, d, mu=0.01, order=32):
        N = len(x)
        M = order
        w = np.zeros(M)
        y = np.zeros(N)
        e = np.zeros(N)
        
        if N < M:
            raise ValueError("Length of input signal must be greater than filter order.")
        
        for n in range(M, N):
            x_n = x[n-M:n]  # Segment of length M
            d_n = d[n]
            y[n] = np.dot(w, x_n)
            e[n] = d_n - y[n]
            w = w + 2 * mu * e[n] * x_n

        return y, e

    # Open the EDF file
    with pyedflib.EdfReader(file_path) as f:
        n_channels = f.signals_in_file
        fs = f.getSampleFrequency(0)  # Assuming all signals have the same sample frequency
        data = np.zeros((n_channels, f.getNSamples()[0]))  # Preallocate space for data
        for i in range(n_channels):
            data[i, :] = f.readSignal(i)
        
        # Determine the length to read (first quarter)
        total_samples = data.shape[1]
        quarter_samples = total_samples // 16
        
        # Use only the first quarter of the data
        data = data[:, :quarter_samples]
        time = np.arange(quarter_samples) / fs  # Time in seconds
    
    # Define filter design parameters
    low_cutoff = 45  # Low-pass filter cutoff frequency in Hz
    high_cutoff = 5  # High-pass filter cutoff frequency in Hz
    notch_freq = 50  # For 50 Hz power line interference
    fs = 1000
    # Apply filters to the first channel (example)
    data_lowpass = butter_filter(data[0], low_cutoff, fs, btype='low')
    data_highpass = butter_filter(data[0], high_cutoff, fs, btype='high')
    data_notch = notch_filter(data_highpass, notch_freq, fs)

    # R-peak Detection
    r_peaks, _ = find_peaks(data_notch, height=None, distance=int(0.6 * fs), prominence=0.5)
    r_peaks = np.array(r_peaks, dtype=int)  # Ensure r_peaks is an integer array

    # Create a better maternal ECG reference signal using R-peaks
    global maternal_ecg
    maternal_ecg = np.interp(np.arange(len(data_notch)), r_peaks, data_notch[r_peaks])
    global maternal_file

    # Apply LMS Adaptive Filtering to remove maternal ECG
    global fetal_ecg
    fetal_ecg, residual = adaptive_filter(data_notch, maternal_ecg)
    global fetal_file

    # R-peak Detection for Fetal ECG (after maternal removal)
    fetal_r_peaks, _ = find_peaks(fetal_ecg, height=None, distance=int(0.3 * fs), prominence=0.5)

    # Heart Rate Calculation (Fetal ECG)
    global fetal_heart_rate
    fetal_heart_rate = calculate_heart_rate(fetal_r_peaks, fs)
    print(f'Estimated Fetal Heart Rate: {fetal_heart_rate:.2f} BPM')
    global maternal_heart_rate
    maternal_heart_rate = calculate_heart_rate(r_peaks, fs)
    print(f'Estimated Maternal Heart Rate: {maternal_heart_rate:.2f} BPM')

    # Fetal ECG should have a higher heart rate than maternal
    if fetal_heart_rate > maternal_heart_rate:
        print(f"Fetal heart rate is faster than maternal heart rate, as expected.")
    else:
        print(f"Fetal heart rate is not faster than maternal heart rate. Please recheck.")        

    global fetal_ecg_cleaned
    # Fetal ECG after removing maternal signal
    fetal_ecg_cleaned = data_notch - residual

    # Time-Frequency Analysis
    # Fast Fourier Transform (FFT)
    chunk_size = 1024  # Define your chunk size
    f, fft_values = compute_fft(ecg_signal, fs, chunk_size)
    global mf
    mf = f[np.argmax(fft_values)]
    print(f'Peak frequency for Maternal: {mf:.2f} Hz\n')
    
    chunk_size = 1024  # Define your chunk size
    f, fft_values = compute_fft(data_notch, fs, chunk_size)
    global ff
    ff = f[np.argmax(fft_values)]
    print(f'Peak frequency for fetal: {ff:.2f} Hz\n')

    # Short-Time Fourier Transform (STFT)
    f_stft, t_stft, Zxx = stft(fft_values, fs=fs, nperseg=256)

    return f,r_peaks,maternal_ecg,fetal_ecg,fetal_r_peaks,fetal_heart_rate,maternal_heart_rate,fetal_ecg_cleaned,fft_values,f_stft, t_stft, Zxx,time,data_notch

# Example to play the audio signal
def play_audio(filename, label):
    print(f"Now playing: {label}...")  # Announcement before playing
    fs, data = wavfile.read(filename)
    data = data.astype(np.float32) / 32768.0  # Convert to float32 in the range [-1, 1]
    sd.play(data, fs)
    sd.wait()  # Wait until the sound finishes playing

def amplify_signal(signal, gain=20.0):
    amplified_signal = signal * gain
    # Ensure the signal stays within the range [-1, 1]
    return np.clip(amplified_signal, -1, 1)

def normalize_audio(signal):
        max_amplitude = np.max(np.abs(signal))
        if max_amplitude == 0:
            return signal  # Avoid division by zero
        return signal / max_amplitude

def create_audio_from_ecg(ecg_signal, fs, gain=2.0, target_bpm=None, current_bpm=None, filename="heartbeat.mp3"):
        # Normalize and amplify the ECG signal
        normalized_ecg = normalize_audio(ecg_signal)
        amplified_ecg = amplify_signal(normalized_ecg, gain=gain)
        audio_signal = np.int16(amplified_ecg * 32767)
        
        # Adjust playback speed if target and current BPM are provided
        if target_bpm and current_bpm:
            speed_factor = target_bpm / current_bpm
            audio_signal = resample(audio_signal, int(len(audio_signal) * speed_factor))

        wavfile.write(filename, fs, audio_signal)
        return filename

class ECGApp(QMainWindow):
    # Define filter parameters
    def butter_filter(self,data, cutoff, fs, btype='low', order=4):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype=btype, analog=False)
        y = filtfilt(b, a, data)
        return y

    def notch_filter(self,data, notch_freq, fs, quality_factor=30.0):
        nyquist = 0.5 * fs
        normal_notch = notch_freq / nyquist
        b, a = iirnotch(normal_notch, quality_factor)
        y = filtfilt(b, a, data)
        return y
    
    def detect_r_peaks(self, ecg_signal, distance=50):
        # Simple thresholding method
        threshold = 0.5 * np.max(ecg_signal)  # Set threshold as half the max value
        r_peaks = []
        
        # Find candidates for R-peaks
        for i in range(1, len(ecg_signal) - 1):
            # Check if the current sample is greater than the threshold and greater than its neighbors
            if ecg_signal[i] > threshold and ecg_signal[i] > ecg_signal[i - 1] and ecg_signal[i] > ecg_signal[i + 1]:
                # Append the index of the detected R peak
                r_peaks.append(i)
        
        # Convert to numpy array for easier indexing
        r_peaks = np.array(r_peaks)
        
        # Remove close peaks based on the specified minimum distance
        if len(r_peaks) > 0:
            filtered_r_peaks = [r_peaks[0]]  # Start with the first detected peak
            for peak in r_peaks[1:]:
                # Only add peaks that are at least 'distance' samples apart
                if peak - filtered_r_peaks[-1] > distance:
                    filtered_r_peaks.append(peak)
        
            return np.array(filtered_r_peaks)

        return np.array([])
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fetal ECG Separation From Maternal ECG")
        self.setGeometry(100, 100, 1600, 900)
        self.setWindowIcon(QIcon('icons/ecg_icon.png'))

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.init_toolbar()
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.setTabShape(QTabWidget.Rounded)
        self.main_layout.addWidget(self.tabs)

        self.init_dashboard_tab()
        self.init_visualization_tab()
        self.init_processing_tab()
        
    def create_styled_button(self, text, icon_path, slot):
        button = QPushButton(text)
        button.setIcon(QIcon(icon_path))
        button.setStyleSheet("""
            QPushButton {
                border-radius: 15px; 
                background-color: #4CAF50; 
                color: white; 
                padding: 10px; 
                transition: background-color 0.3s, border 0.3s;
            }
            QPushButton:hover {
                background-color: #45a049;
                border: 1px solid #3e8e41;
            }
            QPushButton:pressed {
                background-color: #3e8e41;
            }
        """)
        button.clicked.connect(slot)
        return button
                
    class SimpleDialog(QDialog):
        def __init__(self):
            super().__init__()
            global name_input,age_input,dob_input
            # Input fields for patient information
            self.name_input = QLineEdit(self)
            self.name_input.setPlaceholderText("Enter Patient Name")

            self.age_input = QLineEdit(self)
            self.age_input.setPlaceholderText("Enter Age")

            self.dob_input = QDateEdit(self)
            self.dob_input.setDate(QDate.currentDate())

            # Submit button to generate the report
            self.submit_button = QPushButton("Generate Report", self)
            self.submit_button.clicked.connect(self.generate_report)

            # Layout for the dialog
            layout = QVBoxLayout(self)
            layout.addWidget(self.name_input)
            layout.addWidget(self.age_input)
            layout.addWidget(self.dob_input)
            layout.addWidget(self.submit_button)
            
        def generate_report(self):
            # Check if patient information is valid
            self.accept()
            if not self.name_input or not self.age_input or not self.dob_input:
                QMessageBox.warning(self, "Invalid Data", "Please fill in all patient information before generating the report.")
                return

            # Show save file dialog
            filename, _ = QFileDialog.getSaveFileName(self, "Save Report", "", "PDF Files (*.pdf)")
            
            if filename:
                # Generate the PDF report with patient info
                pdf_make.generate_pdf_report(filename, self.name_input.text(), self.age_input.text(), self.dob_input.text(),maternal_heart_rate,fetal_heart_rate,mf,ff)

                # Show message that report has been saved
                QMessageBox.information(self, "Report Generated", f"Report saved to {filename}.")

                # Ask the user if they want to open the file
                reply = QMessageBox.question(self, 'Open File', 
                                             f"Do you want to open the file '{filename}'?", 
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

                if reply == QMessageBox.Yes:
                    if os.path.exists(filename):
                        os.startfile(filename)  # Open the file
                    else:
                        QMessageBox.warning(self, 'Error', f"The file '{filename}' does not exist.")
            else:
                QMessageBox.warning(self, "File Name Not Chosen", "Please select the file location.")
    
    def init_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)

        # Load Data Button
        load_icon = QIcon('icons/load.png')
        load_button = QPushButton(load_icon, "Load ECG Data", self)
        load_button.clicked.connect(self.load_data)
        toolbar.addWidget(load_button)

        # Play Audio Button
        play_icon = QIcon('icons/play.png')
        play_button = QPushButton(play_icon, "Play ECG Audio", self)
        play_button.clicked.connect(self.play_audio_placeholder)
        toolbar.addWidget(play_button)

        # Export Report Button
        export_icon = QIcon('icons/export.png')
        export_button = QPushButton(export_icon, "Export Report", self)
        export_button.clicked.connect(self.generate_report)
        toolbar.addWidget(export_button)

    def create_toolbar_action(self, icon_path, text, slot):
        action = QAction(QIcon(icon_path), text, self)
        action.triggered.connect(slot)
        return action

    def init_dashboard_tab(self):
        self.dashboard_tab = QWidget()
        self.tabs.addTab(self.dashboard_tab, "Dashboard")
        layout = QHBoxLayout(self.dashboard_tab)

        # Overview Group Box
        overview_group = QGroupBox("Overview")
        overview_layout = QVBoxLayout()
        self.total_channels_label = QLabel("Total Channels: N/A")
        self.sampling_freq_label = QLabel("Sampling Frequency: N/A Hz")
        self.maternal_hr_label = QLabel("Maternal Heart Rate: N/A BPM")
        self.fetal_hr_label = QLabel("Fetal Heart Rate: N/A BPM")

        overview_layout.addWidget(self.total_channels_label)
        overview_layout.addWidget(self.sampling_freq_label)
        overview_layout.addWidget(self.maternal_hr_label)
        overview_layout.addWidget(self.fetal_hr_label)
        
        # Create the "Show Channel Details" button
        show_details_button = self.create_styled_button("Show Channel Details", 'icons/details.png', self.show_channel_details)
        overview_layout.addWidget(show_details_button)

        overview_group.setLayout(overview_layout)

        # Set a fixed or proportional size for the Overview Group Box
        overview_group.setFixedWidth(700)  # Set the width of the Overview box (change value as needed)

        # Activity Group Box
        activity_group = QGroupBox("Recent Activity")
        activity_layout = QVBoxLayout()
        self.activity_log = QTextEdit()
        self.activity_log.setReadOnly(True)
        self.activity_log.setPlaceholderText("Activity logs will appear here...")

        activity_layout.addWidget(self.activity_log)
        activity_group.setLayout(activity_layout)

        layout.addWidget(overview_group)
        layout.addWidget(activity_group)
        
    def create_styled_button(self, text, icon_path, slot):
        button = QPushButton(text)
        button.setIcon(QIcon(icon_path))
        button.setStyleSheet("border-radius: 15px; background-color: #4CAF50; color: white; padding: 10px;")
        button.clicked.connect(slot)
        self.add_hover_effect(button)
        return button

    def add_hover_effect(self, button):
        button.setStyleSheet("""
            QPushButton {
                border-radius: 15px; 
                background-color: #4CAF50; 
                color: white; 
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
                border: 1px solid #3e8e41;
            }
            QPushButton:pressed {
                background-color: #3e8e41;
            }
        """)

    def init_visualization_tab(self):
        self.visualization_tab = QWidget()
        self.tabs.addTab(self.visualization_tab, "ECG Visualization")
        layout = QVBoxLayout(self.visualization_tab)

        # Set gradient background for the tab (QWidget)
        gradient = QLinearGradient(0, 0, 1, self.visualization_tab.height())
        gradient.setColorAt(0.0, QColor(240, 208, 255))  # Light hospital-like color (light blue)

        palette = QPalette()
        palette.setBrush(QPalette.Window, QBrush(gradient))
        self.visualization_tab.setAutoFillBackground(True)
        self.visualization_tab.setPalette(palette)

        # Create the plot widget with a transparent background
        self.plot_widget = pg.PlotWidget(title="Raw ECG Signal")
        self.plot_widget.setBackground(None)  # Transparent background
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.addLegend()

        # Plot the raw ECG signal (initially empty)
        self.raw_ecg_curve = self.plot_widget.plot(pen=pg.mkPen(color='b', width=2), name="Raw ECG Signal")

        # Add the plot widget to the layout
        layout.addWidget(self.plot_widget)

    def init_processing_tab(self):
        self.processing_tab = QWidget()
        self.tabs.addTab(self.processing_tab, "Signal Processing")
        layout = QVBoxLayout(self.processing_tab)

        # Graph Selection Controls Group Box
        graph_group = QGroupBox("Select Graphs to Display")
        graph_layout = QVBoxLayout()

        # ComboBox for selecting graph type
        self.graph_type_combo = QComboBox()
        self.graph_type_combo.addItems(["Raw ECG", "R-Peak Detection", "Low Pass Filtered ECG", "High Pass Filtered ECG", "Notch Filtered ECG", "Fetal ECG after LMS adaptive filter", "Fetal ECG R-Peak Detection","Fetal ECG After Maternal Signal Removal", "Frequency Spectrum of Filtered ECG Signal", "STFT of Fetal ECG"])
        self.graph_type_combo.setStyleSheet("QComboBox { padding: 10px; }")
        self.graph_type_combo.currentTextChanged.connect(self.update_graph_display)

        # Button to display selected graph
        show_graph_button = QPushButton("Show Graph")
        show_graph_button.clicked.connect(self.update_graph_display)
        show_graph_button.setIcon(QIcon('icons/show_graph.png'))
        show_graph_button.setStyleSheet("""
            QPushButton {
                background-color: #3F51B5; 
                color: white; 
                padding: 10px; 
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #303F9F;
            }
        """)

        # Checkbox for showing grid
        self.grid_checkbox = QCheckBox("Show Grid")
        self.grid_checkbox.setChecked(True)
        self.grid_checkbox.stateChanged.connect(self.toggle_grid)
        
        # Adding Widgets to Graph Group
        graph_layout.addWidget(self.graph_type_combo)
        graph_layout.addWidget(show_graph_button)
        graph_layout.addWidget(self.grid_checkbox)

        graph_group.setLayout(graph_layout)
        layout.addWidget(graph_group)

        # Area to show the selected graph with interactive features
        self.graph_display_area = pg.PlotWidget(title="Selected Graph")
        self.graph_display_area.setBackground('w')
        self.graph_display_area.showGrid(x=True, y=True)
        self.graph_display_area.setLabel('left', 'Amplitude')
        self.graph_display_area.setLabel('bottom', 'Time (s)')
        self.graph_display_area.addLegend()
        layout.addWidget(self.graph_display_area)

        # Dashboard & Real-time Simulation Controls
        self.dashboard = QWidget()
        dashboard_layout = QHBoxLayout()

        self.dashboard.setLayout(dashboard_layout)
        layout.addWidget(self.dashboard)

    def update_dashboard(self, t, ecg_signal, signal_labels,fetal_heart_rate,maternal_heart_rate):
        self.total_channels_label.setText(f"Total Channels: {len(signal_labels)}")
        self.sampling_freq_label.setText(f"Sampling Frequency: {self.sampling_freq} Hz")
        self.maternal_hr_label.setText("Maternal Heart Rate:"+str(maternal_heart_rate)+"BPM") 
        self.fetal_hr_label.setText("Fetal Heart Rate:"+str(fetal_heart_rate)+"BPM")
        self.activity_log.append(f"Loaded ECG data has {len(t)} Samples.")
    
    def update_graph_display(self):
        try:
            selected_graph = self.graph_type_combo.currentText()

            # Clear the current graph
            self.graph_display_area.clear()

            # Number of points to display (1/16 or 1/32 of total data)
            if len(self.ecg_signal) < 400000:
                num_points_to_display = len(self.ecg_signal)// 16
            else:
                num_points_to_display = len(self.ecg_signal)// 32
            t_display = self.t[:num_points_to_display]

            # Logic to display the selected graph
            if selected_graph == "Raw ECG":
                ecg_signal_display = self.ecg_signal[:num_points_to_display]
                self.graph_display_area.plot(t_display, ecg_signal_display, pen=pg.mkPen(color='#006400', width=2), name="Raw ECG")
                exporter = ImageExporter(self.graph_display_area.plotItem)
                exporter.export('raw_ecg_signal.png')
                
            elif selected_graph == "R-Peak Detection":
                ecg_signal_display = self.ecg_signal[:num_points_to_display]
                r_peaks = self.detect_r_peaks(ecg_signal_display)
                self.graph_display_area.plot(t_display, ecg_signal_display, pen=pg.mkPen(color='b', width=2), name="ECG with R-peaks")
                exporter = ImageExporter(self.graph_display_area.plotItem)
                exporter.export('filtered_maternal_ecg_signal_peak.png')

                # Mark the R-peaks
                r_peaks_display = [peak for peak in r_peaks if peak < num_points_to_display]
                self.graph_display_area.plot(t_display[r_peaks_display], ecg_signal_display[r_peaks_display], pen=None,
                                             symbol='o', symbolBrush=('m'), symbolSize=10, name="R-Peaks")

            elif selected_graph == "Low Pass Filtered ECG":
                low_passed_ecg = self.butter_filter(self.ecg_signal, cutoff=50, fs=1000, btype='low')
                low_passed_display = low_passed_ecg[:num_points_to_display]
                self.graph_display_area.plot(t_display, low_passed_display, pen=pg.mkPen(color='g', width=2), name="Low Pass Filtered ECG")

            elif selected_graph == "High Pass Filtered ECG":
                low_passed_ecg = self.butter_filter(self.ecg_signal, cutoff=50, fs=1000, btype='low')
                high_passed_ecg = self.butter_filter(low_passed_ecg, cutoff=1, fs=1000, btype='high')
                high_passed_display = high_passed_ecg[:num_points_to_display]
                self.graph_display_area.plot(t_display, high_passed_display, pen=pg.mkPen(color='b', width=2), name="High Pass Filtered ECG")

            elif selected_graph == "Notch Filtered ECG":
                low_passed_ecg = self.butter_filter(self.ecg_signal, cutoff=50, fs=1000, btype='low')
                high_passed_ecg = self.butter_filter(low_passed_ecg, cutoff=1, fs=1000, btype='high')
                notch_filtered_ecg = self.notch_filter(high_passed_ecg, notch_freq=50, fs=1000)
                notch_display = notch_filtered_ecg[:num_points_to_display]
                self.graph_display_area.plot(t_display, notch_display, pen=pg.mkPen(color='m', width=2), name="Notch Filtered ECG")
                exporter = ImageExporter(self.graph_display_area.plotItem)
                exporter.export('filtered_maternal_ecg_signal.png')

            else:
                pass

            global data_notch,fetal_ecg_cleaned
            f,r_peaks,maternal_ecg,fetal_ecg,fetal_r_peaks,fetal_heart_rate,maternal_heart_rate,fetal_ecg_cleaned,fft_values,f_stft, t_stft, Zxx,time,data_notch = run_adaptive(file_path)
            self.update_dashboard(self.t, self.ecg_signal, signal_labels,fetal_heart_rate,maternal_heart_rate)
            print(fetal_heart_rate)
            if fetal_heart_rate > 0:
                if selected_graph == "Fetal ECG after LMS adaptive filter":
                    self.graph_display_area.plot(time, fetal_ecg, pen='b')
                
                elif selected_graph == "Fetal ECG R-Peak Detection":
                    self.graph_display_area.plot(time, fetal_ecg, pen='b')
                    self.graph_display_area.plot(time[fetal_r_peaks], fetal_ecg[fetal_r_peaks], symbol='o', pen=None, symbolSize=10, symbolBrush='r')
                    exporter = ImageExporter(self.graph_display_area.plotItem)
                    exporter.export('extracted_fetal_ecg_signal.png')

                elif selected_graph == "Fetal ECG After Maternal Signal Removal":
                    fetal_ecg_cleaned = self.butter_filter(fetal_ecg_cleaned, cutoff=50, fs=1000, btype='low')
                    fetal_ecg_cleaned = self.butter_filter(fetal_ecg_cleaned, cutoff=1, fs=1000, btype='high')
                    fetal_ecg_cleaned = self.notch_filter(fetal_ecg_cleaned, notch_freq=50, fs=1000)
                    self.graph_display_area.plot(time, fetal_ecg_cleaned, pen='r')
                    exporter = ImageExporter(self.graph_display_area.plotItem)
                    exporter.export('extracted_fetal_ecg_signal.png')

                elif selected_graph == "Frequency Spectrum of Filtered ECG Signal":
                    self.graph_display_area.plot(f, fft_values, pen='g')
                    self.graph_display_area.setLabel('bottom', 'Frequency (Hz)')
                    self.graph_display_area.setLabel('left', 'Magnitude')
                    exporter = ImageExporter(self.graph_display_area.plotItem)
                    exporter.export('fft_fetal_ecg_signal.png')
                    
                elif selected_graph == "STFT of Fetal ECG":
                    for i in range(Zxx.shape[1]):  # Loop through each time slice
                        self.graph_display_area.plot(f_stft, np.abs(Zxx[:, i]), pen='b', fillLevel=0, fillBrush=(0, 0, 255, 100))
                        self.graph_display_area.setLabel('left', 'Magnitude')
                        self.graph_display_area.setLabel('bottom', 'Frequency (Hz)')
                else:
                    pass
            else:
                QMessageBox.warning(None, "Only Maternal Data", "The Data provide has only maternal ECG signal.")
                self.close()
                QApplication.quit()
        except:
            QMessageBox.warning(self, "No Data", "Please load an EDF file first.")
            
    def toggle_grid(self):
        self.graph_display_area.showGrid(x=self.grid_checkbox.isChecked(), y=self.grid_checkbox.isChecked())

    def calculate_heart_rate(self, r_peaks, fs):
        # Calculate the time intervals between R-peaks
        if len(r_peaks) < 2:
            return 0  # Not enough R-peaks to calculate heart rate
        rr_intervals = np.diff(r_peaks) / fs  # Convert to seconds
        heart_rate = 60 / np.mean(rr_intervals)  # Convert to beats per minute
        return heart_rate

    def load_data(self):
        global file_path
        file_path, _ = QFileDialog.getOpenFileName(self, "Load EDF File", "", "EDF Files (*.edf)")
        if file_path:
            global signal_labels
            self.t, self.ecg_signal, self.sampling_freq, signal_labels = load_ecg_data(file_path)
            self.update_dashboard(self.t, self.ecg_signal, signal_labels,"Yet To Calculate ","Yet To Calculate ")
            self.plot_ecg_signal(self.t, self.ecg_signal)
            self.channel_reports = []  # Store reports for each channel
            with pyedflib.EdfReader(file_path) as f:
                n_channels = f.signals_in_file
                fs = f.getSampleFrequency(0)  # Assuming all signals have the same sample frequency
                global data
                data = np.zeros((n_channels, f.getNSamples()[0]))  # Preallocate space for data
                for i in range(n_channels):
                    data[i, :] = f.readSignal(i)

                total_samples = data.shape[1]
                quarter_samples = total_samples
                
                # Use only the first quarter of the data
                data = data[:, :quarter_samples]
                time = np.arange(quarter_samples) / fs  # Time in seconds

                self.total_channels_label.setText(f"Total Channels: {n_channels}")
                self.sampling_freq_label.setText(f"Sampling Frequency: {fs} Hz")
                
                # Analyze each channel
                for ch in range(n_channels):
                    raw_data = data[ch]
                    r_peaks, _ = find_peaks(raw_data, height=None, distance=int(0.6 * fs), prominence=0.5)
                    heart_rate = self.calculate_heart_rate(r_peaks, fs)

                    # Store the report for the current channel
                    report = (
                        f"--- Channel {ch + 1} Report ---\n"
                        f"Total Samples: {len(raw_data)}\n"
                        f"Heart Rate: {heart_rate:.2f} BPM\n"
                        f"Peak R-Points Detected: {len(r_peaks)}\n"
                        f"First 10 Raw Data Points: {raw_data[:10]}\n"
                    )
                    self.channel_reports.append(report)

                self.activity_log.append(f"Loaded ECG data from {len(data)} Channels.")

    
    def generate_report(self):
        # Run the application
        app = QApplication([])
        dialog = self.SimpleDialog()
        dialog.exec_()

    def show_channel_details(self):
        if hasattr(self, 'channel_reports'):
            report_text = "\n".join(self.channel_reports)
            QMessageBox.information(self, "Channel Reports", report_text)
        else:
            QMessageBox.warning(self, "No Data", "Please load an EDF file first.")

    def plot_ecg_signal(self, t, ecg_signal):
        # Calculate the number of data points to display (1/16)
        if len(ecg_signal)>400000:
            num_points_to_display = len(ecg_signal) // 32
        else:
            num_points_to_display = len(ecg_signal) // 16
        # Slice the time and signal arrays to get only 1/16 of the data
        t_display = t[:num_points_to_display]
        ecg_signal_display = ecg_signal[:num_points_to_display]

        # Clear the previous plot
        self.plot_widget.clear()
        
        # Plot the sliced data
        self.ecg_curve = self.plot_widget.plot(t_display, ecg_signal_display, pen=pg.mkPen(color='#006400', width=2), name="ECG Signal")


    def play_audio_placeholder(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Play ECG Audio")
        dialog.setMinimumSize(300, 200)  # Adjust minimum size as needed

        layout = QVBoxLayout(dialog)

        label = QLabel("Which audio do you want to play?")
        label.setFont(QFont("Arial", 14))  # Adjust font as needed
        layout.addWidget(label)

        combo_box = QComboBox(dialog)
        combo_box.addItems(["Maternal ECG Signal", "Fetal ECG Signal"])
        combo_box.setFixedHeight(50)
        combo_box.setStyleSheet("font-size: 16px;")
        layout.addWidget(combo_box)

        play_button = QPushButton("Play", dialog)
        play_button.setFixedSize(100, 40)
        play_button.setStyleSheet("font-size: 16px;")
        layout.addWidget(play_button)

        play_button.clicked.connect(lambda: self.play_audio(combo_box.currentText()))

        dialog.exec_()  # Show the dialog

    def play_audio(self, selected_audio):
        try:
            fs =1000
            if selected_audio == "Maternal ECG Signal":
                raw_audio_filename = create_audio_from_ecg(data_notch, fs, gain=20.0, filename="raw_heartbeat.wav")
                play_audio(raw_audio_filename, "Raw ECG Signal")
            else:
                if fetal_heart_rate>0:
                    fetal_audio_filename = create_audio_from_ecg(fetal_ecg_cleaned, fs, gain=20.0, 
                                                                  target_bpm=fetal_heart_rate * (1/(fetal_heart_rate+(fetal_heart_rate-maternal_heart_rate)))*100, 
                                                                  current_bpm=fetal_heart_rate)
                    play_audio(fetal_audio_filename, "Fetal ECG Signal")
                else:
                    QMessageBox.information(self, "No Fetal ECG", "No Fetal ECG present in this dataset")
        except NameError:
            QMessageBox.warning(self, "Graphs Error", "Filter Procedures are not done.")
        except:
            QMessageBox.warning(self, "No Data", "Please load an EDF file first.")
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ecg_app = ECGApp()
    ecg_app.show()
    sys.exit(app.exec_())
