from datetime import datetime
from fpdf import FPDF
import os

try:
    def generate_pdf_report(filename,name,age,dob,mh,fh):
        class AdvancedReport(FPDF):
            def header(self):
                self.set_font("Arial", "B", 12)
                self.cell(0, 10, 'Fetal and Maternal ECG Analysis Report', 0, 1, 'C')
                self.cell(0, 10, f"Generated on: {datetime.now().strftime('%Y-%m-%d %I:%M %p')}", 0, 1, 'C')
                # Draw page border
                self.rect(10, 10, 190, 277)  # x, y, width, height (190 for A4 width, 277 for A4 height minus margins)

            def footer(self):
                self.set_y(-13)
                self.set_font("Arial", "I", 8)
                self.cell(0, 10, 'Page %s' % self.page_no(), 0, 0, 'C')
                # Draw page border again for consistency in footer
                self.rect(10, 10, 190, 277)

        # Function to generate a well-organized advanced report
        def generate_advanced_report(patient_name, patient_age, maternal_hr, fetal_hr, doctor_notes="No significant abnormalities observed."):       
            pdf = AdvancedReport()
            pdf.set_auto_page_break(auto=True, margin=15)
            
            # Cover Page
            pdf.add_page()
            pdf.set_font("Arial", "B", 24)
            pdf.cell(0, 50, ln=True)  # Padding for alignment
            pdf.cell(0, 10, "Fetal and Maternal ECG Analysis Report", ln=True, align='C')
            pdf.set_font("Arial", "", 14)
            pdf.ln(20)
            pdf.cell(0, 10, f"Patient Name: {patient_name}", ln=True, align='C')
            pdf.cell(0, 10, f"Age: {patient_age}", ln=True, align='C')
            pdf.cell(0, 10, f"Date of Birth: {dob}", ln=True, align='C')
            pdf.ln(10)
            pdf.set_font("Arial", "I", 12)
            pdf.cell(0, 10, f"Report Date: {datetime.now().strftime('%d-%m-%Y')}", ln=True, align='C')
            pdf.cell(0, 10, f"Report Time: {datetime.now().strftime('%I:%M %p')}", ln=True, align='C')
            pdf.ln(35)
            pdf.cell(0, 10, "Report automatically Generated", ln=True, align='C')
            pdf.ln(7)
            pdf.cell(0, 10, "Project Done by:", ln=True, align='C')
            pdf.cell(0, 10, "Adarsh-CB.EN.U4ECE23004", ln=True, align='C')
            pdf.cell(0, 10, "Uday-CB.EN.U4ECE23010", ln=True, align='C')
            pdf.cell(0, 10, "Ezhilkirthik.M-CB.EN.U4ECE23016", ln=True, align='C')
            
            # Add a page break for the main report content
            pdf.add_page()
            
            # Patient details section
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Patient Information", ln=True, border=1)
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 10, f"Name: {patient_name}", ln=True)
            pdf.cell(0, 10, f"Age: {patient_age}", ln=True)
            pdf.cell(0, 10, f"Date: {datetime.now().strftime('%Y-%m-%d')}", ln=True)
            pdf.cell(0, 10, f"Time: {datetime.now().strftime('%I:%M %p')}", ln=True)  # Adjusted to 12-hour format

            # Maternal ECG Results
            pdf.ln(10)
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Maternal ECG Results", ln=True, border=1)
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 10, f"Maternal Heart Rate: {maternal_hr:.2f} bpm", ln=True)
            pdf.cell(0, 10, "R-Peak Detection Plot:", ln=True)
            pdf.image("filtered_maternal_ecg_signal.png", w=160)

            # Fetal ECG Results
            pdf.ln(60)
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Fetal ECG Results", ln=True, border=1)
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 10, f"Fetal Heart Rate: {fetal_hr:.2f} bpm", ln=True)
            pdf.cell(0, 10, "R-Peak Detection Plot for Fetal ECG:", ln=True)
            pdf.image("extracted_fetal_ecg_signal.png", w=160)

            # ECG Signal Plots Section
            pdf.ln(10)
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "ECG Signal Plots", ln=True, border=1)

            # Raw ECG signal plot
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 10, "1. Raw ECG Signal", ln=True)
            pdf.image("raw_ecg_signal.png", w=160)

            # Filtered Maternal ECG signal plot
            pdf.cell(0, 10, "2. Filtered Maternal ECG Signal", ln=True)
            pdf.image("filtered_maternal_ecg_signal.png", w=160)

            # Extracted Fetal ECG signal plot
            pdf.cell(0, 10, "3. Extracted Fetal ECG Signal", ln=True)
            pdf.image("extracted_fetal_ecg_signal.png", w=160)

            # FFT Analysis of Fetal ECG plot
            pdf.set_font("Arial", "", 12)
            pdf.cell(0, 47.6, "4. Time-Frequency Analysis (FFT) of Fetal ECG", ln=True)
            pdf.image("fft_fetal_ecg_signal.png", w=160)

            # Doctor's Notes
            pdf.ln(10)
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "Doctor's Observations", ln=True, border=1)
            pdf.set_font("Arial", "", 12)
            pdf.multi_cell(0, 10, f"Observations: {doctor_notes}")

            pdf.output(filename)
            print(f"Report generated and saved as {filename}")

            images = [
                    "filtered_maternal_ecg_signal.png",
                    "extracted_fetal_ecg_signal.png",
                    "raw_ecg_signal.png",
                    "fft_fetal_ecg_signal.png","filtered_maternal_ecg_signal_peak.png"
                ]

            # Deleting the images after generating the PDF
            for img in images:
                if os.path.exists(img):
                    os.remove(img)
                    print(f"{img} deleted.")
                else:
                    print(f"{img} not found, skipping deletion.")

        # Example to generate an advanced formatted report
        generate_advanced_report(patient_name=name, patient_age=age, maternal_hr=mh, fetal_hr=fh)

except:
    QMessageBox.warning(self, "No Data", "Please load an EDF file first.")
