import fitz
import os
import re
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

class ReportAgent:
    def __init__(self):
        self.llm = ChatOllama(model="llama3.2:3b")

    def clean_markdown(self, text):
        """Removes ** or __ markdown and other symbols for PDF compatibility."""
        # Simple regex to remove bold markers; or you can replace with nothing
        # for a clean look, as basic fitz textboxes don't support mixed formatting easily.
        cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', text) 
        return cleaned

    def generate_clinical_report(self, diagnosis, confidence, name, age, report_time):
        template = """
        ROLE: Senior Clinical Ophthalmologist.
        TASK: Generate a formal Diabetic Retinopathy (DR) Screening Report based on AI-assisted fundus analysis.
        
        STRICT GUIDELINES:
        1. Tone: Professional, clinical, and objective.
        2. Format: Use "SECTION HEADER:" for all titles.
        3. No Conversational Filler: Do not say "Here is the report" or "Based on the data."
        4. No Placeholders: Use {report_time} for all dates; never use brackets like "[Date]".
        
        PATIENT INPUTS:
        - Name: {name}
        - Age: {age}
        - Date of Analysis: {report_time}
        - AI Classification: {diagnosis}
        - Confidence Score: {confidence}%
        
        REPORT STRUCTURE TO FOLLOW:

        SECTION 1: PATIENT IDENTIFICATION
        Patient Name: {name}
        Patient Age: {age}
        Accession Date: {report_time}
        
        SECTION 2: DIAGNOSTIC SUMMARY
        Screening Result: {diagnosis}
        System Confidence: {confidence}%
        Clinical Interpretation: [Provide a 5-sentence medical explanation of the {diagnosis} stage and its typical impact on the retina.]
        
        SECTION 3: CLINICAL RECOMMENDATIONS
        [Provide 4 actionable, bulleted medical steps specifically tailored to a {diagnosis} finding, such as follow-up frequency or blood sugar management.]

        """
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm
        
        try:
            # Clean the confidence string to ensure it's just a number
            clean_conf = str(confidence).replace('%', '')
            
            response = chain.invoke({
                "diagnosis": diagnosis, 
                "confidence": clean_conf, 
                "name": name, 
                "age": age, 
                "report_time": report_time
            })
            return response.content
        except Exception as e:
            return f"SECTION 1: ERROR\nReport generation failed for {name}."

    def create_pdf_report(self, report_text, diagnosis, output_path, original_img_path, processed_img_path):
        """
        Args:
            original_img_path: Path to the raw uploaded image
            processed_img_path: Path to the image after Ben Graham processing
        """
        if not report_text:
            report_text = "No clinical data generated."

        # Clean the Markdown bolding
        report_text = self.clean_markdown(report_text)

        doc = fitz.open()
        page = doc.new_page()
        
        # 1. Header
        page.insert_text((50, 40), "OCULIS AI: DIAGNOSTIC REPORT", 
                         fontsize=18, fontname="helvetica-bold", color=(0, 0, 0.5))
        
        # 2. Patient & Condition Summary
        page.insert_text((50, 70), f"Condition Stage: {diagnosis}", 
                         fontsize=12, fontname="helvetica-bold")

        # 3. Insert Images (Side by Side)
        # We define rectangles for where the images should go
        # [x0, y0, x1, y1]
        try:
            page.insert_text((50, 100), "Retinal Scans (Original vs. Processed):", fontsize=10, fontname="helvetica-bold")
            
            # Original Image
            rect_orig = fitz.Rect(50, 110, 280, 260) 
            page.insert_image(rect_orig, filename=original_img_path)
            
            # Processed Image
            rect_proc = fitz.Rect(300, 110, 530, 260)
            page.insert_image(rect_proc, filename=processed_img_path)
        except Exception as e:
            page.insert_text((50, 110), f"Error loading images: {e}", color=(1, 0, 0))

        # 4. Report Body (Starting below the images)
        # We start at y=280 to leave room for the scans
        rect_text = fitz.Rect(50, 280, 550, 780) 
        
        rc = page.insert_textbox(rect_text, report_text, 
                                fontsize=10, 
                                fontname="helvetica", 
                                align=0) 
        
        # Handle overflow
        if rc < 0:
            page = doc.new_page()
            page.insert_textbox(fitz.Rect(50, 50, 550, 750), "(Continued...)\n" + report_text[1200:], 
                                fontsize=10, fontname="helvetica")

        # 5. Footer
        page.insert_text((50, 800), "DISCLAIMER: This report is AI-generated for screening purposes only.", 
                         fontsize=8, fontname="helvetica-oblique") 

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        doc.save(output_path)
        doc.close()
        return output_path