import os
import time
from typing import TypedDict
from langgraph.graph import StateGraph, END
from flask_mail import Message

# 1. Define the Shared Memory (State)
class AgentState(TypedDict):
    patient_data: dict      # Basic info
    analysis_results: dict  # AI prediction data
    report_path: str        
    email_sent: bool        

class OculisMAS:
    def __init__(self, mail, report_agent, predict_fn):
        self.mail = mail
        self.report_agent = report_agent
        self.predict_fn = predict_fn
        self.workflow = self._create_workflow()

    # --- AGENT 1: Vision Specialist (Optimized) ---
    def vision_agent(self, state: AgentState):
        p_data = state['patient_data']
        img_path = p_data['image_path']
        
        # Prepare the processed image path regardless of which branch we take
        proc_filename = f"proc_{os.path.basename(img_path)}"
        proc_path = os.path.join(os.path.dirname(img_path), proc_filename)

        # SPEED FIX: If we have precomputed results, we still need to generate the processed image for the PDF
        if p_data.get('verified_severity') and p_data.get('precomputed_conf'):
            print("MAS: Skipping AI Inference, but generating processed image for PDF...")
            
            # We import the processing function here or use it via a helper
            # Since predict_fn usually returns (level, conf, processed_img)
            # We just run the processing part manually to save time
            from PIL import Image
            import predict # Assuming ben_graham_processing is accessible
            
            orig_img = Image.open(img_path).convert("RGB")
            # Only run the CV2 processing, not the heavy Model inference
            proc_img = predict.ben_graham_processing(orig_img)
            proc_img.save(proc_path)

            return {
                "analysis_results": {
                    "dr_level": p_data['verified_severity'],
                    "confidence": p_data['precomputed_conf'],
                    "proc_path": proc_path
                }
            }

        # Fallback: Run full AI
        print("MAS: Running Full Vision AI Model...")
        dr_level, confidence, proc_img = self.predict_fn(img_path)
        proc_img.save(proc_path)
        
        return {
            "analysis_results": {
                "dr_level": dr_level, 
                "confidence": confidence, 
                "proc_path": proc_path
            }
        }

    # --- AGENT 2: Medical Reporter ---
    def reporter_agent(self, state: AgentState):
        p_data = state['patient_data']
        results = state['analysis_results']
        
        pdf_path = os.path.join(os.path.dirname(p_data['image_path']), f"report_{p_data['id']}.pdf")
        report_time = time.strftime("%Y-%m-%d %H:%M:%S")

        print(f"MAS: LLM is generating report for {p_data['name']}...")
        
        # Generate Clinical Text (LLM)
        severity = p_data.get('verified_severity') or results['dr_level']
        clinical_text = self.report_agent.generate_clinical_report(
            severity, results['confidence'], p_data['name'], p_data['age'], report_time
        )
        
        # Create PDF
        self.report_agent.create_pdf_report(
            clinical_text, severity, pdf_path, 
            p_data['image_path'], results.get('proc_path', p_data['image_path'])
        )
        return {"report_path": pdf_path}

    # --- AGENT 3: Communication Agent ---
    def communication_agent(self, state: AgentState):
        p_data = state['patient_data']
        print(f"MAS: Sending email to {p_data['email']}...")
        try:
            msg = Message(
                subject=f"Retina Screening Report - {p_data['name']}",
                recipients=[p_data['email']],
                body=f"Dear {p_data['name']},\n\nYour report is ready. Please find the attachment for your report. Feel free to use our chatbot for more queries!"
            )
            with open(state['report_path'], 'rb') as fp:
                msg.attach(os.path.basename(state['report_path']), "application/pdf", fp.read())
            
            self.mail.send(msg)
            return {"email_sent": True}
        except Exception as e:
            print(f"CommAgent Error: {e}")
            return {"email_sent": False}

    def _create_workflow(self):
        graph = StateGraph(AgentState)
        graph.add_node("vision_specialist", self.vision_agent)
        graph.add_node("medical_scribe", self.reporter_agent)
        graph.add_node("courier", self.communication_agent)
        
        graph.set_entry_point("vision_specialist")
        graph.add_edge("vision_specialist", "medical_scribe")
        graph.add_edge("medical_scribe", "courier")
        graph.add_edge("courier", END)
        
        return graph.compile()