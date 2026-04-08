import os
from dotenv import load_dotenv
load_dotenv()
import random
import mysql.connector
import time
import uuid
from flask import Flask, request, jsonify, render_template, url_for
from flask_mail import Mail, Message
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

# --- AI & RAG IMPORTS ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- MAS & PREDICTION IMPORTS ---
from predict import get_prediction  
from report_gen import ReportAgent
from agents import OculisMAS  

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY') 

# --- CONFIGURATION ---
db_config = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME'),
    'auth_plugin': 'mysql_native_password'
}

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- FLASK-MAIL CONFIGURATION ---
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER')

mail = Mail(app)

# --- AI MODEL & AGENT INITIALIZATION ---
# Using llama3.2:3b via Ollama
llm = ChatOllama(model="llama3.2:3b")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = None

# Initialize the Report Tool and the Multi-Agent System
report_tool = ReportAgent() 
oculis_mas = OculisMAS(mail, report_tool, get_prediction)

# --- TEMPORARY STORAGE ---
users_pending = {} 
otp_store = {}

def get_db_connection():
    return mysql.connector.connect(**db_config)

# --- PAGE ROUTES ---
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/login_page")
def login_page():
    return render_template("login.html")

@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")

@app.route("/patient")
def patient():
    return render_template("patients.html")

@app.route("/about-us")
def aboutus():
    return render_template("aboutus.html")

@app.route("/feedback")
def feedback():
    return render_template("feedback.html")

@app.route("/admin")
def admin():
    return render_template("admin.html")

# --- AUTHENTICATION ROUTES ---
@app.route('/register', methods=['POST'])
def register_admin():
    name = request.form.get('full_name')
    email = request.form.get('email')
    phone = request.form.get('phone_number')
    plain_password = request.form.get('password')
    hashed_pw = generate_password_hash(plain_password)
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        query = "INSERT INTO users (full_name, email, phone_number, password_hash, user_role) VALUES (%s, %s, %s, %s, %s)"
        cursor.execute(query, (name, email, phone, hashed_pw, 'admin'))
        conn.commit()
        cursor.close()
        conn.close()
        return f"""<div style='text-align:center; padding:50px;'><h1 style='color:#020082;'>Success!</h1><p>Admin account created.</p><a href='/admin'>Back</a></div>"""
    except mysql.connector.Error as err:
        return f"<h1 style='color:red;'>Database Error: {err}</h1>"

@app.route("/signup", methods=["POST"])
def signup():
    data = request.get_json()
    full_name, email, phone, password = data.get("full_name"), data.get("email"), data.get("phone"), data.get("password")
    role = data.get("role", "user").lower() 
    if not all([full_name, email, phone, password]):
        return jsonify({"status": "error", "message": "Please fill all fields"}), 400
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT email FROM users WHERE email = %s", (email,))
    if cursor.fetchone():
        return jsonify({"status": "error", "message": "Email already registered"}), 400
    otp = random.randint(1000, 9999)
    otp_store[email] = otp
    users_pending[email] = {"full_name": full_name, "phone": phone, "password": generate_password_hash(password), "role": role}
    msg = Message(subject="OCULIS AI - OTP Verification", recipients=[email], html=f"<h2>{otp}</h2>")
    mail.send(msg)
    return jsonify({"status": "otp_sent"})

@app.route("/verify_otp", methods=["POST"])
def verify_otp():
    data = request.get_json()
    email, otp_input = data.get("email"), data.get("otp")
    if email in otp_store and str(otp_store[email]) == str(otp_input):
        u = users_pending.get(email)
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (full_name, email, phone_number, password_hash, user_role) VALUES (%s, %s, %s, %s, %s)",
                       (u['full_name'], email, u['phone'], u['password'], u['role']))
        conn.commit()
        return jsonify({"status": "success", "redirect_url": url_for('login_page')})
    return jsonify({"status": "error", "message": "Invalid OTP"}), 400

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    email, password = data.get("email").strip(), data.get("password")
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
    user = cursor.fetchone()
    if user and check_password_hash(user['password_hash'], password):
        role = str(user.get('user_role', 'user')).lower()
        if email == "admin@oculis.ai": return jsonify({"status": "success", "redirect_url": url_for('admin')})
        elif role == 'admin': return jsonify({"status": "success", "redirect_url": url_for('patient')})
        else: return jsonify({"status": "success", "redirect_url": url_for('chatbot')})
    return jsonify({"status": "error", "message": "Invalid credentials"}), 401

# --- CHATBOT RAG LOGIC ---
@app.route('/upload', methods=['POST'])
def upload_pdf():
    global vector_store
    file = request.files['file']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], "temp_report.pdf")
    file.save(file_path)
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(documents)
    vector_store = FAISS.from_documents(chunks, embeddings)
    return jsonify({"message": "Report indexed!"})

@app.route('/ask', methods=['POST'])
def ask_ai():
    user_query = request.json.get("message")
    context = ""
    if vector_store:
        context = "\n".join([doc.page_content for doc in vector_store.similarity_search(user_query, k=3)])
    response = llm.invoke(f"Context: {context}\n\nQuestion: {user_query}")
    return jsonify({"response": response.content})

# --- PATIENT ANALYSIS LOGIC (MAS IMPLEMENTATION) ---

@app.route("/predict_only", methods=["POST"])
def predict_only():
    """Step 1: Rapid AI Screening Node"""
    try:
        file = request.files.get("fundus_image")
        temp_filename = secure_filename(f"temp_{uuid.uuid4()}_{file.filename}")
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        file.save(temp_path)
        
        
        dr_level, confidence, _ = get_prediction(temp_path)
        
        return jsonify({
            "status": "success", 
            "dr_level": dr_level, 
            "confidence": f"{confidence}%", 
            "temp_filename": temp_filename
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/add_patient", methods=["POST"])
def add_patient():
    """Step 2: Orchestrated MAS Workflow (Report -> Email)"""
    try:
        p_id = request.form.get("patient_id")
        temp_filename = request.form.get("temp_filename")
        conf_val = request.form.get("confidence", "90.0").replace('%', '')
        
        # Prepare file path
        source_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        final_filename = secure_filename(f"{p_id}_{temp_filename.replace('temp_', '')}")
        final_save_path = os.path.join(app.config['UPLOAD_FOLDER'], final_filename)
        
        if os.path.exists(source_path):
            os.rename(source_path, final_save_path)

        # 1. Trigger the Multi-Agent System (Optimized Input)
        state_input = {
            "patient_data": {
                "id": p_id,
                "name": request.form.get("name"),
                "email": request.form.get("email"),
                "age": request.form.get("age"),
                "verified_severity": request.form.get("verified_severity"),
                "precomputed_conf": conf_val, # Critical for speed
                "image_path": final_save_path
            }
        }
        
        # Invoke LangGraph Workflow (Now runs in seconds)
        final_state = oculis_mas.workflow.invoke(state_input)

        # 2. Synchronize Database
        db_image_path = f"uploads/{final_filename}"
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO patients (patient_id, name, email, age, weight, gender, phone_number, doctor_name, fundus_image_path)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (p_id, request.form.get("name"), request.form.get("email"), request.form.get("age"), 
              request.form.get("weight"), request.form.get("gender"), request.form.get("phone"), 
              request.form.get("doctor_name"), db_image_path))
        
        p_internal_id = cursor.lastrowid
        
        # Store analysis results in DB
        cursor.execute("""
            INSERT INTO dr_analysis (patient_id, image_path, dr_level, confidence, notes)
            VALUES (%s, %s, %s, %s, %s)
        """, (p_internal_id, db_image_path, str(request.form.get("verified_severity")), 
              final_state['analysis_results']['confidence'], 
              f"PDF: uploads/{os.path.basename(final_state['report_path'])}"))
        
        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({
            "status": "success",
            "email_status": "Sent" if final_state.get("email_sent") else "Failed",
            "pdf_url": url_for('static', filename=f'uploads/{os.path.basename(final_state["report_path"])}')
        })

    except Exception as e:
        print(f"Error in add_patient route: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)