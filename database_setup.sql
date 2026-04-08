CREATE DATABASE IF NOT EXISTS oculisai;
USE oculisai;

-- 1. Users Table (For Admins and Staff)
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    full_name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    phone_number VARCHAR(20),
    password_hash VARCHAR(255) NOT NULL,
    user_role VARCHAR(50) DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Patients Table
CREATE TABLE IF NOT EXISTS patients (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id VARCHAR(50) UNIQUE NOT NULL, -- The custom ID you pass from the form
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255),
    age INT,
    weight VARCHAR(10),
    gender VARCHAR(20),
    phone_number VARCHAR(20),
    doctor_name VARCHAR(255),
    fundus_image_path VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Diabetic Retinopathy Analysis Table
CREATE TABLE IF NOT EXISTS dr_analysis (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT, -- Foreign Key linking to patients.id
    image_path VARCHAR(255),
    dr_level VARCHAR(100),
    confidence VARCHAR(50),
    notes TEXT, -- Stores the PDF path or clinical remarks
    analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
);