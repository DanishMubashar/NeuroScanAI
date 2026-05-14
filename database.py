import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import hashlib

class Database:
    def __init__(self, db_path="neuroscan.db"):
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_database(self):
        """Initialize all database tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Doctors table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS doctors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                name TEXT NOT NULL,
                email TEXT,
                specialization TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Patients table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cnic TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                age INTEGER,
                gender TEXT,
                contact TEXT,
                address TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Visits table (each patient can have multiple visits)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS visits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER NOT NULL,
                doctor_id INTEGER NOT NULL,
                visit_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                mri_image_path TEXT,
                prediction_results TEXT,
                tumor_type TEXT,
                confidence REAL,
                tumor_area REAL,
                tumor_width REAL,
                tumor_height REAL,
                tumor_center_x INTEGER,
                tumor_center_y INTEGER,
                tumor_radius INTEGER,
                brain_region TEXT,
                hemisphere TEXT,
                risk_level TEXT,
                doctor_notes TEXT,
                report_pdf_path TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients(id),
                FOREIGN KEY (doctor_id) REFERENCES doctors(id)
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_patients_cnic ON patients(cnic)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_visits_patient ON visits(patient_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_visits_date ON visits(visit_date)')
        
        # Insert default doctor if not exists
        cursor.execute('''
            INSERT OR IGNORE INTO doctors (username, password, name, email, specialization)
            VALUES (?, ?, ?, ?, ?)
        ''', ('dr_raj', hashlib.sha256('doctor123'.encode()).hexdigest(), 'Dr. Raj Sharma', 'dr.raj@neuroscan.com', 'Neurologist'))
        
        conn.commit()
        conn.close()
    
    # ==================== DOCTOR OPERATIONS ====================
    
    def authenticate_doctor(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate doctor login"""
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM doctors WHERE username = ? AND password = ?', (username, hashed_password))
        doctor = cursor.fetchone()
        conn.close()
        return dict(doctor) if doctor else None
    
    def get_doctor_by_id(self, doctor_id: int) -> Optional[Dict]:
        """Get doctor by ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM doctors WHERE id = ?', (doctor_id,))
        doctor = cursor.fetchone()
        conn.close()
        return dict(doctor) if doctor else None
    
    # ==================== PATIENT OPERATIONS ====================
    
    def add_patient(self, patient_data: Dict) -> int:
        """Add new patient to database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO patients (cnic, name, age, gender, contact, address)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            patient_data['cnic'],
            patient_data['name'],
            patient_data['age'],
            patient_data['gender'],
            patient_data.get('contact', ''),
            patient_data.get('address', '')
        ))
        
        patient_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return patient_id
    
    def get_patient_by_cnic(self, cnic: str) -> Optional[Dict]:
        """Get patient by CNIC"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM patients WHERE cnic = ?', (cnic,))
        patient = cursor.fetchone()
        conn.close()
        return dict(patient) if patient else None
    
    def get_patient_by_id(self, patient_id: int) -> Optional[Dict]:
        """Get patient by ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM patients WHERE id = ?', (patient_id,))
        patient = cursor.fetchone()
        conn.close()
        return dict(patient) if patient else None
    
    def get_all_patients(self) -> List[Dict]:
        """Get all patients"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM patients ORDER BY created_at DESC')
        patients = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return patients
    
    def search_patients(self, search_term: str) -> List[Dict]:
        """Search patients by name or CNIC"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM patients 
            WHERE name LIKE ? OR cnic LIKE ?
            ORDER BY created_at DESC
        ''', (f'%{search_term}%', f'%{search_term}%'))
        patients = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return patients
    
    # ==================== VISIT OPERATIONS ====================
    
    def add_visit(self, visit_data: Dict) -> int:
        """Add new patient visit/scan"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO visits (
                patient_id, doctor_id, mri_image_path, prediction_results,
                tumor_type, confidence, tumor_area, tumor_width, tumor_height,
                tumor_center_x, tumor_center_y, tumor_radius, brain_region,
                hemisphere, risk_level, doctor_notes, report_pdf_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            visit_data['patient_id'],
            visit_data['doctor_id'],
            visit_data.get('mri_image_path', ''),
            json.dumps(visit_data.get('prediction_results', {})),
            visit_data.get('tumor_type', ''),
            visit_data.get('confidence', 0),
            visit_data.get('tumor_area', 0),
            visit_data.get('tumor_width', 0),
            visit_data.get('tumor_height', 0),
            visit_data.get('tumor_center_x', 0),
            visit_data.get('tumor_center_y', 0),
            visit_data.get('tumor_radius', 0),
            visit_data.get('brain_region', ''),
            visit_data.get('hemisphere', ''),
            visit_data.get('risk_level', ''),
            visit_data.get('doctor_notes', ''),
            visit_data.get('report_pdf_path', '')
        ))
        
        visit_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return visit_id
    
    def get_patient_visits(self, patient_id: int) -> List[Dict]:
        """Get all visits for a patient"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT v.*, d.name as doctor_name
            FROM visits v
            JOIN doctors d ON v.doctor_id = d.id
            WHERE v.patient_id = ?
            ORDER BY v.visit_date DESC
        ''', (patient_id,))
        visits = []
        for row in cursor.fetchall():
            visit = dict(row)
            if visit.get('prediction_results'):
                try:
                    visit['prediction_results'] = json.loads(visit['prediction_results'])
                except:
                    pass
            visits.append(visit)
        conn.close()
        return visits
    
    def get_latest_visit(self, patient_id: int) -> Optional[Dict]:
        """Get most recent visit for a patient"""
        visits = self.get_patient_visits(patient_id)
        return visits[0] if visits else None
    
    def get_previous_visit(self, patient_id: int, current_visit_id: int) -> Optional[Dict]:
        """Get previous visit excluding current"""
        visits = self.get_patient_visits(patient_id)
        for visit in visits:
            if visit['id'] != current_visit_id:
                return visit
        return None
    
    # ==================== ANALYTICS OPERATIONS ====================
    
    def get_dashboard_stats(self) -> Dict:
        """Get dashboard statistics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Total patients
        cursor.execute('SELECT COUNT(*) as count FROM patients')
        total_patients = cursor.fetchone()['count']
        
        # Total visits/scans
        cursor.execute('SELECT COUNT(*) as count FROM visits')
        total_scans = cursor.fetchone()['count']
        
        # Tumor cases (non-no-tumor)
        cursor.execute('SELECT COUNT(*) as count FROM visits WHERE tumor_type != "notumor"')
        tumor_cases = cursor.fetchone()['count']
        
        # No tumor cases
        cursor.execute('SELECT COUNT(*) as count FROM visits WHERE tumor_type = "notumor"')
        no_tumor_cases = cursor.fetchone()['count']
        
        # Today's scans
        cursor.execute('''
            SELECT COUNT(*) as count FROM visits 
            WHERE DATE(visit_date) = DATE('now')
        ''')
        today_scans = cursor.fetchone()['count']
        
        # Gender distribution
        cursor.execute('''
            SELECT gender, COUNT(*) as count FROM patients 
            WHERE gender IS NOT NULL GROUP BY gender
        ''')
        gender_stats = {row['gender']: row['count'] for row in cursor.fetchall()}
        
        # Average age
        cursor.execute('SELECT AVG(age) as avg_age FROM patients WHERE age IS NOT NULL')
        avg_age = cursor.fetchone()['avg_age'] or 0
        
        # Monthly scan trends
        cursor.execute('''
            SELECT strftime('%Y-%m', visit_date) as month, COUNT(*) as count
            FROM visits
            GROUP BY month
            ORDER BY month DESC
            LIMIT 6
        ''')
        monthly_trends = [dict(row) for row in cursor.fetchall()]
        
        # Tumor type distribution
        cursor.execute('''
            SELECT tumor_type, COUNT(*) as count
            FROM visits
            WHERE tumor_type IS NOT NULL AND tumor_type != ''
            GROUP BY tumor_type
        ''')
        tumor_distribution = {row['tumor_type']: row['count'] for row in cursor.fetchall()}
        
        # Age group analysis
        cursor.execute('''
            SELECT 
                CASE 
                    WHEN age < 18 THEN '0-17'
                    WHEN age BETWEEN 18 AND 30 THEN '18-30'
                    WHEN age BETWEEN 31 AND 45 THEN '31-45'
                    WHEN age BETWEEN 46 AND 60 THEN '46-60'
                    ELSE '60+'
                END as age_group,
                COUNT(*) as count
            FROM patients
            WHERE age IS NOT NULL
            GROUP BY age_group
            ORDER BY MIN(age)
        ''')
        age_groups = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            'total_patients': total_patients,
            'total_scans': total_scans,
            'tumor_cases': tumor_cases,
            'no_tumor_cases': no_tumor_cases,
            'today_scans': today_scans,
            'gender_stats': gender_stats,
            'avg_age': round(avg_age, 1),
            'monthly_trends': monthly_trends,
            'tumor_distribution': tumor_distribution,
            'age_groups': age_groups
        }
    
    def get_tumor_progression(self, patient_id: int) -> Dict:
        """Get tumor progression data for a patient"""
        visits = self.get_patient_visits(patient_id)
        
        if len(visits) < 2:
            return None
        
        # Filter visits that have tumor area data
        tumor_visits = [v for v in visits if v.get('tumor_area', 0) > 0]
        
        if len(tumor_visits) < 2:
            return None
        
        latest = tumor_visits[0]
        previous = tumor_visits[1]
        
        area_change = latest['tumor_area'] - previous['tumor_area']
        percentage_change = (area_change / previous['tumor_area']) * 100 if previous['tumor_area'] > 0 else 0
        
        return {
            'previous_area': previous['tumor_area'],
            'current_area': latest['tumor_area'],
            'area_change': area_change,
            'percentage_change': percentage_change,
            'direction': 'increased' if area_change > 0 else 'decreased' if area_change < 0 else 'stable',
            'previous_date': previous['visit_date'],
            'current_date': latest['visit_date'],
            'previous_tumor_type': previous['tumor_type'],
            'current_tumor_type': latest['tumor_type']
        }
