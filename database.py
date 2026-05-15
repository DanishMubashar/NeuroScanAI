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
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_patients_name ON patients(name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_visits_patient ON visits(patient_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_visits_date ON visits(visit_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_visits_tumor_type ON visits(tumor_type)')
        
        # Insert default doctor if not exists
        cursor.execute('SELECT COUNT(*) as count FROM doctors')
        doctor_count = cursor.fetchone()['count']
        
        if doctor_count == 0:
            # Default doctor with hashed password
            hashed_password = hashlib.sha256('doctor123'.encode()).hexdigest()
            cursor.execute('''
                INSERT INTO doctors (username, password, name, email, specialization)
                VALUES (?, ?, ?, ?, ?)
            ''', ('dr_raj', hashed_password, 'Dr. Raj Sharma', 'dr.raj@neuroscan.com', 'Neurologist'))
            
            # Add one more doctor
            hashed_password2 = hashlib.sha256('doctor456'.encode()).hexdigest()
            cursor.execute('''
                INSERT INTO doctors (username, password, name, email, specialization)
                VALUES (?, ?, ?, ?, ?)
            ''', ('dr_priya', hashed_password2, 'Dr. Priya Singh', 'dr.priya@neuroscan.com', 'Radiologist'))
        
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
    
    def get_all_doctors(self) -> List[Dict]:
        """Get all doctors"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id, name, username, email, specialization FROM doctors ORDER BY name')
        doctors = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return doctors
    
    def add_doctor(self, doctor_data: Dict) -> int:
        """Add new doctor"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        hashed_password = hashlib.sha256(doctor_data['password'].encode()).hexdigest()
        
        cursor.execute('''
            INSERT INTO doctors (username, password, name, email, specialization)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            doctor_data['username'],
            hashed_password,
            doctor_data['name'],
            doctor_data.get('email', ''),
            doctor_data.get('specialization', '')
        ))
        
        doctor_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return doctor_id
    
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
    
    def get_all_patients(self, limit=100) -> List[Dict]:
        """Get all patients"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM patients ORDER BY created_at DESC LIMIT ?', (limit,))
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
    
    def update_patient(self, patient_id: int, patient_data: Dict) -> bool:
        """Update patient information"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE patients 
            SET name = ?, age = ?, gender = ?, contact = ?, address = ?
            WHERE id = ?
        ''', (
            patient_data['name'],
            patient_data['age'],
            patient_data['gender'],
            patient_data.get('contact', ''),
            patient_data.get('address', ''),
            patient_id
        ))
        
        updated = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return updated
    
    def delete_patient(self, patient_id: int) -> bool:
        """Delete patient and all associated visits"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # First delete all visits for this patient
        cursor.execute('DELETE FROM visits WHERE patient_id = ?', (patient_id,))
        # Then delete the patient
        cursor.execute('DELETE FROM patients WHERE id = ?', (patient_id,))
        
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted
    
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
            SELECT v.*, d.name as doctor_name, d.specialization as doctor_specialization
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
    
    def get_visit_by_id(self, visit_id: int) -> Optional[Dict]:
        """Get specific visit by ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT v.*, d.name as doctor_name, p.name as patient_name, p.cnic as patient_cnic
            FROM visits v
            JOIN doctors d ON v.doctor_id = d.id
            JOIN patients p ON v.patient_id = p.id
            WHERE v.id = ?
        ''', (visit_id,))
        visit = cursor.fetchone()
        conn.close()
        return dict(visit) if visit else None
    
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
    
    def update_visit_report_path(self, visit_id: int, report_path: str) -> bool:
        """Update PDF report path for a visit"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('UPDATE visits SET report_pdf_path = ? WHERE id = ?', (report_path, visit_id))
        updated = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return updated
    
    def update_visit_notes(self, visit_id: int, notes: str) -> bool:
        """Update doctor notes for a visit"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('UPDATE visits SET doctor_notes = ? WHERE id = ?', (notes, visit_id))
        updated = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return updated
    
    # ==================== TUMOR PROGRESSION OPERATIONS ====================
    
    def get_tumor_progression(self, patient_id: int) -> Optional[Dict]:
        """Get tumor progression data for a patient"""
        visits = self.get_patient_visits(patient_id)
        
        if len(visits) < 2:
            return None
        
        # Filter visits that have tumor area data
        tumor_visits = [v for v in visits if v.get('tumor_area', 0) > 0 and v.get('tumor_type') != 'notumor']
        
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
            'current_tumor_type': latest['tumor_type'],
            'previous_visit_id': previous['id'],
            'current_visit_id': latest['id']
        }
    
    def get_tumor_growth_rate(self, patient_id: int) -> Optional[Dict]:
        """Calculate tumor growth rate over time"""
        visits = self.get_patient_visits(patient_id)
        tumor_visits = [v for v in visits if v.get('tumor_area', 0) > 0 and v.get('tumor_type') != 'notumor']
        
        if len(tumor_visits) < 2:
            return None
        
        # Calculate growth rate per day
        growth_rates = []
        for i in range(len(tumor_visits) - 1):
            current = tumor_visits[i]
            previous = tumor_visits[i + 1]
            
            # Calculate days between visits
            current_date = datetime.strptime(current['visit_date'], '%Y-%m-%d %H:%M:%S')
            previous_date = datetime.strptime(previous['visit_date'], '%Y-%m-%d %H:%M:%S')
            days_diff = (current_date - previous_date).days
            
            if days_diff > 0:
                area_change = current['tumor_area'] - previous['tumor_area']
                daily_growth_rate = area_change / days_diff
                growth_rates.append(daily_growth_rate)
        
        if growth_rates:
            return {
                'average_daily_growth': sum(growth_rates) / len(growth_rates),
                'max_daily_growth': max(growth_rates),
                'min_daily_growth': min(growth_rates),
                'number_of_comparisons': len(growth_rates)
            }
        
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
            LIMIT 12
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
        
        # Risk level distribution
        cursor.execute('''
            SELECT risk_level, COUNT(*) as count
            FROM visits
            WHERE risk_level IS NOT NULL
            GROUP BY risk_level
        ''')
        risk_distribution = {row['risk_level']: row['count'] for row in cursor.fetchall()}
        
        # Average confidence by tumor type
        cursor.execute('''
            SELECT tumor_type, AVG(confidence) as avg_confidence
            FROM visits
            WHERE tumor_type IS NOT NULL AND confidence IS NOT NULL
            GROUP BY tumor_type
        ''')
        avg_confidence_by_type = {row['tumor_type']: row['avg_confidence'] for row in cursor.fetchall()}
        
        # Doctor statistics
        cursor.execute('''
            SELECT d.name, COUNT(v.id) as scan_count
            FROM doctors d
            LEFT JOIN visits v ON d.id = v.doctor_id
            GROUP BY d.id
            ORDER BY scan_count DESC
        ''')
        doctor_stats = [dict(row) for row in cursor.fetchall()]
        
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
            'age_groups': age_groups,
            'risk_distribution': risk_distribution,
            'avg_confidence_by_type': avg_confidence_by_type,
            'doctor_stats': doctor_stats
        }
    
    def get_patient_statistics(self) -> Dict:
        """Get detailed patient statistics"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Patients with multiple visits
        cursor.execute('''
            SELECT patient_id, COUNT(*) as visit_count
            FROM visits
            GROUP BY patient_id
            HAVING visit_count > 1
        ''')
        multiple_visits = len(cursor.fetchall())
        
        # Patients with tumor detected
        cursor.execute('''
            SELECT COUNT(DISTINCT patient_id) as count
            FROM visits
            WHERE tumor_type != "notumor"
        ''')
        patients_with_tumor = cursor.fetchone()['count']
        
        # Average scans per patient
        cursor.execute('''
            SELECT AVG(visit_count) as avg_scans
            FROM (
                SELECT patient_id, COUNT(*) as visit_count
                FROM visits
                GROUP BY patient_id
            )
        ''')
        avg_scans_per_patient = cursor.fetchone()['avg_scans'] or 0
        
        conn.close()
        
        return {
            'patients_with_multiple_visits': multiple_visits,
            'patients_with_tumor': patients_with_tumor,
            'avg_scans_per_patient': round(avg_scans_per_patient, 1)
        }
    
    def export_to_dataframe(self, table_name: str) -> pd.DataFrame:
        """Export table to pandas DataFrame"""
        conn = self.get_connection()
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def get_database_backup(self) -> bytes:
        """Create database backup"""
        import io
        backup_buffer = io.BytesIO()
        
        conn = self.get_connection()
        backup_conn = sqlite3.connect(backup_buffer)
        
        conn.backup(backup_conn)
        
        backup_conn.close()
        conn.close()
        
        backup_buffer.seek(0)
        return backup_buffer.getvalue()
    
    def clear_all_data(self, confirm: bool = False) -> bool:
        """Clear all data from tables (for testing/reset)"""
        if not confirm:
            return False
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('DELETE FROM visits')
            cursor.execute('DELETE FROM patients')
            cursor.execute("DELETE FROM sqlite_sequence WHERE name IN ('patients', 'visits')")
            conn.commit()
            return True
        except Exception as e:
            print(f"Error clearing data: {e}")
            return False
        finally:
            conn.close()
