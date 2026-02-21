import base64
from datetime import datetime, time, timedelta
from io import BytesIO
import os
import sqlite3
import shutil
import socket
import sys
import tempfile
import traceback
import joblib
import pandas as pd
import cv2
import numpy as np
from flask import Flask, json, request, jsonify, send_file
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS



app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

MODEL_DIR = "models"
RESULT_FOLDER = "results"
os.makedirs(RESULT_FOLDER, exist_ok=True)
# ================= Paths =============================
if getattr(sys, "frozen", False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROFILE_PICTURES_DIR = os.path.join(BASE_DIR, "profile_pictures")
os.makedirs(PROFILE_PICTURES_DIR, exist_ok=True)
MODELS_DIR = os.path.join(BASE_DIR, "models")
CLASSES_FILE = os.path.join(BASE_DIR, "classes.txt")
DB_PATH = os.path.join(BASE_DIR, "users.db")

# ================= Database Setup ====================
def ensure_profile_picture_column():
    """Ensure profile_picture_url column exists in users table"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Check if column exists
        cursor.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'profile_picture_url' not in columns:
            # Add the column if it doesn't exist
            cursor.execute('ALTER TABLE users ADD COLUMN profile_picture_url TEXT')
            conn.commit()
            print("✅ Added profile_picture_url column to users table")
    except sqlite3.OperationalError as e:
        print(f"Warning: Could not add profile_picture_url column: {e}")
    finally:
        conn.close()

def init_db():
    """Initialize SQLite database and create users table"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create users table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()
    
    # Ensure profile_picture_url column exists (for existing databases)
    ensure_profile_picture_column()
    
    # Reconnect to continue with other operations
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create prediction_history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            user_email TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            probabilities TEXT NOT NULL,
            model_name TEXT NOT NULL,
            image_type TEXT NOT NULL,
            cropped_image TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')
    
    # Create index for faster queries
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON prediction_history(user_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON prediction_history(created_at)')
    
    conn.commit()
    
    # Check if admin user exists, if not create one
    admin_email = 'pramudithapasindu48@gmail.com'
    admin_password = '1234'
    cursor.execute('SELECT id FROM users WHERE email = ?', (admin_email,))
    if not cursor.fetchone():
        admin_hash = generate_password_hash(admin_password)
        cursor.execute('''
            INSERT INTO users (email, password_hash, is_admin, created_at)
            VALUES (?, ?, ?, ?)
        ''', (admin_email, admin_hash, 1, datetime.now().isoformat()))
        conn.commit()
    
    conn.close()
    
    # Ensure column exists one more time after all operations
    ensure_profile_picture_column()

# Initialize database on startup
init_db()


# ================= Admin Helper Functions ============
def verify_admin(email):
    """Verify if user is admin"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT is_admin FROM users WHERE email = ?', (email.lower(),))
    result = cursor.fetchone()
    conn.close()
    return result and result[0] == 1


# ================= Admin Routes ======================
@app.route("/api/admin/users", methods=["GET"])
def get_all_users():
    """Get all users (admin only)"""
    try:
        # Get admin email from header or query param
        admin_email = request.headers.get("X-Admin-Email") or request.args.get("admin_email", "").strip().lower()
        
        if not admin_email or not verify_admin(admin_email):
            return jsonify({"error": "Admin access required"}), 403
        
        # Ensure column exists
        ensure_profile_picture_column()
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if profile_picture_url column exists
        cursor.execute("PRAGMA table_info(users)")
        columns = [col[1] for col in cursor.fetchall()]
        has_profile_column = 'profile_picture_url' in columns
        
        if has_profile_column:
            cursor.execute('SELECT id, email, is_admin, profile_picture_url, created_at FROM users ORDER BY created_at DESC')
        else:
            cursor.execute('SELECT id, email, is_admin, created_at FROM users ORDER BY created_at DESC')
        
        users = cursor.fetchall()
        conn.close()
        
        user_list = []
        for user in users:
            if has_profile_column:
                user_id, email, is_admin, profile_picture_url, created_at = user
            else:
                user_id, email, is_admin, created_at = user
                profile_picture_url = None
            
            user_list.append({
                "id": user_id,
                "email": email,
                "is_admin": bool(is_admin),
                "profile_picture_url": profile_picture_url,
                "created_at": created_at
            })
        
        return jsonify({"users": user_list}), 200
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/users/<int:user_id>", methods=["PUT"])
def update_user(user_id):
    """Update user (admin only)"""
    try:
        admin_email = request.headers.get("X-Admin-Email") or request.get_json().get("admin_email", "").strip().lower()
        
        if not admin_email or not verify_admin(admin_email):
            return jsonify({"error": "Admin access required"}), 403
        
        data = request.get_json()
        new_email = data.get("email", "").strip().lower()
        new_password = data.get("password", "").strip()
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if user exists
        cursor.execute('SELECT id FROM users WHERE id = ?', (user_id,))
        if not cursor.fetchone():
            conn.close()
            return jsonify({"error": "User not found"}), 404
        
        # Update email if provided
        if new_email:
            # Check if email already exists
            cursor.execute('SELECT id FROM users WHERE email = ? AND id != ?', (new_email, user_id))
            if cursor.fetchone():
                conn.close()
                return jsonify({"error": "Email already in use"}), 400
            cursor.execute('UPDATE users SET email = ? WHERE id = ?', (new_email, user_id))
        
        # Update password if provided
        if new_password:
            if len(new_password) < 4:
                conn.close()
                return jsonify({"error": "Password must be at least 4 characters"}), 400
            password_hash = generate_password_hash(new_password)
            cursor.execute('UPDATE users SET password_hash = ? WHERE id = ?', (password_hash, user_id))
        
        conn.commit()
        conn.close()
        
        return jsonify({"message": "User updated successfully"}), 200
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/users/<int:user_id>", methods=["DELETE"])
def delete_user(user_id):
    """Delete user (admin only)"""
    try:
        admin_email = request.headers.get("X-Admin-Email") or request.get_json().get("admin_email", "").strip().lower()
        
        if not admin_email or not verify_admin(admin_email):
            return jsonify({"error": "Admin access required"}), 403
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if user exists
        cursor.execute('SELECT id, email FROM users WHERE id = ?', (user_id,))
        user = cursor.fetchone()
        if not user:
            conn.close()
            return jsonify({"error": "User not found"}), 404
        
        # Prevent deleting yourself
        user_email = user[1]
        if user_email.lower() == admin_email.lower():
            conn.close()
            return jsonify({"error": "Cannot delete your own account"}), 400
        
        # Delete user
        cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
        conn.commit()
        conn.close()
        
        return jsonify({"message": "User deleted successfully"}), 200
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/users/<int:user_id>/toggle-admin", methods=["POST"])
def toggle_admin(user_id):
    """Toggle admin status (admin only)"""
    try:
        admin_email = request.headers.get("X-Admin-Email") or request.get_json().get("admin_email", "").strip().lower()
        
        if not admin_email or not verify_admin(admin_email):
            return jsonify({"error": "Admin access required"}), 403
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if user exists
        cursor.execute('SELECT id, email, is_admin FROM users WHERE id = ?', (user_id,))
        user = cursor.fetchone()
        if not user:
            conn.close()
            return jsonify({"error": "User not found"}), 404
        
        user_email = user[1]
        # Prevent demoting yourself
        if user_email.lower() == admin_email.lower():
            conn.close()
            return jsonify({"error": "Cannot change your own admin status"}), 400
        
        # Toggle admin status
        new_admin_status = 0 if user[2] == 1 else 1
        cursor.execute('UPDATE users SET is_admin = ? WHERE id = ?', (new_admin_status, user_id))
        conn.commit()
        conn.close()
        
        return jsonify({
            "message": f"User {'promoted to' if new_admin_status else 'demoted from'} admin successfully",
            "is_admin": bool(new_admin_status)
        }), 200
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/stats", methods=["GET"])
def get_admin_stats():
    """Get admin statistics (admin only)"""
    try:
        admin_email = request.headers.get("X-Admin-Email") or request.args.get("admin_email", "").strip().lower()
        
        if not admin_email or not verify_admin(admin_email):
            return jsonify({"error": "Admin access required"}), 403
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Total users
        cursor.execute('SELECT COUNT(*) FROM users')
        total_users = cursor.fetchone()[0]
        
        # Admin users
        cursor.execute('SELECT COUNT(*) FROM users WHERE is_admin = 1')
        admin_users = cursor.fetchone()[0]
        
        # Regular users
        regular_users = total_users - admin_users
        
        # Users registered today
        today = datetime.now().date().isoformat()
        cursor.execute('SELECT COUNT(*) FROM users WHERE created_at LIKE ?', (f'{today}%',))
        users_today = cursor.fetchone()[0]
        
        # Users registered this week (last 7 days)
        week_ago = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        week_ago = (week_ago - timedelta(days=7)).isoformat()
        cursor.execute('SELECT COUNT(*) FROM users WHERE created_at >= ?', (week_ago,))
        users_week = cursor.fetchone()[0]
        
        # Users registered this month
        month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
        cursor.execute('SELECT COUNT(*) FROM users WHERE created_at >= ?', (month_start,))
        users_month = cursor.fetchone()[0]
        
        # Recent registrations (last 10)
        cursor.execute('SELECT email, created_at FROM users ORDER BY created_at DESC LIMIT 10')
        recent_users = cursor.fetchall()
        
        conn.close()
        
        return jsonify({
            "total_users": total_users,
            "admin_users": admin_users,
            "regular_users": regular_users,
            "users_today": users_today,
            "users_week": users_week,
            "users_month": users_month,
            "recent_users": [
                {"email": email, "created_at": created_at}
                for email, created_at in recent_users
            ]
        }), 200
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ================= History Routes ====================
@app.route("/api/history", methods=["GET"])
def get_user_history():
    """Get user's own prediction history"""
    try:
        user_email = request.headers.get("X-User-Email") or request.args.get("user_email", "").strip().lower()
        
        if not user_email:
            return jsonify({"error": "User email required"}), 400
        
        print(f"📊 Fetching history for user: {user_email}")
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Verify user exists
        cursor.execute('SELECT id FROM users WHERE email = ?', (user_email,))
        user_result = cursor.fetchone()
        if not user_result:
            conn.close()
            print(f"❌ User not found: {user_email}")
            return jsonify({"error": "User not found"}), 404
        
        user_id = user_result[0]
        print(f"✅ User found with ID: {user_id}")
        
        # Get user's history
        cursor.execute('''
            SELECT id, prediction, confidence, probabilities, model_name, image_type, cropped_image, created_at
            FROM prediction_history
            WHERE user_id = ?
            ORDER BY created_at DESC
        ''', (user_id,))
        
        history = cursor.fetchall()
        print(f"📈 Found {len(history)} history entries for user {user_email}")
        conn.close()
        
        history_list = []
        for h in history:
            hist_id, prediction, confidence, probabilities_str, model_name, image_type, cropped_image, created_at = h
            try:
                probabilities = json.loads(probabilities_str)
            except:
                probabilities = {}
            
            # Format cropped_image to include data URL prefix if it's base64
            image_data = cropped_image
            if cropped_image and not cropped_image.startswith("data:"):
                image_data = f"data:image/png;base64,{cropped_image}"
            
            history_list.append({
                "id": hist_id,
                "prediction": prediction,  # The actual prediction value
                "prediction_result": prediction,  # Alias for frontend compatibility
                "confidence": confidence,
                "probabilities": probabilities,
                "model_name": model_name,
                "image_type": image_type,
                "cropped_image": image_data,
                "image_data": image_data,  # Alias for frontend compatibility
                "created_at": created_at,
                "user_id": user_id,
                "user_email": user_email
            })
        
        return jsonify({"history": history_list}), 200
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/history", methods=["GET"])
def get_all_history():
    """Get all users' prediction history (admin only)"""
    try:
        admin_email = request.headers.get("X-Admin-Email") or request.args.get("admin_email", "").strip().lower()
        user_email_filter = request.args.get("user_email", "").strip().lower()
        
        if not admin_email or not verify_admin(admin_email):
            return jsonify({"error": "Admin access required"}), 403
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get all history with optional user filter
        if user_email_filter:
            cursor.execute('''
                SELECT id, user_id, user_email, prediction, confidence, probabilities, model_name, image_type, cropped_image, created_at
                FROM prediction_history
                WHERE LOWER(user_email) = ?
                ORDER BY created_at DESC
            ''', (user_email_filter,))
        else:
            cursor.execute('''
                SELECT id, user_id, user_email, prediction, confidence, probabilities, model_name, image_type, cropped_image, created_at
                FROM prediction_history
                ORDER BY created_at DESC
            ''')
        
        history = cursor.fetchall()
        conn.close()
        
        history_list = []
        for h in history:
            hist_id, user_id, user_email, prediction, confidence, probabilities_str, model_name, image_type, cropped_image, created_at = h
            try:
                probabilities = json.loads(probabilities_str)
            except:
                probabilities = {}
            
            # Format cropped_image to include data URL prefix if it's base64
            image_data = cropped_image
            if cropped_image and not cropped_image.startswith("data:"):
                image_data = f"data:image/png;base64,{cropped_image}"
            
            history_list.append({
                "id": hist_id,
                "user_id": user_id,
                "user_email": user_email,
                "prediction": prediction,
                "confidence": confidence,
                "probabilities": probabilities,
                "model_name": model_name,
                "image_type": image_type,
                "cropped_image": image_data,
                "created_at": created_at
            })
        
        return jsonify({"history": history_list}), 200
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/history", methods=["DELETE"])
def delete_all_history():
    """Delete all prediction history (admin only)"""
    try:
        admin_email = request.headers.get("X-Admin-Email") or request.get_json().get("admin_email", "").strip().lower()
        
        if not admin_email or not verify_admin(admin_email):
            return jsonify({"error": "Admin access required"}), 403
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM prediction_history')
        conn.commit()
        deleted_count = cursor.rowcount
        conn.close()
        
        return jsonify({"message": f"Deleted {deleted_count} prediction records"}), 200
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/history/report", methods=["GET"])
def download_user_report():
    """Download user's prediction history as CSV report"""
    try:
        user_email = request.headers.get("X-User-Email") or request.args.get("user_email", "").strip().lower()
        
        if not user_email:
            return jsonify({"error": "User email required"}), 400
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Verify user exists
        cursor.execute('SELECT id FROM users WHERE email = ?', (user_email,))
        user_result = cursor.fetchone()
        if not user_result:
            conn.close()
            return jsonify({"error": "User not found"}), 404
        
        user_id = user_result[0]
        
        # Get user's history
        cursor.execute('''
            SELECT prediction, confidence, probabilities, model_name, image_type, created_at
            FROM prediction_history
            WHERE user_id = ?
            ORDER BY created_at DESC
        ''', (user_id,))
        
        history = cursor.fetchall()
        conn.close()
        
        # Create CSV
        output = BytesIO()
        df_data = []
        for h in history:
            prediction, confidence, probabilities_str, model_name, image_type, created_at = h
            try:
                probabilities = json.loads(probabilities_str)
            except:
                probabilities = {}
            
            df_data.append({
                "Date": created_at,
                "Prediction": prediction,
                "Confidence": f"{confidence*100:.2f}%",
                "Model": model_name,
                "Image Type": image_type,
                **{f"Prob_{k}": f"{v*100:.2f}%" for k, v in probabilities.items()}
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'prediction_history_{user_email}_{datetime.now().strftime("%Y%m%d")}.csv'
        )
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/admin/history/report", methods=["GET"])
def download_admin_report():
    """Download all users' prediction history as CSV report (admin only)"""
    try:
        admin_email = request.headers.get("X-Admin-Email") or request.args.get("admin_email", "").strip().lower()
        
        if not admin_email or not verify_admin(admin_email):
            return jsonify({"error": "Admin access required"}), 403
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get all history
        cursor.execute('''
            SELECT user_email, prediction, confidence, probabilities, model_name, image_type, created_at
            FROM prediction_history
            ORDER BY created_at DESC
        ''')
        
        history = cursor.fetchall()
        conn.close()
        
        # Create CSV
        output = BytesIO()
        df_data = []
        for h in history:
            user_email, prediction, confidence, probabilities_str, model_name, image_type, created_at = h
            try:
                probabilities = json.loads(probabilities_str)
            except:
                probabilities = {}
            
            df_data.append({
                "User Email": user_email,
                "Date": created_at,
                "Prediction": prediction,
                "Confidence": f"{confidence*100:.2f}%",
                "Model": model_name,
                "Image Type": image_type,
                **{f"Prob_{k}": f"{v*100:.2f}%" for k, v in probabilities.items()}
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'all_users_prediction_history_{datetime.now().strftime("%Y%m%d")}.csv'
        )
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ================= Profile Routes ====================
@app.route("/api/profile/upload-picture", methods=["POST"])
def upload_profile_picture():
    """Upload user profile picture"""
    try:
        # Ensure column exists
        ensure_profile_picture_column()
        
        user_email = request.headers.get("X-User-Email") or request.form.get("user_email", "").strip().lower()
        
        if not user_email:
            return jsonify({"error": "User email required"}), 400
        
        # Verify user exists
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM users WHERE email = ?', (user_email,))
        user_result = cursor.fetchone()
        if not user_result:
            conn.close()
            return jsonify({"error": "User not found"}), 404
        
        user_id = user_result[0]
        
        # Get uploaded file
        file = request.files.get("file")
        if not file:
            conn.close()
            return jsonify({"error": "No file uploaded"}), 400
        
        # Validate file type
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            conn.close()
            return jsonify({"error": "Invalid file type. Allowed: jpg, jpeg, png, gif"}), 400
        
        # Save profile picture
        filename = f"profile_{user_id}_{int(datetime.now().timestamp())}{file_ext}"
        file_path = os.path.join(PROFILE_PICTURES_DIR, filename)
        file.save(file_path)
        
        # Generate URL (relative path that can be served)
        profile_picture_url = f"/api/profile/picture/{filename}"
        
        # Update user's profile picture URL in database
        cursor.execute('UPDATE users SET profile_picture_url = ? WHERE id = ?', (profile_picture_url, user_id))
        conn.commit()
        
        # Check if profile_picture_url column exists
        cursor.execute("PRAGMA table_info(users)")
        columns = [col[1] for col in cursor.fetchall()]
        has_profile_column = 'profile_picture_url' in columns
        
        # Get updated user info
        if has_profile_column:
            cursor.execute('SELECT id, email, is_admin, profile_picture_url FROM users WHERE id = ?', (user_id,))
            user_data = cursor.fetchone()
            profile_picture_url = user_data[3] if user_data else None
        else:
            cursor.execute('SELECT id, email, is_admin FROM users WHERE id = ?', (user_id,))
            user_data = cursor.fetchone()
            profile_picture_url = None
        
        conn.close()
        
        return jsonify({
            "message": "Profile picture uploaded successfully",
            "user": {
                "id": user_data[0],
                "email": user_data[1],
                "is_admin": bool(user_data[2]),
                "profile_picture_url": profile_picture_url
            }
        }), 200
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/profile/picture/<filename>", methods=["GET"])
def get_profile_picture(filename):
    """Serve profile picture"""
    try:
        file_path = os.path.join(PROFILE_PICTURES_DIR, filename)
        if os.path.exists(file_path):
            return send_file(file_path)
        else:
            return jsonify({"error": "Profile picture not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/profile/update", methods=["POST"])
def update_profile():
    """Update user profile"""
    try:
        # Ensure column exists
        ensure_profile_picture_column()
        
        user_email = request.headers.get("X-User-Email") or request.get_json().get("user_email", "").strip().lower()
        
        if not user_email:
            return jsonify({"error": "User email required"}), 400
        
        data = request.get_json()
        new_email = data.get("email", "").strip().lower()
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Verify user exists
        cursor.execute('SELECT id FROM users WHERE email = ?', (user_email,))
        user_result = cursor.fetchone()
        if not user_result:
            conn.close()
            return jsonify({"error": "User not found"}), 404
        
        user_id = user_result[0]
        
        # Update email if provided and different
        if new_email and new_email != user_email:
            # Check if new email already exists
            cursor.execute('SELECT id FROM users WHERE email = ? AND id != ?', (new_email, user_id))
            if cursor.fetchone():
                conn.close()
                return jsonify({"error": "Email already in use"}), 400
            cursor.execute('UPDATE users SET email = ? WHERE id = ?', (new_email, user_id))
            user_email = new_email  # Update for response
        
        conn.commit()
        
        # Check if profile_picture_url column exists
        cursor.execute("PRAGMA table_info(users)")
        columns = [col[1] for col in cursor.fetchall()]
        has_profile_column = 'profile_picture_url' in columns
        
        # Get updated user info
        if has_profile_column:
            cursor.execute('SELECT id, email, is_admin, profile_picture_url FROM users WHERE id = ?', (user_id,))
            user_data = cursor.fetchone()
            profile_picture_url = user_data[3] if user_data else None
        else:
            cursor.execute('SELECT id, email, is_admin FROM users WHERE id = ?', (user_id,))
            user_data = cursor.fetchone()
            profile_picture_url = None
        
        conn.close()
        
        return jsonify({
            "message": "Profile updated successfully",
            "user": {
                "id": user_data[0],
                "email": user_data[1],
                "is_admin": bool(user_data[2]),
                "profile_picture_url": profile_picture_url
            }
        }), 200
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500



# ---------------- Load All Models ----------------
# Assuming you may have multiple models for regions and groups, named conventionally
available_models = {
    "svm": {
        "region": joblib.load(os.path.join(MODEL_DIR, "SVM_region.joblib")),
        "group": joblib.load(os.path.join(MODEL_DIR, "SVM_group.joblib"))
    },
    "randomforest": {  # example random forest
        "region": joblib.load(os.path.join(MODEL_DIR, "RandomForest_region.joblib")),
        "group": joblib.load(os.path.join(MODEL_DIR, "RandomForest_group.joblib"))
    },
    "knn": {  # example KNN
        "region": joblib.load(os.path.join(MODEL_DIR, "KNN_region.joblib")),
        "group": joblib.load(os.path.join(MODEL_DIR, "KNN_group.joblib"))
    },
    "logisticregression": {  # example logistic regression
        "region": joblib.load(os.path.join(MODEL_DIR, "LogisticRegression_region.joblib")),
        "group": joblib.load(os.path.join(MODEL_DIR, "LogisticRegression_group.joblib"))
    }
}

scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

feature_columns = (
    ["R_mean","G_mean","B_mean","H_mean","S_mean","V_mean"] +
    ["Texture_mean","Texture_std","Texture_skew","Texture_kurtosis"] +
    ["Edge_mean"] +
    [f"LBP_{i}" for i in range(256)]
)

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message": "Backend is running fine",
        "model_loaded": True
    }), 200

@app.route("/predict_region_group", methods=["POST"])
def predict():
    # Get model name from header, default to SVM if not provided
    model_name = request.headers.get("X-Model-Name", "svm").lower()
    if model_name not in available_models:
        return jsonify({"error": f"Model '{model_name}' not available"}), 400

    model_region = available_models[model_name]["region"]
    model_group = available_models[model_name]["group"]

    data = request.get_json()
    if not data or "rows" not in data:
        return jsonify({"error": "No data provided"}), 400

    df = pd.DataFrame(data["rows"])

    # Fill missing features with 0
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    try:
        X = df[feature_columns].astype(float).values
    except Exception as e:
        return jsonify({"error": f"Feature conversion error: {str(e)}"}), 400

    # Scale features
    try:
        X_scaled = scaler.transform(X)
    except Exception as e:
        return jsonify({"error": f"Scaler transform error: {str(e)}"}), 500

    # Predict
    try:
        predicted_region = model_region.predict(X_scaled)
        predicted_group = model_group.predict(X_scaled)
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

    df["predicted_region"] = predicted_region
    df["predicted_group"] = predicted_group

    # Optional: fill NaNs in original fields with 0
    for col in ["R_mean","G_mean","B_mean","H_mean","S_mean","V_mean"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return jsonify({"results": df.to_dict(orient="records")})

#======================================================
# ---------------- Feature extraction functions ----------------
def extract_color_features(img):
    r_mean = np.mean(img[:,:,2])
    g_mean = np.mean(img[:,:,1])
    b_mean = np.mean(img[:,:,0])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_mean = np.mean(hsv[:,:,0])
    s_mean = np.mean(hsv[:,:,1])
    v_mean = np.mean(hsv[:,:,2])
    return [r_mean, g_mean, b_mean, h_mean, s_mean, v_mean]

def extract_texture_features(gray):
    mean = np.mean(gray)
    std = np.std(gray)
    skewness = np.mean((gray - mean)**3) / (std**3 + 1e-8)
    kurtosis = np.mean((gray - mean)**4) / (std**4 + 1e-8)
    return [mean, std, skewness, kurtosis]

def extract_lbp_like(gray):
    lbp = np.zeros_like(gray)
    for i in range(1, gray.shape[0]-1):
        for j in range(1, gray.shape[1]-1):
            c = gray[i,j]
            b = 0
            b |= (gray[i-1,j-1] > c) << 7
            b |= (gray[i-1,j  ] > c) << 6
            b |= (gray[i-1,j+1] > c) << 5
            b |= (gray[i,j+1]   > c) << 4
            b |= (gray[i+1,j+1] > c) << 3
            b |= (gray[i+1,j]   > c) << 2
            b |= (gray[i+1,j-1] > c) << 1
            b |= (gray[i,j-1]   > c) << 0
            lbp[i,j] = b
    hist = cv2.calcHist([lbp],[0],None,[256],[0,256]).flatten()
    hist = hist / (hist.sum() + 1e-8)
    return hist.tolist()

def extract_edge_features(gray):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobelx**2 + sobely**2)
    return [np.mean(mag)]

# ---------------- API Endpoint ----------------
@app.route("/extract_features", methods=["POST"])
def extract_features_api():
    """
    Expects FormData POST:
    - images: multiple image files
    - output_csv: CSV file name
    """
    try:
        files = request.files.getlist("images")
        output_csv_name = request.form.get("output_csv", "handcrafted_features.csv")
        if not files:
            return jsonify({"error": "No images uploaded"}), 400

        # Create a temp folder to save uploaded files
        temp_dir = tempfile.mkdtemp()

        features, labels, regions, groups, group_labels, paths = [], [], [], [], [], []
        group_classes = ["OP", "BOP", "BOPF"]

        # Save uploaded files
        for f in files:
            save_path = os.path.join(temp_dir, f.filename)
            f.save(save_path)
            # ================= Process each image =================
            try:
                img = cv2.imread(save_path)
                if img is None:
                    continue
                img = cv2.resize(img, (224,224))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except:
                continue

            # Extract group from filename (example: NU_OP_001.jpg)
            name = os.path.splitext(f.filename)[0]
            parts = name.split("_")
            if len(parts) < 2:
                continue
            group = parts[1]
            if group not in group_classes:
                continue
            group_id = group_classes.index(group)
            region = parts[0]

            f_color = extract_color_features(img)
            f_texture = extract_texture_features(gray)
            f_edge = extract_edge_features(gray)
            f_lbp = extract_lbp_like(gray)
            feature_vector = f_color + f_texture + f_edge + f_lbp

            features.append(feature_vector)
            labels.append(region)
            regions.append(region)
            groups.append(group)
            group_labels.append(group_id)
            paths.append(f.filename)

        if not features:
            shutil.rmtree(temp_dir)
            return jsonify({"error": "No valid images found"}), 400

        # Save CSV in temp folder
        columns = (
            ["R_mean","G_mean","B_mean","H_mean","S_mean","V_mean"] +
            ["Texture_mean","Texture_std","Texture_skew","Texture_kurtosis"] +
            ["Edge_mean"] +
            [f"LBP_{i}" for i in range(256)]
        )
        df = pd.DataFrame(features, columns=columns)
        df["region_label"] = labels
        df["region"] = df["region_label"].apply(lambda x: "Dimbula Region" if x == "DI" else "Uva Region" if x == "UV" else "Nuwara Eliya Region" if x == "NU" else "Sabaragamuwa Region" if x == "SB" else "Kandy Region" if x == "KA" else "Ruhuna Region" if x == "RU" else "Udapussellawa Region")
        df["group"] = groups
        df["group_label"] = group_labels
        df["path"] = paths


        output_csv_path = os.path.join(temp_dir, output_csv_name)
        df.to_csv(output_csv_path, index=False)

        # Send CSV file back
        response = send_file(output_csv_path, as_attachment=True)
        return response

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500
# ================= Server Setup ======================
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def find_tea_circle(img):
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    B = cv2.GaussianBlur(lab[:,:,2], (9,9), 0)
    _, mask = cv2.threshold(B, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cnt = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(cnt) > 0.01*h*w:
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            cx,cy,r = int(round(cx)), int(round(cy)), int(round(r))
            r = max(5, min(r, cx, cy, w-cx-1, h-cy-1))
            return cx,cy,r
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),(9,9),2)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min(h,w)//4,param1=80,param2=40,minRadius=min(h,w)//8,maxRadius=0)
    if circles is not None:
        x,y,r = max(np.round(circles[0]).astype(int), key=lambda c: c[2])
        r = max(5, min(r, x, y, w-x-1, h-y-1))
        return x,y,r
    return None

def remove_reflection(img, circle_mask):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(hsv)
    bright = cv2.inRange(V,220,255)
    low_sat = cv2.inRange(S,0,60)
    spec = cv2.bitwise_and(bright, low_sat)
    spec = cv2.bitwise_and(spec, circle_mask)
    k = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
    spec = cv2.morphologyEx(spec, cv2.MORPH_OPEN,k,1)
    spec = cv2.morphologyEx(spec, cv2.MORPH_CLOSE,k,1)
    if np.count_nonzero(spec)==0:
        return img
    return cv2.inpaint(img, spec, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

def crop_circle_png(img,x,y,r):
    h,w = img.shape[:2]
    mask = np.zeros((h,w),np.uint8)
    cv2.circle(mask,(x,y),int(0.80*r),255,-1)
    inner_radius = int(r*0.42)
    ring_mask = np.zeros((h,w),np.uint8)
    cv2.circle(ring_mask,(x,y),inner_radius+15,255,-1)
    cv2.circle(ring_mask,(x,y),inner_radius,0,-1)
    ring_mask = cv2.bitwise_and(ring_mask, mask)
    ring_pixels = img[ring_mask==255]
    if len(ring_pixels)>0:
        inner_mask = np.zeros((h,w),np.uint8)
        cv2.circle(inner_mask,(x,y),inner_radius,255,-1)
        indices = np.argwhere(inner_mask==255)
        for idx in indices:
            y_idx,x_idx = idx
            img[y_idx,x_idx] = ring_pixels[np.random.randint(len(ring_pixels))]
    else:
        mean_color=[128,90,60]
        cv2.circle(img,(x,y),inner_radius,mean_color,-1)
    cleaned = remove_reflection(img, mask)
    b,g,rch = cv2.split(cleaned)
    alpha = mask
    rgba = cv2.merge([b,g,rch,alpha])
    x1,x2 = max(0,x-r), min(w,x+r)
    y1,y2 = max(0,y-r), min(h,y+r)
    return rgba[y1:y2,x1:x2]

#======================================================

@app.route("/crop_reflection", methods=["POST"])
def crop_reflection_route():
    file_path = None
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error":"No file uploaded"}),400
        # Sanitize filename to prevent directory traversal issues
        safe_filename = os.path.basename(file.filename) if file.filename else "uploaded_image.jpg"
        safe_filename = safe_filename.replace("\\", "_").replace("/", "_").replace("..", "_")
        if not safe_filename or safe_filename.strip() == "":
            safe_filename = f"upload_{int(time.time())}.jpg"
        file_path = os.path.join(UPLOAD_DIR, safe_filename)
        file.save(file_path)
        
        img_bgr = cv2.imread(file_path)
        if img_bgr is None:
            return jsonify({"error":"Invalid image"}),400
        circle = find_tea_circle(img_bgr)
        if circle is None:
            return jsonify({"error":"No tea circle detected"}),400
        x,y,r = circle
        cropped_rgba = crop_circle_png(img_bgr,x,y,r)
        _, buffer = cv2.imencode(".png", cropped_rgba)
        cropped_base64 = base64.b64encode(buffer).decode("utf-8")
        return jsonify({"cropped_image":f"data:image/png;base64,{cropped_base64}"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass


# ================= Authentication Routes ============
@app.route("/register", methods=["POST"])
def register():
    """Register a new user"""
    try:
        data = request.get_json()
        email = data.get("email", "").strip().lower()
        password = data.get("password", "").strip()
        
        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400
        
        if len(password) < 4:
            return jsonify({"error": "Password must be at least 4 characters long"}), 400
        
        # Ensure column exists before querying
        ensure_profile_picture_column()
        
        # Check if user already exists
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
        if cursor.fetchone():
            conn.close()
            return jsonify({"error": "Email already registered"}), 400
        
        # Create new user
        password_hash = generate_password_hash(password)
        is_admin = 1 if email == 'pramudithapasindu48@gmail.com' else 0
        
        cursor.execute('''
            INSERT INTO users (email, password_hash, is_admin, created_at)
            VALUES (?, ?, ?, ?)
        ''', (email, password_hash, is_admin, datetime.now().isoformat()))
        conn.commit()
        user_id = cursor.lastrowid
        
        # Get profile picture URL if exists (before closing connection)
        cursor.execute("PRAGMA table_info(users)")
        columns = [col[1] for col in cursor.fetchall()]
        has_profile_column = 'profile_picture_url' in columns
        
        if has_profile_column:
            cursor.execute('SELECT profile_picture_url FROM users WHERE id = ?', (user_id,))
            result = cursor.fetchone()
            profile_picture_url = result[0] if result else None
        else:
            profile_picture_url = None
        
        conn.close()
        
        return jsonify({
            "message": "User registered successfully",
            "user": {
                "id": user_id,
                "email": email,
                "is_admin": bool(is_admin),
                "profile_picture_url": profile_picture_url
            }
        }), 201
        
    except sqlite3.Error as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/login", methods=["POST"])
def login():
    """Authenticate user and return user info"""
    try:
        data = request.get_json()
        email = data.get("email", "").strip().lower()
        password = data.get("password", "").strip()
        
        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400
        
        # Ensure column exists before querying
        ensure_profile_picture_column()
        
        # Check user credentials
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if profile_picture_url column exists
        cursor.execute("PRAGMA table_info(users)")
        columns = [col[1] for col in cursor.fetchall()]
        has_profile_column = 'profile_picture_url' in columns
        
        if has_profile_column:
            cursor.execute('SELECT id, email, password_hash, is_admin, profile_picture_url FROM users WHERE email = ?', (email,))
        else:
            cursor.execute('SELECT id, email, password_hash, is_admin FROM users WHERE email = ?', (email,))
        
        user = cursor.fetchone()
        conn.close()
        
        if not user:
            return jsonify({"error": "Invalid credentials"}), 401
        
        if has_profile_column:
            user_id, user_email, password_hash, is_admin, profile_picture_url = user
        else:
            user_id, user_email, password_hash, is_admin = user
            profile_picture_url = None
        
        # Verify password
        if not check_password_hash(password_hash, password):
            return jsonify({"error": "Invalid credentials"}), 401
        
        return jsonify({
            "message": "Login successful",
            "user": {
                "id": user_id,
                "email": user_email,
                "is_admin": bool(is_admin),
                "profile_picture_url": profile_picture_url
            }
        }), 200
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

#======================================================
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

if __name__ == "__main__":
    local_ip = get_local_ip()
    print(f"✅ Server running on:")
    print(f"  Localhost: http://127.0.0.1:5000")
    print(f"  Network:   http://{local_ip}:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)