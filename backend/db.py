# backend/db.py
# MySQL connection — yahan apna password daalo

import mysql.connector
from mysql.connector import Error

DB_CONFIG = {
    "host":     "localhost",
    "user":     "root",
    "password": "YOUR_MYSQL_PASSWORD",  # <-- apna MySQL password yahan
    "database": "deepfake_detector",
    "charset":  "utf8mb4"
}

def get_connection():
    """Har API call pe fresh connection deta hai"""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except Error as e:
        print(f"DB Connection Error: {e}")
        return None

def query(sql, params=None, fetch=False):
    """
    fetch=False  → INSERT / UPDATE / DELETE
    fetch=True   → SELECT (returns list of dicts)
    fetch='one'  → SELECT single row (returns dict or None)
    """
    conn = get_connection()
    if not conn:
        return None

    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(sql, params or ())

        if fetch == 'one':
            result = cursor.fetchone()
        elif fetch:
            result = cursor.fetchall()
        else:
            conn.commit()
            result = cursor.lastrowid  # INSERT ke baad new id milti hai

        return result
    except Error as e:
        print(f"Query Error: {e}")
        conn.rollback()
        return None
    finally:
        cursor.close()
        conn.close()