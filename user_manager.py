import sqlite3
import os
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

DB_NAME = "users.db"

class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

def init_db():
    """Initialize the SQLite database with a default admin user."""
    # Check if database file exists
    if not os.path.exists(DB_NAME):
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute('''CREATE TABLE user 
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password TEXT)''')
        
        # Create default admin
        # FIX: Removed method='sha256' to let Werkzeug use its secure default (scrypt/pbkdf2)
        default_pass = generate_password_hash("admin123") 
        
        c.execute("INSERT INTO user (username, password) VALUES (?, ?)", ("admin", default_pass))
        
        conn.commit()
        conn.close()
        print("✓ Database initialized with default user: admin / admin123")
    else:
        print("✓ Database already exists.")

def get_user(user_id):
    """Retrieve user by ID for Flask-Login session management."""
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT id, username FROM user WHERE id=?", (user_id,))
        res = c.fetchone()
        conn.close()
        if res:
            return User(id=res[0], username=res[1])
    except Exception as e:
        print(f"Error fetching user: {e}")
    return None

def verify_user(username, password):
    """Verify credentials during login."""
    try:
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("SELECT id, username, password FROM user WHERE username=?", (username,))
        res = c.fetchone()
        conn.close()
        
        if res and check_password_hash(res[2], password):
            return User(id=res[0], username=res[1])
    except Exception as e:
        print(f"Login error: {e}")
    return None