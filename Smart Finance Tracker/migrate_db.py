# migrate_db.py
import sqlite3

def migrate():
    print("--- Starting Database Migration ---")
    conn = sqlite3.connect('finance.db')
    cursor = conn.cursor()

    try:
        # 1. Add 'full_name' column
        cursor.execute("ALTER TABLE users ADD COLUMN full_name VARCHAR DEFAULT 'User'")
        print("✅ Added 'full_name' column.")
    except sqlite3.OperationalError:
        print("ℹ️ 'full_name' already exists.")

    try:
        # 2. Add 'country' column
        cursor.execute("ALTER TABLE users ADD COLUMN country VARCHAR DEFAULT 'Pakistan'")
        print("✅ Added 'country' column.")
    except sqlite3.OperationalError:
        print("ℹ️ 'country' already exists.")

    try:
        # 3. Add 'currency' column
        cursor.execute("ALTER TABLE users ADD COLUMN currency VARCHAR DEFAULT 'PKR'")
        print("✅ Added 'currency' column.")
    except sqlite3.OperationalError:
        print("ℹ️ 'currency' already exists.")

    try:
        # 4. Add 'profile_pic' column
        cursor.execute("ALTER TABLE users ADD COLUMN profile_pic VARCHAR DEFAULT '/static/default_profile.png'")
        print("✅ Added 'profile_pic' column.")
    except sqlite3.OperationalError:
        print("ℹ️ 'profile_pic' already exists.")

    conn.commit()
    conn.close()
    print("--- Migration Complete! Your data is safe. ---")

if __name__ == "__main__":
    migrate()