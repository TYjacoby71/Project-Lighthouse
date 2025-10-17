
#!/usr/bin/env python3
"""Run database migrations."""

from app import create_app, db
from flask_migrate import upgrade, init, migrate

def run_migrations():
    app = create_app()
    with app.app_context():
        try:
            # Try to upgrade first
            upgrade()
            print("Database migrations applied successfully!")
        except Exception as e:
            print(f"Migration failed: {e}")
            print("Initializing new migration repository...")
            try:
                init()
                migrate(message="Initial migration")
                upgrade()
                print("Database initialized and migrated successfully!")
            except Exception as e2:
                print(f"Failed to initialize: {e2}")
                print("Creating tables directly...")
                db.create_all()
                print("Database tables created successfully!")

if __name__ == '__main__':
    run_migrations()
