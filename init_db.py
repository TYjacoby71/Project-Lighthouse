
#!/usr/bin/env python3
"""Initialize the database with tables."""

from app import create_app, db

def init_database():
    app = create_app()
    with app.app_context():
        db.create_all()
        print("Database initialized successfully!")

if __name__ == '__main__':
    init_database()
