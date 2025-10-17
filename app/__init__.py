from __future__ import annotations

import os
from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from authlib.integrations.flask_client import OAuth
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from dotenv import load_dotenv
from sqlalchemy import inspect, text

# Extensions
db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()
limiter = Limiter(key_func=get_remote_address)
oauth = OAuth()


def create_app(test_config: dict | None = None) -> Flask:
    # Load environment
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)

    app = Flask(__name__, instance_relative_config=True)

    # Basic config
    # Use SQLite DB under instance folder by default so it persists and matches shipped DB
    default_sqlite_uri = f"sqlite:///{os.path.join(app.instance_path, 'lighthouse.db')}"
    app.config.from_mapping(
        SECRET_KEY=os.getenv('SECRET_KEY', 'dev-secret-key'),
        SQLALCHEMY_DATABASE_URI=os.getenv('DATABASE_URL', default_sqlite_uri),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        MAX_CONTENT_LENGTH=int(os.getenv('UPLOAD_MAX_SIZE_MB', '25')) * 1024 * 1024,
        AWS_REGION=os.getenv('AWS_REGION', ''),
        AWS_S3_BUCKET=os.getenv('AWS_S3_BUCKET', ''),
        GEMINI_API_KEY=os.getenv('GEMINI_API_KEY', ''),
        ORIGIN=os.getenv('ORIGIN', 'http://localhost:5000'),
    )

    if test_config:
        app.config.update(test_config)

    # Ensure instance folder exists
    try:
        os.makedirs(app.instance_path, exist_ok=True)
    except OSError:
        pass

    # Init extensions
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    limiter.init_app(app)
    oauth.init_app(app)

    # Optional: register Google OAuth if configured
    google_client_id = os.getenv('GOOGLE_CLIENT_ID')
    google_client_secret = os.getenv('GOOGLE_CLIENT_SECRET')
    if google_client_id and google_client_secret:
        oauth.register(
            name='google',
            client_id=google_client_id,
            client_secret=google_client_secret,
            server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
            client_kwargs={'scope': 'openid email profile'},
        )

    # Register blueprints
    from .auth.routes import auth_bp
    from .main.routes import main_bp
    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)

    # Models import for Alembic
    from . import models  # noqa: F401

    # Dev-friendly auto-migration for SQLite: create new tables/columns if missing
    try:
        with app.app_context():
            engine = db.engine
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            with engine.connect() as conn:
                # conversations table (SQLite simple DDL)
                if 'conversations' not in tables:
                    conn.execute(text(
                        "CREATE TABLE conversations ("
                        "id INTEGER PRIMARY KEY,"
                        "organization_id INTEGER NOT NULL,"
                        "subject VARCHAR(512),"
                        "created_at DATETIME NOT NULL"
                        ")"
                    ))

                # communications new columns (SQLite supports ADD COLUMN without constraints)
                comm_cols = {c['name'] for c in inspector.get_columns('communications')}
                add_cols = [
                    ('conversation_id', 'INTEGER'),
                    ('word_count', 'INTEGER'),
                    ('char_count', 'INTEGER'),
                    ('readability_score', 'FLOAT'),
                    ('sentiment_score', 'FLOAT'),
                    ('auto_detected_entities', 'TEXT'),
                    ('pos_counts', 'TEXT'),
                    ('embedding', 'TEXT'),
                ]
                for col, ddl in add_cols:
                    if col not in comm_cols:
                        conn.execute(text(f"ALTER TABLE communications ADD COLUMN {col} {ddl}"))
    except Exception:
        # Best effort for local dev; use proper migrations in production
        pass

    # Secure cookies and basic security headers
    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE='Lax',
        SESSION_COOKIE_SECURE=os.getenv('SESSION_COOKIE_SECURE', 'false').lower() == 'true',
    )

    @app.after_request
    def set_security_headers(resp):
        csp = (
            "default-src 'self'; "
            "img-src 'self' data:; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "script-src 'self' https://cdn.jsdelivr.net https://cdn.jsdelivr.net/npm/chart.js; "
            "font-src 'self' data:; "
            "connect-src 'self'; "
            "frame-ancestors 'none'"
        )
        resp.headers.setdefault('Content-Security-Policy', csp)
        resp.headers.setdefault('X-Content-Type-Options', 'nosniff')
        resp.headers.setdefault('X-Frame-Options', 'DENY')
        resp.headers.setdefault('Referrer-Policy', 'strict-origin-when-cross-origin')
        return resp

    return app
