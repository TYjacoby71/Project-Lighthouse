from __future__ import annotations

import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import LoginManager
from dotenv import load_dotenv

# Extensions
db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()


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

    # Register blueprints
    from .auth.routes import auth_bp
    from .main.routes import main_bp
    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)

    # Models import for Alembic
    from . import models  # noqa: F401

    return app
