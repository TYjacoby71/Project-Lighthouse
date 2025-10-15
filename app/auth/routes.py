from __future__ import annotations

from flask import Blueprint, render_template, redirect, url_for, request, flash
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

from .. import db
from ..models import User, Organization

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        name = request.form.get('name', '').strip()
        org_name = request.form.get('organization', '').strip()

        if not email or not password or not org_name:
            flash('Email, password, and organization are required.', 'danger')
            return render_template('auth/register.html')

        if db.session.query(User).filter_by(email=email).first():
            flash('Email already registered.', 'danger')
            return render_template('auth/register.html')

        organization = Organization(name=org_name)
        db.session.add(organization)
        db.session.flush()

        user = User(email=email, name=name, password_hash=generate_password_hash(password), organization_id=organization.id)
        db.session.add(user)
        db.session.commit()

        flash('Registration successful. Please log in.', 'success')
        return redirect(url_for('auth.login'))

    return render_template('auth/register.html')


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        user = db.session.query(User).filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('main.dashboard'))
        flash('Invalid credentials.', 'danger')
    return render_template('auth/login.html')


@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))
