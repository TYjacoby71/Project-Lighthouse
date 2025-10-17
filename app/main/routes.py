from __future__ import annotations

from datetime import datetime
import os
import uuid
import hashlib
import json

from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, current_app
from flask_login import login_required, current_user
import boto3
from sqlalchemy import or_, and_

from .. import db
from ..models import Communication

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    return redirect(url_for('auth.login'))


@main_bp.route('/dashboard')
@login_required
def dashboard():
    q = request.args.get('q', '').strip()
    sender = request.args.get('sender', '').strip()
    recipients = request.args.get('recipients', '').strip()
    start = request.args.get('start', '').strip()
    end = request.args.get('end', '').strip()

    query = db.session.query(Communication).filter(
        Communication.organization_id == current_user.organization_id
    )
    if sender:
        query = query.filter(Communication.sender.ilike(f"%{sender}%"))
    if recipients:
        query = query.filter(Communication.recipients.ilike(f"%{recipients}%"))
    if q:
        like = f"%{q}%"
        query = query.filter(
            or_(
                Communication.content_md.ilike(like),
                Communication.original_filename.ilike(like),
                Communication.sender.ilike(like),
                Communication.recipients.ilike(like),
            )
        )
    if start:
        try:
            start_dt = datetime.fromisoformat(start)
            query = query.filter(Communication.sent_at >= start_dt)
        except ValueError:
            pass
    if end:
        try:
            end_dt = datetime.fromisoformat(end)
            query = query.filter(Communication.sent_at <= end_dt)
        except ValueError:
            pass

    communications = query.order_by(Communication.created_at.desc()).limit(100).all()
    return render_template('main/dashboard.html', communications=communications)


@main_bp.route('/communication/<int:communication_id>')
@login_required
def communication_detail(communication_id: int):
    communication = db.session.get(Communication, communication_id)
    if not communication or communication.organization_id != current_user.organization_id:
        flash('Not found.', 'danger')
        return redirect(url_for('main.dashboard'))

    # Render Markdown content on the server for now
    from markdown import markdown
    import bleach

    allowed_tags = bleach.sanitizer.ALLOWED_TAGS.union({'p', 'pre', 'code', 'ul', 'ol', 'li', 'blockquote', 'hr', 'br'})
    html_content = bleach.clean(
        markdown(communication.content_md or ''),
        tags=allowed_tags,
        strip=True,
    )

    return render_template('main/detail.html', communication=communication, html_content=html_content)


@main_bp.route('/api/communications/upload', methods=['POST'])
@login_required
def api_upload():
    # Validate inputs
    sender = request.form.get('sender', '').strip()
    recipients = request.form.get('recipients', '').strip()
    date_str = request.form.get('date', '').strip()
    text_content = request.form.get('text_content', '').strip()
    file = request.files.get('file')

    if not sender or not recipients:
        return jsonify({'error': 'sender and recipients are required'}), 400

    sent_at = None
    if date_str:
        try:
            sent_at = datetime.fromisoformat(date_str)
        except ValueError:
            return jsonify({'error': 'Invalid date format. Use ISO 8601 (YYYY-MM-DD or with time).'}), 400

    if not file and not text_content:
        return jsonify({'error': 'Provide a PDF file or text content'}), 400

    # Compute sha256 of original file or text
    if file:
        file_bytes = file.read()
        file.seek(0)
        sha256_hash = hashlib.sha256(file_bytes).hexdigest()
        original_filename = file.filename
    else:
        file_bytes = text_content.encode('utf-8')
        sha256_hash = hashlib.sha256(file_bytes).hexdigest()
        original_filename = 'pasted_text.txt'

    # Upload original to S3 under org prefix, or local fallback if AWS config missing
    storage_uuid = str(uuid.uuid4())
    org_prefix = str(current_user.organization_id)
    aws_region = (current_app.config.get('AWS_REGION') or '').strip()
    aws_bucket = (current_app.config.get('AWS_S3_BUCKET') or '').strip()
    use_s3 = bool(aws_region and aws_bucket)

    if use_s3:
        s3 = boto3.client('s3', region_name=aws_region)
        storage_path = f"{org_prefix}/{storage_uuid}"
        try:
            s3.put_object(
                Bucket=aws_bucket,
                Key=storage_path,
                Body=file_bytes,
                ACL='private',
                ServerSideEncryption='AES256',
                Metadata={'original_filename': original_filename, 'sha256': sha256_hash},
            )
        except Exception:
            current_app.logger.exception('S3 upload failed')
            return jsonify({'error': 'Upload failed'}), 500
    else:
        # Local dev fallback
        uploads_dir = os.path.join(current_app.instance_path, 'uploads', org_prefix)
        os.makedirs(uploads_dir, exist_ok=True)
        storage_filename = storage_uuid
        storage_path_local = os.path.join(uploads_dir, storage_filename)
        with open(storage_path_local, 'wb') as f:
            f.write(file_bytes)
        storage_path = f"local:{storage_path_local}"

    # Extract text and convert to Markdown
    content_md = None
    if file and (original_filename.lower().endswith('.pdf')):
        try:
            import fitz  # PyMuPDF
            with fitz.open(stream=file_bytes, filetype='pdf') as doc:
                text = "\n\n".join(page.get_text("text") for page in doc)
        except Exception:
            text = ''
    else:
        text = text_content or file_bytes.decode('utf-8', errors='ignore')

    # Basic cleanup
    if text:
        cleaned = '\n'.join([line.strip() for line in text.splitlines()])
        from markdownify import markdownify as md
        paragraphs = [p for p in cleaned.split('\n')]
        htmlish = ''.join(f'<p>{p}</p>' if p else '<br/>' for p in paragraphs)
        content_md = md(htmlish)

    # NLP metrics: word/char, readability, sentiment, POS, entities
    nlp_metrics = {
        'word_count': len(text.split()) if text else 0,
        'char_count': len(text) if text else 0,
        'readability_score': None,
        'pos_counts': {},
        'named_entities': [],
        'sentiment_score': 0.0,
    }
    try:
        # readability
        try:
            import textstat
            nlp_metrics['readability_score'] = float(textstat.flesch_kincaid_grade(text or ''))
        except Exception:
            pass

        # sentiment via VADER or TextBlob fallback
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            vs = SentimentIntensityAnalyzer()
            nlp_metrics['sentiment_score'] = float(vs.polarity_scores(text or '').get('compound', 0.0))
        except Exception:
            try:
                from textblob import TextBlob
                nlp_metrics['sentiment_score'] = float(TextBlob(text or '').sentiment.polarity)
            except Exception:
                pass

        # spaCy NER + POS counts
        try:
            import spacy
            nlp = spacy.load('en_core_web_sm')
            doc = nlp(text or '')
            ents = []
            for ent in doc.ents:
                if ent.label_ in {'PERSON', 'ORG', 'GPE', 'DATE', 'MONEY'}:
                    ents.append({'text': ent.text, 'label': ent.label_})
            nlp_metrics['named_entities'] = ents
            pos_counts = {'NOUN': 0, 'VERB': 0, 'ADJ': 0, 'ADV': 0}
            for token in doc:
                if token.pos_ in pos_counts:
                    pos_counts[token.pos_] += 1
            nlp_metrics['pos_counts'] = pos_counts
        except Exception:
            pass
    except Exception:
        pass

    # Gemini analysis (placeholder request structure)
    ai_analysis = {
        'theme': None,
        'keywords': [],
        'emotional_tone': None,
        'linguistic_patterns': [],
        'metrics': nlp_metrics,
    }
    try:
        import requests
        gemini_key = current_app.config.get('GEMINI_API_KEY')
        if gemini_key:
            prompt = {
                'system': 'You are a precise analyst of written communications. Return strict JSON.',
                'user': (
                    "Analyze the following text. Return JSON with: "
                    "theme (string), keywords (array of strings), emotional_tone (string), "
                    "linguistic_patterns (array of strings, e.g., blame-shifting, gaslighting). "
                    "Text:\n" + (text or '')
                ),
            }
            # Placeholder endpoint; replace with official Gemini 1.5 Pro API
            resp = requests.post(
                'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent',
                params={'key': gemini_key},
                json={'contents': [{'parts': [{'text': prompt['user']}]}]},
                timeout=30,
            )
            if resp.ok:
                data = resp.json()
                # Very naive extraction; in production parse according to Gemini schema
                text_out = json.dumps(data)[:8192]
                ai_analysis['theme'] = ai_analysis['theme'] or 'auto'
                ai_analysis['keywords'] = ai_analysis['keywords'] or []
                ai_analysis['emotional_tone'] = ai_analysis['emotional_tone'] or 'neutral'
                ai_analysis['linguistic_patterns'] = ai_analysis['linguistic_patterns'] or []
    except Exception:
        pass

    # Save record
    comm = Communication(
        organization_id=current_user.organization_id,
        sender=sender,
        recipients=recipients,
        sent_at=sent_at,
        sha256_hash=sha256_hash,
        storage_path=storage_path,
        original_filename=original_filename,
        content_md=content_md,
        ai_analysis=ai_analysis,
        word_count=nlp_metrics.get('word_count'),
        char_count=nlp_metrics.get('char_count'),
        readability_score=nlp_metrics.get('readability_score'),
        sentiment_score=nlp_metrics.get('sentiment_score'),
        auto_detected_entities=nlp_metrics.get('named_entities'),
        pos_counts=nlp_metrics.get('pos_counts'),
    )
    db.session.add(comm)
    db.session.commit()

    # Update FAISS vector index (placeholder)
    try:
        from langchain.embeddings import FakeEmbeddings
        from langchain_community.vectorstores import FAISS
        embeddings = FakeEmbeddings(size=1536)
        index_dir = os.path.join('/workspace/var/indexes', str(current_user.organization_id))
        os.makedirs(index_dir, exist_ok=True)
        texts = [content_md or (text or '')]
        metadatas = [{'id': comm.id}]
        if os.path.exists(os.path.join(index_dir, 'index.faiss')):
            db_faiss = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
            db_faiss.add_texts(texts, metadatas=metadatas)
            db_faiss.save_local(index_dir)
        else:
            db_faiss = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
            db_faiss.save_local(index_dir)
    except Exception:
        pass

    return jsonify({'id': comm.id}), 201


@main_bp.route('/upload')
@login_required
def upload():
    return render_template('main/upload.html')


@main_bp.route('/api/communications/<int:communication_id>/original')
@login_required
def presigned_url(communication_id: int):
    comm = db.session.get(Communication, communication_id)
    if not comm or comm.organization_id != current_user.organization_id:
        return jsonify({'error': 'Not found'}), 404

    # If stored locally (dev fallback), return a direct download route
    if comm.storage_path.startswith('local:'):
        return jsonify({'url': url_for('main.download_local', communication_id=communication_id, _external=True)})

    s3 = boto3.client('s3', region_name=current_app.config['AWS_REGION'])
    try:
        url = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': current_app.config['AWS_S3_BUCKET'], 'Key': comm.storage_path},
            ExpiresIn=300,
        )
        return jsonify({'url': url})
    except Exception:
        return jsonify({'error': 'Failed to create URL'}), 500


@main_bp.route('/api/communications/<int:communication_id>/integrity')
@login_required
def integrity_check(communication_id: int):
    comm = db.session.get(Communication, communication_id)
    if not comm or comm.organization_id != current_user.organization_id:
        return jsonify({'error': 'Not found'}), 404

    try:
        if comm.storage_path.startswith('local:'):
            path = comm.storage_path[len('local:'):]
            with open(path, 'rb') as f:
                data = f.read()
        else:
            s3 = boto3.client('s3', region_name=current_app.config['AWS_REGION'])
            obj = s3.get_object(Bucket=current_app.config['AWS_S3_BUCKET'], Key=comm.storage_path)
            data = obj['Body'].read()
        h = hashlib.sha256(data).hexdigest()
        return jsonify({'ok': h == comm.sha256_hash, 'stored': comm.sha256_hash, 'computed': h})
    except Exception as e:
        return jsonify({'error': 'Integrity check failed'}), 500


@main_bp.route('/api/communications/<int:communication_id>/download')
@login_required
def download_local(communication_id: int):
    # Only for local fallback
    from flask import send_file
    comm = db.session.get(Communication, communication_id)
    if not comm or comm.organization_id != current_user.organization_id:
        return jsonify({'error': 'Not found'}), 404
    if not comm.storage_path.startswith('local:'):
        return jsonify({'error': 'Not a local object'}), 400
    path = comm.storage_path[len('local:'):]
    try:
        return send_file(path, as_attachment=True, download_name=comm.original_filename)
    except Exception:
        return jsonify({'error': 'Failed to serve file'}), 500


@main_bp.route('/chat')
@login_required
def chat():
    return render_template('main/chat.html')


@main_bp.route('/api/chat', methods=['POST'])
@login_required
def api_chat():
    data = request.get_json(silent=True) or {}
    question = (data.get('question') or '').strip()
    if not question:
        return jsonify({'error': 'Question is required'}), 400

    # Retrieve from FAISS index
    contexts = []
    try:
        from langchain.embeddings import FakeEmbeddings
        from langchain_community.vectorstores import FAISS
        embeddings = FakeEmbeddings(size=1536)
        index_dir = os.path.join('/workspace/var/indexes', str(current_user.organization_id))
        if os.path.exists(os.path.join(index_dir, 'index.faiss')):
            db_faiss = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
            docs = db_faiss.similarity_search(question, k=4)
            contexts = [d.page_content for d in docs]
        else:
            # fallback: last few communications
            rows = (
                db.session.query(Communication)
                .filter(Communication.organization_id == current_user.organization_id)
                .order_by(Communication.created_at.desc())
                .limit(4)
                .all()
            )
            contexts = [(r.content_md or '') for r in rows]
    except Exception:
        pass

    answer = None
    try:
        import requests
        gemini_key = current_app.config.get('GEMINI_API_KEY')
        if gemini_key:
            prompt_text = (
                "You are a helpful analyst. Use only the context below to answer.\n\n" +
                "\n\n".join([f"Context {i+1}:\n{c}" for i, c in enumerate(contexts)]) +
                "\n\nQuestion: " + question
            )
            resp = requests.post(
                'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent',
                params={'key': gemini_key},
                json={'contents': [{'parts': [{'text': prompt_text}]}]},
                timeout=30,
            )
            if resp.ok:
                data = resp.json()
                # naive extraction - depends on API response shape
                answer = data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text')
    except Exception:
        pass

    if not answer:
        # Local fallback
        answer = "Analysis unavailable. Please configure GEMINI_API_KEY."
    return jsonify({'answer': answer, 'contexts': contexts})


@main_bp.route('/sender/<string:sender_name>')
@login_required
def sender_stats(sender_name: str):
    rows = (
        db.session.query(Communication)
        .filter(
            Communication.organization_id == current_user.organization_id,
            Communication.sender == sender_name,
        )
        .order_by(Communication.sent_at.asc().nulls_last())
        .all()
    )
    # Build stats structures for charts
    timeline = []
    tones = {}
    patterns = {}
    keywords = {}
    for r in rows:
        metrics = (r.ai_analysis or {}).get('metrics') or {}
        timeline.append({'t': (r.sent_at.isoformat() if r.sent_at else None), 'sentiment': metrics.get('sentiment_score', 0.0)})
        tone = (r.ai_analysis or {}).get('emotional_tone') or 'unknown'
        tones[tone] = tones.get(tone, 0) + 1
        for p in (r.ai_analysis or {}).get('linguistic_patterns', []) or []:
            patterns[p] = patterns.get(p, 0) + 1
        for k in (r.ai_analysis or {}).get('keywords', []) or []:
            keywords[k] = keywords.get(k, 0) + 1
    return render_template('main/sender_stats.html', sender_name=sender_name, timeline=timeline, tones=tones, patterns=patterns, keywords=keywords)


@main_bp.route('/api/communications/<int:communication_id>/draft', methods=['POST'])
@login_required
def draft_response(communication_id: int):
    comm = db.session.get(Communication, communication_id)
    if not comm or comm.organization_id != current_user.organization_id:
        return jsonify({'error': 'Not found'}), 404
    base_prompt = (
        "You are a paralegal assistant specializing in clear, non-emotional communication. "
        "Help the user draft a response to the message below. Return plain text.\n\n" \
        + (comm.content_md or '')
    )
    text_out = None
    try:
        import requests
        gemini_key = current_app.config.get('GEMINI_API_KEY')
        if gemini_key:
            resp = requests.post(
                'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent',
                params={'key': gemini_key},
                json={'contents': [{'parts': [{'text': base_prompt}]}]},
                timeout=30,
            )
            if resp.ok:
                data = resp.json()
                text_out = data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text')
    except Exception:
        pass
    if not text_out:
        text_out = "AI draft unavailable. Configure GEMINI_API_KEY."
    return jsonify({'draft': text_out})
