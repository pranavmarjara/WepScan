import os
import logging
from flask import Flask, request, render_template, redirect, url_for, flash, session, jsonify
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
import cv2
import numpy as np
from datetime import datetime
import uuid
from detect import WepScanDetector

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "wepscan_secret_key_2024")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Initialize detector
detector = WepScanDetector()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_session_history():
    """Get detection history from session"""
    if 'detection_history' not in session:
        session['detection_history'] = []
    return session['detection_history']

def add_to_history(detection_result):
    """Add detection result to session history"""
    history = get_session_history()
    history.insert(0, detection_result)  # Add to beginning
    # Keep only last 10 results
    session['detection_history'] = history[:10]

@app.route('/')
def index():
    """Main page with upload form and history"""
    history = get_session_history()
    return render_template('index.html', history=history)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing"""
    try:
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '' or file.filename is None:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if not allowed_file(file.filename):
            flash('Invalid file type. Please upload PNG, JPG, JPEG, or TIF files only.', 'error')
            return redirect(request.url)
        
        if file and file.filename and allowed_file(file.filename):
            # Generate unique filename
            filename = secure_filename(file.filename)
            unique_id = str(uuid.uuid4())
            file_extension = filename.rsplit('.', 1)[1].lower()
            unique_filename = f"{unique_id}.{file_extension}"
            
            # Save uploaded file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            # Process the image
            try:
                detection_result = detector.detect_weapons(filepath)
                
                # Generate processed image filename
                processed_filename = f"processed_{unique_filename}"
                processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
                
                # Draw bounding boxes and save processed image
                detector.draw_detections(filepath, detection_result['detections'], processed_path)
                
                # Prepare result data
                result_data = {
                    'id': unique_id,
                    'original_filename': filename,
                    'processed_filename': processed_filename,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'detections': detection_result['detections'],
                    'threat_level': detection_result['threat_level'],
                    'alert_triggered': detection_result['alert_triggered']
                }
                
                # Add to session history
                add_to_history(result_data)
                
                # Log console alert if threats detected
                if result_data['alert_triggered']:
                    threat_items = [d for d in result_data['detections'] if d['confidence'] >= 0.5]
                    threat_names = [d['label'] for d in threat_items]
                    app.logger.warning(f"🚨 SECURITY ALERT: Weapons detected in {filename}")
                    app.logger.warning(f"   Detected items: {', '.join(threat_names)}")
                    app.logger.warning(f"   Threat level: {result_data['threat_level']}")
                    print(f"\n🚨 SECURITY ALERT: Weapons detected in {filename}")
                    print(f"   Detected items: {', '.join(threat_names)}")
                    print(f"   Threat level: {result_data['threat_level']}\n")
                
                return render_template('results.html', result=result_data)
                
            except Exception as e:
                app.logger.error(f"Error processing image: {str(e)}")
                flash(f'Error processing image: {str(e)}', 'error')
                return redirect(url_for('index'))
    
    except Exception as e:
        app.logger.error(f"Upload error: {str(e)}")
        flash(f'Upload failed: {str(e)}', 'error')
        return redirect(url_for('index'))
    
    return redirect(url_for('index'))

@app.route('/history')
def history():
    """View detection history"""
    history = get_session_history()
    return render_template('index.html', history=history, show_history=True)

@app.route('/clear_history')
def clear_history():
    """Clear detection history"""
    session['detection_history'] = []
    flash('Detection history cleared', 'success')
    return redirect(url_for('index'))

@app.errorhandler(413)
def too_large(e):
    flash('File too large. Maximum size is 16MB.', 'error')
    return redirect(url_for('index'))

@app.errorhandler(404)
def not_found(e):
    return render_template('index.html'), 404

@app.errorhandler(500)
def server_error(e):
    app.logger.error(f"Server error: {str(e)}")
    flash('Internal server error occurred', 'error')
    return render_template('index.html'), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
