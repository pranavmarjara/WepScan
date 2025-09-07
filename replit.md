# WepScan - AI Weapon Detection System

## Overview

WepScan is a Flask-based web application that simulates an AI-powered weapon detection system for X-ray luggage screening. The application allows users to upload X-ray images and receive mock detection results for weapons, explosives, and suspicious objects. It features a modern web interface with real-time threat assessment, session-based history tracking, and a responsive design optimized for security personnel workflows.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Template Engine**: Jinja2 templates with Bootstrap 5 dark theme for responsive UI
- **Static Assets**: CSS and JavaScript files served from Flask's static folder
- **User Interface**: Modern, security-focused design with drag-and-drop file upload, real-time feedback, and threat level indicators
- **Client-side Logic**: Vanilla JavaScript for file handling, upload zone interactions, and dynamic UI updates

### Backend Architecture
- **Web Framework**: Flask with session-based state management
- **File Processing**: OpenCV for image manipulation and processing
- **Detection Engine**: Mock YOLO-style detector (`WepScanDetector`) that simulates weapon detection results
- **Session Management**: Flask sessions for user history tracking without persistent storage
- **File Handling**: Secure file upload with extension validation and size limits (16MB max)

### Data Storage Solutions
- **File Storage**: Local filesystem storage in `static/uploads` and `static/processed` directories
- **Session Data**: In-memory session storage for detection history (no persistent database)
- **Configuration**: Environment variables for sensitive data like session secrets

### Security Features
- **File Validation**: Whitelist-based file extension checking (png, jpg, jpeg, tif, tiff)
- **Secure Filenames**: Werkzeug's secure_filename for preventing path traversal attacks
- **File Size Limits**: 16MB maximum upload size to prevent resource exhaustion
- **Proxy Support**: ProxyFix middleware for deployment behind reverse proxies

### Detection System
- **Mock AI Engine**: Simulated YOLOv8-style detection with confidence scoring and bounding box generation
- **Threat Classification**: Multi-level threat assessment (high, medium, low, safe)
- **Alert System**: Configurable confidence thresholds for security alerts
- **Result Visualization**: Bounding box overlays with color-coded threat levels

## External Dependencies

### Core Framework Dependencies
- **Flask**: Web application framework for routing, templating, and session management
- **Werkzeug**: WSGI utilities for secure file handling and proxy support
- **OpenCV (cv2)**: Computer vision library for image processing and manipulation
- **NumPy**: Numerical computing for array operations and image data handling

### Frontend Dependencies
- **Bootstrap 5**: CSS framework with dark theme for responsive UI components
- **Font Awesome 6**: Icon library for security-themed iconography
- **Custom CSS/JS**: Application-specific styling and interactive behaviors

### Development Dependencies
- **Python Standard Library**: UUID generation, datetime handling, os operations, and logging
- **Environment Variables**: Configuration management for deployment flexibility

### Future Integration Points
- **YOLOv8/Ultralytics**: Real AI model training and inference (currently simulated)
- **GDXray Dataset**: X-ray image dataset for training weapon detection models
- **Database Systems**: Potential integration for persistent storage of scan results and user data
- **Cloud Storage**: Alternative to local file storage for production deployments