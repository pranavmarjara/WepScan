from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class Settings(db.Model):
    """Application settings model"""
    __tablename__ = 'settings'
    
    id = db.Column(db.Integer, primary_key=True)
    detection_sensitivity = db.Column(db.Integer, default=75, nullable=False)  # 0-100%
    danger_threshold = db.Column(db.Integer, default=60, nullable=False)       # 0-100%
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    @classmethod
    def get_current_settings(cls):
        """Get current settings or create defaults if none exist"""
        settings = cls.query.first()
        if not settings:
            settings = cls(detection_sensitivity=75, danger_threshold=60)
            db.session.add(settings)
            db.session.commit()
        return settings
    
    def update_settings(self, detection_sensitivity, danger_threshold):
        """Update settings values"""
        self.detection_sensitivity = max(0, min(100, int(detection_sensitivity)))
        self.danger_threshold = max(0, min(100, int(danger_threshold)))
        self.updated_at = datetime.utcnow()
        db.session.commit()
        return self

    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'detection_sensitivity': self.detection_sensitivity,
            'danger_threshold': self.danger_threshold,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }