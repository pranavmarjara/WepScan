// WepScan Frontend JavaScript

document.addEventListener('DOMContentLoaded', function() {
    initializeUploadZone();
    initializeFileInput();
    initializeTooltips();
    updateConfidenceBars();
    initializeAlerts();
});

function initializeUploadZone() {
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');
    
    if (!uploadZone || !fileInput) return;
    
    // Drag and drop functionality
    uploadZone.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });
    
    uploadZone.addEventListener('dragleave', function(e) {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
    });
    
    uploadZone.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelection(files[0]);
        }
    });
    
    // Click to upload
    uploadZone.addEventListener('click', function() {
        fileInput.click();
    });
}

function initializeFileInput() {
    const fileInput = document.getElementById('fileInput');
    const fileName = document.getElementById('fileName');
    
    if (!fileInput) return;
    
    fileInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            handleFileSelection(e.target.files[0]);
        }
    });
}

function handleFileSelection(file) {
    const fileName = document.getElementById('fileName');
    const scanningStatus = document.getElementById('scanningStatus');
    const fileSizeLimit = 16 * 1024 * 1024; // 16MB
    
    // Validate file type
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/tiff', 'image/tif'];
    if (!allowedTypes.includes(file.type)) {
        showAlert('Invalid file type. Please upload PNG, JPG, JPEG, or TIF files only.', 'danger');
        return;
    }
    
    // Validate file size
    if (file.size > fileSizeLimit) {
        showAlert('File too large. Maximum size is 16MB.', 'danger');
        return;
    }
    
    // Update UI
    if (fileName) {
        fileName.textContent = `Selected: ${file.name} (${formatFileSize(file.size)})`;
        fileName.style.display = 'block';
    }
    
    // Show scanning status
    if (scanningStatus) {
        scanningStatus.style.display = 'block';
    }
    
    // Auto-submit the form
    setTimeout(() => {
        document.getElementById('uploadForm').submit();
    }, 500);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function updateConfidenceBars() {
    const confidenceBars = document.querySelectorAll('.confidence-fill');
    
    confidenceBars.forEach(bar => {
        const confidence = parseFloat(bar.dataset.confidence);
        
        // Animate bar width
        setTimeout(() => {
            bar.style.width = (confidence * 100) + '%';
        }, 100);
        
        // Set color based on confidence
        if (confidence >= 0.7) {
            bar.classList.add('confidence-high');
        } else if (confidence >= 0.5) {
            bar.classList.add('confidence-medium');
        } else {
            bar.classList.add('confidence-low');
        }
    });
}

function initializeTooltips() {
    // Initialize Bootstrap tooltips if available
    if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
}

function initializeAlerts() {
    // Auto-hide success alerts after 5 seconds
    const successAlerts = document.querySelectorAll('.alert-success');
    successAlerts.forEach(alert => {
        setTimeout(() => {
            fadeOutAlert(alert);
        }, 5000);
    });
    
    // Auto-hide info alerts after 7 seconds
    const infoAlerts = document.querySelectorAll('.alert-info');
    infoAlerts.forEach(alert => {
        setTimeout(() => {
            fadeOutAlert(alert);
        }, 7000);
    });
}

function fadeOutAlert(alertElement) {
    alertElement.style.transition = 'opacity 0.5s ease';
    alertElement.style.opacity = '0';
    setTimeout(() => {
        alertElement.remove();
    }, 500);
}

function showAlert(message, type = 'info') {
    const alertContainer = document.getElementById('alertContainer');
    if (!alertContainer) return;
    
    const alertElement = document.createElement('div');
    alertElement.className = `alert alert-${type} alert-dismissible fade show`;
    alertElement.role = 'alert';
    alertElement.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    alertContainer.appendChild(alertElement);
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        fadeOutAlert(alertElement);
    }, 5000);
}

function showLoadingSpinner() {
    const spinner = document.getElementById('loadingSpinner');
    if (spinner) {
        spinner.style.display = 'block';
    }
}

function hideLoadingSpinner() {
    const spinner = document.getElementById('loadingSpinner');
    if (spinner) {
        spinner.style.display = 'none';
    }
}

function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(function() {
        showAlert('Copied to clipboard!', 'success');
    }).catch(function(err) {
        console.error('Could not copy text: ', err);
        showAlert('Failed to copy to clipboard', 'danger');
    });
}

// Utility functions for threat level styling
function getThreatLevelClass(threatLevel) {
    const levelMap = {
        'CRITICAL': 'threat-critical',
        'HIGH': 'threat-high',
        'MEDIUM': 'threat-medium',
        'LOW': 'threat-low',
        'SAFE': 'threat-safe'
    };
    return levelMap[threatLevel] || 'threat-safe';
}

function getThreatLevelIcon(threatLevel) {
    const iconMap = {
        'CRITICAL': '🚨',
        'HIGH': '⚠️',
        'MEDIUM': '⚡',
        'LOW': '⚪',
        'SAFE': '✅'
    };
    return iconMap[threatLevel] || '❓';
}

// Real-time clock for security monitoring feel
function updateClock() {
    const clockElement = document.getElementById('systemClock');
    if (clockElement) {
        const now = new Date();
        const timeString = now.toLocaleTimeString('en-US', {
            hour12: false,
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
        const dateString = now.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        });
        clockElement.innerHTML = `${dateString} ${timeString}`;
    }
}

// Update clock every second
setInterval(updateClock, 1000);
updateClock(); // Initial call

// Form submission with loading state
document.addEventListener('submit', function(e) {
    if (e.target.id === 'uploadForm') {
        const submitBtn = e.target.querySelector('button[type="submit"]');
        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.innerHTML = `
                <span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>
                Scanning...
            `;
        }
        showLoadingSpinner();
    }
});

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl+U or Cmd+U to focus upload
    if ((e.ctrlKey || e.metaKey) && e.key === 'u') {
        e.preventDefault();
        const fileInput = document.getElementById('fileInput');
        if (fileInput) {
            fileInput.click();
        }
    }
    
    // Escape to clear selection
    if (e.key === 'Escape') {
        const fileInput = document.getElementById('fileInput');
        const fileName = document.getElementById('fileName');
        const uploadBtn = document.getElementById('uploadBtn');
        
        if (fileInput) fileInput.value = '';
        if (fileName) fileName.style.display = 'none';
        if (uploadBtn) {
            uploadBtn.disabled = true;
            uploadBtn.textContent = 'Select File First';
            uploadBtn.classList.remove('btn-wepscan-primary');
            uploadBtn.classList.add('btn-secondary');
        }
    }
});

// Performance monitoring
let performanceStartTime = Date.now();

window.addEventListener('load', function() {
    const loadTime = Date.now() - performanceStartTime;
    console.log(`WepScan loaded in ${loadTime}ms`);
});

// Console branding
console.log(`
██╗    ██╗███████╗██████╗ ███████╗ ██████╗ █████╗ ███╗   ██╗
██║    ██║██╔════╝██╔══██╗██╔════╝██╔════╝██╔══██╗████╗  ██║
██║ █╗ ██║█████╗  ██████╔╝███████╗██║     ███████║██╔██╗ ██║
██║███╗██║██╔══╝  ██╔═══╝ ╚════██║██║     ██╔══██║██║╚██╗██║
╚███╔███╔╝███████╗██║     ███████║╚██████╗██║  ██║██║ ╚████║
 ╚══╝╚══╝ ╚══════╝╚═╝     ╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═══╝

WepScan v1.0 - AI-Powered Weapon Detection System
Security Status: Active | Monitoring: X-Ray Scanners
`);
