// WepScan Frontend JavaScript

document.addEventListener('DOMContentLoaded', function() {
    initializeThemeSystem();
    initializeSidebar();
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

// Theme Management System
let currentTheme = 'dark'; // Default theme

function initializeThemeSystem() {
    // Get saved theme from localStorage or detect system preference
    const savedTheme = localStorage.getItem('wepscan-theme');
    const systemTheme = getSystemTheme();
    
    if (savedTheme) {
        currentTheme = savedTheme;
    } else if (systemTheme) {
        currentTheme = 'system';
    }
    
    applyTheme(currentTheme);
    updateThemeUI();
    setupThemeEventListeners();
    
    // Listen for system theme changes
    if (window.matchMedia) {
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function(e) {
            if (currentTheme === 'system') {
                applyTheme('system');
            }
        });
    }
}

function getSystemTheme() {
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
        return 'dark';
    } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
        return 'light';
    }
    return null;
}

function applyTheme(theme) {
    const html = document.documentElement;
    
    // Remove existing theme attributes
    html.removeAttribute('data-theme');
    
    if (theme === 'system') {
        // Let CSS media queries handle system theme
        html.setAttribute('data-theme', 'system');
    } else {
        html.setAttribute('data-theme', theme);
    }
    
    currentTheme = theme;
    
    // Save to localStorage
    localStorage.setItem('wepscan-theme', theme);
    
    // Update theme icon
    updateThemeIcon(theme);
    
    console.log(`Theme changed to: ${theme}`);
}

function updateThemeIcon(theme) {
    const themeIcon = document.getElementById('themeIcon');
    if (!themeIcon) return;
    
    const iconMap = {
        'dark': 'fas fa-moon',
        'light': 'fas fa-sun',
        'system': 'fas fa-desktop'
    };
    
    themeIcon.className = iconMap[theme] || 'fas fa-moon';
}

function updateThemeUI() {
    const themeOptions = document.querySelectorAll('.theme-option');
    themeOptions.forEach(option => {
        option.classList.remove('active');
        if (option.dataset.theme === currentTheme) {
            option.classList.add('active');
        }
    });
    
    updateThemeIcon(currentTheme);
}

function setupThemeEventListeners() {
    const themeOptions = document.querySelectorAll('.theme-option');
    themeOptions.forEach(option => {
        option.addEventListener('click', function(e) {
            e.preventDefault();
            const selectedTheme = this.dataset.theme;
            
            if (selectedTheme !== currentTheme) {
                applyTheme(selectedTheme);
                updateThemeUI();
                
                // Show feedback
                showAlert(`Theme changed to ${selectedTheme.charAt(0).toUpperCase() + selectedTheme.slice(1)}`, 'success');
            }
        });
    });
}

function toggleTheme() {
    const themes = ['dark', 'light', 'system'];
    const currentIndex = themes.indexOf(currentTheme);
    const nextIndex = (currentIndex + 1) % themes.length;
    const nextTheme = themes[nextIndex];
    
    applyTheme(nextTheme);
    updateThemeUI();
}

// Add keyboard shortcut for theme toggle (Ctrl/Cmd + Shift + T)
document.addEventListener('keydown', function(e) {
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'T') {
        e.preventDefault();
        toggleTheme();
    }
});

// Export theme functions for potential external use
window.WepScanTheme = {
    toggle: toggleTheme,
    apply: applyTheme,
    current: () => currentTheme,
    getSystem: getSystemTheme
};

// Sidebar Management System
function initializeSidebar() {
    const mobileMenuToggle = document.getElementById('mobileMenuToggle');
    const floatingSidebar = document.getElementById('floatingSidebar');
    const sidebarOverlay = document.getElementById('sidebarOverlay');
    
    if (mobileMenuToggle && floatingSidebar && sidebarOverlay) {
        // Mobile menu toggle
        mobileMenuToggle.addEventListener('click', function() {
            toggleSidebar();
        });
        
        // Close sidebar when clicking overlay
        sidebarOverlay.addEventListener('click', function() {
            closeSidebar();
        });
        
        // Close sidebar on escape key
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape' && floatingSidebar.classList.contains('active')) {
                closeSidebar();
            }
        });
    }
    
    // Update active navigation links
    updateActiveNavigation();
}

function toggleSidebar() {
    const floatingSidebar = document.getElementById('floatingSidebar');
    const sidebarOverlay = document.getElementById('sidebarOverlay');
    
    if (floatingSidebar && sidebarOverlay) {
        const isActive = floatingSidebar.classList.contains('active');
        
        if (isActive) {
            closeSidebar();
        } else {
            openSidebar();
        }
    }
}

function openSidebar() {
    const floatingSidebar = document.getElementById('floatingSidebar');
    const sidebarOverlay = document.getElementById('sidebarOverlay');
    
    if (floatingSidebar && sidebarOverlay) {
        floatingSidebar.classList.add('active');
        sidebarOverlay.classList.add('active');
        document.body.style.overflow = 'hidden';
    }
}

function closeSidebar() {
    const floatingSidebar = document.getElementById('floatingSidebar');
    const sidebarOverlay = document.getElementById('sidebarOverlay');
    
    if (floatingSidebar && sidebarOverlay) {
        floatingSidebar.classList.remove('active');
        sidebarOverlay.classList.remove('active');
        document.body.style.overflow = '';
    }
}

function updateActiveNavigation() {
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.sidebar-nav .nav-link');
    
    navLinks.forEach(link => {
        const linkPath = new URL(link.href).pathname;
        
        if (linkPath === currentPath) {
            link.classList.add('active');
        } else {
            link.classList.remove('active');
        }
    });
}

// Export sidebar functions
window.WepScanSidebar = {
    toggle: toggleSidebar,
    open: openSidebar,
    close: closeSidebar,
    updateActive: updateActiveNavigation
};
