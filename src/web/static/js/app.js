/**
 * Stran Connect Virtual Try-On Application
 * 
 * This JavaScript module handles the frontend functionality for the virtual try-on
 * system, including camera integration, image upload, API communication, and
 * UI interactions with comprehensive typing through JSDoc.
 * 
 * @author Stran Connect Team
 */

/**
 * @typedef {Object} AnalysisResult
 * @property {boolean} success
 * @property {string} face_shape
 * @property {number} confidence
 * @property {Object} measurements
 * @property {Array} recommendations
 * @property {Array} characteristics
 * @property {Array} styling_principles
 * @property {Array} avoid_styles
 * @property {Array} personalization_notes
 * @property {string} [error_message]
 * @property {number} [processing_time]
 */

/**
 * @typedef {Object} HairstyleRecommendation
 * @property {string} name
 * @property {string} category
 * @property {string} description
 * @property {number} suitability_score
 * @property {Array<string>} styling_tips
 * @property {string} maintenance_level
 * @property {string} length
 * @property {Array<string>} texture_compatibility
 * @property {string} [image_url]
 */

class VirtualTryOnApp {
    /**
     * Initialize the Virtual Try-On application
     */
    constructor() {
        this.currentImage = null;
        this.cameraStream = null;
        this.analysisResult = null;
        
        // API endpoints
        this.endpoints = {
            analyze: '/api/analyze',
            upload: '/api/upload',
            capture: '/api/camera/capture',
            validatePreferences: '/api/preferences/validate'
        };
        
        // Initialize the application
        this.init();
    }

    /**
     * Initialize the application and set up event listeners
     */
    init() {
        this.setupEventListeners();
        this.setupSmoothScrolling();
        this.updateAnalyzeButtonState();
        
        console.log('Virtual Try-On App initialized successfully');
    }

    /**
     * Set up all event listeners for the application
     */
    setupEventListeners() {
        // Navigation
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', this.handleNavigation.bind(this));
        });

        // Preferences form changes
        const preferencesForm = document.getElementById('preferences-form');
        if (preferencesForm) {
            preferencesForm.addEventListener('change', this.handlePreferencesChange.bind(this));
        }

        // File input change
        const fileInput = document.getElementById('file-input');
        if (fileInput) {
            fileInput.addEventListener('change', this.handleFileUpload.bind(this));
        }

        // Window resize for responsive camera
        window.addEventListener('resize', this.handleWindowResize.bind(this));
    }

    /**
     * Set up smooth scrolling for navigation links
     */
    setupSmoothScrolling() {
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', (e) => {
                e.preventDefault();
                const target = document.querySelector(anchor.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });
    }

    /**
     * Handle navigation link clicks
     * @param {Event} event - Click event
     */
    handleNavigation(event) {
        event.preventDefault();
        
        // Update active nav link
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });
        event.target.classList.add('active');
        
        // Scroll to section
        const targetId = event.target.getAttribute('href');
        const targetSection = document.querySelector(targetId);
        if (targetSection) {
            targetSection.scrollIntoView({ behavior: 'smooth' });
        }
    }

    /**
     * Handle preferences form changes
     * @param {Event} event - Change event
     */
    async handlePreferencesChange(event) {
        const preferences = this.getPreferences();
        
        try {
            const response = await this.makeRequest(this.endpoints.validatePreferences, {
                method: 'POST',
                body: JSON.stringify({ preferences })
            });
            
            if (response.success) {
                console.log('Preferences validated successfully');
            } else {
                console.warn('Preference validation warnings:', response.validation_result);
            }
        } catch (error) {
            console.error('Error validating preferences:', error);
        }
    }

    /**
     * Handle window resize events
     */
    handleWindowResize() {
        if (this.cameraStream) {
            this.adjustCameraSize();
        }
    }

    /**
     * Scroll to analysis section
     */
    scrollToAnalysis() {
        const analysisSection = document.getElementById('analysis');
        if (analysisSection) {
            analysisSection.scrollIntoView({ behavior: 'smooth' });
        }
    }

    /**
     * Trigger file upload dialog
     */
    triggerFileUpload() {
        const fileInput = document.getElementById('file-input');
        if (fileInput) {
            fileInput.click();
        }
    }

    /**
     * Handle file upload
     * @param {Event} event - File input change event
     */
    async handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        // Validate file
        if (!this.validateFile(file)) return;

        try {
            this.showLoadingState('Processing uploaded image...');
            
            // Convert file to base64
            const base64Data = await this.fileToBase64(file);
            
            // Set current image and update UI
            this.currentImage = base64Data;
            this.displayImage(base64Data);
            this.updateAnalyzeButtonState();
            
            this.hideLoadingState();
            
        } catch (error) {
            this.hideLoadingState();
            this.showError('Failed to process uploaded file: ' + error.message);
        }
    }

    /**
     * Validate uploaded file
     * @param {File} file - File to validate
     * @returns {boolean} Whether file is valid
     */
    validateFile(file) {
        const maxSize = 10 * 1024 * 1024; // 10MB
        const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/webp'];

        if (file.size > maxSize) {
            this.showError('File too large. Maximum size is 10MB.');
            return false;
        }

        if (!allowedTypes.includes(file.type)) {
            this.showError('Invalid file type. Please upload a JPEG, PNG, BMP, or WebP image.');
            return false;
        }

        return true;
    }

    /**
     * Convert file to base64
     * @param {File} file - File to convert
     * @returns {Promise<string>} Base64 data URL
     */
    fileToBase64(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    }

    /**
     * Start camera for image capture
     */
    async startCamera() {
        try {
            this.showLoadingState('Starting camera...');
            
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'user'
                }
            });

            this.cameraStream = stream;
            
            const videoElement = document.getElementById('camera-feed');
            const cameraContainer = document.getElementById('camera-container');
            const imagePreview = document.getElementById('image-preview');
            
            if (videoElement && cameraContainer && imagePreview) {
                videoElement.srcObject = stream;
                cameraContainer.style.display = 'block';
                imagePreview.style.display = 'none';
                
                this.adjustCameraSize();
            }
            
            this.hideLoadingState();
            
        } catch (error) {
            this.hideLoadingState();
            console.error('Camera access error:', error);
            
            let errorMessage = 'Could not access camera. ';
            if (error.name === 'NotAllowedError') {
                errorMessage += 'Please allow camera access and try again.';
            } else if (error.name === 'NotFoundError') {
                errorMessage += 'No camera found on this device.';
            } else {
                errorMessage += 'Please check your camera settings.';
            }
            
            this.showError(errorMessage);
        }
    }

    /**
     * Adjust camera size based on container
     */
    adjustCameraSize() {
        const videoElement = document.getElementById('camera-feed');
        const container = document.getElementById('camera-container');
        
        if (videoElement && container) {
            const containerWidth = container.offsetWidth;
            const aspectRatio = 16 / 9;
            const height = containerWidth / aspectRatio;
            
            videoElement.style.width = containerWidth + 'px';
            videoElement.style.height = height + 'px';
        }
    }

    /**
     * Capture photo from camera
     */
    capturePhoto() {
        const videoElement = document.getElementById('camera-feed');
        if (!videoElement || !this.cameraStream) {
            this.showError('Camera not available');
            return;
        }

        try {
            // Create canvas to capture image
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            
            canvas.width = videoElement.videoWidth;
            canvas.height = videoElement.videoHeight;
            
            // Draw video frame to canvas
            context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
            
            // Convert to base64
            const base64Data = canvas.toDataURL('image/jpeg', 0.9);
            
            // Set current image and update UI
            this.currentImage = base64Data;
            this.displayImage(base64Data);
            this.stopCamera();
            this.updateAnalyzeButtonState();
            
            console.log('Photo captured successfully');
            
        } catch (error) {
            console.error('Error capturing photo:', error);
            this.showError('Failed to capture photo: ' + error.message);
        }
    }

    /**
     * Stop camera and hide camera interface
     */
    stopCamera() {
        if (this.cameraStream) {
            this.cameraStream.getTracks().forEach(track => track.stop());
            this.cameraStream = null;
        }

        const cameraContainer = document.getElementById('camera-container');
        const imagePreview = document.getElementById('image-preview');
        
        if (cameraContainer && imagePreview) {
            cameraContainer.style.display = 'none';
            imagePreview.style.display = 'flex';
        }
    }

    /**
     * Display image in preview area
     * @param {string} base64Data - Base64 image data
     */
    displayImage(base64Data) {
        const imagePreview = document.getElementById('image-preview');
        if (!imagePreview) return;

        imagePreview.innerHTML = `
            <img src="${base64Data}" alt="Uploaded image" class="preview-image fade-in">
        `;
    }

    /**
     * Get user preferences from form
     * @returns {Object} User preferences
     */
    getPreferences() {
        const form = document.getElementById('preferences-form');
        if (!form) return {};

        const formData = new FormData(form);
        const preferences = {};

        for (const [key, value] of formData.entries()) {
            if (value) {
                preferences[key] = value;
            }
        }

        return preferences;
    }

    /**
     * Update analyze button state based on whether image is available
     */
    updateAnalyzeButtonState() {
        const analyzeButton = document.getElementById('analyze-button');
        if (!analyzeButton) return;

        if (this.currentImage) {
            analyzeButton.disabled = false;
            analyzeButton.textContent = 'Analyze Face Shape';
        } else {
            analyzeButton.disabled = true;
            analyzeButton.textContent = 'Upload or capture an image first';
        }
    }

    /**
     * Analyze uploaded/captured image
     */
    async analyzeImage() {
        if (!this.currentImage) {
            this.showError('Please upload or capture an image first');
            return;
        }

        try {
            this.showLoadingState('Analyzing your facial features...');
            
            const preferences = this.getPreferences();
            const requestData = {
                image_data: this.currentImage,
                preferences: preferences,
                top_n: 5
            };

            const result = await this.makeRequest(this.endpoints.analyze, {
                method: 'POST',
                body: JSON.stringify(requestData)
            });

            this.hideLoadingState();

            if (result.success) {
                this.analysisResult = result;
                this.displayAnalysisResults(result);
                this.showRecommendations(result);
                
                // Scroll to results
                setTimeout(() => {
                    document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
                }, 500);
                
            } else {
                this.showError(result.error_message || 'Analysis failed');
            }

        } catch (error) {
            this.hideLoadingState();
            console.error('Analysis error:', error);
            this.showError('Failed to analyze image: ' + error.message);
        }
    }

    /**
     * Display analysis results
     * @param {AnalysisResult} result - Analysis result data
     */
    displayAnalysisResults(result) {
        // Show results section
        const resultsSection = document.getElementById('results');
        if (resultsSection) {
            resultsSection.style.display = 'block';
            resultsSection.classList.add('fade-in');
        }

        // Update face shape information
        this.updateFaceShapeDisplay(result);
        this.updateMeasurementsDisplay(result.measurements);
    }

    /**
     * Update face shape display
     * @param {AnalysisResult} result - Analysis result
     */
    updateFaceShapeDisplay(result) {
        const shapeVisual = document.getElementById('detected-shape-visual');
        const shapeName = document.getElementById('detected-shape-name');
        const shapeConfidence = document.getElementById('detected-shape-confidence');
        const characteristics = document.getElementById('shape-characteristics');

        if (shapeVisual) {
            shapeVisual.innerHTML = this.getFaceShapeIcon(result.face_shape);
        }

        if (shapeName) {
            shapeName.textContent = this.formatFaceShapeName(result.face_shape);
        }

        if (shapeConfidence) {
            const percentage = Math.round(result.confidence * 100);
            shapeConfidence.textContent = `Confidence: ${percentage}%`;
        }

        if (characteristics && result.characteristics) {
            characteristics.innerHTML = result.characteristics
                .map(char => `<span class="characteristic-tag">${char}</span>`)
                .join('');
        }
    }

    /**
     * Update measurements display
     * @param {Object} measurements - Facial measurements
     */
    updateMeasurementsDisplay(measurements) {
        const measurementsGrid = document.getElementById('measurements-grid');
        if (!measurementsGrid || !measurements) return;

        const measurementLabels = {
            'face_width': 'Face Width',
            'face_height': 'Face Height',
            'forehead_width': 'Forehead Width',
            'jaw_width': 'Jaw Width',
            'cheekbone_width': 'Cheekbone Width',
            'face_ratio': 'Face Ratio',
            'forehead_ratio': 'Forehead Ratio',
            'jaw_ratio': 'Jaw Ratio',
            'cheekbone_ratio': 'Cheekbone Ratio'
        };

        measurementsGrid.innerHTML = Object.entries(measurements)
            .filter(([key]) => measurementLabels[key])
            .map(([key, value]) => `
                <div class="measurement-item">
                    <span class="measurement-label">${measurementLabels[key]}</span>
                    <span class="measurement-value">${this.formatMeasurement(key, value)}</span>
                </div>
            `).join('');
    }

    /**
     * Show hairstyle recommendations
     * @param {AnalysisResult} result - Analysis result
     */
    showRecommendations(result) {
        const recommendationsSection = document.getElementById('recommendations');
        if (!recommendationsSection) return;

        // Show section
        recommendationsSection.style.display = 'block';
        recommendationsSection.classList.add('slide-up');

        // Update subtitle
        const subtitle = document.getElementById('recommendations-subtitle');
        if (subtitle) {
            subtitle.textContent = `Styles that complement your ${this.formatFaceShapeName(result.face_shape)} face shape`;
        }

        // Update styling principles
        this.updateStylingPrinciples(result);

        // Update hairstyle grid
        this.updateHairstyleGrid(result.recommendations);

        // Update personalization notes
        this.updatePersonalizationNotes(result.personalization_notes);
    }

    /**
     * Update styling principles section
     * @param {AnalysisResult} result - Analysis result
     */
    updateStylingPrinciples(result) {
        const doList = document.getElementById('styling-do-list');
        const dontList = document.getElementById('styling-dont-list');

        if (doList && result.styling_principles) {
            doList.innerHTML = result.styling_principles
                .map(principle => `<li>${principle}</li>`)
                .join('');
        }

        if (dontList && result.avoid_styles) {
            dontList.innerHTML = result.avoid_styles
                .map(style => `<li>${style}</li>`)
                .join('');
        }
    }

    /**
     * Update hairstyle recommendations grid
     * @param {Array<HairstyleRecommendation>} recommendations - Hairstyle recommendations
     */
    updateHairstyleGrid(recommendations) {
        const grid = document.getElementById('hairstyles-grid');
        if (!grid || !recommendations) return;

        grid.innerHTML = recommendations.map(style => this.createHairstyleCard(style)).join('');
    }

    /**
     * Create hairstyle card HTML
     * @param {HairstyleRecommendation} style - Hairstyle data
     * @returns {string} HTML string for hairstyle card
     */
    createHairstyleCard(style) {
        const score = Math.round(style.suitability_score * 100);
        
        return `
            <div class="hairstyle-card">
                <div class="suitability-score">${score}%</div>
                <h4>${style.name}</h4>
                <div class="hairstyle-meta">
                    <span class="meta-tag">${style.length}</span>
                    <span class="meta-tag">${style.maintenance_level} maintenance</span>
                    <span class="meta-tag">${style.category}</span>
                </div>
                <p class="hairstyle-description">${style.description}</p>
                ${style.texture_compatibility.length > 0 ? `
                    <div class="texture-compatibility">
                        <strong>Best for:</strong> ${style.texture_compatibility.join(', ')} hair
                    </div>
                ` : ''}
                <div class="styling-tips">
                    <h5>Styling Tips:</h5>
                    <ul>
                        ${style.styling_tips.map(tip => `<li>${tip}</li>`).join('')}
                    </ul>
                </div>
            </div>
        `;
    }

    /**
     * Update personalization notes
     * @param {Array<string>} notes - Personalization notes
     */
    updatePersonalizationNotes(notes) {
        const notesContent = document.getElementById('notes-content');
        if (!notesContent || !notes) return;

        notesContent.innerHTML = notes
            .map(note => `<div class="note-item">${note}</div>`)
            .join('');
    }

    /**
     * Get face shape icon HTML
     * @param {string} shapeType - Face shape type
     * @returns {string} HTML for face shape icon
     */
    getFaceShapeIcon(shapeType) {
        const icons = {
            'oval': '<i class="fas fa-circle"></i>',
            'round': '<i class="fas fa-circle"></i>',
            'square': '<i class="fas fa-square"></i>',
            'heart': '<i class="fas fa-heart"></i>',
            'diamond': '<i class="fas fa-gem"></i>',
            'long': '<i class="fas fa-rectangle-portrait"></i>'
        };
        
        return icons[shapeType] || '<i class="fas fa-user"></i>';
    }

    /**
     * Format face shape name for display
     * @param {string} shapeType - Face shape type
     * @returns {string} Formatted name
     */
    formatFaceShapeName(shapeType) {
        return shapeType.charAt(0).toUpperCase() + shapeType.slice(1);
    }

    /**
     * Format measurement value for display
     * @param {string} key - Measurement key
     * @param {number} value - Measurement value
     * @returns {string} Formatted measurement
     */
    formatMeasurement(key, value) {
        if (key.includes('ratio')) {
            return value.toFixed(2);
        } else {
            return Math.round(value) + 'px';
        }
    }

    /**
     * Show loading state
     * @param {string} message - Loading message
     */
    showLoadingState(message = 'Processing...') {
        const loadingIndicator = document.getElementById('loading-indicator');
        if (loadingIndicator) {
            loadingIndicator.style.display = 'block';
            const messageElement = loadingIndicator.querySelector('p');
            if (messageElement) {
                messageElement.textContent = message;
            }
        }

        // Disable analyze button
        const analyzeButton = document.getElementById('analyze-button');
        if (analyzeButton) {
            analyzeButton.disabled = true;
        }
    }

    /**
     * Hide loading state
     */
    hideLoadingState() {
        const loadingIndicator = document.getElementById('loading-indicator');
        if (loadingIndicator) {
            loadingIndicator.style.display = 'none';
        }

        // Re-enable analyze button if image is available
        this.updateAnalyzeButtonState();
    }

    /**
     * Show error message
     * @param {string} message - Error message
     */
    showError(message) {
        // Create or update error toast
        let errorToast = document.getElementById('error-toast');
        
        if (!errorToast) {
            errorToast = document.createElement('div');
            errorToast.id = 'error-toast';
            errorToast.className = 'error-toast';
            document.body.appendChild(errorToast);
        }

        errorToast.innerHTML = `
            <div class="error-content">
                <i class="fas fa-exclamation-triangle"></i>
                <span>${message}</span>
                <button onclick="this.parentElement.parentElement.remove()" class="error-close">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;

        errorToast.style.display = 'block';
        errorToast.classList.add('show');

        // Auto-hide after 5 seconds
        setTimeout(() => {
            if (errorToast && errorToast.parentElement) {
                errorToast.classList.remove('show');
                setTimeout(() => {
                    errorToast.remove();
                }, 300);
            }
        }, 5000);
    }

    /**
     * Make HTTP request to API
     * @param {string} url - Request URL
     * @param {Object} options - Fetch options
     * @returns {Promise<Object>} Response data
     */
    async makeRequest(url, options = {}) {
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json'
            }
        };

        const requestOptions = { ...defaultOptions, ...options };

        try {
            const response = await fetch(url, requestOptions);
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.error_message || `HTTP ${response.status}: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Request failed:', error);
            throw error;
        }
    }
}

// Global functions for HTML onclick handlers
let app;

/**
 * Scroll to analysis section
 */
function scrollToAnalysis() {
    if (app) app.scrollToAnalysis();
}

/**
 * Trigger file upload
 */
function triggerFileUpload() {
    if (app) app.triggerFileUpload();
}

/**
 * Handle file upload
 * @param {Event} event - File input change event
 */
function handleFileUpload(event) {
    if (app) app.handleFileUpload(event);
}

/**
 * Start camera
 */
function startCamera() {
    if (app) app.startCamera();
}

/**
 * Capture photo
 */
function capturePhoto() {
    if (app) app.capturePhoto();
}

/**
 * Stop camera
 */
function stopCamera() {
    if (app) app.stopCamera();
}

/**
 * Analyze image
 */
function analyzeImage() {
    if (app) app.analyzeImage();
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    app = new VirtualTryOnApp();
    
    // Add error toast styles if not already present
    if (!document.getElementById('error-toast-styles')) {
        const style = document.createElement('style');
        style.id = 'error-toast-styles';
        style.textContent = `
            .error-toast {
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 1000;
                background: #ef4444;
                color: white;
                border-radius: 8px;
                padding: 16px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                transform: translateX(100%);
                transition: transform 0.3s ease-in-out;
                max-width: 400px;
                display: none;
            }
            
            .error-toast.show {
                transform: translateX(0);
            }
            
            .error-content {
                display: flex;
                align-items: center;
                gap: 12px;
            }
            
            .error-close {
                background: none;
                border: none;
                color: white;
                cursor: pointer;
                font-size: 16px;
                margin-left: auto;
            }
            
            .error-close:hover {
                opacity: 0.8;
            }
        `;
        document.head.appendChild(style);
    }
});

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VirtualTryOnApp;
}