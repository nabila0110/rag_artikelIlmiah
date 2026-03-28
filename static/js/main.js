// Utility functions
const utils = {
    debounce: function(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },
    
    // Format date
    formatDate: function(dateString) {
        const options = { year: 'numeric', month: 'long', day: 'numeric' };
        return new Date(dateString).toLocaleDateString('id-ID', options);
    },
    
    // Copy to clipboard
    copyToClipboard: function(text) {
        navigator.clipboard.writeText(text).then(() => {
            window.showAppMessage('Berhasil', 'Copied to clipboard!', 'success');
        });
    },
    
    // Highlight query terms in text
    highlightTerms: function(text, query) {
        const terms = query.toLowerCase().split(' ');
        let highlighted = text;
        
        terms.forEach(term => {
            if (term.length > 3) {
                const regex = new RegExp(`(${term})`, 'gi');
                highlighted = highlighted.replace(regex, '<mark>$1</mark>');
            }
        });
        
        return highlighted;
    }
};

// Search history management
const searchHistory = {
    key: 'rag_search_history',
    maxItems: 10,
    
    save: function(query) {
        let history = this.get();
        
        // Remove duplicates
        history = history.filter(item => item !== query);
        
        // Add to beginning
        history.unshift(query);
        
        // Limit to maxItems
        history = history.slice(0, this.maxItems);
        
        localStorage.setItem(this.key, JSON.stringify(history));
    },
    
    get: function() {
        const stored = localStorage.getItem(this.key);
        return stored ? JSON.parse(stored) : [];
    },
    
    clear: function() {
        localStorage.removeItem(this.key);
    },
    
    display: function() {
        const history = this.get();
        if (history.length === 0) return '';
        
        return `
            <div class="search-history mt-3">
                <small class="text-muted">Pencarian terakhir:</small>
                <div class="d-flex flex-wrap gap-2 mt-2">
                    ${history.map(query => `
                        <button class="btn btn-sm btn-outline-secondary history-item" 
                                data-query="${query}">
                            ${query}
                        </button>
                    `).join('')}
                </div>
            </div>
        `;
    }
};

const themeManager = {
    key: 'rag_theme_mode',
    defaultTheme: 'dark',

    apply: function(theme) {
        const normalizedTheme = ['dark', 'light', 'plain'].includes(theme) ? theme : 'dark';
        document.body.setAttribute('data-theme', normalizedTheme);
        localStorage.setItem(this.key, normalizedTheme);
    },

    init: function() {
        const savedTheme = localStorage.getItem(this.key) || this.defaultTheme;
        this.apply(savedTheme);

        const toggleButton = document.getElementById('themeToggle');
        if (!toggleButton) {
            return;
        }

        toggleButton.addEventListener('click', () => {
            const currentTheme = document.body.getAttribute('data-theme') || this.defaultTheme;
            const themeOrder = ['dark', 'light', 'plain'];
            const currentIndex = themeOrder.indexOf(currentTheme);
            const nextTheme = themeOrder[(currentIndex + 1) % themeOrder.length];
            this.apply(nextTheme);
        });
    }
};

const appMessage = {
    init: function() {
        this.overlay = document.getElementById('appMessageOverlay');
        this.title = document.getElementById('appMessageTitle');
        this.body = document.getElementById('appMessageBody');
        this.icon = document.getElementById('appMessageIcon');
        this.okBtn = document.getElementById('appMessageOkBtn');
        if (!this.overlay || !this.title || !this.body || !this.icon || !this.okBtn) {
            return;
        }

        this.okBtn.addEventListener('click', () => this.hide());
        this.overlay.addEventListener('click', (e) => {
            if (e.target === this.overlay) {
                this.hide();
            }
        });
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && !this.overlay.classList.contains('d-none')) {
                this.hide();
            }
        });
    },

    show: function(title, message, type = 'error') {
        if (!this.overlay || !this.title || !this.body || !this.icon || !this.okBtn) {
            return;
        }
        const normalizedType = ['error', 'warning', 'success', 'info'].includes(type) ? type : 'error';
        const defaultTitle = {
            error: 'Ooops!',
            warning: 'Perhatian!',
            success: 'Berhasil!',
            info: 'Informasi'
        };
        const iconClass = {
            error: 'fas fa-triangle-exclamation',
            warning: 'fas fa-triangle-exclamation',
            success: 'fas fa-check',
            info: 'fas fa-circle-info'
        };

        this.title.textContent = title || defaultTitle[normalizedType];
        this.body.textContent = message || '';
        this.icon.innerHTML = `<i class="${iconClass[normalizedType]}"></i>`;
        this.overlay.classList.remove('d-none', 'type-error', 'type-warning', 'type-success', 'type-info', 'is-visible');
        this.overlay.classList.add(`type-${normalizedType}`);
        this.overlay.classList.remove('d-none');
        requestAnimationFrame(() => {
            this.overlay.classList.add('is-visible');
        });
        this.okBtn.focus();
    },

    hide: function() {
        if (!this.overlay) {
            return;
        }
        this.overlay.classList.remove('is-visible');
        setTimeout(() => {
            this.overlay.classList.add('d-none');
        }, 180);
    }
};

// Export functions
window.utils = utils;
window.searchHistory = searchHistory;
window.themeManager = themeManager;
window.showAppMessage = function(title, message, type = 'error') {
    appMessage.show(title, message, type);
};

// Auto-save search queries
document.addEventListener('DOMContentLoaded', () => {
    appMessage.init();
    themeManager.init();

    // Display search history on load
    const searchForm = document.querySelector('#searchForm');
    const searchFormContainer = searchForm ? searchForm.parentElement : null;
    if (searchFormContainer) {
        searchFormContainer.insertAdjacentHTML('beforeend', searchHistory.display());
    }
    
    // Click handler for history items
    document.addEventListener('click', (e) => {
        if (e.target.classList.contains('history-item')) {
            const query = e.target.getAttribute('data-query');
            const queryInput = document.getElementById('queryInput');
            const currentSearchForm = document.getElementById('searchForm');
            if (!queryInput || !currentSearchForm) {
                return;
            }
            queryInput.value = query;
            currentSearchForm.dispatchEvent(new Event('submit'));
        }
    });
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + K to focus search
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        const queryInput = document.getElementById('queryInput');
        if (queryInput) {
            queryInput.focus();
        }
    }
});

// Add loading state to buttons
function setButtonLoading(button, isLoading) {
    if (isLoading) {
        button.disabled = true;
        button.dataset.originalHtml = button.innerHTML;
        button.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Loading...';
    } else {
        button.disabled = false;
        button.innerHTML = button.dataset.originalHtml || button.innerHTML;
    }
}

window.setButtonLoading = setButtonLoading;