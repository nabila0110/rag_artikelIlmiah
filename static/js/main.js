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
            alert('Copied to clipboard!');
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

// Export functions
window.utils = utils;
window.searchHistory = searchHistory;

// Auto-save search queries
document.addEventListener('DOMContentLoaded', () => {
    // Display search history on load
    const searchFormContainer = document.querySelector('#searchForm').parentElement;
    searchFormContainer.insertAdjacentHTML('beforeend', searchHistory.display());
    
    // Click handler for history items
    document.addEventListener('click', (e) => {
        if (e.target.classList.contains('history-item')) {
            const query = e.target.getAttribute('data-query');
            document.getElementById('queryInput').value = query;
            document.getElementById('searchForm').dispatchEvent(new Event('submit'));
        }
    });
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + K to focus search
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        document.getElementById('queryInput').focus();
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