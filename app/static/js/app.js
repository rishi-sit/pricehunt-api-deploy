/**
 * PriceHunt - Price Comparison App
 * Frontend JavaScript with Real-time Streaming
 */

class PriceHuntApp {
    constructor() {
        this.currentMode = 'single';
        this.pincode = '560087';
        this.isLoading = false;
        this.eventSource = null;
        
        // Streaming state
        this.streamingResults = [];
        this.platformsLoading = new Set();
        this.platformsCompleted = new Set();
        this.platformsCached = new Set();  // Track which platforms returned cached data
        this.currentQuery = '';
        this.cachedResultsCount = 0;
        
        this.init();
    }
    
    init() {
        this.bindElements();
        this.bindEvents();
        this.loadFromURL();
    }
    
    bindElements() {
        // Mode toggle
        this.modeBtns = document.querySelectorAll('.mode-btn');
        this.singleSearch = document.getElementById('single-search');
        this.bulkSearch = document.getElementById('bulk-search');
        
        // Search inputs
        this.searchInput = document.getElementById('search-input');
        this.searchBtn = document.getElementById('search-btn');
        this.bulkInput = document.getElementById('bulk-input');
        this.bulkSearchBtn = document.getElementById('bulk-search-btn');
        this.productCount = document.querySelector('.product-count');
        
        // Pincode
        this.pincodeInput = document.getElementById('pincode-input');
        this.currentPincodeDisplay = document.getElementById('current-pincode');
        
        // Sections
        this.loadingSection = document.getElementById('loading-section');
        this.resultsSection = document.getElementById('results-section');
    }
    
    bindEvents() {
        // Mode toggle
        this.modeBtns.forEach(btn => {
            btn.addEventListener('click', () => this.switchMode(btn.dataset.mode));
        });
        
        // Single search
        this.searchBtn.addEventListener('click', () => this.searchSingle());
        this.searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.searchSingle();
        });
        
        // Bulk search
        this.bulkSearchBtn.addEventListener('click', () => this.searchBulk());
        this.bulkInput.addEventListener('input', () => this.updateProductCount());
        
        // Pincode
        this.pincodeInput.addEventListener('change', () => this.updatePincode());
        this.pincodeInput.addEventListener('blur', () => this.updatePincode());
    }
    
    loadFromURL() {
        const params = new URLSearchParams(window.location.search);
        const query = params.get('q');
        const pincode = params.get('pincode');
        
        if (pincode) {
            this.pincodeInput.value = pincode;
            this.updatePincode();
        }
        
        if (query) {
            this.searchInput.value = query;
            this.searchSingle();
        }
    }
    
    switchMode(mode) {
        this.currentMode = mode;
        
        this.modeBtns.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.mode === mode);
        });
        
        if (mode === 'single') {
            this.singleSearch.classList.remove('hidden');
            this.bulkSearch.classList.add('hidden');
        } else {
            this.singleSearch.classList.add('hidden');
            this.bulkSearch.classList.remove('hidden');
        }
    }
    
    updatePincode() {
        const pincode = this.pincodeInput.value.trim();
        if (/^\d{6}$/.test(pincode)) {
            this.pincode = pincode;
            this.currentPincodeDisplay.textContent = pincode;
        }
    }
    
    updateProductCount() {
        const products = this.getProductsFromBulkInput();
        const count = products.length;
        this.productCount.textContent = `${count} product${count !== 1 ? 's' : ''}`;
    }
    
    getProductsFromBulkInput() {
        return this.bulkInput.value
            .split('\n')
            .map(p => p.trim())
            .filter(p => p.length > 0);
    }
    
    async searchSingle() {
        const query = this.searchInput.value.trim();
        if (!query || this.isLoading) return;
        
        // Update URL
        const url = new URL(window.location);
        url.searchParams.set('q', query);
        url.searchParams.set('pincode', this.pincode);
        window.history.pushState({}, '', url);
        
        // Reset streaming state
        this.streamingResults = [];
        this.platformsLoading = new Set();
        this.platformsCompleted = new Set();
        this.platformsCached = new Set();
        this.currentQuery = query;
        this.cachedResultsCount = 0;
        
        // Close any existing event source
        if (this.eventSource) {
            this.eventSource.close();
        }
        
        this.showStreamingUI(query);
        
        // Connect to SSE endpoint
        const streamUrl = `/api/search/stream?q=${encodeURIComponent(query)}&pincode=${this.pincode}`;
        this.eventSource = new EventSource(streamUrl);
        
        this.eventSource.addEventListener('init', (event) => {
            const data = JSON.parse(event.data);
            data.platforms.forEach(platform => {
                this.platformsLoading.add(platform);
            });
            this.updatePlatformLoadingStates();
        });
        
        this.eventSource.addEventListener('platform', (event) => {
            const data = JSON.parse(event.data);
            this.handlePlatformResults(data);
        });
        
        // Handle refresh events (stale-while-revalidate updates)
        this.eventSource.addEventListener('refresh', (event) => {
            const data = JSON.parse(event.data);
            this.handleRefreshResults(data);
        });
        
        this.eventSource.addEventListener('complete', (event) => {
            const data = JSON.parse(event.data);
            this.eventSource.close();
            this.eventSource = null;
            this.isLoading = false;
            this.finalizeResults(data.all_cached);
        });
        
        this.eventSource.onerror = (error) => {
            console.error('SSE error:', error);
            this.eventSource.close();
            this.eventSource = null;
            this.isLoading = false;
            if (this.streamingResults.length === 0) {
                this.renderError('Connection lost. Please try again.');
            }
        };
    }
    
    showStreamingUI(query) {
        this.isLoading = true;
        this.loadingSection.classList.add('hidden');
        this.resultsSection.classList.remove('hidden');
        
        // Create initial streaming container
        this.resultsSection.innerHTML = `
            <div class="result-group streaming-active" id="streaming-results">
                <div class="result-header">
                    <h2 class="result-query">Results for <span>"${this.escapeHtml(query)}"</span></h2>
                    <div class="result-meta">
                        <span class="results-count">0 products found</span>
                        <span>•</span>
                        <span class="platforms-count">0 platforms</span>
                    </div>
                </div>
                
                <div class="streaming-status" id="streaming-status">
                    <div class="streaming-indicator">
                        <div class="pulse-dot"></div>
                        <span>Searching platforms...</span>
                    </div>
                    <div class="platform-statuses" id="platform-statuses"></div>
                </div>
                
                <div class="best-deal-container" id="best-deal-container"></div>
                
                <div class="products-grid" id="products-grid"></div>
            </div>
        `;
        
        this.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    
    updatePlatformLoadingStates() {
        const container = document.getElementById('platform-statuses');
        if (!container) return;
        
        const platformOrder = ['Amazon Fresh', 'Flipkart Minutes', 'JioMart Quick', 'BigBasket', 'Amazon', 'Flipkart', 'JioMart', 'Zepto'];
        
        container.innerHTML = platformOrder.map(platform => {
            const isLoading = this.platformsLoading.has(platform) && !this.platformsCompleted.has(platform);
            const isCompleted = this.platformsCompleted.has(platform);
            const isCached = this.platformsCached.has(platform);
            
            return `
                <div class="platform-status ${isCompleted ? 'completed' : ''} ${isLoading ? 'loading' : ''} ${isCached ? 'cached' : ''}">
                    <span class="status-indicator">
                        ${isCompleted ? (isCached ? `
                            <svg viewBox="0 0 24 24" fill="currentColor" width="14" height="14">
                                <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
                            </svg>
                        ` : `
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14">
                                <path d="M20 6L9 17l-5-5"/>
                            </svg>
                        `) : `
                            <div class="mini-spinner"></div>
                        `}
                    </span>
                    <span class="platform-name">${platform}</span>
                    ${isCached ? '<span class="cached-badge">cached</span>' : ''}
                </div>
            `;
        }).join('');
    }
    
    handlePlatformResults(data) {
        const { platform, results, count, cached, stale } = data;
        
        // Track if this was a cached result
        if (cached) {
            this.platformsCached.add(platform);
            this.cachedResultsCount += count;
        }
        
        // Update platform status
        this.platformsCompleted.add(platform);
        this.updatePlatformLoadingStates();
        
        // Add results to our collection (mark them as cached if applicable)
        if (results && results.length > 0) {
            results.forEach(result => {
                result._cached = cached;
                result._stale = stale;
                this.streamingResults.push(result);
            });
        }
        
        // Update counts
        this.updateResultCounts();
        
        // Re-render products with animation for new ones
        this.renderStreamingProducts(platform, cached);
        
        // Update best deal
        this.updateBestDeal();
    }
    
    handleRefreshResults(data) {
        const { platform, results, count } = data;
        
        // Remove old results from this platform
        this.streamingResults = this.streamingResults.filter(r => r.platform !== platform);
        
        // Remove from cached set since we now have fresh data
        this.platformsCached.delete(platform);
        
        // Add fresh results
        if (results && results.length > 0) {
            results.forEach(result => {
                result._cached = false;
                result._stale = false;
                this.streamingResults.push(result);
            });
        }
        
        // Update counts
        this.updateResultCounts();
        
        // Re-render products (with refresh animation)
        this.renderStreamingProducts(platform, false, true);
        
        // Update best deal in case prices changed
        this.updateBestDeal();
    }
    
    updateResultCounts() {
        const resultsCount = document.querySelector('.results-count');
        const platformsCount = document.querySelector('.platforms-count');
        
        if (resultsCount) {
            resultsCount.textContent = `${this.streamingResults.length} product${this.streamingResults.length !== 1 ? 's' : ''} found`;
        }
        
        if (platformsCount) {
            const platformsWithResults = new Set(this.streamingResults.map(r => r.platform)).size;
            platformsCount.textContent = `${platformsWithResults} platform${platformsWithResults !== 1 ? 's' : ''}`;
        }
    }
    
    renderStreamingProducts(newPlatform, isCached = false, isRefresh = false) {
        const grid = document.getElementById('products-grid');
        if (!grid) return;
        
        // Get current lowest price for highlighting
        const lowestPrice = this.calculateLowestPrice();
        
        // Sort results by platform order then price
        const platformOrder = {
            "Amazon Fresh": 1,
            "Flipkart Minutes": 2,
            "JioMart Quick": 3,
            "BigBasket": 4,
            "Zepto": 5,
            "Amazon": 6,
            "Flipkart": 7,
            "JioMart": 8,
            "Blinkit": 9,
            "Instamart": 10,
        };
        
        const sortedResults = [...this.streamingResults].sort((a, b) => {
            const orderDiff = (platformOrder[a.platform] || 99) - (platformOrder[b.platform] || 99);
            if (orderDiff !== 0) return orderDiff;
            return (a.price || 0) - (b.price || 0);
        });
        
        // Create cards with animation class for new platform
        grid.innerHTML = sortedResults.map(product => {
            const isNew = product.platform === newPlatform;
            return this.createProductCard(product, lowestPrice, isNew, product._cached, isRefresh && isNew);
        }).join('');
    }
    
    calculateLowestPrice() {
        const availableResults = this.streamingResults.filter(r => r.available !== false && r.price > 0);
        if (availableResults.length === 0) return null;
        return availableResults.reduce((min, r) => r.price < min.price ? r : min, availableResults[0]);
    }
    
    updateBestDeal() {
        const container = document.getElementById('best-deal-container');
        if (!container) return;
        
        const lowestPrice = this.calculateLowestPrice();
        
        if (lowestPrice) {
            const prices = this.streamingResults.filter(r => r.price > 0).map(r => r.price);
            const maxPrice = Math.max(...prices);
            const savings = maxPrice - lowestPrice.price;
            
            const platformColor = this.getPlatformColor(lowestPrice.platform);
            
            // Add animation class if content is changing
            const isUpdate = container.innerHTML !== '';
            
            container.innerHTML = `
                <div class="best-deal ${isUpdate ? 'updating' : 'slide-in'}">
                    <div class="best-deal-badge">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
                        </svg>
                        Best Deal${isUpdate ? ' (Updated!)' : ''}
                    </div>
                    <div class="best-deal-content">
                        <div class="best-deal-info">
                            <h3>${this.escapeHtml(lowestPrice.name)}</h3>
                            <div class="best-deal-platform">
                                <span class="dot" style="background: ${platformColor}"></span>
                                <span>${lowestPrice.platform}</span>
                                ${lowestPrice.delivery_time ? `<span>• ${lowestPrice.delivery_time}</span>` : ''}
                            </div>
                        </div>
                        <div class="best-deal-price">
                            <div class="price">₹${this.formatPrice(lowestPrice.price)}</div>
                            ${savings > 0 ? `
                                <div class="savings">Save up to <span>₹${this.formatPrice(savings)}</span> vs other platforms</div>
                            ` : ''}
                        </div>
                    </div>
                </div>
            `;
        }
    }
    
    finalizeResults(allCached = false) {
        // Update streaming status to show completion
        const streamingStatus = document.getElementById('streaming-status');
        if (streamingStatus) {
            if (allCached) {
                // Show instant results message
                streamingStatus.innerHTML = `
                    <div class="streaming-complete cached-complete">
                        <svg viewBox="0 0 24 24" fill="currentColor" width="20" height="20">
                            <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
                        </svg>
                        <span>Instant results from cache!</span>
                    </div>
                `;
                setTimeout(() => {
                    streamingStatus.classList.add('fade-out');
                    setTimeout(() => streamingStatus.remove(), 300);
                }, 1500);
            } else {
                streamingStatus.classList.add('fade-out');
                setTimeout(() => streamingStatus.remove(), 300);
            }
        }
        
        // Remove streaming-active class
        const resultsGroup = document.getElementById('streaming-results');
        if (resultsGroup) {
            resultsGroup.classList.remove('streaming-active');
        }
        
        // Final update of best deal (remove "Updated!" text)
        const lowestPrice = this.calculateLowestPrice();
        if (lowestPrice) {
            const container = document.getElementById('best-deal-container');
            if (container) {
                const badge = container.querySelector('.best-deal-badge');
                if (badge) {
                    badge.innerHTML = `
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
                        </svg>
                        Best Deal
                    `;
                }
            }
        }
        
        // If no results at all
        if (this.streamingResults.length === 0) {
            const grid = document.getElementById('products-grid');
            if (grid) {
                grid.innerHTML = this.createNoResults();
            }
        }
    }
    
    async searchBulk() {
        const products = this.getProductsFromBulkInput();
        if (products.length === 0 || this.isLoading) return;
        
        this.showLoading();
        
        try {
            const response = await fetch('/api/search/bulk', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    products: products,
                    pincode: this.pincode
                })
            });
            const data = await response.json();
            this.renderResults(data.comparisons);
        } catch (error) {
            console.error('Bulk search error:', error);
            this.renderError('Failed to fetch results. Please try again.');
        } finally {
            this.hideLoading();
        }
    }
    
    showLoading() {
        this.isLoading = true;
        this.loadingSection.classList.remove('hidden');
        this.resultsSection.classList.add('hidden');
        
        // Animate loading platforms
        const platforms = this.loadingSection.querySelectorAll('.loading-platform');
        platforms.forEach((p, i) => {
            setTimeout(() => p.style.opacity = '1', i * 200);
        });
    }
    
    hideLoading() {
        this.isLoading = false;
        this.loadingSection.classList.add('hidden');
    }
    
    renderResults(comparisons) {
        this.resultsSection.classList.remove('hidden');
        this.resultsSection.innerHTML = '';
        
        comparisons.forEach((comparison, index) => {
            const group = this.createResultGroup(comparison, index);
            this.resultsSection.appendChild(group);
        });
        
        // Scroll to results
        this.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    
    createResultGroup(comparison, index) {
        const group = document.createElement('div');
        group.className = 'result-group';
        group.style.animationDelay = `${index * 0.1}s`;
        
        const hasResults = comparison.results && comparison.results.length > 0;
        
        group.innerHTML = `
            <div class="result-header">
                <h2 class="result-query">Results for <span>"${this.escapeHtml(comparison.query)}"</span></h2>
                <div class="result-meta">
                    <span>${comparison.results?.length || 0} products found</span>
                    <span>•</span>
                    <span>${comparison.total_platforms || 0} platforms</span>
                </div>
            </div>
            
            ${hasResults && comparison.lowest_price ? this.createBestDealCard(comparison.lowest_price, comparison.results) : ''}
            
            ${hasResults ? `
                <div class="products-grid">
                    ${comparison.results.map(product => this.createProductCard(product, comparison.lowest_price)).join('')}
                </div>
            ` : this.createNoResults()}
        `;
        
        return group;
    }
    
    createBestDealCard(lowestPrice, allResults) {
        // Calculate potential savings
        const prices = allResults
            .filter(r => r.price > 0)
            .map(r => r.price);
        const maxPrice = Math.max(...prices);
        const savings = maxPrice - lowestPrice.price;
        
        const platformColor = this.getPlatformColor(lowestPrice.platform);
        
        return `
            <div class="best-deal">
                <div class="best-deal-badge">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
                    </svg>
                    Best Deal
                </div>
                <div class="best-deal-content">
                    <div class="best-deal-info">
                        <h3>${this.escapeHtml(lowestPrice.name)}</h3>
                        <div class="best-deal-platform">
                            <span class="dot" style="background: ${platformColor}"></span>
                            <span>${lowestPrice.platform}</span>
                            ${lowestPrice.delivery_time ? `<span>• ${lowestPrice.delivery_time}</span>` : ''}
                        </div>
                    </div>
                    <div class="best-deal-price">
                        <div class="price">₹${this.formatPrice(lowestPrice.price)}</div>
                        ${savings > 0 ? `
                            <div class="savings">Save up to <span>₹${this.formatPrice(savings)}</span> vs other platforms</div>
                        ` : ''}
                    </div>
                </div>
            </div>
        `;
    }
    
    createProductCard(product, lowestPrice, isNew = false, isCached = false, isRefresh = false) {
        const isLowest = lowestPrice && product.price === lowestPrice.price && product.platform === lowestPrice.platform;
        const platformColor = this.getPlatformColor(product.platform);
        const platformInitial = product.platform.charAt(0);
        
        let animationClass = '';
        if (isNew && !isRefresh) animationClass = 'slide-in-card';
        if (isRefresh) animationClass = 'refresh-card';
        
        return `
            <div class="product-card ${isLowest ? 'lowest' : ''} ${animationClass} ${isCached ? 'cached-result' : ''}">
                ${isCached ? `
                    <div class="cached-indicator" title="Loaded from cache - instant results!">
                        <svg viewBox="0 0 24 24" fill="currentColor" width="12" height="12">
                            <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/>
                        </svg>
                    </div>
                ` : ''}
                <div class="product-platform">
                    <div class="platform-tag" data-platform="${product.platform}">
                        <div class="icon">${platformInitial}</div>
                        <span>${product.platform}</span>
                    </div>
                    ${product.delivery_time ? `<span class="delivery-tag">${product.delivery_time}</span>` : ''}
                </div>
                
                <div class="product-name">${this.escapeHtml(product.name)}</div>
                
                <div class="product-pricing">
                    <span class="product-price">₹${this.formatPrice(product.price)}</span>
                    ${product.original_price && product.original_price > product.price ? `
                        <span class="product-original-price">₹${this.formatPrice(product.original_price)}</span>
                    ` : ''}
                    ${product.discount ? `<span class="product-discount">${product.discount}</span>` : ''}
                </div>
                
                <div class="product-footer">
                    ${product.rating ? `
                        <div class="product-rating">
                            <svg viewBox="0 0 24 24">
                                <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
                            </svg>
                            <span>${product.rating}</span>
                        </div>
                    ` : '<div></div>'}
                    ${product.url ? `
                        <a href="${product.url}" target="_blank" rel="noopener" class="product-link">
                            View
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M7 17L17 7M17 7H7M17 7v10"/>
                            </svg>
                        </a>
                    ` : ''}
                </div>
            </div>
        `;
    }
    
    createNoResults() {
        return `
            <div class="no-results">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                    <circle cx="11" cy="11" r="8"/>
                    <path d="m21 21-4.35-4.35"/>
                    <path d="M8 11h6"/>
                </svg>
                <h3>No results found</h3>
                <p>We couldn't find any products matching your search. Try a different keyword or check back later.</p>
            </div>
        `;
    }
    
    renderError(message) {
        this.resultsSection.classList.remove('hidden');
        this.resultsSection.innerHTML = `
            <div class="no-results">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                    <circle cx="12" cy="12" r="10"/>
                    <path d="M12 8v4M12 16h.01"/>
                </svg>
                <h3>Something went wrong</h3>
                <p>${this.escapeHtml(message)}</p>
            </div>
        `;
    }
    
    getPlatformColor(platform) {
        const colors = {
            'Amazon': '#FF9900',
            'Amazon Fresh': '#5EA03E',
            'Flipkart': '#2874F0',
            'Flipkart Minutes': '#FFCE00',
            'JioMart': '#0078AD',
            'JioMart Quick': '#0078AD',
            'BigBasket': '#84C225',
            'Zepto': '#8B5CF6',
            'Instamart': '#FC8019',
            'Blinkit': '#F8CB46'
        };
        return colors[platform] || '#888888';
    }
    
    formatPrice(price) {
        if (typeof price !== 'number') return '0';
        return price.toLocaleString('en-IN', {
            minimumFractionDigits: 0,
            maximumFractionDigits: 2
        });
    }
    
    escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    window.priceHuntApp = new PriceHuntApp();
});

