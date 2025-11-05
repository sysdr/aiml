// Real-time Dashboard JavaScript
let eventSource = null;
let isDemoRunning = false;

// Initialize dashboard on load
document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
    setupEventListeners();
    loadInitialMetrics();
});

function initializeDashboard() {
    console.log('Dashboard initialized');
    updateStatus('Ready', false);
}

function setupEventListeners() {
    document.getElementById('startBtn').addEventListener('click', startDemo);
    document.getElementById('stopBtn').addEventListener('click', stopDemo);
}

async function loadInitialMetrics() {
    try {
        const response = await fetch('/api/metrics');
        const data = await response.json();
        updateDashboard(data);
        startEventStream();
    } catch (error) {
        console.error('Error loading initial metrics:', error);
        showError('Failed to load metrics. Please refresh the page.');
    }
}

function startDemo() {
    fetch('/api/start', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            console.log('Demo started:', data);
            updateStatus('Running', true);
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            isDemoRunning = true;
            
            // Show visual feedback that demo started
            const header = document.querySelector('header');
            header.style.border = '2px solid #10b981';
            setTimeout(() => {
                header.style.border = '';
            }, 2000);
            
            // Force an immediate update to show changes
            setTimeout(() => {
                fetch('/api/metrics')
                    .then(r => r.json())
                    .then(d => updateDashboard(d));
            }, 1000);
        })
        .catch(error => {
            console.error('Error starting demo:', error);
            showError('Failed to start demo.');
        });
}

function stopDemo() {
    fetch('/api/stop', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            console.log('Demo stopped:', data);
            updateStatus('Stopped', false);
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            isDemoRunning = false;
        })
        .catch(error => {
            console.error('Error stopping demo:', error);
            showError('Failed to stop demo.');
        });
}

function startEventStream() {
    // Close existing stream if any
    if (eventSource) {
        eventSource.close();
    }

    // Create new EventSource for Server-Sent Events
    eventSource = new EventSource('/api/metrics/stream');

    eventSource.onmessage = function(event) {
        try {
            const data = JSON.parse(event.data);
            updateDashboard(data);
        } catch (error) {
            console.error('Error parsing event data:', error);
        }
    };

    eventSource.onerror = function(error) {
        console.error('EventSource error:', error);
        // Attempt to reconnect after 3 seconds
        setTimeout(() => {
            if (!eventSource || eventSource.readyState === EventSource.CLOSED) {
                startEventStream();
            }
        }, 3000);
    };
}

function updateDashboard(data) {
    if (!data || !data.feature_pairs || data.feature_pairs.length === 0) {
        showNoData();
        return;
    }

    updateStatusText(data.last_update);
    renderOverview(data);
    renderFeaturePairCards(data.feature_pairs);
    renderSummaryTable(data.feature_pairs);
    
    // Add visual feedback that data is updating
    if (isDemoRunning) {
        addUpdateAnimation();
    }
}

function updateStatus(text, isRunning) {
    const indicator = document.getElementById('statusIndicator');
    const statusText = document.getElementById('statusText');
    
    statusText.textContent = text;
    indicator.className = 'status-indicator' + (isRunning ? ' running' : ' stopped');
}

function updateStatusText(lastUpdate) {
    if (lastUpdate) {
        const date = new Date(lastUpdate);
        const timeStr = date.toLocaleTimeString();
        document.getElementById('lastUpdate').textContent = `Last update: ${timeStr}`;
    }
}

function renderOverview(data) {
    const grid = document.getElementById('overviewGrid');
    grid.innerHTML = `
        <div class="overview-card">
            <h3>Features</h3>
            <div class="value">${data.n_features}</div>
        </div>
        <div class="overview-card">
            <h3>Samples</h3>
            <div class="value">${data.n_samples}</div>
        </div>
        <div class="overview-card">
            <h3>Avg Correlation</h3>
            <div class="value">${formatNumber(data.avg_correlation)}</div>
        </div>
        <div class="overview-card">
            <h3>Max Correlation</h3>
            <div class="value">${formatNumber(data.max_correlation)}</div>
        </div>
        <div class="overview-card">
            <h3>High Corr Pairs</h3>
            <div class="value">${data.high_corr_pairs}</div>
        </div>
        <div class="overview-card">
            <h3>Features to Remove</h3>
            <div class="value">${data.features_to_remove.length}</div>
        </div>
    `;
}

function renderFeaturePairCards(featurePairs) {
    const grid = document.getElementById('dashboardGrid');
    grid.innerHTML = '';

    featurePairs.forEach(pair => {
        const card = createFeaturePairCard(pair);
        grid.appendChild(card);
    });
}

function createFeaturePairCard(pair) {
    const card = document.createElement('div');
    card.className = 'metric-card';

    const strengthClass = pair.strength.toLowerCase().replace('_', '-');
    const corrClass = pair.correlation >= 0 ? 'positive' : 'negative';
    
    card.innerHTML = `
        <div class="card-header">
            <h3 class="card-title">
                <i class="fas fa-link"></i> ${pair.pair_name}
            </h3>
            <span class="strength-badge ${strengthClass}">
                ${pair.strength_icon} ${pair.strength}
            </span>
        </div>
        
        <div class="correlation-value ${corrClass}">
            ${formatCorrelation(pair.correlation)}
        </div>
        
        <div class="metrics-grid">
            <div class="metric-item highlight-metric">
                <div class="metric-label">Correlation</div>
                <div class="metric-value">${formatCorrelation(pair.correlation)}</div>
            </div>
            
            <div class="metric-item">
                <div class="metric-label">Covariance</div>
                <div class="metric-value">${formatNumber(pair.covariance)}</div>
            </div>
            
            <div class="metric-item">
                <div class="metric-label">Absolute Correlation</div>
                <div class="metric-value">${formatNumber(pair.abs_correlation)}</div>
            </div>
            
            <div class="metric-item">
                <div class="metric-label">High Correlation</div>
                <div class="metric-value">${pair.is_high_correlation ? 'Yes' : 'No'}</div>
            </div>
        </div>
        
        <div style="margin-top: 15px; padding: 12px; background: #f8f9fa; border-radius: 8px; font-size: 14px; color: #495057;">
            <strong>Recommendation:</strong> ${pair.advice}
        </div>
    `;

    return card;
}

function renderSummaryTable(featurePairs) {
    const tbody = document.getElementById('summaryTableBody');
    tbody.innerHTML = '';

    featurePairs.forEach(pair => {
        const row = document.createElement('tr');
        
        const strengthClass = pair.strength.toLowerCase().replace('_', '-');
        
        row.innerHTML = `
            <td><strong>${pair.pair_name}</strong></td>
            <td class="value-highlight">${formatCorrelation(pair.correlation)}</td>
            <td>${formatNumber(pair.covariance)}</td>
            <td>
                <span class="strength-badge ${strengthClass}">
                    ${pair.strength_icon} ${pair.strength}
                </span>
            </td>
            <td style="font-size: 12px;">${pair.advice}</td>
        `;
        
        tbody.appendChild(row);
    });
}

function formatNumber(value) {
    if (value === null || value === undefined || isNaN(value)) {
        return '0';
    }
    
    if (value === Infinity) {
        return '∞';
    }
    
    // Format with appropriate decimal places
    if (Math.abs(value) >= 1000) {
        return value.toLocaleString('en-US', { maximumFractionDigits: 2 });
    } else if (Math.abs(value) >= 1) {
        return value.toFixed(2);
    } else {
        return value.toFixed(4);
    }
}

function formatCorrelation(value) {
    if (value === null || value === undefined || isNaN(value)) {
        return '0.00';
    }
    
    return value.toFixed(3);
}

function showNoData() {
    const grid = document.getElementById('dashboardGrid');
    grid.innerHTML = '<div class="no-data"><i class="fas fa-exclamation-circle"></i> No metrics data available</div>';
}

function showError(message) {
    console.error(message);
    alert(message);
}

function addUpdateAnimation() {
    // Add a subtle pulse animation to cards to show they're updating
    const cards = document.querySelectorAll('.metric-card');
    cards.forEach((card, index) => {
        setTimeout(() => {
            card.style.transition = 'box-shadow 0.3s ease';
            card.style.boxShadow = '0 8px 24px rgba(59, 130, 246, 0.2), 0 2px 6px rgba(0, 0, 0, 0.08)';
            setTimeout(() => {
                card.style.boxShadow = '';
            }, 300);
        }, index * 50);
    });
    
    // Pulse the overview cards
    const overviewCards = document.querySelectorAll('.overview-card');
    overviewCards.forEach((card, index) => {
        setTimeout(() => {
            card.style.transform = 'translateY(-3px) scale(1.02)';
            setTimeout(() => {
                card.style.transform = '';
            }, 200);
        }, index * 30);
    });
}

// Clean up on page unload
window.addEventListener('beforeunload', function() {
    if (eventSource) {
        eventSource.close();
    }
});

