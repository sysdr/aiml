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
    if (!data || !data.datasets || data.datasets.length === 0) {
        showNoData();
        return;
    }

    updateStatusText(data.last_update);
    renderMetricCards(data.datasets);
    renderSummaryTable(data.datasets);
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

function renderMetricCards(datasets) {
    const grid = document.getElementById('dashboardGrid');
    grid.innerHTML = '';

    datasets.forEach(dataset => {
        const card = createMetricCard(dataset);
        grid.appendChild(card);
    });
}

function createMetricCard(dataset) {
    const card = document.createElement('div');
    card.className = 'metric-card';

    const badgeClass = dataset.ml_readiness.toLowerCase().replace('_', '-');
    
    card.innerHTML = `
        <div class="card-header">
            <h3 class="card-title">
                <i class="fas fa-database"></i> ${dataset.dataset_name}
            </h3>
            <span class="ml-badge ${badgeClass}">
                ${dataset.ml_readiness_icon} ${dataset.ml_readiness}
            </span>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-item highlight-metric">
                <div class="metric-label">Mean</div>
                <div class="metric-value">${formatNumber(dataset.mean)}<span class="metric-unit"></span></div>
            </div>
            
            <div class="metric-item">
                <div class="metric-label">Standard Deviation</div>
                <div class="metric-value">${formatNumber(dataset.sample_std)}<span class="metric-unit"></span></div>
            </div>
            
            <div class="metric-item">
                <div class="metric-label">Variance</div>
                <div class="metric-value">${formatNumber(dataset.sample_variance)}<span class="metric-unit"></span></div>
            </div>
            
            <div class="metric-item">
                <div class="metric-label">Coefficient of Variation</div>
                <div class="metric-value">${formatNumber(dataset.cv)}<span class="metric-unit">%</span></div>
            </div>
            
            <div class="metric-item">
                <div class="metric-label">Sample Size</div>
                <div class="metric-value">${dataset.sample_size}<span class="metric-unit"> points</span></div>
            </div>
            
            <div class="metric-item">
                <div class="metric-label">Range</div>
                <div class="metric-value">${formatNumber(dataset.min)} - ${formatNumber(dataset.max)}</div>
            </div>
            
            <div class="metric-item">
                <div class="metric-label">IQR</div>
                <div class="metric-value">${formatNumber(dataset.iqr)}<span class="metric-unit"></span></div>
            </div>
            
            <div class="metric-item">
                <div class="metric-label">Outliers</div>
                <div class="metric-value">${dataset.outlier_count}<span class="metric-unit"> (${formatNumber(dataset.outlier_percentage)}%)</span></div>
            </div>
        </div>
        
        <div style="margin-top: 15px; padding: 12px; background: #f8f9fa; border-radius: 8px; font-size: 14px; color: #495057;">
            <strong>Recommendation:</strong> ${dataset.recommendation}
        </div>
    `;

    return card;
}

function renderSummaryTable(datasets) {
    const tbody = document.getElementById('summaryTableBody');
    tbody.innerHTML = '';

    datasets.forEach(dataset => {
        const row = document.createElement('tr');
        
        const badgeClass = dataset.ml_readiness.toLowerCase().replace('_', '-');
        
        row.innerHTML = `
            <td><strong>${dataset.dataset_name}</strong></td>
            <td class="value-highlight">${formatNumber(dataset.mean)}</td>
            <td class="value-highlight">${formatNumber(dataset.sample_std)}</td>
            <td class="value-highlight">${formatNumber(dataset.sample_variance)}</td>
            <td class="value-highlight">${formatNumber(dataset.cv)}%</td>
            <td>${dataset.outlier_count} (${formatNumber(dataset.outlier_percentage)}%)</td>
            <td>
                <span class="ml-badge ${badgeClass}">
                    ${dataset.ml_readiness_icon} ${dataset.ml_readiness}
                </span>
            </td>
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

function showNoData() {
    const grid = document.getElementById('dashboardGrid');
    grid.innerHTML = '<div class="no-data"><i class="fas fa-exclamation-circle"></i> No metrics data available</div>';
}

function showError(message) {
    console.error(message);
    // You could add a toast notification here
    alert(message);
}

// Clean up on page unload
window.addEventListener('beforeunload', function() {
    if (eventSource) {
        eventSource.close();
    }
});

