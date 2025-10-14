// Gradient Descent Dashboard JavaScript
// Interactive controls and visualizations

class GradientDescentDashboard {
    constructor() {
        this.currentParams = {
            learningRate: 0.000001,
            epochs: 100,
            initialWeight: 0.1,
            initialBias: 0,
            currentWeight: 0.1,
            currentBias: 0
        };
        
        this.trainingHistory = [];
        this.isTraining = false;
        
        this.initializeEventListeners();
        this.loadInitialPlot();
    }
    
    initializeEventListeners() {
        // Slider controls
        document.getElementById('learningRate').addEventListener('input', (e) => {
            this.currentParams.learningRate = parseFloat(e.target.value);
            document.getElementById('learningRateValue').textContent = this.currentParams.learningRate.toExponential(2);
        });
        
        document.getElementById('epochs').addEventListener('input', (e) => {
            this.currentParams.epochs = parseInt(e.target.value);
            document.getElementById('epochsValue').textContent = this.currentParams.epochs;
        });
        
        document.getElementById('initialWeight').addEventListener('input', (e) => {
            this.currentParams.initialWeight = parseFloat(e.target.value);
            document.getElementById('initialWeightValue').textContent = this.currentParams.initialWeight.toFixed(2);
        });
        
        document.getElementById('initialBias').addEventListener('input', (e) => {
            this.currentParams.initialBias = parseFloat(e.target.value);
            document.getElementById('initialBiasValue').textContent = `$${this.currentParams.initialBias.toLocaleString()}`;
        });
        
        // Button controls
        document.getElementById('trainBtn').addEventListener('click', () => this.trainModel());
        document.getElementById('updatePlotBtn').addEventListener('click', () => this.updatePlot());
        document.getElementById('resetBtn').addEventListener('click', () => this.reset());
    }
    
    async loadInitialPlot() {
        try {
            await this.updatePlot();
        } catch (error) {
            console.error('Error loading initial plot:', error);
            this.showError('Failed to load initial visualization');
        }
    }
    
    async updatePlot() {
        const plotContainer = document.getElementById('plotContainer');
        plotContainer.innerHTML = '<div class="spinner-border text-primary" role="status"><span class="visually-hidden">Loading...</span></div><p class="mt-2">Updating visualization...</p>';
        
        try {
            const response = await fetch('/api/plot', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    weight: this.currentParams.currentWeight,
                    bias: this.currentParams.currentBias,
                    learning_rate: this.currentParams.learningRate,
                    epochs: this.currentParams.epochs
                })
            });
            
            const data = await response.json();
            
            plotContainer.innerHTML = `<img src="data:image/png;base64,${data.plot}" alt="Gradient Descent Visualization" class="img-fluid">`;
            
            this.updateModelInfo();
            
        } catch (error) {
            console.error('Error updating plot:', error);
            this.showError('Failed to update visualization');
        }
    }
    
    async trainModel() {
        if (this.isTraining) return;
        
        this.isTraining = true;
        this.trainingHistory = [];
        
        const trainBtn = document.getElementById('trainBtn');
        const progressContainer = document.getElementById('trainingProgress');
        const historyChart = document.getElementById('historyChart');
        
        trainBtn.disabled = true;
        trainBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Training...';
        progressContainer.style.display = 'block';
        historyChart.style.display = 'none';
        
        try {
            const response = await fetch('/api/train', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    learning_rate: this.currentParams.learningRate,
                    epochs: this.currentParams.epochs,
                    initial_weight: this.currentParams.initialWeight,
                    initial_bias: this.currentParams.initialBias
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const responseText = await response.text();
            let data;
            try {
                data = JSON.parse(responseText);
            } catch (parseError) {
                console.error('JSON parse error:', parseError);
                console.error('Response text:', responseText);
                throw new Error('Invalid JSON response from server');
            }
            this.trainingHistory = data.history;
            this.currentParams.currentWeight = data.final_weight;
            this.currentParams.currentBias = data.final_bias;
            
            // Update progress display
            this.updateTrainingProgress();
            
            // Show training history chart
            this.showTrainingHistory();
            
            // Update plot with final results
            await this.updatePlot();
            
        } catch (error) {
            console.error('Error training model:', error);
            this.showError('Failed to train model');
        } finally {
            this.isTraining = false;
            trainBtn.disabled = false;
            trainBtn.innerHTML = '<i class="fas fa-play"></i> Train Model';
        }
    }
    
    updateTrainingProgress() {
        const progressBar = document.getElementById('progressBar');
        const currentEpoch = document.getElementById('currentEpoch');
        const currentLoss = document.getElementById('currentLoss');
        const currentWeightProgress = document.getElementById('currentWeightProgress');
        
        if (this.trainingHistory.length > 0) {
            const lastEpoch = this.trainingHistory[this.trainingHistory.length - 1];
            const progress = (lastEpoch.epoch / this.currentParams.epochs) * 100;
            
            progressBar.style.width = `${Math.min(progress, 100)}%`;
            currentEpoch.textContent = lastEpoch.epoch;
            currentLoss.textContent = lastEpoch.loss.toExponential(2);
            currentWeightProgress.textContent = lastEpoch.weight.toFixed(4);
        }
    }
    
    showTrainingHistory() {
        const historyChart = document.getElementById('historyChart');
        historyChart.style.display = 'block';
        
        const ctx = document.getElementById('historyCanvas').getContext('2d');
        
        // Destroy existing chart if it exists
        if (this.chart) {
            this.chart.destroy();
        }
        
        const epochs = this.trainingHistory.map(h => h.epoch);
        const losses = this.trainingHistory.map(h => h.loss);
        const weights = this.trainingHistory.map(h => h.weight);
        
        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: epochs,
                datasets: [
                    {
                        label: 'Loss',
                        data: losses,
                        borderColor: 'rgb(220, 53, 69)',
                        backgroundColor: 'rgba(220, 53, 69, 0.1)',
                        yAxisID: 'y'
                    },
                    {
                        label: 'Weight',
                        data: weights,
                        borderColor: 'rgb(0, 123, 255)',
                        backgroundColor: 'rgba(0, 123, 255, 0.1)',
                        yAxisID: 'y1'
                    }
                ]
            },
            options: {
                responsive: true,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Epoch'
                        }
                    },
                    y: {
                        type: 'logarithmic',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Loss (Log Scale)'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: 'Weight'
                        },
                        grid: {
                            drawOnChartArea: false,
                        },
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Training Progress: Loss and Weight Evolution'
                    },
                    legend: {
                        display: true,
                        position: 'top'
                    }
                }
            }
        });
    }
    
    updateModelInfo() {
        document.getElementById('currentWeight').textContent = this.currentParams.currentWeight.toFixed(4);
        document.getElementById('currentBias').textContent = `$${this.currentParams.currentBias.toLocaleString()}`;
        document.getElementById('modelFormula').textContent = 
            `Price = ${this.currentParams.currentWeight.toFixed(4)} × Size + $${this.currentParams.currentBias.toLocaleString()}`;
        
        if (this.trainingHistory.length > 0) {
            const finalLoss = this.trainingHistory[this.trainingHistory.length - 1].loss;
            document.getElementById('finalLoss').textContent = finalLoss.toExponential(2);
            document.getElementById('finalLoss').className = 'text-success';
        } else {
            document.getElementById('finalLoss').textContent = 'Not trained';
            document.getElementById('finalLoss').className = 'text-muted';
        }
    }
    
    reset() {
        this.currentParams = {
            learningRate: 0.000001,
            epochs: 100,
            initialWeight: 0.1,
            initialBias: 0,
            currentWeight: 0.1,
            currentBias: 0
        };
        
        this.trainingHistory = [];
        
        // Reset sliders
        document.getElementById('learningRate').value = 0.000001;
        document.getElementById('epochs').value = 100;
        document.getElementById('initialWeight').value = 0.1;
        document.getElementById('initialBias').value = 0;
        
        // Reset displays
        document.getElementById('learningRateValue').textContent = '0.000001';
        document.getElementById('epochsValue').textContent = '100';
        document.getElementById('initialWeightValue').textContent = '0.10';
        document.getElementById('initialBiasValue').textContent = '$0';
        
        // Hide training progress
        document.getElementById('trainingProgress').style.display = 'none';
        document.getElementById('historyChart').style.display = 'none';
        
        // Update plot
        this.updatePlot();
        
        // Destroy chart if it exists
        if (this.chart) {
            this.chart.destroy();
            this.chart = null;
        }
    }
    
    showError(message) {
        const plotContainer = document.getElementById('plotContainer');
        plotContainer.innerHTML = `
            <div class="alert alert-danger" role="alert">
                <i class="fas fa-exclamation-triangle"></i>
                <strong>Error:</strong> ${message}
            </div>
        `;
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new GradientDescentDashboard();
    
    // Add some helpful console messages
    console.log('🎯 Gradient Descent Dashboard Loaded!');
    console.log('📊 Features available:');
    console.log('   • Interactive parameter adjustment');
    console.log('   • Real-time visualization');
    console.log('   • Training progress tracking');
    console.log('   • Historical data charts');
    console.log('🚀 Ready to explore gradient descent!');
});
