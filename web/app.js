// Modern ROG-Agent Dashboard JavaScript

// DOM Elements
const elements = {
  // Navigation
  sidebar: document.getElementById('sidebar'),
  sidebarToggle: document.getElementById('sidebarToggle'),
  mobileMenuBtn: document.getElementById('mobileMenuBtn'),
  navItems: document.querySelectorAll('.nav-item'),

  // Header
  pageTitle: document.getElementById('pageTitle'),
  themeToggle: document.getElementById('themeToggle'),

  // Sections
  sections: document.querySelectorAll('.section'),
  mainContent: document.querySelector('.main-content'),

  // Dashboard
  totalRuns: document.getElementById('totalRuns'),
  activeTasks: document.getElementById('activeTasks'),
  totalReports: document.getElementById('totalReports'),
  systemHealth: document.getElementById('systemHealth'),
  activityFeed: document.getElementById('activityFeed'),
  statusIndicator: document.getElementById('statusIndicator'),

  // Tasks
  taskTemplate: document.getElementById('taskTemplate'),
  loadTemplateBtn: document.getElementById('loadTemplateBtn'),
  taskInput: document.getElementById('taskInput'),
  runTaskBtn: document.getElementById('runTaskBtn'),
  taskOutput: document.getElementById('taskOutput'),
  taskHistory: document.getElementById('taskHistory'),

  // Task Output Tabs
  tabBtns: document.querySelectorAll('.tab-btn'),
  tabContents: document.querySelectorAll('.tab-content'),

  // Reports
  reportTypeFilter: document.getElementById('reportTypeFilter'),
  reportSearch: document.getElementById('reportSearch'),
  refreshReportsBtn: document.getElementById('refreshReportsBtn'),
  reportsList: document.getElementById('reportsList'),
  reportViewer: document.getElementById('reportViewer'),

  // Data
  ingestPath: document.getElementById('ingestPath'),
  ingestBtn: document.getElementById('ingestBtn'),
  ingestOutput: document.getElementById('ingestOutput'),

  // Settings
  themeSelect: document.getElementById('themeSelect'),
  healthCheckBtn: document.getElementById('healthCheckBtn'),
  healthStatus: document.getElementById('healthStatus'),

  // Modals
  modalOverlay: document.getElementById('modalOverlay'),
  taskModal: document.getElementById('taskModal'),
  taskProgress: document.getElementById('taskProgress'),
  taskStatus: document.getElementById('taskStatus'),
};

// State Management
const state = {
  currentSection: 'dashboard',
  theme: localStorage.getItem('theme') || 'light',
  sidebarCollapsed: false,
  activeTask: null,
  reports: [],
  runs: [],
  systemHealth: null,
};

// Task Templates
const taskTemplates = {
  backtest: {
    title: "Backtest Trading Strategy",
    description: "Design and backtest a quantitative trading strategy",
    template: `Design and backtest a trading strategy with the following requirements:

1. Universe: SPY, QQQ, IWM (S&P 500, Nasdaq 100, Russell 2000 ETFs)
2. Strategy: Mean-reversion based on RSI indicator
3. Entry: RSI < 30 (oversold)
4. Exit: RSI > 70 (overbought) or after 20 trading days
5. Position sizing: Equal weight allocation
6. Backtest period: 2015-2024
7. Risk management: 2% maximum drawdown limit

Please implement this strategy, run the backtest, and provide:
- Performance metrics (Sharpe ratio, max drawdown, win rate)
- Equity curve visualization
- Risk analysis
- Strategy code implementation`
  },

  correlation: {
    title: "Asset Correlation Analysis",
    description: "Analyze correlations between financial assets",
    template: `Perform a comprehensive correlation analysis with the following scope:

1. Assets: SPY, QQQ, TLT, GLD, USO (Stocks, Bonds, Gold, Oil)
2. Time period: Last 5 years of daily data
3. Analysis requirements:
   - Rolling 252-day correlations
   - Correlation matrix with heat map
   - Statistical significance testing
   - Regime-dependent correlations (high/low volatility periods)
   - Lead-lag relationships

Please provide:
- Correlation analysis code
- Statistical test results
- Visualization of correlation dynamics
- Interpretation of findings
- Risk management implications`
  },

  kan: {
    title: "KAN Demo Analysis",
    description: "Run and analyze the Multifidelity Kolmogorov-Arnold Network demo",
    template: `Execute the Multifidelity KAN demo and provide comprehensive analysis:

1. Run the KAN training on the demo dataset
2. Analyze the network architecture and performance
3. Compare with traditional MLP baseline
4. Evaluate training dynamics and convergence
5. Assess generalization capabilities

Please provide:
- Demo execution results
- Performance metrics comparison
- Network architecture analysis
- Training visualization
- Computational efficiency assessment`
  },

  ingest: {
    title: "Document Ingestion",
    description: "Ingest research documents into the knowledge base",
    template: `Ingest academic and research documents for quantitative finance analysis:

Target documents:
- Research papers on quantitative trading strategies
- Financial econometrics literature
- Risk management methodologies
- Machine learning applications in finance

Ingestion requirements:
- Chunk documents into semantically meaningful sections
- Preserve mathematical equations and formulas
- Extract key concepts and methodologies
- Enable semantic search and retrieval

Please specify the document path and execute ingestion with quality validation.`
  },

  custom: {
    title: "Custom Analysis",
    description: "Define your own analysis task",
    template: `Describe your custom quantitative finance analysis task:

Please specify:
1. Research question or hypothesis
2. Data requirements (assets, time period, frequency)
3. Analysis methodology
4. Expected outputs and visualizations
5. Risk considerations

Be specific about your requirements and I'll implement a complete analysis pipeline.`
  }
};

// Utility Functions
function setTheme(theme) {
  document.documentElement.className = theme;
  state.theme = theme;
  localStorage.setItem('theme', theme);

  // Update theme toggle icon
  elements.themeToggle.textContent = theme === 'dark' ? '☀️' : '🌙';

  // Update Prism theme
  const prismTheme = document.getElementById('prism-theme');
  if (theme === 'dark') {
    prismTheme.href = 'https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-dark.min.css';
  } else {
    prismTheme.href = 'https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css';
  }
}

function toggleSidebar() {
  state.sidebarCollapsed = !state.sidebarCollapsed;
  elements.sidebar.classList.toggle('collapsed', state.sidebarCollapsed);
  elements.mainContent.classList.toggle('sidebar-collapsed', state.sidebarCollapsed);
  localStorage.setItem('sidebarCollapsed', state.sidebarCollapsed);
}

function showSection(sectionId) {
  // Update navigation
  elements.navItems.forEach(item => {
    item.classList.toggle('active', item.dataset.section === sectionId);
  });

  // Update sections
  elements.sections.forEach(section => {
    section.classList.toggle('active', section.id === `${sectionId}-section`);
  });

  // Update page title
  const titles = {
    dashboard: 'Dashboard',
    tasks: 'Task Management',
    reports: 'Report Viewer',
    data: 'Data Management',
    settings: 'Settings'
  };
  elements.pageTitle.textContent = titles[sectionId] || 'ROG-Agent';

  state.currentSection = sectionId;
  localStorage.setItem('currentSection', sectionId);
}

function showModal(modal) {
  elements.modalOverlay.classList.add('active');
  modal.style.display = 'block';
}

function hideModal() {
  elements.modalOverlay.classList.remove('active');
  document.querySelectorAll('.modal').forEach(modal => {
    modal.style.display = 'none';
  });
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${text}`);
  }
  return response.json();
}

function addActivityItem(icon, title, time = 'Just now') {
  const item = document.createElement('div');
  item.className = 'activity-item';
  item.innerHTML = `
    <div class="activity-icon">${icon}</div>
    <div class="activity-content">
      <div class="activity-title">${title}</div>
      <div class="activity-time">${time}</div>
    </div>
  `;

  elements.activityFeed.insertBefore(item, elements.activityFeed.firstChild);

  // Keep only last 10 items
  while (elements.activityFeed.children.length > 10) {
    elements.activityFeed.removeChild(elements.activityFeed.lastChild);
  }
}

function updateSystemStatus(health) {
  state.systemHealth = health;
  const indicator = elements.statusIndicator;

  if (health?.status === 'ok') {
    indicator.classList.add('active');
    elements.systemHealth.textContent = 'Healthy';
    elements.systemHealth.style.color = 'var(--text-accent)';
  } else {
    indicator.classList.remove('active');
    elements.systemHealth.textContent = health?.status || 'Unknown';
    elements.systemHealth.style.color = '#ef4444';
  }
}

// Task Management
function loadTaskTemplate(templateId) {
  const template = taskTemplates[templateId];
  if (template) {
    elements.taskInput.value = template.template;
    addActivityItem('📝', `Loaded template: ${template.title}`);
  }
}

async function runTask() {
  const taskText = elements.taskInput.value.trim();
  if (!taskText) {
    alert('Please enter a task description');
    return;
  }

  // Show progress modal
  showModal(elements.taskModal);
  elements.taskProgress.style.width = '0%';
  elements.taskStatus.textContent = 'Initializing task...';

  try {
    elements.taskStatus.textContent = 'Sending task to server...';
    elements.taskProgress.style.width = '25%';

    const result = await fetchJson('/run-task', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ task: taskText }),
    });

    elements.taskStatus.textContent = 'Processing results...';
    elements.taskProgress.style.width = '75%';

    // Update UI with results
    elements.taskOutput.textContent = JSON.stringify(result, null, 2);
    addActivityItem('⚡', 'Task completed successfully');

    // Refresh data
    await loadTasks();
    await loadSystemHealth();

    elements.taskProgress.style.width = '100%';
    elements.taskStatus.textContent = 'Task completed!';

    setTimeout(() => hideModal(), 2000);

  } catch (error) {
    elements.taskStatus.textContent = `Error: ${error.message}`;
    elements.taskProgress.style.width = '0%';
    addActivityItem('❌', `Task failed: ${error.message}`);
  }
}

// Report Management
async function loadReports() {
  try {
    // For now, simulate report loading - in real implementation,
    // this would scan the output directories
    const reports = [
      { name: 'demo_report.tex', type: 'latex', path: 'output/demo_report/report.tex' },
      { name: 'demo_report.csv', type: 'data', path: 'output/demo_report/SPY.csv' },
      { name: 'demo_report.json', type: 'data', path: 'output/demo_report/SPY.json' },
    ];

    state.reports = reports;
    renderReportsList(reports);
  } catch (error) {
    console.error('Failed to load reports:', error);
  }
}

function renderReportsList(reports) {
  const filter = elements.reportTypeFilter.value;
  const search = elements.reportSearch.value.toLowerCase();

  const filtered = reports.filter(report => {
    const matchesFilter = filter === 'all' || report.type === filter;
    const matchesSearch = report.name.toLowerCase().includes(search);
    return matchesFilter && matchesSearch;
  });

  elements.reportsList.innerHTML = '';

  if (filtered.length === 0) {
    elements.reportsList.innerHTML = '<div class="no-reports">No reports found</div>';
    return;
  }

  filtered.forEach(report => {
    const item = document.createElement('div');
    item.className = 'file-item';
    item.innerHTML = `
      <span class="file-icon">${getFileIcon(report.type)}</span>
      <span class="file-name">${report.name}</span>
    `;
    item.addEventListener('click', () => loadReport(report));
    elements.reportsList.appendChild(item);
  });
}

function getFileIcon(type) {
  const icons = {
    latex: '📄',
    data: '📊',
    plots: '📈'
  };
  return icons[type] || '📄';
}

async function loadReport(report) {
  try {
    // In a real implementation, this would fetch the file content
    // For now, show a placeholder
    elements.reportViewer.innerHTML = `
      <div class="report-header">
        <h3>${report.name}</h3>
        <div class="report-actions">
          <button class="btn-secondary" onclick="downloadReport('${report.path}')">Download</button>
        </div>
      </div>
      <div class="report-content">
        <pre><code class="language-${report.type === 'latex' ? 'latex' : 'json'}">
// Report content would be loaded here
// File: ${report.path}
// Type: ${report.type}
        </code></pre>
      </div>
    `;

    // Highlight code if Prism is loaded
    if (window.Prism) {
      setTimeout(() => Prism.highlightAll(), 100);
    }

    addActivityItem('📄', `Loaded report: ${report.name}`);

  } catch (error) {
    elements.reportViewer.innerHTML = `<div class="error">Failed to load report: ${error.message}</div>`;
  }
}

function downloadReport(path) {
  // In a real implementation, this would trigger a download
  addActivityItem('⬇️', `Downloading: ${path}`);
}

// Data Management
async function ingestDocuments() {
  const path = elements.ingestPath.value.trim();
  if (!path) {
    alert('Please enter a document path');
    return;
  }

  elements.ingestOutput.textContent = 'Ingesting documents...';

  try {
    const result = await fetchJson('/ingest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path: path }),
    });

    elements.ingestOutput.textContent = `Ingestion completed!\nChunks: ${result.chunks}\nAdded: ${result.added}`;
    addActivityItem('📚', `Ingested ${result.added} document chunks`);

  } catch (error) {
    elements.ingestOutput.textContent = `Ingestion failed: ${error.message}`;
  }
}

// Runs Management
async function loadTasks() {
  try {
    const data = await fetchJson('/api/tasks');
    state.runs = data.tasks;
    elements.totalRuns.textContent = data.total;

    // Update task history
    renderTaskHistory(data.tasks);

  } catch (error) {
    console.error('Failed to load tasks:', error);
  }
}

function renderTaskHistory(tasks) {
  elements.taskHistory.innerHTML = '';

  if (!tasks || tasks.length === 0) {
    elements.taskHistory.innerHTML = '<div class="no-tasks">No completed tasks yet</div>';
    return;
  }

  tasks.slice(-10).reverse().forEach(task => {
    const item = document.createElement('div');
    item.className = 'task-item';
    item.innerHTML = `
      <div class="task-info">
        <div class="task-name">${task.title}</div>
        <div class="task-time">${new Date(task.updated_at).toLocaleString()}</div>
      </div>
      <button class="btn-secondary" onclick="loadTask('${task.task_id}')">View</button>
    `;
    elements.taskHistory.appendChild(item);
  });
}

async function loadTask(taskId) {
  try {
    const task = await fetchJson(`/api/tasks/${taskId}`);
    elements.taskOutput.textContent = JSON.stringify(task, null, 2);
    showSection('tasks');
    addActivityItem('📋', `Loaded task: ${taskId}`);
  } catch (error) {
    alert(`Failed to load task: ${error.message}`);
  }
}

// System Health
async function loadSystemHealth() {
  try {
    const health = await fetchJson('/health');
    updateSystemStatus(health);
  } catch (error) {
    updateSystemStatus({ status: 'error', detail: error.message });
  }
}

async function runHealthCheck() {
  elements.healthStatus.textContent = 'Running health check...';
  await loadSystemHealth();
  elements.healthStatus.textContent = state.systemHealth?.status === 'ok'
    ? 'System is healthy'
    : 'System has issues';
}

// Event Listeners
function setupEventListeners() {
  // Navigation
  elements.sidebarToggle.addEventListener('click', toggleSidebar);
  elements.mobileMenuBtn.addEventListener('click', () => {
    elements.sidebar.classList.toggle('mobile-open');
  });

  elements.navItems.forEach(item => {
    item.addEventListener('click', () => {
      const section = item.dataset.section;
      showSection(section);
      elements.sidebar.classList.remove('mobile-open');
    });
  });

  // Theme
  elements.themeToggle.addEventListener('click', () => {
    const newTheme = state.theme === 'dark' ? 'light' : 'dark';
    setTheme(newTheme);
  });

  elements.themeSelect.addEventListener('change', (e) => {
    setTheme(e.target.value);
  });

  // Tasks
  elements.loadTemplateBtn.addEventListener('click', () => {
    const templateId = elements.taskTemplate.value;
    if (templateId && templateId !== '') {
      loadTaskTemplate(templateId);
    }
  });

  elements.runTaskBtn.addEventListener('click', runTask);

  // Task output tabs
  elements.tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const tabId = btn.dataset.tab;

      // Update tab buttons
      elements.tabBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      // Update tab content
      elements.tabContents.forEach(content => {
        content.classList.toggle('active', content.id === `${tabId}-tab`);
      });
    });
  });

  // Reports
  elements.reportTypeFilter.addEventListener('change', () => renderReportsList(state.reports));
  elements.reportSearch.addEventListener('input', () => renderReportsList(state.reports));
  elements.refreshReportsBtn.addEventListener('click', loadReports);

  // Data
  elements.ingestBtn.addEventListener('click', ingestDocuments);

  // Settings
  elements.healthCheckBtn.addEventListener('click', runHealthCheck);

  // Modals
  elements.modalOverlay.addEventListener('click', hideModal);
  document.querySelectorAll('.modal-close').forEach(btn => {
    btn.addEventListener('click', hideModal);
  });

  // Keyboard shortcuts
  document.addEventListener('keydown', (e) => {
    if (e.ctrlKey || e.metaKey) {
      switch (e.key) {
        case '1':
          e.preventDefault();
          showSection('dashboard');
          break;
        case '2':
          e.preventDefault();
          showSection('tasks');
          break;
        case '3':
          e.preventDefault();
          showSection('reports');
          break;
        case '4':
          e.preventDefault();
          showSection('data');
          break;
        case '5':
          e.preventDefault();
          showSection('settings');
          break;
      }
    }
  });
}

// Initialization
async function init() {
  // Load saved preferences
  const savedTheme = localStorage.getItem('theme') || 'light';
  const savedSection = localStorage.getItem('currentSection') || 'dashboard';
  const savedSidebarCollapsed = localStorage.getItem('sidebarCollapsed') === 'true';

  // Apply theme
  setTheme(savedTheme);
  elements.themeSelect.value = savedTheme;

  // Apply sidebar state
  if (savedSidebarCollapsed) {
    toggleSidebar();
  }

  // Setup event listeners
  setupEventListeners();

  // Load initial data
  await Promise.all([
    loadSystemHealth(),
    loadTasks(),
    loadReports(),
  ]);

  // Show initial section
  showSection(savedSection);

  // Add welcome activity
  addActivityItem('🚀', 'ROG-Agent Dashboard initialized');
}

// Global functions for onclick handlers
window.loadTask = loadTask;
window.downloadReport = downloadReport;

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
