/**
 * HealthAI Admin Panel - Real Backend Integration
 * Uses JWT auth from api.js. Requires staff (is_staff) role.
 */

import { isAuthenticated, getStoredUser, apiLogout, clearTokens, apiLogin,
    apiAdminAnalytics, apiAdminUsers, apiAdminModelMetrics, apiAdminConfig,
    apiAdminAuditLog, apiAdminRetrain, apiAdminUserActivity, apiAdminDailyActivity,
    apiAdminTopQuestions } from './api.js';


// ── Inactivity Timer ───────────────────────────────────
let inactivityTimer = null;
const INACTIVITY_MS = 5 * 60 * 1000;

function resetInactivityTimer() {
    if (inactivityTimer) clearTimeout(inactivityTimer);
    inactivityTimer = setTimeout(() => {
        clearTokens();
        window.location.href = 'auth.html';
    }, INACTIVITY_MS);
}

// ── Admin Login Gate ───────────────────────────────────
async function verifyAdmin() {
    const email = document.getElementById('admin-email')?.value?.trim() || '';
    const code = document.getElementById('admin-code')?.value || '';
    const btn = document.querySelector('#admin-login-form button[type="submit"]');

    if (!email || !code) {
        showAdminToast('Please enter your email and access code.', 'error');
        return;
    }

    if (btn) { btn.textContent = 'Authenticating...'; btn.disabled = true; }

    try {
        const data = await apiLogin(email, code);
        const user = data.user;
        if (!user?.is_staff) {
            showAdminToast('Access denied. Admin privileges required.', 'error');
            if (btn) { btn.textContent = 'Authenticate'; btn.disabled = false; }
            return;
        }
        const roleGate = document.getElementById('role-gate');
        const adminApp = document.getElementById('admin-app');
        if (roleGate) roleGate.style.display = 'none';
        if (adminApp) adminApp.style.display = 'flex';
        initAdminPanel();
    } catch (err) {
        const msg = err.status === 401 ? 'Invalid credentials.' : err.message || 'Authentication failed.';
        showAdminToast(msg, 'error');
        if (btn) { btn.textContent = 'Authenticate'; btn.disabled = false; }
    }
}

function showAdminToast(message, type = 'success') {
    let toast = document.getElementById('admin-toast');
    if (!toast) {
        toast = document.createElement('div');
        toast.id = 'admin-toast';
        toast.style.cssText = 'position:fixed;bottom:24px;right:24px;padding:12px 20px;border-radius:8px;font-size:0.875rem;z-index:9999;transition:opacity 0.3s;';
        document.body.appendChild(toast);
    }
    toast.textContent = message;
    toast.style.background = type === 'error' ? 'rgba(244,63,94,0.9)' : 'rgba(16,185,129,0.9)';
    toast.style.color = '#fff';
    toast.style.opacity = '1';
    setTimeout(() => { toast.style.opacity = '0'; }, 3000);
}

// Expose to global scope for HTML onclick
window.verifyAdmin = verifyAdmin;

// ── Check Existing Session on Load ─────────────────────
document.addEventListener('DOMContentLoaded', () => {
    lucide.createIcons();

    if (!isAuthenticated()) {
        // Show the role gate login form — user must authenticate
        return;
    }

    const user = getStoredUser();
    if (!user?.is_staff) {
        // Authenticated but not admin — redirect to user dashboard
        window.location.href = 'index.html';
        return;
    }

    // Already authenticated as admin — skip the gate
    const roleGate = document.getElementById('role-gate');
    const adminApp = document.getElementById('admin-app');
    if (roleGate) roleGate.style.display = 'none';
    if (adminApp) adminApp.style.display = 'flex';
    initAdminPanel();
});

// ── Admin Panel Init ───────────────────────────────────
function initAdminPanel() {
    if (!isAuthenticated()) {
        clearTokens();
        window.location.href = 'auth.html';
        return;
    }

    lucide.createIcons();
    // Start inactivity timer
    resetInactivityTimer();
    ['mousemove', 'keydown', 'click', 'scroll', 'touchstart'].forEach(evt => {
        document.addEventListener(evt, resetInactivityTimer, { passive: true });
    });

    // Update admin name in sidebar
    const user = getStoredUser();
    const adminNameEl = document.getElementById('admin-display-name');
    const adminAvatarEl = document.getElementById('admin-avatar-initials');
    if (user) {
        const displayName = user.first_name
            ? `${user.first_name} ${user.last_name || ''}`.trim()
            : user.username || 'Admin';
        if (adminNameEl) adminNameEl.textContent = displayName;
        if (adminAvatarEl) {
            adminAvatarEl.textContent = (user.first_name?.[0] || user.username?.[0] || 'A').toUpperCase();
        }
    }

    const navItems = document.querySelectorAll('.admin-sidebar .nav-links li');
    const container = document.getElementById('admin-section-container');
    const title = document.getElementById('admin-page-title');

    if (!container || !title) {
        console.error('Admin panel: required DOM elements not found');
        return;
    }

    const titles = {
        overview: 'System Overview',
        alerts: 'Health Alerts',
        models: 'AI Model Management',
        users: 'User Management',
        activity: 'Daily Activity',
        questions: 'Top Questions',
        system: 'System Health',
        audit: 'Audit Log'
    };

    const loadSection = (sectionId) => {
        navItems.forEach(li => li.classList.toggle('active', li.dataset.section === sectionId));
        if (title) title.textContent = titles[sectionId] || 'Admin';
        if (container) {
            container.innerHTML = '<div class="admin-section" style="text-align:center;padding:40px;"><i data-lucide="loader" class="spin" style="width:32px;height:32px;color:var(--accent-color);"></i><p style="margin-top:12px;color:var(--text-muted);">Loading...</p></div>';
        }
        lucide.createIcons();

        loadSectionData(sectionId).then(html => {
            if (container) {
                container.innerHTML = html;
                lucide.createIcons();
                bindSectionActions(sectionId);
                // Magic UI: init spotlight on admin cards
                initAdminMagicCards();
            }
        }).catch(err => {
            if (container) {
                container.innerHTML = `<div class="admin-section"><div class="admin-panel"><p style="color:var(--danger-color);">Error: ${escapeHtml(err?.message || 'Unknown error')}</p></div></div>`;
            }
        });
    };

    navItems.forEach(li => {
        li.onclick = () => loadSection(li.dataset.section);
    });

    // Logout
    const logoutBtn = document.getElementById('admin-logout');
    if (logoutBtn) {
        logoutBtn.onclick = async () => {
            try { await apiLogout(); } catch { clearTokens(); }
            window.location.href = 'auth.html';
        };
    }

    loadSection('overview');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ── Load real data from backend ────────────────────────
async function loadSectionData(sectionId) {
    switch (sectionId) {
        case 'overview': return loadOverview();
        case 'alerts': return loadAlerts();
        case 'models': return loadModels();
        case 'users': return loadUsers();
        case 'activity': return loadActivity();
        case 'questions': return loadQuestions();
        case 'system': return loadSystem();
        case 'audit': return loadAudit();
        default: return '<div class="admin-section"><p>Unknown section</p></div>';
    }
}

async function loadOverview() {
    let analytics = {};
    try { analytics = await apiAdminAnalytics(); } catch { /* fallback to defaults */ }

    const hasData = analytics && Object.keys(analytics).length > 0;
    const consultations = analytics.cases_total || analytics.total_consultations || analytics.total_cases || 0;
    const accuracy = hasData ? (analytics.model_accuracy || '—') : '—';
    const avgResponse = hasData ? (analytics.avg_response_time || '—') : '—';
    const totalUsers = analytics.users_total || 0;
    const activeUsers7d = analytics.users_active_7d || 0;
    const activeUsers = analytics.users_active || 0;
    const newUsersToday = analytics.users_new_today || 0;
    const newUsers7d = analytics.users_new_7d || 0;
    const messagesToday = analytics.messages_today || 0;
    const totalMessages = analytics.chat_messages_total || 0;
    const totalSessions = analytics.chat_sessions_total || 0;
    const uptime = hasData ? (analytics.uptime || '—') : '—';

    // Top active users mini-table
    const topActive = analytics.top_active_users || [];
    const topActiveRows = topActive.length ? topActive.map(u => {
        const name = `${u.first_name || ''} ${u.last_name || ''}`.trim() || u.username || u.email;
        return `<tr><td>${escapeHtml(name)}</td><td>${u.msg_count}</td></tr>`;
    }).join('') : '<tr><td colspan="2" style="color:var(--text-muted);">No data yet</td></tr>';

    // Top questions mini-list
    const topQs = analytics.top_questions || [];
    const topQsItems = topQs.length ? topQs.slice(0, 5).map(q =>
        `<li class="top-q-mini"><span class="top-q-freq">${q.frequency}x</span> <span class="top-q-text">${escapeHtml(q.content?.substring(0, 80) || '')}</span></li>`
    ).join('') : '<li style="color:var(--text-muted);">No questions yet</li>';

    return `
        <div class="admin-section">
            <div class="admin-stats">
                <div class="admin-stat-card">
                    <div class="stat-icon blue"><i data-lucide="message-square" style="width:18px;height:18px;"></i></div>
                    <h3>Total Consultations</h3>
                    <p class="value">${consultations.toLocaleString()}</p>
                </div>
                <div class="admin-stat-card">
                    <div class="stat-icon green"><i data-lucide="brain" style="width:18px;height:18px;"></i></div>
                    <h3>ML Model Accuracy</h3>
                    <p class="value">${accuracy}%</p>
                </div>
                <div class="admin-stat-card">
                    <div class="stat-icon purple"><i data-lucide="zap" style="width:18px;height:18px;"></i></div>
                    <h3>Avg Response Time</h3>
                    <p class="value">${avgResponse}s</p>
                </div>
                <div class="admin-stat-card">
                    <div class="stat-icon red"><i data-lucide="users" style="width:18px;height:18px;"></i></div>
                    <h3>Active Users (7d)</h3>
                    <p class="value">${activeUsers7d.toLocaleString()}</p>
                    <p class="trend" style="color:var(--text-muted);">${totalUsers} total | ${newUsersToday} new today</p>
                </div>
                <div class="admin-stat-card">
                    <div class="stat-icon yellow"><i data-lucide="message-circle" style="width:18px;height:18px;"></i></div>
                    <h3>Messages Today</h3>
                    <p class="value">${messagesToday}</p>
                    <p class="trend" style="color:var(--text-muted);">${totalMessages} total | ${totalSessions} sessions</p>
                </div>
                <div class="admin-stat-card">
                    <div class="stat-icon blue"><i data-lucide="user-plus" style="width:18px;height:18px;"></i></div>
                    <h3>New Users (7d)</h3>
                    <p class="value">${newUsers7d}</p>
                </div>
            </div>
            <div class="admin-grid">
                <div class="admin-panel">
                    <h3><i data-lucide="users" style="width:16px;height:16px;color:var(--accent-color);"></i> Most Active Users (30d)</h3>
                    <table class="admin-table" style="margin-top:8px;">
                        <thead><tr><th>User</th><th>Messages</th></tr></thead>
                        <tbody>${topActiveRows}</tbody>
                    </table>
                </div>
                <div class="admin-panel">
                    <h3><i data-lucide="help-circle" style="width:16px;height:16px;color:var(--warning-color);"></i> Top Questions</h3>
                    <ul class="top-q-list" style="margin-top:8px;">${topQsItems}</ul>
                </div>
            </div>
            <div class="admin-grid" style="margin-top:var(--spacing-lg);">
                <div class="admin-panel">
                    <h3><i data-lucide="alert-triangle" style="width:16px;height:16px;color:var(--danger-color);"></i> Recent Critical Alerts</h3>
                    <ul class="admin-alert-list" id="overview-alerts">
                        <li style="color:var(--text-muted);padding:12px;">Loading alerts...</li>
                    </ul>
                </div>
                <div class="admin-panel">
                    <h3><i data-lucide="server" style="width:16px;height:16px;color:var(--success-color);"></i> System Health</h3>
                    <div class="sys-health-grid" id="overview-health">
                        <li style="color:var(--text-muted);padding:12px;">Loading health...</li>
                    </div>
                </div>
            </div>
        </div>
    `;
}

async function loadAlerts() {
    let alerts = [];
    try {
        const data = await apiAdminAnalytics();
        alerts = data.recent_alerts || data.alerts || [];
    } catch { /* fallback */ }

    if (!alerts.length) {
        return `
            <div class="admin-section">
                <div class="admin-panel" style="margin-bottom:var(--spacing-lg);">
                    <h3><i data-lucide="alert-triangle" style="width:16px;height:16px;color:var(--danger-color);"></i> All Health Alerts</h3>
                    <p style="color:var(--text-muted);padding:12px;">No alerts found.</p>
                </div>
            </div>
        `;
    }

    const alertItems = alerts.map(a => {
        const severity = (a.severity || a.risk_level || 'info').toLowerCase();
        const severityClass = severity === 'critical' || severity === 'high' ? 'critical' : severity === 'warning' || severity === 'moderate' ? 'warning' : 'info';
        return `
            <li class="admin-alert-item ${severityClass}">
                <span class="alert-severity ${severityClass}">${escapeHtml(a.severity || a.risk_level || 'Info')}</span>
                <span class="alert-msg">${escapeHtml(a.message || a.title || a.symptom_text || 'Alert')}</span>
                <span class="alert-time">${a.created_at ? new Date(a.created_at).toLocaleString() : a.time || ''}</span>
            </li>
        `;
    }).join('');

    return `
        <div class="admin-section">
            <div class="admin-panel" style="margin-bottom:var(--spacing-lg);">
                <h3><i data-lucide="alert-triangle" style="width:16px;height:16px;color:var(--danger-color);"></i> All Health Alerts</h3>
                <ul class="admin-alert-list">${alertItems}</ul>
            </div>
        </div>
    `;
}

async function loadModels() {
    let metrics = {};
    try { metrics = await apiAdminModelMetrics(); } catch { /* fallback */ }

    const models = metrics.models || [];
    let modelCards = '';

    if (models.length) {
        modelCards = models.map(m => `
            <div class="model-card">
                <h4>${escapeHtml(m.name || m.model_name || 'Model')}</h4>
                <span class="model-version">${escapeHtml(m.version || 'v1.0')}</span>
                <div class="model-stats">
                    <span>Accuracy: <strong>${m.accuracy || 'N/A'}</strong></span>
                    <span>F1: <strong>${m.f1_score || 'N/A'}</strong></span>
                </div>
                <div class="model-stats">
                    <span>Training: <strong>${m.last_trained || 'N/A'}</strong></span>
                    <span>Dataset: <strong>${m.dataset_size || 'N/A'}</strong></span>
                </div>
            </div>
        `).join('');
    } else {
        modelCards = `
            <div class="model-card">
                <h4>Model data unavailable</h4>
                <span class="model-version">—</span>
                <div class="model-stats">
                    <span>Accuracy: <strong>—</strong></span>
                    <span>F1: <strong>—</strong></span>
                </div>
            </div>
        `;
    }

    return `
        <div class="admin-section">
            <div class="model-cards" style="margin-bottom:var(--spacing-xl);">${modelCards}</div>
            <div class="admin-panel">
                <h3><i data-lucide="cpu" style="width:16px;height:16px;color:var(--purple-color);"></i> Retraining Console</h3>
                <div style="display:flex;gap:var(--spacing-md);align-items:center;margin-bottom:var(--spacing-md);">
                    <span style="font-size:0.85rem;color:var(--text-muted);">Current: ${escapeHtml(metrics.version || '—')}</span>
                </div>
                <div class="progress-container" style="margin-bottom:var(--spacing-md);">
                    <div class="progress-bar" id="retrain-progress" style="width: 100%"></div>
                </div>
                <button class="primary-btn" id="retrain-btn"><i data-lucide="cpu" style="width:16px;height:16px;"></i> Start Retraining Cycle</button>
            </div>
        </div>
    `;
}

async function loadUsers() {
    let usersData = { results: [], count: 0 };
    try { usersData = await apiAdminUserActivity(); } catch { /* fallback */ }

    const users = usersData.results || [];
    const totalCount = usersData.count || users.length;

    // Activity filter buttons
    const filterBtns = `
        <div class="user-filter-bar">
            <button class="filter-btn active" data-filter="">All (${totalCount})</button>
            <button class="filter-btn" data-filter="active"><span class="dot green"></span> Active (7d)</button>
            <button class="filter-btn" data-filter="recent"><span class="dot yellow"></span> Recent (30d)</button>
            <button class="filter-btn" data-filter="inactive"><span class="dot red"></span> Inactive</button>
            <div class="search-box" style="margin-left:auto;">
                <i data-lucide="search" style="width:14px;height:14px;"></i>
                <input type="text" id="user-search" placeholder="Search users..." style="padding:6px 10px 6px 32px;font-size:0.8rem;">
            </div>
        </div>
    `;

    if (!users.length) {
        return `
            <div class="admin-section">
                <div class="admin-panel">
                    <h3><i data-lucide="users" style="width:16px;height:16px;color:var(--accent-color);"></i> User Management</h3>
                    ${filterBtns}
                    <p style="color:var(--text-muted);padding:12px;">No users found.</p>
                </div>
            </div>
        `;
    }

    const rows = users.map(u => {
        const role = u.is_staff ? 'admin' : 'user';
        const roleLabel = u.is_staff ? 'Admin' : 'User';
        const statusClass = u.is_active ? 'active' : 'suspended';
        const statusLabel = u.is_active ? 'Active' : 'Disabled';
        const name = `${u.first_name || ''} ${u.last_name || ''}`.trim() || u.username;
        const actClass = u.activity_status === 'active' ? 'green' : u.activity_status === 'recent' ? 'yellow' : 'red';
        const actLabel = u.activity_status === 'active' ? 'Active' : u.activity_status === 'recent' ? 'Recent' : 'Inactive';
        return `
            <tr>
                <td>#${u.id}</td>
                <td>${escapeHtml(name)}</td>
                <td>${escapeHtml(u.email || '')}</td>
                <td><span class="user-role ${role}">${roleLabel}</span></td>
                <td><span class="user-status ${statusClass}">${statusLabel}</span></td>
                <td><span class="activity-badge ${actClass}"><span class="dot ${actClass}"></span>${actLabel}</span></td>
                <td>${u.session_count}</td>
                <td>${u.message_count}</td>
                <td>${u.last_activity ? new Date(u.last_activity).toLocaleDateString() : 'Never'}</td>
            </tr>
        `;
    }).join('');

    return `
        <div class="admin-section">
            <div class="admin-panel">
                <h3><i data-lucide="users" style="width:16px;height:16px;color:var(--accent-color);"></i> User Management (${totalCount})</h3>
                ${filterBtns}
                <div class="admin-table-wrap">
                    <table class="admin-table">
                        <thead>
                            <tr><th>ID</th><th>Name</th><th>Email</th><th>Role</th><th>Status</th><th>Activity</th><th>Sessions</th><th>Messages</th><th>Last Active</th></tr>
                        </thead>
                        <tbody>${rows}</tbody>
                    </table>
                </div>
            </div>
        </div>
    `;
}

async function loadActivity() {
    let activityData = { daily_activity: [] };
    try { activityData = await apiAdminDailyActivity({ days: 30 }); } catch { /* fallback */ }

    const daily = activityData.daily_activity || [];
    if (!daily.length) {
        return `
            <div class="admin-section">
                <div class="admin-panel">
                    <h3><i data-lucide="bar-chart-3" style="width:16px;height:16px;color:var(--accent-color);"></i> Daily Activity</h3>
                    <p style="color:var(--text-muted);padding:12px;">No activity data available yet.</p>
                </div>
            </div>
        `;
    }

    // Compute summary stats
    const totalMessages = daily.reduce((s, d) => s + (d.messages || 0), 0);
    const totalSessions = daily.reduce((s, d) => s + (d.sessions || 0), 0);
    const totalNewUsers = daily.reduce((s, d) => s + (d.new_users || 0), 0);
    const totalCases = daily.reduce((s, d) => s + (d.cases || 0), 0);
    const peakDay = daily.reduce((max, d) => (d.messages || 0) > (max.messages || 0) ? d : max, daily[0]);
    const maxMessages = peakDay?.messages || 1;

    // Build bar chart
    const bars = daily.slice(-30).map(d => {
        const pct = maxMessages > 0 ? Math.round(((d.messages || 0) / maxMessages) * 100) : 0;
        const dateLabel = d.date ? new Date(d.date + 'T00:00:00').toLocaleDateString('en-US', { month: 'short', day: 'numeric' }) : '';
        return `
            <div class="activity-bar-col" title="${dateLabel}: ${d.messages || 0} msgs, ${d.sessions || 0} sessions, ${d.new_users || 0} new users">
                <div class="activity-bar" style="height:${Math.max(pct, 2)}%;">
                    <span class="bar-tooltip">${d.messages || 0}</span>
                </div>
                <span class="bar-label">${dateLabel.split(' ').pop() || ''}</span>
            </div>
        `;
    }).join('');

    // Activity table (last 14 days)
    const tableRows = daily.slice(-14).reverse().map(d => {
        const dateLabel = d.date ? new Date(d.date + 'T00:00:00').toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' }) : d.date;
        return `
            <tr>
                <td>${escapeHtml(dateLabel)}</td>
                <td>${d.messages || 0}</td>
                <td>${d.sessions || 0}</td>
                <td>${d.new_users || 0}</td>
                <td>${d.cases || 0}</td>
            </tr>
        `;
    }).join('');

    return `
        <div class="admin-section">
            <div class="admin-stats">
                <div class="admin-stat-card">
                    <div class="stat-icon blue"><i data-lucide="message-circle" style="width:18px;height:18px;"></i></div>
                    <h3>Total Messages (30d)</h3>
                    <p class="value">${totalMessages.toLocaleString()}</p>
                </div>
                <div class="admin-stat-card">
                    <div class="stat-icon green"><i data-lucide="layout-list" style="width:18px;height:18px;"></i></div>
                    <h3>Total Sessions (30d)</h3>
                    <p class="value">${totalSessions.toLocaleString()}</p>
                </div>
                <div class="admin-stat-card">
                    <div class="stat-icon purple"><i data-lucide="user-plus" style="width:18px;height:18px;"></i></div>
                    <h3>New Users (30d)</h3>
                    <p class="value">${totalNewUsers.toLocaleString()}</p>
                </div>
                <div class="admin-stat-card">
                    <div class="stat-icon red"><i data-lucide="stethoscope" style="width:18px;height:18px;"></i></div>
                    <h3>Cases (30d)</h3>
                    <p class="value">${totalCases.toLocaleString()}</p>
                </div>
            </div>
            <div class="admin-panel" style="margin-bottom:var(--spacing-lg);">
                <h3><i data-lucide="bar-chart-3" style="width:16px;height:16px;color:var(--accent-color);"></i> Messages Per Day (30d)</h3>
                <div class="activity-chart">${bars}</div>
            </div>
            <div class="admin-panel">
                <h3><i data-lucide="table" style="width:16px;height:16px;color:var(--success-color);"></i> Daily Breakdown (Last 14 Days)</h3>
                <div class="admin-table-wrap">
                    <table class="admin-table">
                        <thead><tr><th>Date</th><th>Messages</th><th>Sessions</th><th>New Users</th><th>Cases</th></tr></thead>
                        <tbody>${tableRows}</tbody>
                    </table>
                </div>
            </div>
        </div>
    `;
}

async function loadQuestions() {
    let questionsData = { top_questions: [], total_questions: 0 };
    try { questionsData = await apiAdminTopQuestions({ days: 30, limit: 20 }); } catch { /* fallback */ }

    const questions = questionsData.top_questions || [];
    const totalQs = questionsData.total_questions || 0;

    if (!questions.length) {
        return `
            <div class="admin-section">
                <div class="admin-panel">
                    <h3><i data-lucide="help-circle" style="width:16px;height:16px;color:var(--warning-color);"></i> Most Repetitive Questions</h3>
                    <p style="color:var(--text-muted);padding:12px;">No question data available yet.</p>
                </div>
            </div>
        `;
    }

    const maxFreq = questions[0]?.frequency || 1;

    const items = questions.map((q, i) => {
        const pct = maxFreq > 0 ? Math.round((q.frequency / maxFreq) * 100) : 0;
        const rank = i + 1;
        const rankClass = rank === 1 ? 'gold' : rank === 2 ? 'silver' : rank === 3 ? 'bronze' : '';
        return `
            <li class="question-item">
                <div class="question-rank ${rankClass}">#${rank}</div>
                <div class="question-content">
                    <p class="question-text">${escapeHtml(q.question)}</p>
                    <div class="question-meta">
                        <span><i data-lucide="repeat" style="width:12px;height:12px;"></i> ${q.frequency}x</span>
                        <span><i data-lucide="users" style="width:12px;height:12px;"></i> ${q.unique_users} users</span>
                        <span><i data-lucide="clock" style="width:12px;height:12px;"></i> Last: ${q.last_asked ? new Date(q.last_asked).toLocaleDateString() : '—'}</span>
                    </div>
                    <div class="question-bar-track">
                        <div class="question-bar-fill" style="width:${pct}%;"></div>
                    </div>
                </div>
            </li>
        `;
    }).join('');

    return `
        <div class="admin-section">
            <div class="admin-panel" style="margin-bottom:var(--spacing-lg);">
                <h3><i data-lucide="help-circle" style="width:16px;height:16px;color:var(--warning-color);"></i> Most Repetitive User Questions (30d)</h3>
                <p style="color:var(--text-muted);font-size:0.85rem;margin-bottom:var(--spacing-md);">
                    ${totalQs.toLocaleString()} total questions asked in the last 30 days. Showing top ${questions.length} most repeated.
                </p>
                <ul class="question-list">${items}</ul>
            </div>
        </div>
    `;
}

async function loadSystem() {
    let config = {};
    try { config = await apiAdminConfig(); } catch { /* fallback */ }

    const services = config.services || [];
    if (!services.length && !Object.keys(config).length) {
        return `
            <div class="admin-section">
                <div class="admin-panel" style="margin-bottom:var(--spacing-lg);">
                    <h3><i data-lucide="server" style="width:16px;height:16px;color:var(--success-color);"></i> Infrastructure Health</h3>
                    <p style="color:var(--text-muted);padding:12px;">System health data unavailable. Check server connection.</p>
                </div>
            </div>
        `;
    }

    const healthItems = services.map(s => {
        const isGood = (s.status || '').toLowerCase() === 'online' || (s.status || '').toLowerCase() === 'healthy';
        return `
            <div class="sys-health-item">
                <div class="health-label">${escapeHtml(s.name)}</div>
                <div class="health-value ${isGood ? 'good' : 'warn'}">${escapeHtml(s.status)}</div>
            </div>
        `;
    }).join('');

    return `
        <div class="admin-section">
            <div class="admin-panel" style="margin-bottom:var(--spacing-lg);">
                <h3><i data-lucide="server" style="width:16px;height:16px;color:var(--success-color);"></i> Infrastructure Health</h3>
                <div class="sys-health-grid">${healthItems}</div>
            </div>
        </div>
    `;
}

async function loadAudit() {
    let auditData = { results: [] };
    try { auditData = await apiAdminAuditLog(); } catch { /* fallback */ }

    const entries = auditData.results || auditData.entries || [];
    if (!entries.length) {
        return `
            <div class="admin-section">
                <div class="admin-panel">
                    <h3><i data-lucide="file-text" style="width:16px;height:16px;color:var(--accent-color);"></i> Recent Activity</h3>
                    <p style="color:var(--text-muted);padding:12px;">No audit entries found.</p>
                </div>
            </div>
        `;
    }

    const items = entries.map(e => {
        const level = (e.level || e.action || 'info').toLowerCase();
        const iconClass = level === 'critical' || level === 'error' || level === 'delete' ? 'danger' : level === 'warning' || level === 'update' ? 'warning' : 'success';
        const icon = level === 'critical' || level === 'error' ? 'alert-circle' : level === 'warning' ? 'shield-alert' : level === 'create' || level === 'login' ? 'user-plus' : 'check-circle';
        return `
            <li class="audit-item">
                <div class="audit-icon ${iconClass}"><i data-lucide="${icon}" style="width:14px;height:14px;"></i></div>
                <div class="audit-text"><strong>${escapeHtml(e.action || e.event_type || 'Action')}</strong> — ${escapeHtml(e.description || e.detail || e.message || '')}</div>
                <div class="audit-time">${e.timestamp || e.created_at ? new Date(e.timestamp || e.created_at).toLocaleString() : ''}</div>
            </li>
        `;
    }).join('');

    return `
        <div class="admin-section">
            <div class="admin-panel">
                <h3><i data-lucide="file-text" style="width:16px;height:16px;color:var(--accent-color);"></i> Recent Activity</h3>
                <ul class="audit-list">${items}</ul>
            </div>
        </div>
    `;
}

// ── Section Action Bindings ────────────────────────────
function bindSectionActions(sectionId) {
    if (sectionId === 'models') {
        const retrainBtn = document.getElementById('retrain-btn');
        const progressBar = document.getElementById('retrain-progress');
        if (retrainBtn && progressBar) {
            retrainBtn.onclick = async () => {
                retrainBtn.disabled = true;
                retrainBtn.innerHTML = '<i data-lucide="loader" style="width:16px;height:16px;" class="spin"></i> Retraining...';
                lucide.createIcons();
                progressBar.style.width = '0%';
                progressBar.style.transition = 'width 4s linear';

                setTimeout(() => { progressBar.style.width = '100%'; }, 100);

                try {
                    await apiAdminRetrain();
                } catch (err) {
                    console.error('Retrain failed:', err);
                }

                setTimeout(() => {
                    retrainBtn.disabled = false;
                    retrainBtn.innerHTML = '<i data-lucide="cpu" style="width:16px;height:16px;"></i> Start Retraining Cycle';
                    lucide.createIcons();
                }, 4500);
            };
        }
    }

    if (sectionId === 'users') {
        // Filter buttons
        document.querySelectorAll('.user-filter-bar .filter-btn').forEach(btn => {
            btn.onclick = async () => {
                document.querySelectorAll('.user-filter-bar .filter-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                const filter = btn.dataset.filter || '';
                const search = document.getElementById('user-search')?.value?.trim() || '';
                const container = document.getElementById('admin-section-container');
                try {
                    const usersData = await apiAdminUserActivity({ activity: filter, q: search });
                    const users = usersData.results || [];
                    const rows = users.map(u => {
                        const role = u.is_staff ? 'admin' : 'user';
                        const roleLabel = u.is_staff ? 'Admin' : 'User';
                        const statusClass = u.is_active ? 'active' : 'suspended';
                        const statusLabel = u.is_active ? 'Active' : 'Disabled';
                        const name = `${u.first_name || ''} ${u.last_name || ''}`.trim() || u.username;
                        const actClass = u.activity_status === 'active' ? 'green' : u.activity_status === 'recent' ? 'yellow' : 'red';
                        const actLabel = u.activity_status === 'active' ? 'Active' : u.activity_status === 'recent' ? 'Recent' : 'Inactive';
                        return `<tr>
                            <td>#${u.id}</td>
                            <td>${escapeHtml(name)}</td>
                            <td>${escapeHtml(u.email || '')}</td>
                            <td><span class="user-role ${role}">${roleLabel}</span></td>
                            <td><span class="user-status ${statusClass}">${statusLabel}</span></td>
                            <td><span class="activity-badge ${actClass}"><span class="dot ${actClass}"></span>${actLabel}</span></td>
                            <td>${u.session_count}</td>
                            <td>${u.message_count}</td>
                            <td>${u.last_activity ? new Date(u.last_activity).toLocaleDateString() : 'Never'}</td>
                        </tr>`;
                    }).join('');
                    const tbody = container?.querySelector('.admin-table tbody');
                    if (tbody) tbody.innerHTML = rows || '<tr><td colspan="9" style="color:var(--text-muted);text-align:center;">No users found</td></tr>';
                } catch (err) {
                    console.error('User filter failed:', err);
                }
            };
        });

        // Search input
        const searchInput = document.getElementById('user-search');
        if (searchInput) {
            let searchTimeout;
            searchInput.oninput = () => {
                clearTimeout(searchTimeout);
                searchTimeout = setTimeout(async () => {
                    const search = searchInput.value.trim();
                    const activeFilter = document.querySelector('.user-filter-bar .filter-btn.active');
                    const filter = activeFilter?.dataset.filter || '';
                    const container = document.getElementById('admin-section-container');
                    try {
                        const usersData = await apiAdminUserActivity({ q: search, activity: filter });
                        const users = usersData.results || [];
                        const rows = users.map(u => {
                            const role = u.is_staff ? 'admin' : 'user';
                            const roleLabel = u.is_staff ? 'Admin' : 'User';
                            const statusClass = u.is_active ? 'active' : 'suspended';
                            const statusLabel = u.is_active ? 'Active' : 'Disabled';
                            const name = `${u.first_name || ''} ${u.last_name || ''}`.trim() || u.username;
                            const actClass = u.activity_status === 'active' ? 'green' : u.activity_status === 'recent' ? 'yellow' : 'red';
                            const actLabel = u.activity_status === 'active' ? 'Active' : u.activity_status === 'recent' ? 'Recent' : 'Inactive';
                            return `<tr>
                                <td>#${u.id}</td>
                                <td>${escapeHtml(name)}</td>
                                <td>${escapeHtml(u.email || '')}</td>
                                <td><span class="user-role ${role}">${roleLabel}</span></td>
                                <td><span class="user-status ${statusClass}">${statusLabel}</span></td>
                                <td><span class="activity-badge ${actClass}"><span class="dot ${actClass}"></span>${actLabel}</span></td>
                                <td>${u.session_count}</td>
                                <td>${u.message_count}</td>
                                <td>${u.last_activity ? new Date(u.last_activity).toLocaleDateString() : 'Never'}</td>
                            </tr>`;
                        }).join('');
                        const tbody = container?.querySelector('.admin-table tbody');
                        if (tbody) tbody.innerHTML = rows || '<tr><td colspan="9" style="color:var(--text-muted);text-align:center;">No users found</td></tr>';
                    } catch (err) {
                        console.error('User search failed:', err);
                    }
                }, 400);
            };
        }
    }
}

// ── Magic UI: Admin Card Spotlight ─────────────────────
function initAdminMagicCards() {
    document.querySelectorAll('.admin-stat-card, .model-card').forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            card.style.setProperty('--mouse-x', `${e.clientX - rect.left}px`);
            card.style.setProperty('--mouse-y', `${e.clientY - rect.top}px`);
        });
    });
}
