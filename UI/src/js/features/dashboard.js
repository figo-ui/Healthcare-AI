/**
 * Feature: Dashboard — Real data only, no hardcoded demo values
 */
import { apiGetProfile, apiListSessions, apiHealthCheck, getStoredUser } from '../api.js';
import { t } from '../../i18n/i18n.js';

function getTimeAgo(date) {
    const seconds = Math.floor((Date.now() - date.getTime()) / 1000);
    if (seconds < 60) return t('just_now');
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}${t('min_ago')}`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}${t('hr_ago')}`;
    const days = Math.floor(hours / 24);
    if (days < 7) return `${days}${t('day_ago')}`;
    return date.toLocaleDateString();
}

function escapeHtml(text) {
    const d = document.createElement('div');
    d.textContent = text;
    return d.innerHTML;
}

function buildActivityChart(sessions) {
    const container = document.getElementById('activity-chart-container');
    const labelsEl = document.getElementById('activity-chart-labels');
    if (!container || !sessions.length) {
        if (container) container.innerHTML = `<p style="color:var(--text-muted);font-size:0.8rem;padding:20px 0;">${t('no_activity')}</p>`;
        return;
    }

    // Group sessions by day (last 7 days)
    const days = 7;
    const counts = new Array(days).fill(0);
    const labels = [];
    const now = new Date();

    for (let i = days - 1; i >= 0; i--) {
        const d = new Date(now);
        d.setDate(d.getDate() - i);
        labels.push(d.toLocaleDateString('en', { weekday: 'short' }));
        const dayStr = d.toDateString();
        counts[days - 1 - i] = sessions.filter(s => {
            const sd = new Date(s.updated_at || s.created_at);
            return sd.toDateString() === dayStr;
        }).length;
    }

    const max = Math.max(...counts, 1);
    const w = 400, h = 120, pad = 10;
    const stepX = (w - pad * 2) / (days - 1);

    const points = counts.map((c, i) => {
        const x = pad + i * stepX;
        const y = h - pad - ((c / max) * (h - pad * 2));
        return `${x},${y}`;
    }).join(' ');

    const areaPoints = `${pad},${h - pad} ` + points + ` ${pad + (days - 1) * stepX},${h - pad}`;

    container.innerHTML = `
        <svg viewBox="0 0 ${w} ${h}" class="line-chart" style="width:100%;height:auto;">
            <defs>
                <linearGradient id="dash-grad" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" style="stop-color:var(--accent-color);stop-opacity:0.25"/>
                    <stop offset="100%" style="stop-color:var(--accent-color);stop-opacity:0"/>
                </linearGradient>
            </defs>
            <polygon points="${areaPoints}" fill="url(#dash-grad)"/>
            <polyline points="${points}" fill="none" stroke="var(--accent-color)" stroke-width="2.5"
                stroke-linecap="round" stroke-linejoin="round" class="chart-line-anim"/>
            ${counts.map((c, i) => {
                const x = pad + i * stepX;
                const y = h - pad - ((c / max) * (h - pad * 2));
                return c > 0 ? `<circle cx="${x}" cy="${y}" r="3.5" fill="var(--accent-color)"/>` : '';
            }).join('')}
        </svg>
    `;

    if (labelsEl) {
        labelsEl.innerHTML = labels.map(l => `<span>${l}</span>`).join('');
    }
}

export const initDashboard = async (loadView) => {
    // ── 1. Greeting ──────────────────────────────────────
    const hour = new Date().getHours();
    const greetingWord = hour < 12 ? t('good_morning') : hour < 18 ? t('good_afternoon') : t('good_evening');
    const greetingIcon = hour < 12 ? 'sun' : hour < 18 ? 'cloud-sun' : 'moon';

    const greetingEl = document.getElementById('welcome-greeting');
    const subtitleEl = document.getElementById('welcome-subtitle');
    const iconEl = document.getElementById('greeting-icon');
    if (iconEl) iconEl.setAttribute('data-lucide', greetingIcon);

    // ── 2. Load profile ───────────────────────────────────
    let user = getStoredUser();
    try {
        const profile = await apiGetProfile();
        user = profile.user || user;

        const firstName = user?.first_name || user?.username || 'there';
        if (greetingEl) greetingEl.innerHTML = `<span class="sparkles-text">${greetingWord}</span>, ${escapeHtml(firstName)}`;

        // Profile summary tile
        const profileSummary = document.getElementById('profile-summary');
        if (profileSummary) {
            const langMap = {
                en: 'English', am: 'አማርኛ (Amharic)', om: 'Afaan Oromoo',
            };
            const lang = langMap[profile.preferred_language] || profile.preferred_language || 'English';
            const verified = profile.email_verified ? t('verified') : t('unverified');
            profileSummary.innerHTML = `
                <p style="font-weight:600;font-size:0.9rem;margin-bottom:4px;">${escapeHtml(user?.email || '')}</p>
                <p style="font-size:0.75rem;color:var(--text-muted);margin-bottom:2px;">${t('language_label')} ${lang}</p>
                <p style="font-size:0.75rem;color:${profile.email_verified ? 'var(--success-color)' : 'var(--warning-color)'};">${verified}</p>
            `;
        }

        // Account status tile
        const accountEl = document.getElementById('stat-account');
        const accountSub = document.getElementById('stat-account-sub');
        if (accountEl) accountEl.innerHTML = user?.is_staff ? `<span style="color:var(--purple-color)">${t('admin')}</span>` : `<span style="color:var(--success-color)">${t('active')}</span>`;
        if (accountSub) accountSub.textContent = user?.is_staff ? t('staff_account') : t('standard_user');

    } catch {
        const firstName = user?.first_name || user?.username || 'there';
        if (greetingEl) greetingEl.innerHTML = `<span class="sparkles-text">${greetingWord}</span>, ${escapeHtml(firstName)}`;
        const profileSummary = document.getElementById('profile-summary');
        if (profileSummary) profileSummary.innerHTML = `<p style="color:var(--text-muted);font-size:0.8rem;">${t('could_not_load_profile')}</p>`;
    }

    // ── 3. Load sessions ──────────────────────────────────
    let sessions = [];
    try {
        sessions = await apiListSessions();

        // Consultations count
        const consultEl = document.getElementById('stat-consultations');
        const consultSub = document.getElementById('stat-consultations-sub');
        if (consultEl) consultEl.innerHTML = `${sessions.length} <span class="unit">${t('total')}</span>`;
        if (consultSub) consultSub.textContent = sessions.length === 1 ? `1 ${t('session')}` : `${sessions.length} ${t('sessions')}`;

        // Badge
        const badge = document.getElementById('sessions-badge');
        if (badge && sessions.length > 0) {
            badge.textContent = sessions.length;
            badge.style.display = 'inline-block';
        }

        // Recent sessions list
        const list = document.getElementById('recent-sessions-list');
        if (list) {
            if (sessions.length === 0) {
                list.innerHTML = `<li style="padding:12px 0;color:var(--text-muted);font-size:0.85rem;">${t('no_sessions')}</li>`;
            } else {
                list.innerHTML = '';
                sessions.slice(0, 4).forEach(session => {
                    const date = new Date(session.updated_at || session.created_at);
                    const li = document.createElement('li');
                    li.className = 'prio-low';
                    li.style.cursor = 'pointer';
                    li.innerHTML = `
                        <div class="prio-indicator"></div>
                        <div class="prio-text">
                            <h4>${escapeHtml(session.title || t('health_consultation'))}</h4>
                            <p>${getTimeAgo(date)}</p>
                        </div>
                    `;
                    list.appendChild(li);
                });
            }
        }

        // Subtitle
        if (subtitleEl) {
            subtitleEl.textContent = sessions.length > 0
                ? `${t('sessions')}: ${sessions.length}. ${getTimeAgo(new Date(sessions[0]?.updated_at || sessions[0]?.created_at))}.`
                : t('welcome_subtitle_empty');
        }

        // Activity chart
        buildActivityChart(sessions);

        // Last risk level — try to get from most recent session metadata
        const riskEl = document.getElementById('stat-risk');
        const riskSub = document.getElementById('stat-risk-sub');
        if (riskEl) {
            // We don't have risk in session list — show session count trend instead
            const today = sessions.filter(s => {
                const d = new Date(s.updated_at || s.created_at);
                return d.toDateString() === new Date().toDateString();
            }).length;
            riskEl.innerHTML = `${today} <span class="unit">${t('today')}</span>`;
            if (riskSub) riskSub.textContent = t('sessions_today');
        }

    } catch {
        const consultEl = document.getElementById('stat-consultations');
        if (consultEl) consultEl.innerHTML = '—';
        if (subtitleEl) subtitleEl.textContent = t('could_not_load_sessions');
    }

    // ── 4. System health check ────────────────────────────
    try {
        await apiHealthCheck();
        const statusEl = document.getElementById('system-status-text');
        const statusSub = document.getElementById('system-status-sub');
        if (statusEl) statusEl.innerHTML = `<span style="color:var(--success-color)">${t('online')}</span>`;
        if (statusSub) statusSub.textContent = t('all_systems_operational');
    } catch {
        const statusEl = document.getElementById('system-status-text');
        const statusSub = document.getElementById('system-status-sub');
        if (statusEl) statusEl.innerHTML = `<span style="color:var(--danger-color)">${t('offline')}</span>`;
        if (statusSub) statusSub.textContent = t('cannot_reach_server');
    }

    // ── 5. Quick action buttons ───────────────────────────
    if (typeof loadView === 'function') {
        document.querySelectorAll('.quick-action-btn[data-page]').forEach(btn => {
            btn.onclick = () => loadView(btn.dataset.page);
        });

        // New consultation button
        const newBtn = document.getElementById('new-consultation-btn');
        if (newBtn) {
            newBtn.onclick = () => loadView('symptoms');
        }
    }

    // Chart period switching
    document.querySelectorAll('.chart-period span[data-period]').forEach(btn => {
        btn.onclick = () => {
            document.querySelectorAll('.chart-period span').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            // Filter sessions by period
            const period = btn.dataset.period;
            let filtered = sessions;
            if (period === 'week') {
                const cutoff = Date.now() - 7 * 24 * 60 * 60 * 1000;
                filtered = sessions.filter(s => new Date(s.updated_at || s.created_at).getTime() > cutoff);
            } else if (period === 'month') {
                const cutoff = Date.now() - 30 * 24 * 60 * 60 * 1000;
                filtered = sessions.filter(s => new Date(s.updated_at || s.created_at).getTime() > cutoff);
            }
            buildActivityChart(filtered);
        };
    });

    lucide.createIcons();
};
