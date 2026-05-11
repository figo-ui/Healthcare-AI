/**
 * HealthAI - App Orchestrator
 */

import { initDashboard } from './features/dashboard.js';
import { initChat } from './features/chat.js';
import { initSymptoms } from './features/symptoms.js';
import { initFacilities } from './features/facilities.js';
import { initEmergency } from './features/emergency.js';
import { isAuthenticated, apiLogout, getStoredUser, clearTokens } from './api.js';
import { t, applyToDOM } from '../i18n/i18n.js';

const viewContainer = document.getElementById('view-container');
const pageTitle = document.getElementById('page-title');
const navItems = document.querySelectorAll('.nav-links li');

const titles = {
    dashboard: 'page_dashboard',
    chat: 'nav_chat',
    symptoms: 'nav_symptoms',
    facilities: 'nav_facilities',
};

async function loadView(viewId) {
    try {
        const response = await fetch(`src/views/${viewId}.html`);
        if (!response.ok) throw new Error(`Failed to load view: ${viewId}`);
        const html = await response.text();
        
        if (viewContainer) viewContainer.innerHTML = html;
        if (pageTitle) pageTitle.innerText = t(titles[viewId]) || viewId;

        // Toggle layout mode — chat and map manage their own scroll
        const isFullHeight = viewId === 'chat' || viewId === 'facilities';
        if (viewContainer) viewContainer.classList.toggle('full-height-view', isFullHeight);
        
        // Update Nav Active State
        navItems.forEach(item => {
            item.classList.toggle('active', item.dataset.page === viewId);
        });

        // Initialize feature-specific modules
        switch(viewId) {
            case 'dashboard': await initDashboard(loadView); break;
            case 'chat': await initChat(); break;
            case 'symptoms': initSymptoms(loadView); break;
            case 'facilities': initFacilities(); break;
        }

        // Re-initialize Icons for the newly loaded view
        lucide.createIcons();

        // Apply i18n translations to any data-i18n elements in the new view
        applyToDOM();

        // Magic UI: init magic-card spotlight tracking
        initMagicCards();

        // Close mobile sidebar after navigation
        const sidebar = document.getElementById('sidebar');
        const backdrop = document.getElementById('sidebar-backdrop');
        if (sidebar) sidebar.classList.remove('mobile-open');
        if (backdrop) backdrop.classList.remove('visible');

    } catch (error) {
        console.error(error);
        if (viewContainer) {
            viewContainer.innerHTML = `
                <div class="glass-panel" style="padding: 40px; text-align: center;">
                    <i data-lucide="alert-circle" style="color: var(--danger-color); width: 48px; height: 48px;"></i>
                    <h2 style="margin-top: 15px;">System Error</h2>
                    <p style="color: var(--text-muted);">${error.message}</p>
                    <button class="primary-btn" onclick="location.reload()" style="margin-top: 20px;">Reload Application</button>
                </div>
            `;
        }
        lucide.createIcons();
    }
}

// Populate user info in sidebar
function updateUserDisplay() {
    const user = getStoredUser();
    if (!user) return;

    // Update sidebar user info
    const userNameEl = document.querySelector('.user-info .user-name');
    const userEmailEl = document.querySelector('.user-info .user-email');

    const displayName = user.first_name
        ? `${user.first_name} ${user.last_name || ''}`.trim()
        : user.username || user.email || 'User';

    if (userNameEl) userNameEl.textContent = displayName;
    if (userEmailEl) userEmailEl.textContent = user.email || '';

    // Update avatar initials
    const avatarEl = document.querySelector('.user-avatar');
    if (avatarEl) {
        const initials = (user.first_name?.[0] || user.username?.[0] || user.email?.[0] || 'U').toUpperCase();
        avatarEl.textContent = initials;
    }
}

// ── Language Switcher — handled by inline script in index.html ───────
export function getAppLanguage() {
    return localStorage.getItem('healthai_language') || 'en';
}

// Global Nav Interaction
navItems.forEach(item => {
    item.onclick = () => loadView(item.dataset.page);
});

// Initialization
document.addEventListener('DOMContentLoaded', () => {
    // Auth guard — redirect to login if not authenticated
    if (!isAuthenticated()) {
        window.location.href = 'auth.html';
        return;
    }

    lucide.createIcons();
    updateUserDisplay();
    initEmergency();

    // ── Theme toggle ──────────────────────────────────────
    const themeToggle = document.getElementById('theme-toggle');
    const themeIcon = document.getElementById('theme-icon');
    const savedTheme = localStorage.getItem('healthai_theme') || 'dark';
    if (savedTheme === 'light') {
        document.body.classList.add('light-mode');
        if (themeIcon) themeIcon.setAttribute('data-lucide', 'moon');
    }
    if (themeToggle) {
        themeToggle.onclick = () => {
            const isLight = document.body.classList.toggle('light-mode');
            localStorage.setItem('healthai_theme', isLight ? 'light' : 'dark');
            if (themeIcon) themeIcon.setAttribute('data-lucide', isLight ? 'moon' : 'sun');
            lucide.createIcons();
        };
    }
    
    // Mobile Sidebar Toggle
    const mobileToggle = document.getElementById('mobile-toggle');
    const sidebar = document.getElementById('sidebar');
    const backdrop = document.getElementById('sidebar-backdrop');
    if (mobileToggle && sidebar) {
        mobileToggle.onclick = () => {
            sidebar.classList.toggle('mobile-open');
            if (backdrop) backdrop.classList.toggle('visible', sidebar.classList.contains('mobile-open'));
        };
    }
    if (backdrop) {
        backdrop.onclick = () => {
            sidebar.classList.remove('mobile-open');
            backdrop.classList.remove('visible');
        };
    }

    // Notification Dropdown
    const notifBtn = document.getElementById('notifications-btn');
    const notifDropdown = document.getElementById('notification-dropdown');
    if (notifBtn && notifDropdown) {
        notifBtn.onclick = (e) => {
            e.stopPropagation();
            notifDropdown.classList.toggle('show');
        };
        document.addEventListener('click', () => {
            notifDropdown.classList.remove('show');
        });
    }

    // Wire up Logout — calls backend logout API
    const logoutBtn = document.querySelector('.logout-btn');
    if (logoutBtn) {
        logoutBtn.onclick = async (e) => {
            e.preventDefault();
            if (confirm('Are you sure you want to sign out?')) {
                logoutBtn.innerHTML = '<i data-lucide="loader" style="width:16px;height:16px;" class="spin"></i>';
                lucide.createIcons();
                try {
                    await apiLogout();
                } catch {
                    clearTokens();
                }
                window.location.href = 'auth.html';
            }
        };
    }

    // Initial Render
    loadView('dashboard');
});

// ── Magic UI: Card Spotlight ──────────────────────────
function initMagicCards() {
    document.querySelectorAll('.magic-card').forEach(card => {
        card.addEventListener('mousemove', (e) => {
            const rect = card.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            card.style.setProperty('--mouse-x', `${x}px`);
            card.style.setProperty('--mouse-y', `${y}px`);
        });
    });
}
