/**
 * Auth Page Logic - Real Django JWT Authentication
 */

import { apiLogin, apiRegister, apiSocialLogin, apiSocialProviders, isAuthenticated, clearTokens } from './api.js';

// ── Security Config ────────────────────────────────────
const SECURITY = {
    MAX_ATTEMPTS: 5,
    LOCKOUT_MINUTES: 15,
};

// ── Failed Attempt Tracking ────────────────────────────
function getFailedAttempts() {
    const data = sessionStorage.getItem('healthai_failed');
    if (!data) return { count: 0, lockedUntil: 0 };
    try { return JSON.parse(data); } catch { return { count: 0, lockedUntil: 0 }; }
}

function recordFailedAttempt() {
    const data = getFailedAttempts();
    data.count++;
    if (data.count >= SECURITY.MAX_ATTEMPTS) {
        data.lockedUntil = Date.now() + SECURITY.LOCKOUT_MINUTES * 60 * 1000;
    }
    sessionStorage.setItem('healthai_failed', JSON.stringify(data));
}

function isLockedOut() {
    const data = getFailedAttempts();
    if (data.lockedUntil && Date.now() < data.lockedUntil) {
        return Math.ceil((data.lockedUntil - Date.now()) / 60000);
    }
    if (data.lockedUntil && Date.now() >= data.lockedUntil) {
        sessionStorage.removeItem('healthai_failed');
    }
    return 0;
}

function clearFailedAttempts() {
    sessionStorage.removeItem('healthai_failed');
}

// ── Toast Helper ────────────────────────────────────────
function showToast(message, type = 'success') {
    const toast = document.getElementById('auth-toast');
    const toastText = document.getElementById('auth-toast-text');
    if (!toast || !toastText) {
        console.error('Toast elements not found');
        alert(message);
        return Promise.resolve();
    }
    
    const icon = toast.querySelector('i');
    
    toastText.textContent = message;
    toast.style.display = 'flex';
    
    if (icon) {
        if (type === 'error') {
            icon.setAttribute('data-lucide', 'alert-circle');
            icon.style.color = 'var(--danger-color)';
        } else {
            icon.setAttribute('data-lucide', 'check-circle');
            icon.style.color = 'var(--success-color)';
        }
        lucide.createIcons();
    }
    
    requestAnimationFrame(() => toast.classList.add('show'));
    
    return new Promise(resolve => {
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => resolve(), 400);
        }, 2500);
    });
}

// ── Auth Toggle (Login ↔ Register) ─────────────────────
function toggleAuth(event) {
    event.preventDefault();
    
    const title = document.getElementById('auth-title');
    const subtitle = document.getElementById('auth-subtitle');
    const submitBtn = document.getElementById('submit-btn');
    const registerFields = document.getElementById('register-fields');
    const footerText = document.getElementById('auth-footer-text');
    const passwordStrength = document.getElementById('password-strength');

    const isLogin = title.innerText === 'Welcome Back';
    
    title.innerText = isLogin ? 'Create Account' : 'Welcome Back';
    subtitle.innerText = isLogin ? 'Start your health journey today' : 'Secure access to your medical assistant';
    submitBtn.innerText = isLogin ? 'Create Account' : 'Sign In';
    registerFields.style.display = isLogin ? 'block' : 'none';
    passwordStrength.style.display = isLogin ? 'flex' : 'none';
    footerText.innerHTML = isLogin 
        ? 'Already have an account? <a href="#" onclick="toggleAuth(event)">Sign in</a>'
        : 'Don\'t have an account? <a href="#" onclick="toggleAuth(event)">Create account</a>';
    
    lucide.createIcons();
}

// ── Determine redirect based on user role ──────────────
function getRedirectPath(user) {
    if (user?.is_staff) return 'admin.html';
    return 'index.html';
}

// ── Real Login via Django API ──────────────────────────
async function handleLogin() {
    const lockedMins = isLockedOut();
    if (lockedMins) {
        showToast(`Account locked. Try again in ${lockedMins} min.`, 'error');
        return;
    }

    const form = document.getElementById('auth-form');
    const emailInput = form.querySelector('input[type="email"]');
    const passwordInput = document.getElementById('auth-password');
    const identifier = emailInput?.value?.trim() || '';
    const password = passwordInput?.value || '';

    if (!identifier || !password) {
        showToast('Please enter your email and password.', 'error');
        return;
    }

    const submitBtn = document.getElementById('submit-btn');
    const originalText = submitBtn.innerText;
    submitBtn.innerText = 'Signing in...';
    submitBtn.disabled = true;

    try {
        const data = await apiLogin(identifier, password);
        clearFailedAttempts();

        const firstName = data.user?.first_name || data.user?.username || 'User';
        await showToast(`Welcome, ${firstName}! Redirecting...`);

        setTimeout(() => {
            window.location.href = getRedirectPath(data.user);
        }, 800);
    } catch (err) {
        recordFailedAttempt();
        const message = err.status === 401
            ? 'Invalid email or password.'
            : err.status === 403
            ? 'Account is disabled.'
            : err.message || 'Login failed. Please try again.';
        showToast(message, 'error');
    } finally {
        submitBtn.innerText = originalText;
        submitBtn.disabled = false;
    }
}

// ── Real Register via Django API ───────────────────────
async function handleRegister() {
    const form = document.getElementById('auth-form');
    const nameInput = form.querySelector('#register-fields input[type="text"]');
    const emailInput = form.querySelector('input[type="email"]');
    const passwordInput = document.getElementById('auth-password');

    const fullName = nameInput?.value?.trim() || '';
    const email = emailInput?.value?.trim() || '';
    const password = passwordInput?.value || '';

    if (!email || !password) {
        showToast('Please fill in all required fields.', 'error');
        return;
    }
    if (password.length < 8) {
        showToast('Password must be at least 8 characters.', 'error');
        return;
    }

    const submitBtn = document.getElementById('submit-btn');
    const originalText = submitBtn.innerText;
    submitBtn.innerText = 'Creating account...';
    submitBtn.disabled = true;

    const nameParts = fullName.split(/\s+/);
    const first_name = nameParts[0] || '';
    const last_name = nameParts.slice(1).join(' ') || '';

    try {
        const data = await apiRegister({ email, password, first_name, last_name });
        clearFailedAttempts();

        const firstName = data.user?.first_name || data.user?.username || 'User';
        await showToast(`Account created! Welcome, ${firstName}!`);

        setTimeout(() => {
            window.location.href = getRedirectPath(data.user);
        }, 800);
    } catch (err) {
        const errMsg = err.data;
        let message = err.message || 'Registration failed. Please try again.';
        if (typeof errMsg === 'object') {
            const fieldErrors = Object.values(errMsg).flat().join(' ');
            if (fieldErrors) message = fieldErrors;
        }
        showToast(message, 'error');
    } finally {
        submitBtn.innerText = originalText;
        submitBtn.disabled = false;
    }
}

// ── Form Submit Handler ────────────────────────────────
function handleFormSubmit(event) {
    event.preventDefault();
    const title = document.getElementById('auth-title');
    const isLogin = title.innerText === 'Welcome Back';
    if (isLogin) {
        handleLogin();
    } else {
        handleRegister();
    }
}

// ── Social Login ──────────────────────────────────────
// Client IDs loaded dynamically from backend
let SOCIAL_CLIENT_IDS = {};

// Load configured providers from backend on init
let configuredProviders = [];

async function loadProviders() {
    try {
        const data = await apiSocialProviders();
        configuredProviders = data.providers || [];
        // Populate client IDs from backend
        for (const p of configuredProviders) {
            SOCIAL_CLIENT_IDS[p.id] = p.client_id || '';
        }
    } catch { /* ignore — will show all as unconfigured */ }
}

function isProviderConfigured(providerId) {
    return configuredProviders.some(p => p.id === providerId);
}

async function socialLogin(provider) {
    const providerId = provider.toLowerCase();

    if (!isProviderConfigured(providerId)) {
        showToast(`${provider} login is not configured on the server. Add OAuth keys in Django admin.`, 'error');
        return;
    }

    try {
        let access_token = '';
        let id_token = '';

        switch (providerId) {
            case 'google': {
                if (!SOCIAL_CLIENT_IDS.google) {
                    showToast('Google Client ID not set. Configure SOCIAL_GOOGLE_CLIENT_ID.', 'error');
                    return;
                }
                // Use Google Identity Services (GIS) one-tap / popup
                const tokenResp = await new Promise((resolve, reject) => {
                    const tokenClient = google.accounts.oauth2.initTokenClient({
                        client_id: SOCIAL_CLIENT_IDS.google,
                        scope: 'openid email profile',
                        callback: resolve,
                        error_callback: reject,
                    });
                    tokenClient.requestAccessToken();
                });
                access_token = tokenResp.access_token || '';
                break;
            }
            case 'github': {
                // GitHub OAuth popup flow
                if (!SOCIAL_CLIENT_IDS.github) {
                    showToast('GitHub Client ID not set. Configure SOCIAL_GITHUB_CLIENT_ID.', 'error');
                    return;
                }
                const ghRedirect = encodeURIComponent(window.location.origin + '/auth.html');
                const ghUrl = `https://github.com/login/oauth/authorize?client_id=${SOCIAL_CLIENT_IDS.github}&redirect_uri=${ghRedirect}&scope=user:email`;
                access_token = await popupOAuth(ghUrl, 'github');
                break;
            }
            case 'microsoft': {
                if (!SOCIAL_CLIENT_IDS.microsoft) {
                    showToast('Microsoft Client ID not set.', 'error');
                    return;
                }
                const msRedirect = encodeURIComponent(window.location.origin + '/auth.html');
                const msUrl = `https://login.microsoftonline.com/common/oauth2/v2.0/authorize?client_id=${SOCIAL_CLIENT_IDS.microsoft}&response_type=token&redirect_uri=${msRedirect}&scope=openid email profile`;
                access_token = await popupOAuth(msUrl, 'microsoft');
                break;
            }
            case 'facebook': {
                if (!SOCIAL_CLIENT_IDS.facebook) {
                    showToast('Facebook App ID not set.', 'error');
                    return;
                }
                const fbRedirect = encodeURIComponent(window.location.origin + '/auth.html');
                const fbUrl = `https://www.facebook.com/v18.0/dialog/oauth?client_id=${SOCIAL_CLIENT_IDS.facebook}&redirect_uri=${fbRedirect}&response_type=token&scope=email,public_profile`;
                access_token = await popupOAuth(fbUrl, 'facebook');
                break;
            }
            case 'apple': {
                showToast('Apple Sign-In requires a backend redirect flow. Use email/password for now.', 'error');
                return;
            }
            default:
                showToast(`${provider} login is not yet supported.`, 'error');
                return;
        }

        if (!access_token && !id_token) {
            showToast('Authentication was cancelled or failed.', 'error');
            return;
        }

        // Send token to our backend → get JWT
        const data = await apiSocialLogin(providerId, access_token, id_token);
        const firstName = data.user?.first_name || data.user?.username || 'User';
        await showToast(`Welcome, ${firstName}! Redirecting...`);
        setTimeout(() => { window.location.href = getRedirectPath(data.user); }, 800);

    } catch (err) {
        const msg = err.message || `${provider} login failed.`;
        showToast(msg, 'error');
    }
}

// ── Popup OAuth helper ──────────────────────────────────
function popupOAuth(url, provider) {
    return new Promise((resolve, reject) => {
        const width = 500, height = 600;
        const left = (screen.width - width) / 2;
        const top = (screen.height - height) / 2;
        const popup = window.open(url, `${provider}_oauth`, `width=${width},height=${height},top=${top},left=${left}`);

        if (!popup) {
            reject(new Error('Popup blocked. Allow popups for this site.'));
            return;
        }

        const interval = setInterval(() => {
            try {
                if (popup.closed) {
                    clearInterval(interval);
                    reject(new Error('Authentication cancelled.'));
                    return;
                }
                const href = popup.location.href;
                if (href) {
                    const hash = popup.location.hash || popup.location.search;
                    const params = new URLSearchParams(hash.replace('#', '').replace('?', ''));
                    const token = params.get('access_token');
                    if (token) {
                        popup.close();
                        clearInterval(interval);
                        resolve(token);
                    }
                }
            } catch (e) {
                // Cross-origin — wait for redirect back
            }
        }, 200);

        // Timeout after 2 minutes
        setTimeout(() => {
            clearInterval(interval);
            if (!popup.closed) popup.close();
            reject(new Error('Authentication timed out.'));
        }, 120000);
    });
}

// Expose to global scope for HTML onclick handlers
window.socialLogin = socialLogin;
window.toggleAuth = toggleAuth;

// ── Init ────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    lucide.createIcons();

    // Load configured social providers from backend
    loadProviders();

    // If already authenticated, redirect away from auth page
    if (isAuthenticated()) {
        window.location.href = 'index.html';
        return;
    }

    // Check if already locked out on page load
    const lockedMins = isLockedOut();
    if (lockedMins) {
        showToast(`Account locked. Try again in ${lockedMins} min.`, 'error');
    }

    // Override form submit
    const form = document.getElementById('auth-form');
    if (form) {
        form.onsubmit = handleFormSubmit;
    }

    // Password strength checker
    const passwordInput = document.getElementById('auth-password');
    if (passwordInput) {
        passwordInput.addEventListener('input', () => {
            const val = passwordInput.value;
            const fill = document.getElementById('strength-fill');
            const text = document.getElementById('strength-text');
            const container = document.getElementById('password-strength');
            
            if (!val) { container.style.display = 'none'; return; }
            container.style.display = 'flex';
            
            let score = 0;
            if (val.length >= 8) score++;
            if (val.length >= 10) score++;
            if (/[A-Z]/.test(val)) score++;
            if (/[0-9]/.test(val)) score++;
            if (/[^A-Za-z0-9]/.test(val)) score++;
            
            fill.className = 'strength-fill';
            text.className = 'strength-text';
            
            if (score <= 1) { fill.classList.add('weak'); text.classList.add('weak'); text.textContent = 'Weak'; }
            else if (score === 2) { fill.classList.add('fair'); text.classList.add('fair'); text.textContent = 'Fair'; }
            else if (score === 3) { fill.classList.add('good'); text.classList.add('good'); text.textContent = 'Good'; }
            else { fill.classList.add('strong'); text.classList.add('strong'); text.textContent = 'Strong'; }
        });
    }

    // Toggle password visibility
    const toggleBtn = document.getElementById('toggle-password');
    if (toggleBtn && passwordInput) {
        toggleBtn.onclick = () => {
            const isPassword = passwordInput.type === 'password';
            passwordInput.type = isPassword ? 'text' : 'password';
            toggleBtn.innerHTML = isPassword 
                ? '<i data-lucide="eye-off" style="width:16px;height:16px;"></i>' 
                : '<i data-lucide="eye" style="width:16px;height:16px;"></i>';
            lucide.createIcons();
        };
    }
});
