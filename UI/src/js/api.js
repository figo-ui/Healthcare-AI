/**
 * HealthAI - Centralized API Client
 * Handles authentication tokens, request interceptors, and all backend endpoints.
 */

const API_BASE_URL = import.meta.env?.VITE_API_URL || 'http://127.0.0.1:8000/api/v1';

// ── Token Storage ──────────────────────────────────────
const TOKEN_KEYS = {
    access: 'healthai_access_token',
    refresh: 'healthai_refresh_token',
    user: 'healthai_user',
    profile: 'healthai_profile',
};

export function getAccessToken() {
    return localStorage.getItem(TOKEN_KEYS.access);
}

export function getRefreshToken() {
    return localStorage.getItem(TOKEN_KEYS.refresh);
}

export function setTokens(access, refresh) {
    localStorage.setItem(TOKEN_KEYS.access, access);
    localStorage.setItem(TOKEN_KEYS.refresh, refresh);
}

export function clearTokens() {
    localStorage.removeItem(TOKEN_KEYS.access);
    localStorage.removeItem(TOKEN_KEYS.refresh);
    localStorage.removeItem(TOKEN_KEYS.user);
    localStorage.removeItem(TOKEN_KEYS.profile);
}

export function getStoredUser() {
    try {
        return JSON.parse(localStorage.getItem(TOKEN_KEYS.user) || 'null');
    } catch { return null; }
}

export function setStoredUser(user) {
    localStorage.setItem(TOKEN_KEYS.user, JSON.stringify(user));
}

export function getStoredProfile() {
    try {
        return JSON.parse(localStorage.getItem(TOKEN_KEYS.profile) || 'null');
    } catch { return null; }
}

export function setStoredProfile(profile) {
    localStorage.setItem(TOKEN_KEYS.profile, JSON.stringify(profile));
}

export function isAuthenticated() {
    return !!getAccessToken();
}

// ── Core Request Helper ────────────────────────────────
async function apiRequest(path, options = {}) {
    const url = `${API_BASE_URL}${path}`;
    const headers = { ...options.headers };

    // Attach auth header if we have a token (skip for login/register)
    const token = getAccessToken();
    if (token && !options.noAuth) {
        headers['Authorization'] = `Bearer ${token}`;
    }

    // Don't set Content-Type for FormData — browser sets boundary automatically
    const isFormData = options.body instanceof FormData;
    if (!isFormData && options.body && typeof options.body === 'object') {
        headers['Content-Type'] = 'application/json';
        options.body = JSON.stringify(options.body);
    }

    const response = await fetch(url, {
        ...options,
        headers,
        credentials: 'include',
    });

    // Auto-refresh on 401
    if (response.status === 401 && !options._isRetry && getRefreshToken()) {
        const refreshed = await refreshAccessToken();
        if (refreshed) {
            return apiRequest(path, { ...options, _isRetry: true });
        }
        // Refresh failed — force re-login
        clearTokens();
        window.location.href = 'auth.html';
        throw new Error('Session expired. Please sign in again.');
    }

    // Parse response
    let data;
    const contentType = response.headers.get('content-type') || '';
    if (contentType.includes('application/json')) {
        data = await response.json();
    } else {
        data = await response.text();
    }

    if (!response.ok) {
        const errMsg = (data && data.error) || (data && data.detail) || response.statusText || 'Request failed';
        const err = new Error(errMsg);
        err.status = response.status;
        err.data = data;
        throw err;
    }

    return data;
}

// ── Token Refresh ──────────────────────────────────────
async function refreshAccessToken() {
    const refresh = getRefreshToken();
    if (!refresh) return false;

    try {
        const res = await fetch(`${API_BASE_URL}auth/refresh/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'include',
            body: JSON.stringify({ refresh }),
        });
        if (!res.ok) return false;
        const data = await res.json();
        if (data.tokens?.access) {
            setTokens(data.tokens.access, data.tokens.refresh || refresh);
            return true;
        }
        return false;
    } catch {
        return false;
    }
}

// ── Auth Endpoints ─────────────────────────────────────
export async function apiLogin(identifier, password) {
    const data = await apiRequest('/auth/login/', {
        method: 'POST',
        noAuth: true,
        body: { identifier, password },
    });
    if (data.tokens) {
        setTokens(data.tokens.access, data.tokens.refresh);
    }
    if (data.user) setStoredUser(data.user);
    if (data.profile) setStoredProfile(data.profile);
    return data;
}

export async function apiRegister({ email, password, first_name, last_name, phone_number }) {
    const data = await apiRequest('/auth/register/', {
        method: 'POST',
        noAuth: true,
        body: { email, password, first_name, last_name, phone_number },
    });
    if (data.tokens) {
        setTokens(data.tokens.access, data.tokens.refresh);
    }
    if (data.user) setStoredUser(data.user);
    if (data.profile) setStoredProfile(data.profile);
    return data;
}

export async function apiSocialLogin(provider, access_token, id_token) {
    const data = await apiRequest('/auth/social/login/', {
        method: 'POST',
        noAuth: true,
        body: { provider, access_token, id_token },
    });
    if (data.tokens) {
        setTokens(data.tokens.access, data.tokens.refresh);
    }
    if (data.user) setStoredUser(data.user);
    if (data.profile) setStoredProfile(data.profile);
    return data;
}

export async function apiSocialProviders() {
    return apiRequest('/auth/social/providers/', { noAuth: true });
}

export async function apiLogout() {
    const refresh = getRefreshToken();
    try {
        await apiRequest('/auth/logout/', {
            method: 'POST',
            body: { refresh },
        });
    } finally {
        clearTokens();
    }
}

// ── Profile ────────────────────────────────────────────
export async function apiGetProfile() {
    const data = await apiRequest('/profile/');
    setStoredProfile(data);
    if (data.user) setStoredUser(data.user);
    return data;
}

export async function apiUpdateProfile(updates) {
    return apiRequest('/profile/', {
        method: 'PATCH',
        body: updates,
    });
}

// ── Chat Sessions ──────────────────────────────────────
export async function apiListSessions(params = {}) {
    const query = new URLSearchParams(params).toString();
    return apiRequest(`/chat/sessions/${query ? '?' + query : ''}`);
}

export async function apiCreateSession(title = 'Health Consultation') {
    return apiRequest('/chat/sessions/', {
        method: 'POST',
        body: { title },
    });
}

export async function apiGetSessionMessages(sessionId) {
    return apiRequest(`/chat/sessions/${sessionId}/messages/`);
}

export async function apiAnalyzeSymptoms(sessionId, { symptom_text, symptom_tags, consent_given, image, location_lat, location_lng, facility_type, specialization, search_radius_km, model_profile, language_override }) {
    const formData = new FormData();
    formData.append('symptom_text', symptom_text);
    formData.append('consent_given', consent_given ? 'true' : 'false');

    // Attach language — use explicit override, then saved preference, then default 'en'
    const lang = language_override
        || (typeof localStorage !== 'undefined' && localStorage.getItem('healthai_language'))
        || 'en';
    if (lang && lang !== 'en') formData.append('language_override', lang);

    if (symptom_tags?.length) {
        formData.append('symptom_tags', JSON.stringify(symptom_tags));
    }
    if (image) formData.append('image', image);
    if (location_lat != null) formData.append('location_lat', String(location_lat));
    if (location_lng != null) formData.append('location_lng', String(location_lng));
    if (facility_type) formData.append('facility_type', facility_type);
    if (specialization) formData.append('specialization', specialization);
    if (search_radius_km) formData.append('search_radius_km', String(search_radius_km));
    if (model_profile) formData.append('model_profile', model_profile);

    return apiRequest(`/chat/sessions/${sessionId}/analyze/`, {
        method: 'POST',
        body: formData,
    });
}

export async function apiGetChatHistory(params = {}) {
    const query = new URLSearchParams(params).toString();
    return apiRequest(`/chat/history/${query ? '?' + query : ''}`);
}

// ── Quick Prompts ──────────────────────────────────────
export async function apiGetQuickPrompts() {
    return apiRequest('/quick-prompts/', { noAuth: true });
}

// ── Facilities / Location ──────────────────────────────
export async function apiNearbyFacilities({ location_lat, location_lng, facility_type, specialization, radius_km }) {
    const params = { location_lat, location_lng };
    if (facility_type) params.facility_type = facility_type;
    if (specialization) params.specialization = specialization;
    if (radius_km) params.radius_km = radius_km;
    const query = new URLSearchParams(params).toString();
    return apiRequest(`/location/nearby/?${query}`, { noAuth: true });
}

export async function apiGetDirections({ origin_lat, origin_lng, destination_lat, destination_lng, place_id }) {
    const params = {};
    if (place_id) params.place_id = place_id;
    if (origin_lat != null) params.origin_lat = origin_lat;
    if (origin_lng != null) params.origin_lng = origin_lng;
    if (destination_lat != null) params.destination_lat = destination_lat;
    if (destination_lng != null) params.destination_lng = destination_lng;
    const query = new URLSearchParams(params).toString();
    return apiRequest(`/location/directions/?${query}`, { noAuth: true });
}

export async function apiEmergencyContacts(countryCode = '') {
    const query = countryCode ? `?country_code=${countryCode}` : '';
    return apiRequest(`/location/emergency/${query}`, { noAuth: true });
}

// ── Health Check ───────────────────────────────────────
export async function apiHealthCheck() {
    return apiRequest('/health/', { noAuth: true });
}

// ── Admin Endpoints ────────────────────────────────────
export async function apiAdminAnalytics() {
    return apiRequest('/admin/analytics/');
}

export async function apiAdminUsers(params = {}) {
    const query = new URLSearchParams(params).toString();
    return apiRequest(`/admin/users/${query ? '?' + query : ''}`);
}

export async function apiAdminUserDetail(userId) {
    return apiRequest(`/admin/users/${userId}/`);
}

export async function apiAdminUpdateUser(userId, updates) {
    return apiRequest(`/admin/users/${userId}/`, {
        method: 'PATCH',
        body: updates,
    });
}

export async function apiAdminFacilities(params = {}) {
    const query = new URLSearchParams(params).toString();
    return apiRequest(`/admin/facilities/${query ? '?' + query : ''}`);
}

export async function apiAdminModelMetrics() {
    return apiRequest('/admin/model-metrics/');
}

export async function apiAdminConfig() {
    return apiRequest('/admin/config/');
}

export async function apiAdminAuditLog(params = {}) {
    const query = new URLSearchParams(params).toString();
    return apiRequest(`/admin/audit-log/${query ? '?' + query : ''}`);
}

export async function apiAdminRetrain() {
    return apiRequest('/admin/retrain/', { method: 'POST' });
}

export async function apiAdminUserActivity(params = {}) {
    const query = new URLSearchParams(params).toString();
    return apiRequest(`/admin/user-activity/${query ? '?' + query : ''}`);
}

export async function apiAdminDailyActivity(params = {}) {
    const query = new URLSearchParams(params).toString();
    return apiRequest(`/admin/daily-activity/${query ? '?' + query : ''}`);
}

export async function apiAdminTopQuestions(params = {}) {
    const query = new URLSearchParams(params).toString();
    return apiRequest(`/admin/top-questions/${query ? '?' + query : ''}`);
}

export async function apiDetectLanguage(text, preferred = '') {
    return apiRequest('/detect-language/', {
        method: 'POST',
        body: { text, preferred },
    });
}

export async function apiSupportedLanguages() {
    return apiRequest('/supported-languages/');
}

export async function apiAdminDialogueTemplates() {
    return apiRequest('/admin/dialogue-templates/');
}

export async function apiExportProfile(format = 'json') {
    return apiRequest(`/export/profile/?format=${format}`);
}

export async function apiExportChatHistory(format = 'json') {
    return apiRequest(`/chat/export/?format=${format}`);
}

export async function apiPasswordResetRequest(email) {
    return apiRequest('/auth/password-reset/', {
        method: 'POST',
        noAuth: true,
        body: { email },
    });
}

export async function apiPasswordResetConfirm(token, password) {
    return apiRequest('/auth/password-reset/confirm/', {
        method: 'POST',
        noAuth: true,
        body: { token, password },
    });
}

export async function apiVerifyEmail(token) {
    return apiRequest('/auth/verify-email/', {
        method: 'POST',
        noAuth: true,
        body: { token },
    });
}

export async function apiResendVerification() {
    return apiRequest('/auth/resend-verification/', { method: 'POST' });
}

// ── SSE Helper for Async Analysis ──────────────────────
export function listenAnalysisSSE(caseId, statusToken, onResult, onError) {
    const url = `${API_BASE_URL}/analyze/${caseId}/stream/?token=${statusToken}`;
    const eventSource = new EventSource(url);

    eventSource.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.status === 'completed') {
                eventSource.close();
                onResult(data);
            } else if (data.status === 'failed' || data.status === 'timeout' || data.status === 'unknown') {
                eventSource.close();
                onError(data);
            }
            // intermediate status updates (queued, running) — just keep listening
        } catch (e) {
            eventSource.close();
            onError({ status: 'error', message: 'Invalid SSE data' });
        }
    };

    eventSource.onerror = () => {
        eventSource.close();
        onError({ status: 'error', message: 'SSE connection failed' });
    };

    return eventSource;
}
