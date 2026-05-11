/**
 * Feature: AI Chatbot — Real backend, no hardcoded messages
 */

import {
    apiListSessions,
    apiCreateSession,
    apiGetSessionMessages,
    apiAnalyzeSymptoms,
    apiGetQuickPrompts,
    apiDetectLanguage,
    getStoredUser,
    listenAnalysisSSE,
} from '../api.js';
import { simplifyMedicalText, isPuterAvailable } from '../utils/puter_simplify.js';
import { t } from '../../i18n/i18n.js';

// Read the active language set by the language switcher
function getActiveLanguage() {
    return localStorage.getItem('healthai_language') || 'en';
}

// Auto-detect language from user text using the backend API
let _detectedLangCache = null;
async function detectLanguageFromText(text) {
    if (!text || text.trim().length < 5) return null;
    try {
        const result = await apiDetectLanguage(text);
        if (result?.detected_language) {
            _detectedLangCache = result.detected_language;
            return result.detected_language;
        }
    } catch (err) {
        console.debug('Language detection failed:', err);
    }
    return null;
}

// Resolve the effective language: if 'auto', detect from text; otherwise use saved
async function resolveLanguage(text) {
    const saved = getActiveLanguage();
    if (saved === 'auto') {
        const detected = await detectLanguageFromText(text);
        return detected || 'en';
    }
    return saved;
}

let currentSessionId = null;
let isSending = false;
let attachedImage = null;

// ── Helpers ────────────────────────────────────────────
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function riskColor(level) {
    const l = (level || '').toLowerCase();
    if (l === 'high') return 'var(--danger-color)';
    if (l === 'moderate' || l === 'medium') return 'var(--warning-color)';
    return 'var(--success-color)';
}

function riskPercent(score) {
    const s = parseFloat(score);
    if (isNaN(s)) return 30;
    return Math.min(100, Math.max(5, s * 100));
}

function buildAnalysisCard(analysis) {
    if (!analysis) return '';
    const conditions = analysis.probable_conditions || [];
    const riskLevel = analysis.risk_level || 'Low';
    const riskScore = analysis.risk_score;
    const redFlags = analysis.red_flags || [];
    const prevention = analysis.prevention_advice || [];
    const recommendation = analysis.recommendation_text || '';

    let html = '<div class="medical-card" style="margin-top:8px;">';
    html += `<h5><i data-lucide="activity" style="width:14px;height:14px;"></i> ${t('analysis_results')}</h5>`;
    html += `<div class="risk-meter" style="margin:8px 0;"><div class="meter-fill" style="width:${riskPercent(riskScore)}%;background:${riskColor(riskLevel)};"></div></div>`;
    html += `<p class="risk-label" style="color:${riskColor(riskLevel)};font-weight:600;">${t('risk')}: ${riskLevel}${riskScore != null ? ` (${t('score')} ${parseFloat(riskScore).toFixed(2)})` : ''}</p>`;

    if (conditions.length) {
        html += `<div style="margin-top:8px;"><strong>${t('probable_conditions')}:</strong><ul style="margin:4px 0 0 16px;">`;
        for (const c of conditions.slice(0, 5)) {
            const prob = c.probability != null ? ` (${(parseFloat(c.probability) * 100).toFixed(1)}%)` : '';
            html += `<li>${escapeHtml(c.condition || t('unknown'))}${prob}</li>`;
        }
        html += '</ul></div>';
    }
    if (redFlags.length) {
        html += '<div style="margin-top:8px;padding:8px;background:rgba(244,63,94,0.08);border:1px solid rgba(244,63,94,0.2);border-radius:6px;">';
        html += `<strong style="color:var(--danger-color);">${t('red_flags')}:</strong><ul style="margin:4px 0 0 16px;">`;
        for (const rf of redFlags) html += `<li style="color:var(--danger-color);">${escapeHtml(rf)}</li>`;
        html += '</ul></div>';
    }
    if (recommendation) {
        html += `<div style="margin-top:8px;"><strong>${t('recommendation')}:</strong> ${escapeHtml(recommendation)}</div>`;
    }
    if (prevention.length) {
        html += `<div style="margin-top:8px;"><strong>${t('prevention_advice')}:</strong><ul style="margin:4px 0 0 16px;">`;
        for (const p of prevention) html += `<li>${escapeHtml(p)}</li>`;
        html += '</ul></div>';
    }
    html += '</div>';
    return html;
}

// ── Show/hide empty state ──────────────────────────────
function setEmptyState(visible) {
    const emptyState = document.getElementById('chat-empty-state');
    if (!emptyState) return;
    emptyState.style.display = visible ? 'flex' : 'none';
}

// ── Add a message bubble ───────────────────────────────
function addMessage(text, type, options = {}) {
    const messagesContainer = document.getElementById('chat-messages');
    if (!messagesContainer) return null;

    // Hide empty state once we have messages
    setEmptyState(false);

    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${type} fade-in`;

    const time = options.time
        ? new Date(options.time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        : new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    let html = `<div class="msg-bubble"><p class="msg-text">${escapeHtml(text)}</p>`;
    if (options.analysisCard) html += options.analysisCard;
    // Add Simplify button for AI messages if Puter is available
    if (type === 'ai' && isPuterAvailable()) {
        html += `<button class="simplify-btn" title="Rewrite in simple English" style="background:none;border:1px solid var(--border-color);border-radius:6px;padding:2px 8px;font-size:0.7rem;color:var(--text-muted);cursor:pointer;margin-top:4px;display:inline-flex;align-items:center;gap:3px;"><i data-lucide="sparkles" style="width:11px;height:11px;"></i> ${t('simplify')}</button>`;
    }
    html += `<span class="msg-time">${time}</span></div>`;

    if (options.quickReplies && options.quickReplies.length) {
        html += '<div class="quick-replies">';
        for (const q of options.quickReplies) {
            html += `<button class="quick-reply-btn" data-query="${escapeHtml(q)}">${escapeHtml(q)}</button>`;
        }
        html += '</div>';
    }

    msgDiv.innerHTML = html;
    messagesContainer.appendChild(msgDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    lucide.createIcons();
    bindQuickReplies(msgDiv);
    // Bind Simplify button for AI messages
    if (type === 'ai') {
        const simplifyBtn = msgDiv.querySelector('.simplify-btn');
        if (simplifyBtn) {
            simplifyBtn.onclick = async () => {
                const textEl = msgDiv.querySelector('.msg-text');
                if (!textEl || textEl.dataset.simplified) return;
                simplifyBtn.textContent = t('simplifying');
                simplifyBtn.disabled = true;
                const original = textEl.textContent;
                const simplified = await simplifyMedicalText(original);
                if (simplified !== original) {
                    textEl.textContent = simplified;
                    textEl.dataset.simplified = 'true';
                    simplifyBtn.textContent = t('simplified');
                    simplifyBtn.style.color = 'var(--success-color)';
                } else {
                    simplifyBtn.textContent = t('simplify');
                    simplifyBtn.disabled = false;
                }
            };
        }
    }
    return msgDiv;
}

function bindQuickReplies(container) {
    container.querySelectorAll('.quick-reply-btn').forEach(btn => {
        btn.onclick = () => {
            const query = btn.dataset.query;
            // Remove all quick reply rows in this message
            container.querySelectorAll('.quick-replies').forEach(r => r.remove());
            addMessage(query, 'user');
            sendAnalysisMessage(query);
        };
    });
}

// ── Load sidebar history ───────────────────────────────
async function loadChatHistory() {
    const historyContainer = document.getElementById('chat-history-list');
    if (!historyContainer) return;

    try {
        const sessions = await apiListSessions();
        historyContainer.innerHTML = '';

        if (!sessions || !sessions.length) {
            historyContainer.innerHTML = `<p style="padding:16px 12px;color:var(--text-muted);font-size:0.82rem;">${t('no_conversations')}</p>`;
            return;
        }

        sessions.forEach(session => {
            const item = document.createElement('div');
            item.className = `history-item${session.id === currentSessionId ? ' active' : ''}`;
            item.dataset.sessionId = session.id;
            item.innerHTML = `
                <i data-lucide="message-circle"></i>
                <div class="item-info">
                    <span>${escapeHtml(session.title || t('health_consultation'))}</span>
                    <p>${new Date(session.updated_at || session.created_at).toLocaleDateString()}</p>
                </div>
            `;
            item.onclick = () => switchToSession(session.id);
            historyContainer.appendChild(item);
        });
        lucide.createIcons();
    } catch (err) {
        console.error('Failed to load chat history:', err);
    }
}

// ── Switch to an existing session ─────────────────────
async function switchToSession(sessionId) {
    currentSessionId = sessionId;
    const messagesContainer = document.getElementById('chat-messages');
    if (!messagesContainer) return;

    document.querySelectorAll('.history-item').forEach(h => {
        h.classList.toggle('active', parseInt(h.dataset.sessionId) === sessionId);
    });

    messagesContainer.innerHTML = `<div class="date-divider"><span>${t('conversation')}</span></div>`;
    setEmptyState(false);

    try {
        const data = await apiGetSessionMessages(sessionId);
        const messages = (data && data.messages) ? data.messages : [];

        if (!messages.length) {
            setEmptyState(true);
            return;
        }

        messages.forEach(msg => {
            const role = msg.role === 'user' ? 'user' : 'ai';
            const opts = { time: msg.created_at };
            if (role === 'ai' && msg.metadata?.result) {
                opts.analysisCard = buildAnalysisCard(msg.metadata.result);
            }
            addMessage(msg.content, role, opts);
        });
    } catch (err) {
        console.error('Failed to load session messages:', err);
        messagesContainer.innerHTML = `<div class="date-divider"><span>${t('error_loading_messages')}</span></div>`;
    }
}

// ── Start a new empty session ──────────────────────────
function startNewChat() {
    currentSessionId = null;
    const messagesContainer = document.getElementById('chat-messages');
    if (messagesContainer) {
        messagesContainer.innerHTML = '';
        setEmptyState(true);
    }
    document.querySelectorAll('.history-item').forEach(h => h.classList.remove('active'));
}

// ── Send message ───────────────────────────────────────
async function sendAnalysisMessage(userText) {
    if (isSending) return;
    isSending = true;

    const typingIndicator = document.getElementById('typing-indicator');
    const messagesContainer = document.getElementById('chat-messages');

    if (typingIndicator) typingIndicator.style.display = 'flex';
    if (messagesContainer) messagesContainer.scrollTop = messagesContainer.scrollHeight;

    try {
        if (!currentSessionId) {
            const session = await apiCreateSession(userText.slice(0, 64));
            currentSessionId = session?.id ?? null;
            if (currentSessionId) await loadChatHistory();
        }

        // Resolve language: auto-detect if set to 'auto', otherwise use saved preference
        const effectiveLang = await resolveLanguage(userText);

        const result = await apiAnalyzeSymptoms(currentSessionId, {
            symptom_text: userText,
            symptom_tags: [],
            consent_given: true,
            image: attachedImage || undefined,
            model_profile: import.meta.env?.VITE_DEFAULT_MODEL_PROFILE || 'Clinical Balanced',
            language_override: effectiveLang,
        });

        attachedImage = null;
        clearImagePreview();

        // ── Async response (HTTP 202): analysis is running in background ──
        if (result?.status === 'processing' && result?.case_id && result?.status_token) {
            // Keep typing indicator visible while analysis runs
            listenAnalysisSSE(
                result.case_id,
                result.status_token,
                (sseData) => {
                    if (typingIndicator) typingIndicator.style.display = 'none';
                    const assistantMsg = sseData?.assistant_message;
                    const analysis = sseData?.analysis || sseData?.result;
                    const opts = {};

                    if (analysis && (analysis.probable_conditions?.length || analysis.risk_level)) {
                        opts.analysisCard = buildAnalysisCard(analysis);
                    }
                    if (analysis?.risk_level === 'High' || analysis?.needs_urgent_care) {
                        opts.quickReplies = [t('find_emergency_clinic'), t('show_more_details')];
                    }

                    const responseText = assistantMsg?.content || t('analysis_complete');
                    addMessage(responseText, 'ai', opts);
                    loadChatHistory();
                    isSending = false;
                },
                (errData) => {
                    if (typingIndicator) typingIndicator.style.display = 'none';
                    addMessage(errData?.message || t('analysis_failed'), 'ai');
                    isSending = false;
                }
            );
            return; // Don't set isSending = false yet — SSE callback handles it
        }

        // ── Sync response (HTTP 200): result is ready immediately ──
        if (typingIndicator) typingIndicator.style.display = 'none';

        const assistantMsg = result?.assistant_message;
        const analysis = result?.analysis;
        const opts = {};

        if (analysis && (analysis.probable_conditions?.length || analysis.risk_level)) {
            opts.analysisCard = buildAnalysisCard(analysis);
        }
        if (analysis?.risk_level === 'High' || analysis?.needs_urgent_care) {
            opts.quickReplies = [t('find_emergency_clinic'), t('show_more_details')];
        }

        // Use the actual assistant message content from the backend
        const responseText = assistantMsg?.content || result?.message || t('analysis_complete');
        addMessage(responseText, 'ai', opts);
        await loadChatHistory();
    } catch (err) {
        if (typingIndicator) typingIndicator.style.display = 'none';
        addMessage(err?.message || t('something_wrong'), 'ai');
    } finally {
        isSending = false;
    }
}

// ── Image attachment ───────────────────────────────────
function clearImagePreview() {
    const preview = document.getElementById('image-preview-bar');
    if (preview) preview.remove();
    const imageInput = document.getElementById('image-input');
    if (imageInput) imageInput.value = '';
}

function showImagePreview(file) {
    clearImagePreview();
    const footer = document.querySelector('.chat-footer');
    if (!footer) return;
    const bar = document.createElement('div');
    bar.id = 'image-preview-bar';
    bar.style.cssText = 'display:flex;align-items:center;gap:8px;padding:6px 12px;background:rgba(14,165,233,0.08);border-radius:8px;margin-bottom:6px;font-size:0.8rem;color:var(--accent-color);';
    bar.innerHTML = `<i data-lucide="image" style="width:14px;height:14px;"></i> ${escapeHtml(file.name)} <button onclick="this.parentElement.remove()" style="margin-left:auto;background:none;border:none;color:var(--text-muted);cursor:pointer;">✕</button>`;
    footer.insertBefore(bar, footer.firstChild);
    lucide.createIcons();
}

// ── Init ───────────────────────────────────────────────
export const initChat = async () => {
    const input = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const messagesContainer = document.getElementById('chat-messages');
    const attachBtn = document.getElementById('attach-btn');
    const imageInput = document.getElementById('image-input');

    // Show empty state initially
    setEmptyState(true);

    // Send on button click or Enter
    const sendMessage = () => {
        if (!input?.value?.trim() || isSending) return;
        const userText = input.value.trim();

        // Client-side semantic validation
        if (userText.length < 10) {
            const errEl = document.getElementById('chat-input-error');
            if (errEl) { errEl.textContent = t('describe_detail'); errEl.style.display = 'block'; }
            return;
        }
        if (/^[\d\s\.\,\-]+$/.test(userText)) {
            const errEl = document.getElementById('chat-input-error');
            if (errEl) { errEl.textContent = t('describe_words'); errEl.style.display = 'block'; }
            return;
        }
        // Clear any previous error
        const errEl = document.getElementById('chat-input-error');
        if (errEl) { errEl.textContent = ''; errEl.style.display = 'none'; }

        addMessage(userText, 'user');
        input.value = '';
        input.style.height = '44px';
        sendAnalysisMessage(userText);
    };

    if (sendBtn) sendBtn.onclick = sendMessage;
    if (input) {
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
        });
        input.addEventListener('input', () => {
            input.style.height = '44px';
            input.style.height = Math.min(input.scrollHeight, 120) + 'px';
        });
    }

    // Image attach
    if (attachBtn && imageInput) {
        attachBtn.onclick = () => imageInput.click();
        imageInput.onchange = (e) => {
            const file = e.target.files?.[0];
            if (file) {
                attachedImage = file;
                showImagePreview(file);
            }
        };
    }

    // New chat buttons (sidebar + header)
    document.querySelector('.new-chat-btn')?.addEventListener('click', startNewChat);
    document.getElementById('new-chat-header-btn')?.addEventListener('click', startNewChat);

    // Export chat
    document.getElementById('export-chat-btn')?.addEventListener('click', async () => {
        if (!currentSessionId) return;
        try {
            const data = await apiGetSessionMessages(currentSessionId);
            const messages = (data && data.messages) ? data.messages : [];
            const text = messages.map(m => `[${m.role.toUpperCase()}] ${m.content}`).join('\n\n');
            const blob = new Blob([text], { type: 'text/plain' });
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = `healthai-chat-${currentSessionId}.txt`;
            a.click();
        } catch { /* ignore */ }
    });

    // Load quick prompts into empty state
    try {
        const data = await apiGetQuickPrompts();
        const prompts = (data && data.prompts) ? data.prompts : [];
        const quickRepliesEl = document.getElementById('initial-quick-replies');
        if (quickRepliesEl && prompts.length) {
            prompts.slice(0, 4).forEach(p => {
                if (typeof p === 'string' && p.trim()) {
                    const btn = document.createElement('button');
                    btn.className = 'quick-reply-btn';
                    btn.textContent = p;
                    btn.onclick = () => {
                        addMessage(p, 'user');
                        sendAnalysisMessage(p);
                    };
                    quickRepliesEl.appendChild(btn);
                }
            });
        }
    } catch { /* ignore */ }

    // Load sessions into sidebar
    await loadChatHistory();

    lucide.createIcons();
};
