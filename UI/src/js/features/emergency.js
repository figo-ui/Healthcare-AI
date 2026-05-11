/**
 * Feature: Emergency Assistance - Real Backend Integration
 */
import { apiEmergencyContacts } from '../api.js';
import { t } from '../../i18n/i18n.js';

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ── Emergency Modal ────────────────────────────────────
function buildEmergencyModal(contacts) {
    const existing = document.getElementById('emergency-modal');
    if (existing) existing.remove();

    const defaultContacts = [
        { name: 'Emergency Services', phone_number: '911', description: 'Police, Fire, Ambulance' },
        { name: 'Poison Control', phone_number: '1-800-222-1222', description: 'Poison emergencies' },
        { name: 'Crisis Hotline', phone_number: '988', description: 'Mental health crisis' },
    ];

    const list = (contacts && contacts.length) ? contacts : defaultContacts;

    const items = list.map(c => `
        <div class="emergency-contact-item">
            <div class="ec-info">
                <strong>${escapeHtml(c.name || c.service || t('nav_emergency'))}</strong>
                <span>${escapeHtml(c.description || c.type || '')}</span>
            </div>
            <a href="tel:${escapeHtml(c.phone_number || c.phone || c.number || '')}"
               class="ec-call-btn"
               style="display:flex;align-items:center;gap:6px;padding:8px 16px;background:var(--danger-color);color:#fff;border-radius:8px;text-decoration:none;font-weight:600;font-size:0.875rem;">
                <i data-lucide="phone-call" style="width:14px;height:14px;"></i>
                ${escapeHtml(c.phone_number || c.phone || c.number || 'Call')}
            </a>
        </div>
    `).join('');

    const modal = document.createElement('div');
    modal.id = 'emergency-modal';
    modal.style.cssText = `
        position:fixed;inset:0;z-index:9999;display:flex;align-items:center;justify-content:center;
        background:rgba(0,0,0,0.7);backdrop-filter:blur(4px);
    `;
    modal.innerHTML = `
        <div class="glass-panel" style="max-width:480px;width:90%;padding:28px;border-radius:16px;position:relative;border:1px solid rgba(244,63,94,0.3);">
            <button id="emergency-modal-close" style="position:absolute;top:12px;right:12px;background:none;border:none;cursor:pointer;color:var(--text-muted);">
                <i data-lucide="x" style="width:20px;height:20px;"></i>
            </button>
            <div style="display:flex;align-items:center;gap:12px;margin-bottom:20px;">
                <div style="width:44px;height:44px;border-radius:50%;background:rgba(244,63,94,0.15);display:flex;align-items:center;justify-content:center;">
                    <i data-lucide="alert-triangle" style="width:22px;height:22px;color:var(--danger-color);"></i>
                </div>
                <div>
                    <h3 style="margin:0;color:var(--danger-color);">Emergency Contacts</h3>
                    <p style="margin:0;font-size:0.8rem;color:var(--text-muted);">Tap a number to call immediately</p>
                </div>
            </div>
            <div style="display:flex;flex-direction:column;gap:12px;">
                ${items}
            </div>
            <p style="margin-top:16px;font-size:0.75rem;color:var(--text-muted);text-align:center;">
                If this is a life-threatening emergency, call 911 immediately.
            </p>
        </div>
    `;

    document.body.appendChild(modal);
    lucide.createIcons();

    const closeBtn = document.getElementById('emergency-modal-close');
    if (closeBtn) closeBtn.onclick = () => modal.remove();
    modal.addEventListener('click', (e) => { if (e.target === modal) modal.remove(); });
}

export const initEmergency = async () => {
    // Load emergency contacts from backend (cached)
    let contacts = [];
    try {
        const data = await apiEmergencyContacts();
        contacts = (data && (data.contacts || data.emergency_contacts)) || [];
    } catch (err) {
        console.warn('Could not load emergency contacts:', err);
    }

    // Wire up sidebar emergency button
    const emergencyBtn = document.getElementById('emergency-btn');
    if (emergencyBtn) {
        emergencyBtn.onclick = () => buildEmergencyModal(contacts);
    }

    // Wire up any inline emergency call button (e.g. in a view)
    const callBtn = document.getElementById('emergency-call-btn');
    if (callBtn) {
        callBtn.onclick = () => buildEmergencyModal(contacts);
    }

    // SOS pulse animation
    const sosBtn = document.getElementById('sos-btn');
    if (sosBtn) {
        sosBtn.onclick = () => {
            sosBtn.classList.add('pulsing');
            buildEmergencyModal(contacts);
            setTimeout(() => sosBtn.classList.remove('pulsing'), 5000);
        };
    }

    // Populate any static emergency contact list in a view
    const alertList = document.querySelector('.alert-cards') || document.querySelector('.emergency-contacts');
    if (alertList && contacts.length) {
        alertList.innerHTML = '';
        contacts.forEach(contact => {
            const card = document.createElement('div');
            card.className = 'alert-card';
            card.innerHTML = `
                <div class="alert-icon"><i data-lucide="phone"></i></div>
                <div class="alert-details">
                    <h4>${escapeHtml(contact.name || contact.service || t('nav_emergency'))}</h4>
                    <p>${escapeHtml(contact.phone_number || contact.phone || contact.number || '')}</p>
                </div>
                <a href="tel:${escapeHtml(contact.phone_number || contact.phone || contact.number || '')}" class="call-link">
                    <i data-lucide="phone-call"></i>
                </a>
            `;
            alertList.appendChild(card);
        });
        lucide.createIcons();
    }
};
