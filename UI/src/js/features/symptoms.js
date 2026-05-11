/**
 * Feature: Symptom Analysis - Real Backend Integration
 */
import { apiCreateSession, apiAnalyzeSymptoms } from '../api.js';
import { simplifyMedicalText, isPuterAvailable } from '../utils/puter_simplify.js';
import { t } from '../../i18n/i18n.js';

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

export const initSymptoms = (loadView) => {
    let symptomText = '';
    let symptomTags = [];
    let duration = '';
    let severity = 5;
    let preExisting = 'None';
    let imageFile = null;

    // ── Image attach / camera ──────────────────────────────
    const fileInput = document.getElementById('symptom-file-input');
    const cameraInput = document.getElementById('symptom-camera-input');
    const uploadBtn = document.getElementById('symptom-upload-btn');
    const cameraBtn = document.getElementById('symptom-camera-btn');
    const previewDiv = document.getElementById('symptom-image-preview');
    const previewImg = document.getElementById('symptom-preview-img');
    const removeImgBtn = document.getElementById('symptom-remove-img');

    function showImagePreview(file) {
        if (!file || !file.type.startsWith('image/')) return;
        imageFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            if (previewImg) previewImg.src = e.target.result;
            if (previewDiv) previewDiv.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }

    function clearImagePreview() {
        imageFile = null;
        if (previewImg) previewImg.src = '';
        if (previewDiv) previewDiv.style.display = 'none';
        if (fileInput) fileInput.value = '';
        if (cameraInput) cameraInput.value = '';
    }

    if (uploadBtn) uploadBtn.onclick = () => fileInput?.click();
    if (cameraBtn) cameraBtn.onclick = () => cameraInput?.click();
    if (fileInput) fileInput.onchange = (e) => { if (e.target.files[0]) showImagePreview(e.target.files[0]); };
    if (cameraInput) cameraInput.onchange = (e) => { if (e.target.files[0]) showImagePreview(e.target.files[0]); };
    if (removeImgBtn) removeImgBtn.onclick = () => clearImagePreview();

    const nextBtn = document.getElementById('symptom-next-btn');
    if (nextBtn) {
        nextBtn.onclick = () => {
            const textarea = document.getElementById('symptom-text');
            symptomText = textarea?.value?.trim() || '';
            if (!symptomText) return;
            // Collect active chips as tags
            symptomTags = [];
            document.querySelectorAll('.chip[data-symptom].active').forEach(chip => {
                symptomTags.push(chip.dataset.symptom);
            });
            showStep(2);
        };
    }

    // Symptom chips
    const chips = document.querySelectorAll('.chip[data-symptom]');
    const textarea = document.getElementById('symptom-text');
    chips.forEach(chip => {
        chip.onclick = () => {
            chip.classList.toggle('active');
            const symptom = chip.dataset.symptom;
            if (textarea) {
                const current = textarea.value.trim();
                if (chip.classList.contains('active')) {
                    textarea.value = current ? `${current}, ${symptom.toLowerCase()}` : `I am experiencing ${symptom.toLowerCase()}`;
                } else {
                    textarea.value = current.replace(new RegExp(`,?\\s*${symptom.toLowerCase()}`, 'i'), '').trim();
                }
            }
        };
    });

    const updateSteps = (step) => {
        const steps = document.querySelectorAll('.wizard-steps .step');
        const lines = document.querySelectorAll('.wizard-steps .line');
        
        steps.forEach((s, idx) => {
            s.classList.remove('active', 'completed');
            if (idx + 1 === step) s.classList.add('active');
            else if (idx + 1 < step) s.classList.add('completed');
        });

        lines.forEach((l, idx) => {
            l.classList.toggle('active', idx + 1 < step);
        });
    };

    const showStep = (step) => {
        const content = document.getElementById('wizard-content');
        if (!content) return;
        updateSteps(step);

        if (step === 2) {
            content.innerHTML = `
                <div class="phase-2 fade-in">
                    <h2>${t('analysis_parameters')}</h2>
                    <p style="color: var(--text-secondary); margin-bottom: var(--spacing-lg);">${t('refine_analysis_msg')}</p>
                    <div class="form-group">
                        <label>${t('duration_label')}</label>
                        <select id="symptom-duration" class="glass-input">
                            <option value="Less than 24 hours">Less than 24 hours</option>
                            <option value="1-3 days">1-3 days</option>
                            <option value="4-7 days">4-7 days</option>
                            <option value="More than a week">More than a week</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>${t('severity_label')}</label>
                        <input type="range" id="symptom-severity" min="1" max="10" value="5" class="glass-range">
                        <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: var(--text-muted);">
                            <span>${t('mild')}</span><span>${t('moderate')}</span><span>${t('severe')}</span>
                        </div>
                    </div>
                    <div class="form-group">
                        <label>${t('preexisting_label')}</label>
                        <select id="symptom-preexisting" class="glass-input">
                            <option value="None">None</option>
                            <option value="Hypertension">Hypertension</option>
                            <option value="Diabetes">Diabetes</option>
                            <option value="Asthma">Asthma</option>
                            <option value="Heart condition">Heart condition</option>
                        </select>
                    </div>
                    <div class="wizard-actions">
                        <button class="secondary-btn" id="symptom-back-btn">${t('back')}</button>
                        <button class="primary-btn" id="symptom-gen-btn"><i data-lucide="cpu" style="width:16px;height:16px;"></i> ${t('generate_report')}</button>
                    </div>
                </div>
            `;
            lucide.createIcons();

            document.getElementById('symptom-back-btn')?.addEventListener('click', () => showStep(1));
            const genBtn = document.getElementById('symptom-gen-btn');
            if (genBtn) {
                genBtn.onclick = () => {
                    duration = document.getElementById('symptom-duration')?.value || '';
                    severity = parseInt(document.getElementById('symptom-severity')?.value || '5');
                    preExisting = document.getElementById('symptom-preexisting')?.value || 'None';
                    runAnalysis();
                };
            }
        }
    };

    const showLoading = () => {
        const content = document.getElementById('wizard-content');
        if (!content) return;
        updateSteps(3);
        content.innerHTML = `
            <div class="phase-3 fade-in" style="text-align:center;padding:40px 0;">
                <div style="width:64px;height:64px;border-radius:50%;background:rgba(99,102,241,0.15);display:flex;align-items:center;justify-content:center;margin:0 auto var(--spacing-md);">
                    <i data-lucide="loader" style="color: var(--accent-color); width: 36px; height: 36px;" class="spin"></i>
                </div>
                <h2>${t('analyzing_symptoms')}</h2>
                <p style="color: var(--text-secondary);">${t('processing_msg')}</p>
            </div>
        `;
        lucide.createIcons();
    };

    const showResults = (analysis) => {
        const content = document.getElementById('wizard-content');
        if (!content) return;
        const riskLevel = analysis.risk_level || 'Low';
        const riskScore = analysis.risk_score;
        const conditions = analysis.probable_conditions || [];
        const redFlags = analysis.red_flags || [];
        const prevention = analysis.prevention_advice || [];
        const recommendation = analysis.recommendation_text || '';
        const rColor = riskColor(riskLevel);
        const rPercent = riskPercent(riskScore);

        let conditionsHtml = '';
        if (conditions.length) {
            conditionsHtml = `<div style="margin-top:var(--spacing-sm);"><strong>${t('probable_conditions')}:</strong><ul style="margin:4px 0 0 16px;">`;
            for (const c of conditions.slice(0, 5)) {
                const prob = c.probability != null ? ` (${(parseFloat(c.probability) * 100).toFixed(1)}%)` : '';
                conditionsHtml += `<li>${escapeHtml(c.condition || t('unknown'))}${prob}</li>`;
            }
            conditionsHtml += '</ul></div>';
        }

        let redFlagsHtml = '';
        if (redFlags.length) {
            redFlagsHtml = `<div style="margin-top:var(--spacing-md);padding:var(--spacing-md);background:rgba(244,63,94,0.08);border:1px solid rgba(244,63,94,0.2);border-radius:var(--radius-sm);">
                <strong style="color:var(--danger-color);">${t('red_flags')}:</strong><ul style="margin:4px 0 0 16px;">`;
            for (const rf of redFlags) redFlagsHtml += `<li style="color:var(--danger-color);">${escapeHtml(rf)}</li>`;
            redFlagsHtml += '</ul></div>';
        }

        let preventionHtml = '';
        if (prevention.length) {
            preventionHtml = `<div class="recommendations"><h4>${t('prevention_advice')}</h4><ul>`;
            for (const p of prevention) preventionHtml += `<li>${escapeHtml(p)}</li>`;
            preventionHtml += '</ul></div>';
        }

        const iconColor = riskLevel.toLowerCase() === 'high' ? 'var(--danger-color)' : riskLevel.toLowerCase() === 'moderate' ? 'var(--warning-color)' : 'var(--success-color)';
        const iconBg = riskLevel.toLowerCase() === 'high' ? 'rgba(244,63,94,0.15)' : riskLevel.toLowerCase() === 'moderate' ? 'rgba(245,158,11,0.15)' : 'rgba(16,185,129,0.15)';

        content.innerHTML = `
            <div class="phase-3 fade-in">
                <div class="report-header">
                    <div style="width:64px;height:64px;border-radius:50%;background:${iconBg};display:flex;align-items:center;justify-content:center;margin:0 auto var(--spacing-md);">
                        <i data-lucide="check-circle" style="color: ${iconColor}; width: 36px; height: 36px;"></i>
                    </div>
                    <h2>${t('analysis_complete')}</h2>
                    <p style="color: var(--text-secondary);">${t('results_msg')}</p>
                </div>
                <div class="risk-meter"><div class="meter-fill" style="width:${rPercent}%;background:${rColor};"></div></div>
                <p class="risk-label" style="color:${rColor};">${riskLevel} ${t('risk')}${riskScore != null ? ` (${t('score')} ${parseFloat(riskScore).toFixed(2)})` : ''}</p>
                ${conditionsHtml}
                ${recommendation ? `<div style="margin-top:var(--spacing-sm);"><strong>${t('recommendation')}:</strong> ${escapeHtml(recommendation)}</div>` : ''}
                ${redFlagsHtml}
                ${preventionHtml}
                <div style="margin-top: var(--spacing-md); padding: var(--spacing-md); background: rgba(244,63,94,0.05); border: 1px solid rgba(244,63,94,0.2); border-radius: var(--radius-sm);">
                    <p style="font-size: 0.8rem; color: var(--danger-color); font-weight: 600;">⚠ ${t('emergency_warning')}</p>
                </div>
                <div class="wizard-actions">
                    <button class="secondary-btn" onclick="location.reload()">${t('new_analysis')}</button>
                    ${isPuterAvailable() ? '<button class="secondary-btn" id="simplify-results-btn" style="background:rgba(99,102,241,0.12);border-color:rgba(99,102,241,0.3);color:var(--accent-color);"><i data-lucide="sparkles" style="width:16px;height:16px;"></i> Simplify Results</button>' : ''}
                    <button class="primary-btn" id="symptom-find-btn"><i data-lucide="map-pin" style="width:16px;height:16px;"></i> ${t('find_nearby_clinic')}</button>
                </div>
            </div>
        `;
        lucide.createIcons();
        const findBtn = document.getElementById('symptom-find-btn');
        if (findBtn && typeof loadView === 'function') {
            findBtn.onclick = () => loadView('facilities');
        }
        // Bind Simplify button — rewrites conditions, recommendation, prevention in plain English
        const simplifyBtn = document.getElementById('simplify-results-btn');
        if (simplifyBtn) {
            simplifyBtn.onclick = async () => {
                simplifyBtn.textContent = t('simplifying');
                simplifyBtn.disabled = true;
                try {
                    // Collect all the text content to simplify
                    const textsToSimplify = [];
                    if (conditions.length) textsToSimplify.push('Conditions: ' + conditions.slice(0,5).map(c => `${c.condition} (${(parseFloat(c.probability)*100).toFixed(1)}%)`).join(', '));
                    if (recommendation) textsToSimplify.push('Recommendation: ' + recommendation);
                    if (redFlags.length) textsToSimplify.push('Red flags: ' + redFlags.join(', '));
                    if (prevention.length) textsToSimplify.push('Prevention: ' + prevention.join(', '));
                    const combined = textsToSimplify.join('\n');
                    const simplified = await simplifyMedicalText(combined);
                    if (simplified !== combined) {
                        // Replace the results area with simplified version
                        const resultAreas = content.querySelectorAll('.phase-3 > *:not(.report-header):not(.risk-meter):not(.risk-label):not(.wizard-actions)');
                        // Add simplified explanation box
                        const simpleBox = document.createElement('div');
                        simpleBox.style.cssText = 'margin-top:var(--spacing-md);padding:var(--spacing-md);background:rgba(99,102,241,0.06);border:1px solid rgba(99,102,241,0.15);border-radius:var(--radius-sm);';
                        simpleBox.innerHTML = `<h4 style="color:var(--accent-color);margin-bottom:8px;"><i data-lucide="sparkles" style="width:14px;height:14px;"></i> In Simple Words</h4><p style="font-size:0.9rem;line-height:1.6;white-space:pre-line;">${escapeHtml(simplified)}</p>`;
                        content.querySelector('.phase-3').insertBefore(simpleBox, content.querySelector('.wizard-actions'));
                        lucide.createIcons();
                        simplifyBtn.textContent = t('simplified');
                        simplifyBtn.style.color = 'var(--success-color)';
                    } else {
                        simplifyBtn.textContent = t('simplify');
                        simplifyBtn.disabled = false;
                    }
                } catch (err) {
                    simplifyBtn.textContent = t('simplify');
                    simplifyBtn.disabled = false;
                }
            };
        }
    };

    const showError = (message) => {
        const content = document.getElementById('wizard-content');
        if (!content) return;
        content.innerHTML = `
            <div class="phase-3 fade-in" style="text-align:center;padding:40px 0;">
                <div style="width:64px;height:64px;border-radius:50%;background:rgba(244,63,94,0.15);display:flex;align-items:center;justify-content:center;margin:0 auto var(--spacing-md);">
                    <i data-lucide="alert-circle" style="color: var(--danger-color); width: 36px; height: 36px;"></i>
                </div>
                <h2>${t('analysis_failed')}</h2>
                <p style="color: var(--text-secondary);">${escapeHtml(message)}</p>
                <div class="wizard-actions" style="margin-top:var(--spacing-lg);">
                    <button class="primary-btn" onclick="location.reload()">${t('try_again')}</button>
                </div>
            </div>
        `;
        lucide.createIcons();
    };

    const runAnalysis = async () => {
        showLoading();
        try {
            // Send symptom text clean — duration/severity go as metadata
            const session = await apiCreateSession(symptomText.slice(0, 64));
            const result = await apiAnalyzeSymptoms(session.id, {
                symptom_text: symptomText,
                symptom_tags: symptomTags,
                consent_given: true,
                image: imageFile || undefined,
                model_profile: import.meta.env?.VITE_DEFAULT_MODEL_PROFILE || 'Clinical Balanced',
                metadata: {
                    duration: duration || undefined,
                    severity: severity,
                    pre_existing: preExisting !== 'None' ? preExisting : undefined,
                },
            });

            const analysis = result.analysis || {};
            showResults(analysis);
        } catch (err) {
            showError(err.message || t('analysis_failed'));
        }
    };
};
