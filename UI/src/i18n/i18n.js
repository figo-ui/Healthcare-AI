/**
 * i18n — lightweight translation module for Ethiopian languages (en/am/om)
 * Loads translations from JSON via fetch, provides t() function, and applies to DOM
 */

const SUPPORTED = ['en', 'am', 'om'];
const STORAGE_KEY = 'healthai_language';

let _currentLang = 'en';
let _translations = null;

/** Load translations JSON via fetch */
async function loadTranslations() {
    if (_translations) return _translations;
    try {
        const resp = await fetch('./src/i18n/translations.json');
        _translations = await resp.json();
    } catch (err) {
        console.warn('[i18n] Failed to load translations, using fallback:', err);
        _translations = { en: {} };
    }
    return _translations;
}

/** Get the saved language (resolves 'auto' to 'en') */
export function getLang() {
    const saved = localStorage.getItem(STORAGE_KEY) || 'en';
    return saved === 'auto' ? 'en' : saved;
}

/** Set the active language and apply translations to the DOM */
export async function setLang(lang) {
    if (lang === 'auto') lang = 'en';
    if (!SUPPORTED.includes(lang)) lang = 'en';
    _currentLang = lang;
    await loadTranslations();
    applyToDOM();
}

/** Translate a key — falls back to English, then raw key */
export function t(key, vars) {
    if (!_translations) return key;
    let str = _translations[_currentLang]?.[key]
        || _translations.en?.[key]
        || key;
    if (vars) {
        Object.entries(vars).forEach(([k, v]) => {
            str = str.replace(new RegExp(`\\{${k}\\}`, 'g'), v);
        });
    }
    return str;
}

/** Apply translations to all [data-i18n] and [data-i18n-placeholder] elements */
export function applyToDOM() {
    document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.getAttribute('data-i18n');
        if (key) el.textContent = t(key);
    });
    document.querySelectorAll('[data-i18n-placeholder]').forEach(el => {
        const key = el.getAttribute('data-i18n-placeholder');
        if (key) el.setAttribute('placeholder', t(key));
    });
    document.querySelectorAll('[data-i18n-title]').forEach(el => {
        const key = el.getAttribute('data-i18n-title');
        if (key) el.setAttribute('title', t(key));
    });
    document.documentElement.lang = _currentLang;
}

// Initialize on load
_currentLang = getLang();
loadTranslations().then(() => applyToDOM());
