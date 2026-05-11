/**
 * Puter.js AI Simplification — rephrases medical text into plain, grade-4-friendly English.
 *
 * Uses free Puter.js AI models (no API key needed) to simplify clinical language
 * so non-native English speakers and low-literacy users can understand their results.
 */

const SIMPLIFY_PROMPT = `You are a health communication expert. Rewrite the following medical text in very simple English that a 10-year-old could understand.

Rules:
- Use short sentences (max 12 words each)
- Replace ALL medical jargon with everyday words:
  - "myocardial infarction" → "heart attack"
  - "hypertension" → "high blood pressure"
  - "dyspnea" → "trouble breathing"
  - "UTI" → "bladder infection"
  - "cerebrovascular accident" → "stroke"
  - "edema" → "swelling"
  - "syncope" → "fainting"
  - "diagnosis" → "what the doctor thinks is wrong"
  - "prognosis" → "how you will likely get better"
  - "contraindication" → "reason not to use something"
- Keep all important warnings and red flags
- Do NOT remove the disclaimer about not being a real doctor
- Do NOT add new medical claims
- Use "you" instead of "the patient"
- If a risk level is mentioned, keep it clear (Low = not dangerous, Medium = see a doctor soon, High = get help now)

Medical text to simplify:`;


/**
 * Simplify medical text using Puter.js AI.
 * Falls back to the original text if Puter is unavailable.
 *
 * @param {string} text — Medical/clinical text to simplify
 * @param {object} [options] — Options
 * @param {string} [options.model] — Puter model name (default: 'gemini-3.1-flash-lite-preview')
 * @param {number} [options.timeout] — Max ms to wait (default: 8000)
 * @returns {Promise<string>} — Simplified plain-language text, or original on failure
 */
export async function simplifyMedicalText(text, options = {}) {
    const model = options.model || 'gemini-3.1-flash-lite-preview';
    const timeout = options.timeout || 8000;

    if (!text || !text.trim()) return text;

    // Check if Puter is available
    if (typeof puter === 'undefined' || !puter.ai || !puter.ai.chat) {
        console.warn('[puter_simplify] Puter.js not available, returning original text');
        return text;
    }

    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeout);

        const response = await puter.ai.chat(
            `${SIMPLIFY_PROMPT}\n\n${text}`,
            { model }
        );

        clearTimeout(timeoutId);

        // Puter returns the response as a string or object with message.content
        let simplified = '';
        if (typeof response === 'string') {
            simplified = response.trim();
        } else if (response?.message?.content) {
            simplified = response.message.content.trim();
        } else if (response?.toString) {
            simplified = response.toString().trim();
        }

        // Sanity check: if the simplified text is way shorter or empty, return original
        if (!simplified || simplified.length < text.length * 0.2) {
            console.warn('[puter_simplify] Simplified text too short, returning original');
            return text;
        }

        return simplified;
    } catch (err) {
        console.warn('[puter_simplify] Failed to simplify:', err.message || err);
        return text; // Always fall back to original
    }
}


/**
 * Check if Puter.js is available.
 * @returns {boolean}
 */
export function isPuterAvailable() {
    return typeof puter !== 'undefined' && puter.ai && typeof puter.ai.chat === 'function';
}
