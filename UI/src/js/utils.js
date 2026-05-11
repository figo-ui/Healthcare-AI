/**
 * UI Utilities and Helper Functions
 */

export const formatDate = (date) => {
    return new Intl.DateTimeFormat('en-US', { month: 'long', day: 'numeric', year: 'numeric' }).format(date);
};

export const generateId = () => {
    return Math.random().toString(36).substr(2, 9);
};
