/**
 * Feature: Facility Locator - Real Backend + Leaflet Map Integration
 */
import { apiNearbyFacilities, apiGetDirections } from '../api.js';
import { t } from '../../i18n/i18n.js';

let userLocation = null;
let currentFacilityType = '';
let map = null;
let markers = [];
let userMarker = null;

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function facilityTypeIcon(type) {
    const icons = {
        hospital: 'hospital',
        clinic: 'stethoscope',
        pharmacy: 'pill',
        emergency: 'alert-triangle',
        dental: 'smile',
        pediatric: 'baby',
        lab: 'flask-conical',
    };
    return icons[type] || 'building-2';
}

// ── Leaflet Map Setup ────────────────────────────────────
function initMap() {
    if (map) return;

    const mapEl = document.getElementById('facility-map');
    if (!mapEl) return;

    map = L.map('facility-map', {
        zoomControl: false,
        attributionControl: true,
    }).setView([20, 0], 2);

    // Dark-themed tile layer
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
        subdomains: 'abcd',
        maxZoom: 19,
    }).addTo(map);

    L.control.zoom({ position: 'bottomright' }).addTo(map);

    setTimeout(() => map.invalidateSize(), 200);
}

function clearMarkers() {
    markers.forEach(m => map.removeLayer(m));
    markers = [];
}

function addUserMarker(lat, lng) {
    if (userMarker) map.removeLayer(userMarker);
    userMarker = L.circleMarker([lat, lng], {
        radius: 8,
        fillColor: '#0ea5e9',
        color: '#fff',
        weight: 2,
        opacity: 1,
        fillOpacity: 0.9,
    }).addTo(map);
    userMarker.bindPopup('<strong>Your Location</strong>');
}

function addFacilityMarker(facility) {
    const lat = facility.latitude;
    const lng = facility.longitude;
    if (!lat || !lng) return null;

    const typeIconSvg = {
        hospital: '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M8 3v3a2 2 0 0 1-2 2H3m18 0h-3a2 2 0 0 1-2-2V3m0 18v-3a2 2 0 0 1 2-2h3M3 16h3a2 2 0 0 1 2 2v3"/><path d="M12 8v8"/><path d="M8 12h8"/></svg>',
        clinic: '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m8 2 4 4 4-4"/><path d="M6 12h4v8H6z"/><path d="M14 12h4v8h-4z"/><path d="M2 20h20"/></svg>',
        pharmacy: '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m10.5 20.5-5-5a2.12 2.12 0 0 1 3-3l1.5 1.5 1.5-1.5a2.12 2.12 0 0 1 3 3l-5 5z"/><path d="M12 8V2"/><path d="M8 4h8"/></svg>',
        emergency: '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m2 8 4-4 4 4"/><path d="M6 4v12"/><path d="m14 8 4-4 4 4"/><path d="M18 4v12"/><path d="M2 20h20"/></svg>',
        dental: '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a5 5 0 0 1 5 5v3H7V7a5 5 0 0 1 5-5z"/><path d="M8 10v4a2 2 0 0 0 2 2h4a2 2 0 0 0 2-2v-4"/><path d="M9 22V16"/><path d="M15 22V16"/></svg>',
        pediatric: '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="6" r="4"/><path d="M12 10v4"/><path d="M9 22h6"/><path d="M10 14H8a2 2 0 0 0-2 2v0a2 2 0 0 0 2 2h2"/><path d="M14 14h2a2 2 0 0 1 2 2v0a2 2 0 0 1-2 2h-2"/></svg>',
        lab: '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 3h6"/><path d="M10 9V3"/><path d="M14 9V3"/><path d="M5 21h14l-4-8H9z"/></svg>',
    };
    const iconSvg = typeIconSvg[facility.facility_type] || '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="2" width="16" height="20" rx="2"/><path d="M9 22v-4h6v4"/><path d="M8 6h.01"/><path d="M16 6h.01"/><path d="M12 6h.01"/><path d="M12 10h.01"/><path d="M12 14h.01"/><path d="M16 10h.01"/><path d="M16 14h.01"/><path d="M8 10h.01"/><path d="M8 14h.01"/></svg>';

    const icon = L.divIcon({
        className: 'facility-marker',
        html: `<div class="marker-pin">${iconSvg}</div>`,
        iconSize: [30, 40],
        iconAnchor: [15, 40],
        popupAnchor: [0, -40],
    });

    const name = escapeHtml(facility.provider_name || facility.name || 'Facility');
    const dist = facility.distance_km != null ? `${facility.distance_km} mi` : '';
    const phoneIcon = '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72c.127.96.361 1.903.7 2.81a2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0 1 22 16.92z"/></svg>';
    const starIcon = '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>';
    const mapPinIcon = '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/></svg>';
    const rulerIcon = '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.3 15.3a2.4 2.4 0 0 1 0 3.4l-2.6 2.6a2.4 2.4 0 0 1-3.4 0L2.7 8.7a2.41 2.41 0 0 1 0-3.4l2.6-2.6a2.41 2.41 0 0 1 3.4 0Z"/><path d="m14.5 12.5 2-2"/><path d="m11.5 9.5 2-2"/><path d="m8.5 6.5 2-2"/><path d="m17.5 15.5 2-2"/></svg>';
    const navIcon = '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="3 11 22 2 13 21 11 13 3 11"/></svg>';

    const phone = facility.phone_number ? `<br><span class="popup-icon-inline">${phoneIcon}</span> ${escapeHtml(facility.phone_number)}` : '';
    const rating = facility.rating ? `<br><span class="popup-icon-inline">${starIcon}</span> ${facility.rating}` : '';
    const addr = facility.address ? `<br><span class="popup-icon-inline">${mapPinIcon}</span> ${escapeHtml(facility.address)}` : '';
    const mapsLink = facility.maps_url
        ? `<br><a href="${escapeHtml(facility.maps_url)}" target="_blank" rel="noopener" style="color:#0ea5e9;"><span class="popup-icon-inline">${navIcon}</span> Get Directions</a>`
        : '';

    const marker = L.marker([lat, lng], { icon }).addTo(map);
    marker.bindPopup(`
        <div style="font-family:Outfit,sans-serif;min-width:180px;">
            <strong style="font-size:0.95rem;">${name}</strong>
            ${addr}${phone}${rating}${dist ? `<br><span class="popup-icon-inline">${rulerIcon}</span> ${dist}` : ''}${mapsLink}
        </div>
    `);
    markers.push(marker);
    return marker;
}

function fitMapToMarkers() {
    if (!map) return;
    const allPoints = [];
    if (userLocation) allPoints.push([userLocation.lat, userLocation.lng]);
    markers.forEach(m => allPoints.push(m.getLatLng()));
    if (allPoints.length > 0) {
        const bounds = L.latLngBounds(allPoints);
        map.fitBounds(bounds, { padding: [50, 50], maxZoom: 14 });
    }
}

// ── Facility Card Rendering ──────────────────────────────
function renderFacilityCard(facility, index) {
    const distance = facility.distance_km != null ? `${facility.distance_km} miles` : '';
    const phone = facility.phone_number ? ` • ${escapeHtml(facility.phone_number)}` : '';
    const isOpen = facility.is_emergency ? `<span class="status-open">${t('emergency_247')}</span>` : `<span class="status-open">${t('open_now')}</span>`;
    const starSvg = '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>';
    const rating = facility.rating ? `${starSvg} ${facility.rating}` : '';
    const source = facility.source === 'local_registry' ? 'Registry' : facility.source === 'geoapify' ? 'OpenStreetMap' : 'Maps';

    return `
        <div class="clinic-card${index === 0 ? ' active' : ''}" data-place-id="${escapeHtml(facility.place_id || '')}" data-lat="${facility.latitude || ''}" data-lng="${facility.longitude || ''}" data-maps-url="${escapeHtml(facility.maps_url || '')}">
            <div class="clinic-icon"><i data-lucide="${facilityTypeIcon(facility.facility_type)}"></i></div>
            <div class="clinic-details">
                <h4>${escapeHtml(facility.provider_name || facility.name || 'Facility')}</h4>
                <p>${distance}${phone} • ${source}</p>
                ${rating ? `<div class="clinic-rating">${rating}</div>` : ''}
                ${isOpen}
            </div>
        </div>
    `;
}

async function searchFacilities(location, facilityType) {
    const clinicList = document.getElementById('clinic-list');
    if (!clinicList) return;

    clinicList.innerHTML = `<p style="padding:12px;color:var(--text-muted);">${t('searching_facilities')}</p>`;

    try {
        const data = await apiNearbyFacilities({
            location_lat: location.lat,
            location_lng: location.lng,
            facility_type: facilityType || undefined,
            radius_km: 15,
        });

        const facilities = (data && data.facilities) ? data.facilities : [];

        clearMarkers();

        if (!facilities.length) {
            clinicList.innerHTML = '<p style="padding:12px;color:var(--text-muted);">No facilities found nearby. Try a different filter or increase search radius.</p>';
            return;
        }

        clinicList.innerHTML = facilities.map((f, i) => renderFacilityCard(f, i)).join('');
        lucide.createIcons();
        bindClinicCards();

        facilities.forEach(f => addFacilityMarker(f));
        fitMapToMarkers();

    } catch (err) {
        clinicList.innerHTML = `<p style="padding:12px;color:var(--danger-color);">${t('error')}: ${escapeHtml(err?.message || t('unknown_error'))}</p>`;
    }
}

function bindClinicCards() {
    const clinicCards = document.querySelectorAll('.clinic-card');
    clinicCards.forEach((card, index) => {
        card.onclick = () => {
            clinicCards.forEach(c => c.classList.remove('active'));
            card.classList.add('active');

            const lat = parseFloat(card.dataset.lat);
            const lng = parseFloat(card.dataset.lng);
            if (map && !isNaN(lat) && !isNaN(lng)) {
                map.setView([lat, lng], 15, { animate: true });
                if (markers[index]) markers[index].openPopup();
            }

            if (isNaN(lat) || isNaN(lng)) {
                const mapsUrl = card.dataset.mapsUrl;
                if (mapsUrl) window.open(mapsUrl, '_blank');
            }
        };
    });
}

// ── Geolocation ──────────────────────────────────────────
function getUserLocation() {
    return new Promise((resolve, reject) => {
        if (!navigator.geolocation) {
            reject(new Error('Geolocation not supported'));
            return;
        }
        navigator.geolocation.getCurrentPosition(
            (pos) => resolve({ lat: pos.coords.latitude, lng: pos.coords.longitude }),
            (err) => {
                const msgs = { 1: 'Location permission denied', 2: 'Location unavailable', 3: 'Location request timed out' };
                reject(new Error(msgs[err.code] || 'Location error'));
            },
            { enableHighAccuracy: true, timeout: 10000 }
        );
    });
}

// ── Geocode a city/place name → lat/lng using Geoapify ──────────────────
const GEOAPIFY_KEY = import.meta.env?.VITE_GEOAPIFY_KEY || '';

async function geocodePlace(query) {
    if (!GEOAPIFY_KEY) return null;
    const url = `https://api.geoapify.com/v1/geocode/search?text=${encodeURIComponent(query)}&limit=1&apiKey=${GEOAPIFY_KEY}`;
    try {
        const resp = await fetch(url);
        const data = await resp.json();
        const feature = data.features?.[0];
        if (!feature) return null;
        const [lng, lat] = feature.geometry.coordinates;
        const label = feature.properties.formatted || query;
        return { lat, lng, label };
    } catch {
        return null;
    }
}

// ── IP-based location fallback (no permission needed) ────────────────────
async function getLocationByIP() {
    try {
        // Use Geoapify IP geolocation
        if (!GEOAPIFY_KEY) return null;
        const resp = await fetch(`https://api.geoapify.com/v1/ipinfo?apiKey=${GEOAPIFY_KEY}`);
        const data = await resp.json();
        if (data.location?.latitude && data.location?.longitude) {
            return {
                lat: data.location.latitude,
                lng: data.location.longitude,
                label: data.city?.name || data.country?.name || 'Your location',
                isIP: true,
            };
        }
    } catch { /* ignore */ }
    return null;
}

// ── Show location status message ─────────────────────────
function setLocationStatus(message, type = 'info') {
    const clinicList = document.getElementById('clinic-list');
    if (!clinicList) return;
    const color = type === 'error' ? 'var(--danger-color)' : type === 'success' ? 'var(--success-color)' : 'var(--text-muted)';
    clinicList.innerHTML = `<p style="padding:12px;color:${color};font-size:0.875rem;">${message}</p>`;
}

// ── Init ─────────────────────────────────────────────────
export const initFacilities = () => {
    initMap();

    // ── Filter chips ──────────────────────────────────────
    const chips = document.querySelectorAll('.filter-chips .chip');
    chips.forEach(chip => {
        chip.onclick = () => {
            chips.forEach(c => c.classList.remove('active'));
            chip.classList.add('active');
            const type = chip.textContent.trim().toLowerCase();
            currentFacilityType = type === 'all' ? '' : type;
            if (userLocation) searchFacilities(userLocation, currentFacilityType);
        };
    });

    // ── Locate button — re-detect GPS ────────────────────
    const locateBtn = document.getElementById('map-locate-btn');
    if (locateBtn) {
        locateBtn.title = 'Detect my location';
        locateBtn.onclick = async () => {
            locateBtn.disabled = true;
            setLocationStatus(`<i data-lucide="loader" style="width:14px;height:14px;" class="spin"></i> ${t('detecting_location')}`, 'info');
            lucide.createIcons();
            try {
                userLocation = await getUserLocation();
                addUserMarker(userLocation.lat, userLocation.lng);
                map.setView([userLocation.lat, userLocation.lng], 14, { animate: true });
                searchFacilities(userLocation, currentFacilityType);
            } catch (gpsErr) {
                // GPS failed — try IP fallback
                try {
                    const ipLoc = await getLocationByIP();
                    if (ipLoc) {
                        userLocation = { lat: ipLoc.lat, lng: ipLoc.lng };
                        addUserMarker(ipLoc.lat, ipLoc.lng);
                        map.setView([ipLoc.lat, ipLoc.lng], 12, { animate: true });
                        searchFacilities(userLocation, currentFacilityType);
                        setLocationStatus(`📍 ${t('approx_location')}: ${ipLoc.label}`, 'info');
                    } else {
                        setLocationStatus(`⚠ ${gpsErr.message}. ${t('try_city_search')}`, 'error');
                    }
                } catch {
                    setLocationStatus(`⚠ ${gpsErr.message}. ${t('try_city_search')}`, 'error');
                }
            } finally {
                locateBtn.disabled = false;
            }
        };
    }

    // ── Search box — geocode city name if no GPS ─────────
    const searchInput = document.querySelector('.search-input-wrapper input');
    if (searchInput) {
        searchInput.placeholder = t('search_city_placeholder');
        let searchTimeout;
        searchInput.addEventListener('keydown', async (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                clearTimeout(searchTimeout);
                const query = searchInput.value.trim();
                if (!query) return;

                // If we have a location, filter by facility name
                if (userLocation) {
                    searchFacilities(userLocation, currentFacilityType);
                    return;
                }

                // No GPS — geocode the typed place name
                setLocationStatus(`<i data-lucide="loader" style="width:14px;height:14px;" class="spin"></i> Finding "${query}"...`, 'info');
                lucide.createIcons();
                const geo = await geocodePlace(query);
                if (geo) {
                    userLocation = { lat: geo.lat, lng: geo.lng };
                    addUserMarker(geo.lat, geo.lng);
                    map.setView([geo.lat, geo.lng], 13, { animate: true });
                    searchFacilities(userLocation, currentFacilityType);
                } else {
                    setLocationStatus(`Could not find "${query}". Try a different city or area name.`, 'error');
                }
            }
        });

        searchInput.addEventListener('input', () => {
            clearTimeout(searchTimeout);
            // Live filter if we already have a location
            if (userLocation) {
                searchTimeout = setTimeout(() => {
                    searchFacilities(userLocation, currentFacilityType);
                }, 600);
            }
        });
    }

    // ── Auto-detect location on load ─────────────────────
    (async () => {
        setLocationStatus('<i data-lucide="loader" style="width:14px;height:14px;" class="spin"></i> Detecting your location...', 'info');
        lucide.createIcons();

        // 1. Try GPS first
        try {
            userLocation = await getUserLocation();
            addUserMarker(userLocation.lat, userLocation.lng);
            map.setView([userLocation.lat, userLocation.lng], 14);
            searchFacilities(userLocation, currentFacilityType);
            return;
        } catch (gpsErr) {
            console.warn('GPS unavailable:', gpsErr.message);
        }

        // 2. Fallback: IP-based location (no permission needed)
        setLocationStatus('<i data-lucide="loader" style="width:14px;height:14px;" class="spin"></i> Using network location...', 'info');
        lucide.createIcons();
        try {
            const ipLoc = await getLocationByIP();
            if (ipLoc) {
                userLocation = { lat: ipLoc.lat, lng: ipLoc.lng };
                addUserMarker(ipLoc.lat, ipLoc.lng);
                map.setView([ipLoc.lat, ipLoc.lng], 12);
                searchFacilities(userLocation, currentFacilityType);
                setLocationStatus(`📍 Approximate location: ${ipLoc.label}. For precise results, allow location access or type your city.`, 'info');
                return;
            }
        } catch { /* ignore */ }

        // 3. Both failed — show search prompt
        setLocationStatus(
            '📍 Could not detect location automatically.<br>Type your city or area name in the search box above and press <strong>Enter</strong>.',
            'info'
        );
    })();
};
