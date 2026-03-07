// API Base URL — auto-detects local vs deployed environment
const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'http://localhost:8000'
    : '';

// State
let zones = [];
let selectedZoneId = null;
let map = null;
let zoneRectangles = {};

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    initMap();
    initEventListeners();
    startClock();
    await silentReset();   // wipe any persisted data from the last session
    checkHealth();
    showWelcomeMessage();
});

// Initialize Leaflet Map
function initMap() {
    map = L.map('map').setView([29.95, -90.07], 11);

    L.tileLayer('https://basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
        maxZoom: 19
    }).addTo(map);
}

// Event Listeners
function initEventListeners() {
    document.getElementById('resetAppBtn').addEventListener('click', resetApp);
    document.getElementById('chatSendBtn').addEventListener('click', sendChatMessage);
    document.getElementById('chatInput').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendChatMessage();
    });
}

// System Clock
function startClock() {
    function updateClock() {
        const now = new Date();
        const timeString = now.toLocaleTimeString('en-US', { hour12: false });
        document.getElementById('systemClock').textContent = timeString;
    }
    updateClock();
    setInterval(updateClock, 1000);
}

// Welcome message — agent greets the commander on load
function showWelcomeMessage() {
    const welcomeText = `Commander, SilentSignal is online and ready.<br><br>
I monitor cellular and social media signals across the disaster zone to detect areas of sudden silence — likely indicating infrastructure failure and trapped survivors who cannot call for help.<br><br>
<strong>What I can do for you:</strong><br>
• <em>"Start monitoring"</em> — scan signal data and map silence zones<br>
• <em>"Show all zones"</em> — rank priority areas for dispatch<br>
• <em>"Analyze drone footage for Z-001"</em> — scan for signs of life<br>
• <em>"Dispatch to Z-001"</em> — send emergency response<br><br>
Say <strong>"start monitoring"</strong> to begin.`;
    addChatMessage(welcomeText, 'ai');
}

// API Calls
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        if (!response.ok) {
            addChatMessage('Warning: SilentSignal server is not responding. Make sure the API is running.', 'ai');
        }
        // Do NOT auto-load zones — the agent drives all data loading
    } catch (error) {
        addChatMessage('Warning: Cannot reach the SilentSignal server. Make sure it is running on port 8000.', 'ai');
    }
}

async function resetApp() {
    const btn = document.getElementById('resetAppBtn');
    btn.disabled = true;
    btn.textContent = 'Resetting...';

    try {
        await fetch(`${API_BASE}/reset`, { method: 'POST' });
    } catch (e) { /* ignore */ }

    // Clear all UI state
    zones = [];
    selectedZoneId = null;
    Object.values(zoneRectangles).forEach(rect => map.removeLayer(rect));
    zoneRectangles = {};
    document.getElementById('zonesList').innerHTML = '<div class="empty-state">No zones detected. Say "start monitoring" in the chat below.</div>';
    document.getElementById('zoneDetail').innerHTML = '<div class="empty-state">Select a zone to view details</div>';
    document.getElementById('zoneCount').textContent = '0';
    document.getElementById('lastAnalysis').textContent = 'Never';
    document.getElementById('chatHistory').innerHTML = '';
    map.setView([29.95, -90.07], 11);

    btn.disabled = false;
    btn.textContent = '↺ Reset';

    // Re-show welcome message
    showWelcomeMessage();
}

// Silently reset backend data on page load — ensures a clean demo every time
async function silentReset() {
    try {
        await fetch(`${API_BASE}/reset`, { method: 'POST' });
    } catch (e) { /* server may not be up yet — that's fine */ }
}

async function loadZones() {
    try {
        const response = await fetch(`${API_BASE}/zones`);
        if (response.ok) {
            zones = await response.json();
            zones.sort((a, b) => b.drop_score - a.drop_score);
            renderZones();
            renderZonesOnMap();
            updateZoneCount();
        }
    } catch (error) {
        console.error('Load zones failed:', error);
    }
}

async function loadZoneDetail(zoneId) {
    try {
        const response = await fetch(`${API_BASE}/zones/${zoneId}`);
        if (response.ok) {
            const data = await response.json();
            renderZoneDetail(data);
        }
    } catch (error) {
        console.error('Load zone detail failed:', error);
    }
}

async function analyzeDroneMedia(zoneId) {
    const btn = event.target;
    btn.disabled = true;
    btn.textContent = 'Analyzing...';

    try {
        const response = await fetch(`${API_BASE}/zones/${zoneId}/analyze-drone`, { method: 'POST' });
        if (response.ok) {
            await loadZoneDetail(zoneId);
        }
    } catch (error) {
        console.error('Analyze drone media failed:', error);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Analyze Drone Media';
    }
}

async function sendChatMessage() {
    const input = document.getElementById('chatInput');
    const question = input.value.trim();

    if (!question) return;

    addChatMessage(question, 'user');
    input.value = '';

    // Show typing indicator while agent thinks
    const typingId = showTypingIndicator();

    try {
        const response = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        });

        removeTypingIndicator(typingId);

        if (response.ok) {
            const data = await response.json();
            addChatMessage(data.answer, 'ai', data.citations);
            // Handle any actions the agent took (refresh map, select zone, etc.)
            if (data.actions && data.actions.length > 0) {
                await handleAgentActions(data.actions);
            }
        }
    } catch (error) {
        removeTypingIndicator(typingId);
        console.error('Chat failed:', error);
        addChatMessage('Unable to reach SilentSignal AI. Check that the server is running.', 'ai');
    }
}

// Chip shortcut — fills input and sends
function chipCommand(text) {
    document.getElementById('chatInput').value = text;
    sendChatMessage();
}

// Execute frontend actions triggered by the agent
async function handleAgentActions(actions) {
    for (const action of actions) {
        if (action === 'refresh_zones') {
            await loadZones();
            updateLastAnalysisTime();
        } else if (action.startsWith('select_zone:')) {
            const zoneId = action.split(':')[1];
            await loadZones();
            selectZone(zoneId);
        } else if (action.startsWith('refresh_zone:')) {
            const zoneId = action.split(':')[1];
            await loadZoneDetail(zoneId);
        } else if (action.startsWith('dispatch:')) {
            const zoneId = action.split(':')[1];
            // Flash the zone on the map to confirm dispatch
            if (zoneRectangles[zoneId]) {
                zoneRectangles[zoneId].setStyle({ color: '#27ae60', weight: 5 });
                setTimeout(() => {
                    if (zoneRectangles[zoneId]) {
                        zoneRectangles[zoneId].setStyle({ color: '#c0392b', weight: 2 });
                    }
                }, 2000);
            }
        }
    }
}

// Typing indicator
function showTypingIndicator() {
    const chatHistory = document.getElementById('chatHistory');
    const id = 'typing-' + Date.now();
    const div = document.createElement('div');
    div.className = 'chat-message ai';
    div.id = id;
    div.innerHTML = `
        <div class="message-bubble typing-bubble">
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
        </div>
    `;
    chatHistory.appendChild(div);
    chatHistory.scrollTop = chatHistory.scrollHeight;
    return id;
}

function removeTypingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

// Rendering Functions
function renderZones() {
    const container = document.getElementById('zonesList');

    if (zones.length === 0) {
        container.innerHTML = '<div class="empty-state">No zones detected. Say "start monitoring" in the chat below.</div>';
        return;
    }

    container.innerHTML = zones.map(zone => `
        <div class="zone-card ${selectedZoneId === zone.zone_id ? 'selected' : ''}"
             onclick="selectZone('${zone.zone_id}')">
            <div class="zone-card-header">
                <span class="zone-id">${zone.zone_id}</span>
                <span class="zone-confidence">
                    Confidence: <span class="zone-confidence-value">${(zone.confidence * 100).toFixed(0)}%</span>
                </span>
            </div>
            <div class="drop-score-container">
                <span class="drop-score-label">Drop Score</span>
                <div class="drop-score-bar">
                    <div class="drop-score-fill" style="width: ${zone.drop_score * 100}%"></div>
                </div>
            </div>
            <div class="signal-dark-row">
                <span class="signal-dark-label">⏱ Signal Dark:</span>
                <span class="signal-dark-value">${getElapsedTime(zone.start_time)}</span>
            </div>
            <div class="reason-codes">
                ${zone.reason_codes.map(code => `<span class="reason-badge">${code}</span>`).join('')}
            </div>
            <button class="dispatch-btn" onclick="event.stopPropagation(); dispatchToZone('${zone.zone_id}')">
                🚁 DISPATCH
            </button>
        </div>
    `).join('');
}

function renderZonesOnMap() {
    // Clear existing rectangles
    Object.values(zoneRectangles).forEach(rect => map.removeLayer(rect));
    zoneRectangles = {};

    zones.forEach(zone => {
        const bbox = zone.geometry.bbox;
        // Ensure minimum visible size — zones with all cells in same row/col render as lines otherwise
        const minDelta = 0.012;
        const latPad = (bbox[3] - bbox[1]) < minDelta ? (minDelta - (bbox[3] - bbox[1])) / 2 : 0;
        const lonPad = (bbox[2] - bbox[0]) < minDelta ? (minDelta - (bbox[2] - bbox[0])) / 2 : 0;
        const bounds = [[bbox[1] - latPad, bbox[0] - lonPad], [bbox[3] + latPad, bbox[2] + lonPad]];

        // Determine color based on drop_score
        let color;
        if (zone.drop_score > 0.85) {
            color = '#c0392b';
        } else if (zone.drop_score > 0.7) {
            color = '#e67e22';
        } else {
            color = '#f39c12';
        }

        const rectangle = L.rectangle(bounds, {
            color: color,
            fillColor: color,
            fillOpacity: zone.drop_score * 0.6,
            weight: 2
        }).addTo(map);

        rectangle.bindPopup(`
            <strong>${zone.zone_id}</strong><br>
            Drop Score: ${(zone.drop_score * 100).toFixed(1)}%<br>
            Confidence: ${(zone.confidence * 100).toFixed(1)}%
        `);

        rectangle.on('click', () => selectZone(zone.zone_id));

        zoneRectangles[zone.zone_id] = rectangle;
    });

    // Auto-fit map to show all zones
    const rects = Object.values(zoneRectangles);
    if (rects.length > 0) {
        const group = L.featureGroup(rects);
        map.fitBounds(group.getBounds(), { padding: [50, 50] });
    }
}

function renderZoneDetail(data) {
    const container = document.getElementById('zoneDetail');
    const zone = data.zone;
    const events = data.events || [];

    const startTime = zone.start_time ? new Date(zone.start_time).toLocaleString() : 'N/A';
    const endTime = zone.end_time ? new Date(zone.end_time).toLocaleString() : 'Ongoing';

    container.innerHTML = `
        <div class="detail-section">
            <h3>Zone Information</h3>
            <div class="detail-grid">
                <div class="detail-item">
                    <div class="detail-label">Zone ID</div>
                    <div class="detail-value">${zone.zone_id}</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Drop Score</div>
                    <div class="detail-value">${(zone.drop_score * 100).toFixed(1)}%</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Confidence</div>
                    <div class="detail-value">${(zone.confidence * 100).toFixed(1)}%</div>
                </div>
                <div class="detail-item">
                    <div class="detail-label">Start Time</div>
                    <div class="detail-value">${startTime}</div>
                </div>
            </div>
        </div>

        <div class="detail-section">
            <h3>Detected Events (${events.length})</h3>
            ${events.length > 0 ? `
                <table class="events-table">
                    <thead>
                        <tr>
                            <th>Type</th>
                            <th>Likelihood</th>
                            <th>Notes</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${events.map(event => {
                            const emoji = getEventEmoji(event.event_type);
                            const likelihoodClass = getLikelihoodClass(event.likelihood);
                            return `
                                <tr>
                                    <td>
                                        <div class="event-type">
                                            <span>${emoji}</span>
                                            <span>${event.event_type}</span>
                                        </div>
                                    </td>
                                    <td class="${likelihoodClass}">
                                        ${(event.likelihood * 100).toFixed(0)}%
                                    </td>
                                    <td>${event.notes || '-'}</td>
                                </tr>
                            `;
                        }).join('')}
                    </tbody>
                </table>
            ` : '<div class="empty-state">No events detected</div>'}

            <button class="btn btn-primary analyze-drone-btn" onclick="analyzeDroneMedia('${zone.zone_id}')">
                📷 Analyze Drone Media
            </button>
        </div>
    `;
}

function addChatMessage(text, sender, citations = []) {
    const chatHistory = document.getElementById('chatHistory');

    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${sender}`;

    let citationsHtml = '';
    if (citations && citations.length > 0) {
        citationsHtml = `
            <div class="citations">
                <strong>Sources:</strong>
                ${citations.map((cite, i) => `
                    <div class="citation-item">[${i + 1}] ${cite}</div>
                `).join('')}
            </div>
        `;
    }

    messageDiv.innerHTML = `
        <div class="message-bubble">
            ${text}
            ${citationsHtml}
        </div>
    `;

    chatHistory.appendChild(messageDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

// Helper Functions
function selectZone(zoneId) {
    selectedZoneId = zoneId;
    renderZones();
    loadZoneDetail(zoneId);

    // Highlight on map
    Object.entries(zoneRectangles).forEach(([id, rect]) => {
        if (id === zoneId) {
            rect.setStyle({ weight: 4 });
        } else {
            rect.setStyle({ weight: 2 });
        }
    });
}

function dispatchToZone(zoneId) {
    addChatMessage(`Dispatch order confirmed for ${zoneId}. Emergency response unit is en route.`, 'ai');
}

function updateZoneCount() {
    document.getElementById('zoneCount').textContent = zones.length;
}

function updateLastAnalysisTime() {
    const now = new Date();
    const timeString = now.toLocaleTimeString('en-US', { hour12: false });
    document.getElementById('lastAnalysis').textContent = timeString;
}

function getElapsedTime(startTimeStr) {
    if (!startTimeStr) return 'Unknown';
    const diffMs = Date.now() - new Date(startTimeStr).getTime();
    const diffMins = Math.floor(diffMs / 60000);
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m`;
    const h = Math.floor(diffMins / 60);
    const m = diffMins % 60;
    return m > 0 ? `${h}h ${m}m` : `${h}h`;
}

function getEventEmoji(eventType) {
    const emojiMap = {
        'SHOUT': '🔊',
        'TAP_PATTERN': '🔔',
        'HUMAN_FORM': '👤',
        'UNKNOWN': '❓'
    };
    return emojiMap[eventType] || '❓';
}

function getLikelihoodClass(likelihood) {
    if (likelihood > 0.8) return 'likelihood-high';
    if (likelihood > 0.6) return 'likelihood-medium';
    return 'likelihood-low';
}
