/**
 * AccessLens — Main Application JavaScript
 * Handles audit submission, demo loading, result rendering, and UI interactions.
 */

const API_BASE = window.location.origin;

// DOM References
const els = {
    form: document.getElementById('audit-form'),
    urlInput: document.getElementById('url-input'),
    auditBtn: document.getElementById('audit-btn'),
    progressSection: document.getElementById('progress-section'),
    progressBar: document.getElementById('progress-bar'),
    resultsSection: document.getElementById('results-section'),
    hero: document.getElementById('hero'),
    gaugeScore: document.getElementById('gauge-score'),
    gaugeFill: document.getElementById('gauge-fill'),
    gradeBadge: document.getElementById('grade-badge'),
    statTotal: document.getElementById('stat-total'),
    statCritical: document.getElementById('stat-critical'),
    statWarning: document.getElementById('stat-warning'),
    statDuration: document.getElementById('stat-duration'),
    categoriesGrid: document.getElementById('categories-grid'),
    issuesList: document.getElementById('issues-list'),
    filterTabs: document.getElementById('filter-tabs'),
    issueSearch: document.getElementById('issue-search'),
    aiInsightsList: document.getElementById('ai-insights-list'),
    aiBadge: document.getElementById('ai-badge'),
    attentionMapContainer: document.getElementById('attention-map-container'),
    attentionMapImg: document.getElementById('attention-map-img'),
    limitationsList: document.getElementById('limitations-list'),
    newAuditBtn: document.getElementById('new-audit-btn'),
    exportPdfBtn: document.getElementById('export-pdf-btn'),
    navHistory: document.getElementById('nav-history'),
    historySection: document.getElementById('history-section'),
    historyTbody: document.getElementById('history-tbody'),
    historyStats: document.getElementById('history-stats'),
    toastContainer: document.getElementById('toast-container'),
    
    // OpenRouter DOM refs
    orLoading: document.getElementById('or-loading'),
    orContent: document.getElementById('or-content'),
    orError: document.getElementById('or-error'),
    orSummary: document.getElementById('or-summary'),
    orFixesList: document.getElementById('or-fixes-list'),
    orDesignList: document.getElementById('or-design-list'),
    orNarrative: document.getElementById('or-narrative'),
    
    // Screenshot refs
    annotatedScreenshot: document.getElementById('annotated-screenshot'),
    screenshotPlaceholder: document.getElementById('screenshot-placeholder')
};

let currentReport = null;
let currentAuditId = null;
let currentFilter = 'all';

// ============ Event Listeners ============

els.form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const url = els.urlInput.value.trim();
    if (!url) return;
    await runAudit(url);
});

document.querySelectorAll('.demo-btn').forEach(btn => {
    btn.addEventListener('click', () => loadDemo(btn.dataset.demo));
});

els.filterTabs.addEventListener('click', (e) => {
    if (e.target.classList.contains('filter-tab')) {
        document.querySelectorAll('.filter-tab').forEach(t => { t.classList.remove('active'); t.setAttribute('aria-selected', 'false'); });
        e.target.classList.add('active');
        e.target.setAttribute('aria-selected', 'true');
        currentFilter = e.target.dataset.filter;
        renderIssues(currentReport.issues);
    }
});

els.issueSearch.addEventListener('input', () => {
    if (currentReport) renderIssues(currentReport.issues);
});

els.newAuditBtn.addEventListener('click', resetUI);

// ============ Audit Pipeline ============

async function runAudit(url) {
    showProgress();
    els.auditBtn.classList.add('loading');

    try {
        animateSteps();
        const response = await fetch(`${API_BASE}/api/audit`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url, include_ai: true }),
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({ detail: 'Unknown error' }));
            throw new Error(err.detail || `HTTP ${response.status}`);
        }

        const report = await response.json();
        currentReport = report;
        currentAuditId = report.id || null;
        
        showResults(report);
        
        // Fetch AI Insights in background
        if (currentAuditId) {
            fetchOpenRouterInsights(currentAuditId);
        } else {
            fetchOpenRouterInsightsPayload(report);
        }
        
    } catch (err) {
        showToast(`Audit failed: ${err.message}`, 'error');
        resetUI();
    } finally {
        els.auditBtn.classList.remove('loading');
    }
}

async function loadDemo(siteId) {
    showProgress();
    try {
        animateSteps();
        // Try API first, fall back to hardcoded
        let report;
        try {
            const res = await fetch(`${API_BASE}/api/demo/${siteId}`);
            if (res.ok) { report = await res.json(); }
        } catch (_) {}

        if (!report) report = DEMO_DATA[siteId];
        if (!report) { showToast('Demo not found', 'error'); resetUI(); return; }

        currentAuditId = null;
        currentReport = normalizeReport(report);
        await sleep(1500);
        showResults(currentReport);
        
        // Simulate OpenRouter payload
        fetchOpenRouterInsightsPayload(currentReport);
    } catch (err) {
        showToast(`Demo failed: ${err.message}`, 'error');
        resetUI();
    }
}

// ============ OpenRouter AI Insights ============

function resetOpenRouterPanel() {
    els.orLoading.classList.remove('hidden');
    els.orContent.classList.add('hidden');
    els.orError.classList.add('hidden');
}

async function fetchOpenRouterInsights(auditId) {
    resetOpenRouterPanel();
    try {
        const response = await fetch(`${API_BASE}/api/ai-insights`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ audit_id: auditId })
        });
        const data = await response.json();
        renderOpenRouterInsights(data);
    } catch (err) {
        console.error("OpenRouter fetch failed:", err);
        renderOpenRouterInsights({ available: false });
    }
}

async function fetchOpenRouterInsightsPayload(report) {
    resetOpenRouterPanel();
    try {
        const payload = {
            url: report.url,
            score: report.overall_score,
            grade: report.grade,
            issues: report.issues,
            categories: report.categories
        };
        const response = await fetch(`${API_BASE}/api/ai-insights`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await response.json();
        renderOpenRouterInsights(data);
    } catch (err) {
        console.error("OpenRouter fetch failed:", err);
        renderOpenRouterInsights({ available: false });
    }
}

function renderOpenRouterInsights(data) {
    els.orLoading.classList.add('hidden');
    
    if (!data.available) {
        els.orError.classList.remove('hidden');
        if (data.reason) els.orError.textContent = `AI insights unavailable: ${data.reason}`;
        return;
    }
    
    els.orContent.classList.remove('hidden');
    els.orSummary.textContent = data.summary || '';
    
    // Top Fixes
    if (data.top_fixes && data.top_fixes.length) {
        els.orFixesList.innerHTML = data.top_fixes.map(fix => `
            <div class="or-fix-item">
                <div class="or-fix-header">
                    <span class="or-fix-title">${esc(fix.title)}</span>
                    <span class="or-fix-effort">${esc(fix.effort || 'medium')}</span>
                </div>
                <div class="or-fix-desc">${esc(fix.description)}</div>
                ${fix.wcag_criteria ? `<div class="or-fix-wcag">WCAG ${esc(fix.wcag_criteria)}</div>` : ''}
            </div>
        `).join('');
    } else {
        els.orFixesList.innerHTML = '<div>No priority fixes identified.</div>';
    }
    
    // Design Issues / UX Patterns
    let designHtml = '';
    if (data.design_issues && data.design_issues.length) {
        designHtml += data.design_issues.map(issue => `
            <div class="or-design-item">
                <strong>${esc(issue.area || 'Design Issue')}</strong>
                ${esc(issue.problem)} &mdash; ${esc(issue.fix)}
            </div>
        `).join('');
    }
    if (data.ux_patterns && data.ux_patterns.length) {
        designHtml += data.ux_patterns.map(pattern => `
            <div class="or-design-item">
                <strong>UX Pattern</strong>
                ${esc(pattern)}
            </div>
        `).join('');
    }
    els.orDesignList.innerHTML = designHtml || '<div>No design insights identified.</div>';
    
    // Narrative
    els.orNarrative.textContent = data.ai_narrative || "Keep up the great work improving accessibility!";
}

// ============ Rendering ============

function showProgress() {
    els.hero.classList.add('hidden');
    els.resultsSection.classList.add('hidden');
    els.progressSection.classList.remove('hidden');
    els.progressBar.style.width = '0%';
    document.querySelectorAll('.progress-step').forEach(s => { s.classList.remove('active', 'done'); });
}

function animateSteps() {
    const steps = ['step-fetch', 'step-parse', 'step-rules', 'step-ai'];
    steps.forEach((id, i) => {
        setTimeout(() => {
            const el = document.getElementById(id);
            if (el) el.classList.add('active');
            if (i > 0) {
                const prev = document.getElementById(steps[i-1]);
                if (prev) {
                    prev.classList.remove('active');
                    prev.classList.add('done');
                }
            }
            els.progressBar.style.width = `${((i+1)/steps.length)*100}%`;
        }, i * 600);
    });
    setTimeout(() => { 
        const last = document.getElementById(steps[steps.length-1]);
        if(last) last.classList.add('done'); 
    }, steps.length * 600);
}

function showResults(report) {
    els.progressSection.classList.add('hidden');
    els.resultsSection.classList.remove('hidden');
    
    // PDF button
    if (els.exportPdfBtn) {
        if (currentAuditId) {
            els.exportPdfBtn.style.display = 'inline-flex';
            els.exportPdfBtn.onclick = () => window.open(`${API_BASE}/api/audit/pdf/${currentAuditId}`, '_blank');
        } else {
            els.exportPdfBtn.style.display = 'none';
        }
    }
    
    // Render Annotated Screenshot
    if (report.screenshot) {
        els.annotatedScreenshot.src = `data:image/png;base64,${report.screenshot}`;
        els.annotatedScreenshot.style.display = 'block';
        els.screenshotPlaceholder.style.display = 'none';
    } else {
        els.annotatedScreenshot.style.display = 'none';
        
        let errorReason = "Unknown error";
        if (report.limitations && report.limitations.length > 0) {
            const screenLimitation = report.limitations.find(l => l.includes("Screenshot Failed"));
            if (screenLimitation) {
                errorReason = screenLimitation;
            }
        }
        
        els.screenshotPlaceholder.innerHTML = `
            <div style="color: var(--critical); font-weight: bold; margin-bottom: 0.5rem;">No screenshot available</div>
            <div style="font-size: 0.85rem; color: var(--text-tertiary);">${errorReason}</div>
        `;
        els.screenshotPlaceholder.style.display = 'block';
    }

    renderScore(report);
    renderStats(report);
    renderCategories(report.categories || []);
    renderIssues(report.issues || []);
    // Visual Model panel removed — skip renderAIInsights
    els.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderScore(report) {
    const score = Math.round(report.overall_score);
    const grade = report.grade;
    // Animate score counter
    animateCounter(els.gaugeScore, 0, score, 1500);
    // Animate gauge circle
    const circumference = 2 * Math.PI * 85; // r=85
    const offset = circumference - (score / 100) * circumference;
    setTimeout(() => { els.gaugeFill.style.strokeDashoffset = offset; }, 100);

    // Set gauge color
    const gaugeColor = score >= 80 ? 'var(--success)' : score >= 60 ? 'var(--warning)' : 'var(--critical)';
    els.gaugeFill.style.stroke = gaugeColor;

    // Grade badge
    els.gradeBadge.textContent = grade;
    els.gradeBadge.className = `grade-badge grade-${grade}`;
}

function renderStats(report) {
    els.statTotal.querySelector('.stat-value').textContent = report.total_issues || 0;
    els.statCritical.querySelector('.stat-value').textContent = report.critical_count || 0;
    els.statWarning.querySelector('.stat-value').textContent = report.warning_count || 0;
    els.statDuration.querySelector('.stat-value').textContent = `${(report.scan_duration || 0).toFixed(1)}s`;
}

function renderCategories(categories) {
    els.categoriesGrid.innerHTML = categories.map(cat => {
        const color = cat.score >= 80 ? 'var(--success)' : cat.score >= 60 ? 'var(--warning)' : 'var(--critical)';
        return `
            <div class="category-card">
                <div class="category-name">${cat.name}</div>
                <div class="category-bar"><div class="category-fill" style="width:${cat.score}%;background:${color}"></div></div>
                <div class="category-score-text">${Math.round(cat.score)}/100 • ${cat.issue_count} issue${cat.issue_count!==1?'s':''}</div>
            </div>`;
    }).join('');
}

function renderIssues(issues) {
    const search = els.issueSearch.value.toLowerCase();
    const filtered = (issues || []).filter(issue => {
        if (currentFilter !== 'all' && issue.severity !== currentFilter) return false;
        if (search && !issue.title.toLowerCase().includes(search) && !(issue.description||'').toLowerCase().includes(search)) return false;
        return true;
    });

    if (!filtered.length) {
        els.issuesList.innerHTML = '<div style="text-align:center;padding:24px;color:var(--text-muted)">No issues match your filter.</div>';
        return;
    }

    els.issuesList.innerHTML = filtered.map((issue, i) => `
        <div class="issue-card severity-${issue.severity}" role="listitem" onclick="this.classList.toggle('expanded')" id="issue-card-${i}">
            <div class="issue-card-header">
                <div class="issue-title-row">
                    <span class="severity-badge ${issue.severity}">${issue.severity}</span>
                    <span class="issue-card-title">${esc(issue.title)}</span>
                </div>
                <span class="wcag-ref">WCAG ${issue.wcag_criterion || ''}</span>
                <span class="expand-icon">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"></polyline></svg>
                </span>
            </div>
            <div class="issue-details">
                <p class="issue-desc">${esc(issue.description || '')}</p>
                ${issue.element ? `<div class="issue-element">${esc(issue.element)}</div>` : ''}
                ${issue.suggestion ? `<div class="issue-suggestion"><strong>Fix:</strong> ${esc(issue.suggestion)}</div>` : ''}
            </div>
        </div>`).join('');
}

function renderAIInsights(insights) {
    // Visual Model panel has been removed from the UI
    return;
}

function resetUI() {
    els.resultsSection.classList.add('hidden');
    els.progressSection.classList.add('hidden');
    if (els.historySection) els.historySection.classList.add('hidden');
    els.hero.classList.remove('hidden');
    els.urlInput.value = '';
    currentReport = null;
    currentAuditId = null;
    currentFilter = 'all';
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ============ Utilities ============

function animateCounter(el, from, to, duration) {
    const start = performance.now();
    const update = (now) => {
        const p = Math.min((now - start) / duration, 1);
        el.textContent = Math.round(from + (to - from) * easeOut(p));
        if (p < 1) requestAnimationFrame(update);
    };
    requestAnimationFrame(update);
}
function easeOut(t) { return 1 - Math.pow(1 - t, 3); }
function esc(s) { const d = document.createElement('div'); d.textContent = s || ''; return d.innerHTML; }
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function showToast(msg, type = 'info') {
    const t = document.createElement('div');
    t.className = `toast ${type}`;
    t.textContent = msg;
    els.toastContainer.appendChild(t);
    setTimeout(() => t.remove(), 5000);
}

function normalizeReport(raw) {
    return {
        url: raw.url || '',
        overall_score: raw.overall_score || 0,
        grade: raw.grade || 'F',
        total_issues: raw.total_issues || (raw.issues||[]).length,
        critical_count: raw.critical_count || (raw.issues||[]).filter(i=>i.severity==='critical').length,
        warning_count: raw.warning_count || (raw.issues||[]).filter(i=>i.severity==='warning').length,
        issues: raw.issues || [],
        dl_insights: raw.dl_insights || raw.ai_insights || [],
        categories: raw.categories || [
            { name: 'Perceivable', score: Math.min(100, raw.overall_score + 10), issue_count: 0 },
            { name: 'Operable', score: Math.min(100, raw.overall_score + 5), issue_count: 0 },
            { name: 'Understandable', score: Math.min(100, raw.overall_score + 15), issue_count: 0 },
            { name: 'Robust', score: Math.min(100, raw.overall_score + 8), issue_count: 0 },
            { name: 'AI Analysis', score: raw.overall_score, issue_count: (raw.ai_insights||[]).length },
        ],
        scan_duration: raw.scan_duration || 2.4,
        ai_model_used: raw.ai_model_used || false,
        limitations: raw.limitations || [],
    };
}

// ============ Demo Data ============

const DEMO_DATA = {
    "good-site": {
        url: "https://www.gov.uk",
        overall_score: 87, grade: "B", total_issues: 4, critical_count: 0, warning_count: 4, scan_duration: 3.2,
        issues: [
            { title: 'Generic Link Text: "more"', severity: "warning", wcag_criterion: "2.4.4", description: "Some links use generic text.", suggestion: "Replace with descriptive text.", score_impact: 2 },
            { title: "Missing Skip Navigation Link", severity: "warning", wcag_criterion: "2.4.1", description: "No skip-to-content link.", suggestion: "Add skip link.", score_impact: 3 },
            { title: "Table Missing Caption", severity: "warning", wcag_criterion: "1.3.1", description: "A data table lacks caption.", suggestion: "Add <caption>.", score_impact: 2 },
            { title: "Skipped Heading Level (h2 → h4)", severity: "warning", wcag_criterion: "2.4.6", description: "Heading hierarchy skip.", suggestion: "Use h3 instead.", score_impact: 2 },
        ],
        ai_insights: [{ category: "small_targets", confidence: 0.42, title: "Possible Small Touch Targets", description: "Some footer links may be below 44x44px." }],
    },
    "bad-site": {
        url: "https://old-design-example.com",
        overall_score: 34, grade: "F", total_issues: 14, critical_count: 8, warning_count: 6, scan_duration: 4.8,
        issues: [
            { title: "Missing Page Title", severity: "critical", wcag_criterion: "2.4.2", description: "No <title> element.", suggestion: "Add descriptive <title>.", score_impact: 5 },
            { title: "Missing Language Declaration", severity: "critical", wcag_criterion: "3.1.1", description: "No lang on <html>.", suggestion: "Add lang='en'.", score_impact: 5 },
            { title: "Insufficient Color Contrast (2.1:1)", severity: "critical", wcag_criterion: "1.4.3", description: "Light gray on white = 2.1:1.", element: '<p style="color:#999">...', suggestion: "Darken to #767676 for 4.5:1.", score_impact: 4 },
            { title: "Image Missing Alt Text (hero.jpg)", severity: "critical", wcag_criterion: "1.1.1", description: "Hero image has no alt.", suggestion: "Add alt='...'.", score_impact: 5 },
            { title: "Image Missing Alt Text (logo.png)", severity: "critical", wcag_criterion: "1.1.1", description: "Logo missing alt text.", suggestion: "Add alt='Company Logo'.", score_impact: 5 },
            { title: "Image Missing Alt Text (banner.jpg)", severity: "critical", wcag_criterion: "1.1.1", description: "Banner lacks alt text.", suggestion: "Add descriptive alt.", score_impact: 5 },
            { title: "Form Input Missing Label (email)", severity: "critical", wcag_criterion: "3.3.2", description: "Email input has no label.", suggestion: "Add <label>.", score_impact: 4 },
            { title: "Form Input Missing Label (phone)", severity: "critical", wcag_criterion: "3.3.2", description: "Phone input unlabeled.", suggestion: "Add <label>.", score_impact: 4 },
            { title: "No Headings Found", severity: "warning", wcag_criterion: "2.4.6", description: "No heading elements.", suggestion: "Add h1-h6.", score_impact: 3 },
            { title: "Zoom Disabled", severity: "warning", wcag_criterion: "1.4.4", description: "user-scalable=no.", suggestion: "Remove restriction.", score_impact: 5 },
            { title: "Empty Link", severity: "warning", wcag_criterion: "2.4.4", description: "Link has no text.", suggestion: "Add text.", score_impact: 4 },
            { title: "Focus Outline Removed", severity: "warning", wcag_criterion: "2.1.1", description: "outline:none on :focus.", suggestion: "Add visible focus style.", score_impact: 5 },
            { title: "Duplicate ID: 'content'", severity: "warning", wcag_criterion: "4.1.1", description: "ID used twice.", suggestion: "Make unique.", score_impact: 2 },
            { title: "Iframe Missing Title", severity: "warning", wcag_criterion: "4.1.2", description: "No title on iframe.", suggestion: "Add title.", score_impact: 2 },
        ],
        ai_insights: [
            { category: "low_contrast", confidence: 0.92, title: "Low Contrast", description: "Multiple text areas have poor contrast." },
            { category: "missing_alt", confidence: 0.88, title: "Missing Alt Text", description: "Several images lack alt text." },
            { category: "small_text", confidence: 0.71, title: "Small Text", description: "Footer text below 12px." },
        ],
    },
    "medium-site": {
        url: "https://modern-startup.example.com",
        overall_score: 62, grade: "D", total_issues: 8, critical_count: 3, warning_count: 5, scan_duration: 3.6,
        issues: [
            { title: "Insufficient Contrast (3.2:1)", severity: "critical", wcag_criterion: "1.4.3", description: "Subtitle on gradient = 3.2:1.", suggestion: "Increase to 4.5:1.", score_impact: 4 },
            { title: "Image Missing Alt (team-photo.jpg)", severity: "critical", wcag_criterion: "1.1.1", description: "Team photo lacks alt.", suggestion: "Add descriptive alt.", score_impact: 5 },
            { title: "Button Missing Name", severity: "critical", wcag_criterion: "4.1.2", description: "Hamburger menu icon-only.", suggestion: "Add aria-label.", score_impact: 4 },
            { title: "Multiple H1 Headings", severity: "warning", wcag_criterion: "2.4.6", description: "3 h1 elements.", suggestion: "Keep one h1.", score_impact: 2 },
            { title: 'Generic Link: "learn more"', severity: "warning", wcag_criterion: "2.4.4", description: "Multiple 'Learn More' links.", suggestion: "Make descriptive.", score_impact: 2 },
            { title: "Missing Skip Link", severity: "warning", wcag_criterion: "2.4.1", description: "No skip nav.", suggestion: "Add skip link.", score_impact: 3 },
            { title: "Skipped Heading (h1→h3)", severity: "warning", wcag_criterion: "2.4.6", description: "Skips h2.", suggestion: "Use h2.", score_impact: 2 },
            { title: "Newsletter Input Unlabeled", severity: "warning", wcag_criterion: "3.3.2", description: "Placeholder only.", suggestion: "Add <label>.", score_impact: 4 },
        ],
        ai_insights: [
            { category: "low_contrast", confidence: 0.78, title: "Low Contrast on Hero", description: "Text on gradient may fail." },
            { category: "small_targets", confidence: 0.55, title: "Small Social Icons", description: "Footer icons appear small." },
        ],
    },
};

// ============ History Functions ============

if (els.navHistory) {
    els.navHistory.addEventListener('click', async (e) => {
        e.preventDefault();
        els.hero.classList.add('hidden');
        els.resultsSection.classList.add('hidden');
        if (els.historySection) {
            els.historySection.classList.remove('hidden');
            els.historySection.scrollIntoView({ behavior: 'smooth' });
            await loadHistory();
        }
    });
}

async function loadHistory() {
    try {
        els.historyTbody.innerHTML = '<tr><td colspan="6" style="text-align:center;">Loading history...</td></tr>';
        
        const [histRes, statsRes] = await Promise.all([
            fetch(`${API_BASE}/api/history`),
            fetch(`${API_BASE}/api/statistics`)
        ]);
        
        const history = await histRes.json();
        const stats = await statsRes.json();
        
        renderHistoryStats(stats);
        renderHistoryTable(history);
    } catch (err) {
        els.historyTbody.innerHTML = `<tr><td colspan="6" style="text-align:center;color:var(--critical);">Failed to load history: ${err.message}</td></tr>`;
    }
}

function renderHistoryStats(stats) {
    if (!els.historyStats) return;
    els.historyStats.innerHTML = `
        <div class="h-stat"><span class="h-stat-val">${stats.total_audits || 0}</span><span class="h-stat-lbl">Total Audits</span></div>
        <div class="h-stat"><span class="h-stat-val">${stats.average_score || 0}</span><span class="h-stat-lbl">Avg Score</span></div>
    `;
}

function renderHistoryTable(history) {
    if (!els.historyTbody) return;
    if (!history || history.length === 0) {
        els.historyTbody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:var(--text-muted);">No audits run yet.</td></tr>';
        return;
    }
    
    els.historyTbody.innerHTML = history.map(h => {
        let urlDomain = '';
        try { urlDomain = new URL(h.url).hostname; } catch(e) { urlDomain = h.url; }
        return `
        <tr>
            <td>${new Date(h.timestamp).toLocaleDateString()}</td>
            <td><a href="${h.url}" target="_blank" style="color:var(--accent-primary);text-decoration:none;">${urlDomain}</a></td>
            <td>${Math.round(h.overall_score)}</td>
            <td class="grade-col grade-${h.grade.toLowerCase()}">${h.grade}</td>
            <td>${h.total_issues}</td>
            <td>
                <button class="view-report-btn" onclick="window.open('${API_BASE}/api/audit/pdf/${h.id}', '_blank')">PDF</button>
            </td>
        </tr>
    `}).join('');
}
