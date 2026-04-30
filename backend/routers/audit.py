"""
Audit API Router — Handles accessibility audit requests.

ENDPOINTS (Requirement 5 — Backend API):
    POST /api/audit     → Takes URL, returns full JSON audit report
    GET  /api/health    → Health check + model status
    GET  /api/demo/{id} → Pre-computed demo results (Requirement 9)

The /audit endpoint orchestrates the full pipeline:
    URL → Page Fetch → DOM Parse → Rule Engine → CNN Inference → Report
"""
import time
import logging
from fastapi import APIRouter, HTTPException

from backend.models.schemas import AuditRequest, AuditReport
from backend.services.page_fetcher import PageFetcher
from backend.services.dom_analyzer import DOMAnalyzer
from backend.services.rule_engine import RuleEngine
from backend.services.dl_engine import DLEngine
from backend.services.report_generator import ReportGenerator
from backend.services.database import db_service
from backend.services.pdf_generator import pdf_generator
from backend.services.hf_api_service import hf_api_service
from backend.utils.image_annotator import annotate_screenshot

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["audit"])

# ============================================================
# Initialize Services (singleton pattern)
# ============================================================
page_fetcher = PageFetcher()
dom_analyzer = DOMAnalyzer()
rule_engine = RuleEngine()
dl_engine = DLEngine()
report_generator = ReportGenerator()

# Simple in-memory cache
_audit_cache = {}


# ============================================================
# POST /api/audit — Main Audit Endpoint
# ============================================================
@router.post("/audit")
async def run_audit(request: AuditRequest):
    """Run a full accessibility audit on the given URL.
    
    Pipeline:
    1. Fetch and render the page (Selenium + BeautifulSoup)
    2. Analyze DOM structure for accessibility data
    3. Run 15+ WCAG rule-based checks
    4. Run CNN model inference on screenshot (if available)
    5. Aggregate into scored report with issues and suggestions
    
    Args:
        request: AuditRequest with URL and options
        
    Returns:
        AuditReport with score, grade, issues, AI insights, and screenshot
    """
    url = request.url
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    # Check cache
    if url in _audit_cache:
        logger.info(f"Cache hit for {url}")
        return _audit_cache[url]

    start_time = time.time()
    logger.info(f"Starting audit for: {url}")

    try:
        # Step 1: Fetch the page
        logger.info("  Step 1/4: Fetching page...")
        page_data = await page_fetcher.fetch(url)
        if not page_data.success:
            raise HTTPException(
                status_code=422,
                detail=f"Failed to fetch page: {page_data.error}"
            )

        # Step 2: Parse DOM
        logger.info("  Step 2/4: Analyzing DOM...")
        dom_data = dom_analyzer.analyze(page_data.html)

        # Step 3: Run rule-based checks
        logger.info("  Step 3/4: Running WCAG rule checks...")
        issues = rule_engine.run_all_checks(
            dom_data,
            computed_styles=page_data.computed_styles,
        )

        # Step 4: Run AI model (if requested and available)
        dl_insights = []
        attention_heatmap = None
        ai_used = False
        if request.include_ai and page_data.screenshot_b64:
            logger.info("  Step 4/4: Running CNN analysis...")
            dl_insights, attention_heatmap = await dl_engine.analyze_with_explanation(
                screenshot_b64=page_data.screenshot_b64,
            )
            ai_used = dl_engine.is_available

        scan_duration = time.time() - start_time

        # Annotate screenshot with bounding boxes
        annotated_screenshot = annotate_screenshot(
            page_data.screenshot_b64, 
            issues, 
            page_data.element_rects
        )

        # Step 5: Generate report
        report = report_generator.generate(
            url=url,
            issues=issues,
            dl_insights=dl_insights,
            screenshot_b64=annotated_screenshot,
            scan_duration=scan_duration,
            ai_model_used=ai_used,
        )
        report.attention_heatmap = attention_heatmap

        # Cache the result
        _audit_cache[url] = report

        # Save to database
        audit_id = db_service.save_audit(report.dict())
        
        # Attach ID to response so frontend can request AI insights
        report_dict = report.dict()
        report_dict["id"] = audit_id
        
        logger.info(f"Audit complete: score={report.overall_score}, time={scan_duration:.1f}s, DB_ID={audit_id}")
        return report_dict

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audit failed for {url}: {e}")
        raise HTTPException(status_code=500, detail=f"Audit failed: {str(e)}")


# ============================================================
# GET /api/health — Health Check
# ============================================================
@router.get("/health")
async def health_check():
    """Check API health and component status."""
    return {
        "status": "healthy",
        "components": {
            "rule_engine": "ready",
            "cnn_model": "ready" if dl_engine.is_available else "not loaded (train model first)",
            "selenium": "available",
        },
        "version": "1.0.0",
        "project": "AccessLens — AI-Powered Accessibility Audit Tool",
    }


# ============================================================
# GET /api/history — Audit History
# ============================================================
@router.get("/history")
async def get_history(limit: int = 50, offset: int = 0):
    """Get recent audit history."""
    return db_service.get_history(limit, offset)

@router.get("/history/{audit_id}")
async def get_audit_by_id(audit_id: int):
    """Get full audit report by ID."""
    report = db_service.get_audit_by_id(audit_id)
    if not report:
        raise HTTPException(status_code=404, detail="Audit not found")
    return report

@router.get("/statistics")
async def get_statistics():
    """Get aggregate statistics."""
    return db_service.get_statistics()

@router.get("/audit/pdf/{audit_id}")
async def get_pdf(audit_id: int):
    """Download audit report as PDF."""
    from fastapi.responses import StreamingResponse
    report = db_service.get_audit_by_id(audit_id)
    if not report:
        raise HTTPException(status_code=404, detail="Audit not found")
        
    pdf_buffer = pdf_generator.generate(report)
    
    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=AccessLens_Report_{audit_id}.pdf"
        }
    )


# ============================================================
# POST /api/ai-insights — OpenRouter AI Analysis
# ============================================================
@router.post("/ai-insights")
async def get_ai_insights(request: dict):
    """Get AI-powered design and accessibility insights via OpenRouter.
    
    Body: { "audit_id": int } or { "url": str, "score": float, "grade": str, "issues": [...], "categories": [...] }
    """
    audit_id = request.get("audit_id")
    
    # If audit_id provided, load data from DB
    if audit_id:
        # Check for cached insights first
        cached = db_service.get_ai_insights(audit_id)
        if cached:
            return cached
        
        report = db_service.get_audit_by_id(audit_id)
        if not report:
            raise HTTPException(status_code=404, detail="Audit not found")
        
        url = report.get("url", "")
        score = report.get("overall_score", 0)
        grade = report.get("grade", "F")
        issues = report.get("issues", [])
        categories = report.get("categories", [])
    else:
        url = request.get("url", "")
        score = request.get("score", 0)
        grade = request.get("grade", "F")
        issues = request.get("issues", [])
        categories = request.get("categories", [])

    if not hf_api_service.is_available:
        return {"available": False, "reason": "HF_TOKEN not configured in .env"}

    insights = await hf_api_service.get_insights(
        url=url, score=score, grade=grade,
        issues=issues, categories=categories,
    )
    
    # Store insights if audit_id was provided
    if audit_id and insights.get("available"):
        db_service.save_ai_insights(audit_id, insights)
    
    return insights


@router.get("/ai-insights/{audit_id}")
async def get_cached_ai_insights(audit_id: int):
    """Get cached AI insights for a specific audit."""
    insights = db_service.get_ai_insights(audit_id)
    if not insights:
        return {"available": False, "reason": "No AI insights found for this audit"}
    return insights


# ============================================================
# GET /api/demo/{site_id} — Pre-computed Demo Results (Req. 9)
# ============================================================
DEMO_RESULTS = {
    "good-site": {
        "url": "https://www.gov.uk",
        "overall_score": 87.0,
        "grade": "B",
        "description": "Well-structured government website with strong accessibility practices",
        "total_issues": 4,
        "critical_count": 0,
        "warning_count": 4,
        "issues": [
            {
                "title": "Generic Link Text: \"more\"",
                "severity": "warning",
                "wcag_criterion": "2.4.4",
                "description": "Some links use generic text like 'more' instead of descriptive labels.",
                "suggestion": "Replace 'more' with specific text like 'More about our services'.",
                "score_impact": 2.0,
            },
            {
                "title": "Missing Skip Navigation Link",
                "severity": "warning",
                "wcag_criterion": "2.4.1",
                "description": "No skip-to-content link found at the top of the page.",
                "suggestion": "Add a skip link as the first focusable element.",
                "score_impact": 3.0,
            },
            {
                "title": "Table Missing Caption",
                "severity": "warning",
                "wcag_criterion": "1.3.1",
                "description": "A data table lacks a <caption> element describing its content.",
                "suggestion": "Add <caption>Table description</caption> inside the <table>.",
                "score_impact": 2.0,
            },
            {
                "title": "Skipped Heading Level (h2 → h4)",
                "severity": "warning",
                "wcag_criterion": "2.4.6",
                "description": "Heading hierarchy skips from h2 to h4.",
                "suggestion": "Use h3 instead of h4 to maintain proper hierarchy.",
                "score_impact": 2.0,
            },
        ],
        "ai_insights": [
            {
                "category": "small_targets",
                "confidence": 0.42,
                "title": "Possible Small Touch Targets",
                "description": "Some footer links may be slightly below the 44x44px minimum.",
            },
        ],
    },
    "bad-site": {
        "url": "https://old-design-example.com",
        "overall_score": 34.0,
        "grade": "F",
        "description": "Outdated website with numerous accessibility barriers",
        "total_issues": 14,
        "critical_count": 8,
        "warning_count": 6,
        "issues": [
            {
                "title": "Missing Page Title",
                "severity": "critical",
                "wcag_criterion": "2.4.2",
                "description": "The page has no <title> element.",
                "suggestion": "Add a descriptive <title> tag.",
                "score_impact": 5.0,
            },
            {
                "title": "Missing Language Declaration",
                "severity": "critical",
                "wcag_criterion": "3.1.1",
                "description": "No lang attribute on <html>.",
                "suggestion": "Add lang='en' to <html>.",
                "score_impact": 5.0,
            },
            {
                "title": "Insufficient Color Contrast (2.1:1)",
                "severity": "critical",
                "wcag_criterion": "1.4.3",
                "description": "Light gray text (#999) on white background has only 2.1:1 contrast ratio.",
                "suggestion": "Darken text to at least #767676 for 4.5:1 contrast.",
                "score_impact": 4.0,
            },
            {
                "title": "Image Missing Alt Text (hero.jpg)",
                "severity": "critical",
                "wcag_criterion": "1.1.1",
                "description": "Hero image has no alt attribute.",
                "suggestion": "Add alt='Description of hero image'.",
                "score_impact": 5.0,
            },
            {
                "title": "Image Missing Alt Text (logo.png)",
                "severity": "critical",
                "wcag_criterion": "1.1.1",
                "description": "Company logo has no alt text.",
                "suggestion": "Add alt='Company Name Logo'.",
                "score_impact": 5.0,
            },
            {
                "title": "Image Missing Alt Text (banner.jpg)",
                "severity": "critical",
                "wcag_criterion": "1.1.1",
                "description": "Banner image lacks alternative text.",
                "suggestion": "Add descriptive alt text or alt='' if decorative.",
                "score_impact": 5.0,
            },
            {
                "title": "Form Input Missing Label (email)",
                "severity": "critical",
                "wcag_criterion": "3.3.2",
                "description": "Email input has no associated label.",
                "suggestion": "Add <label for='email'>Email</label>.",
                "score_impact": 4.0,
            },
            {
                "title": "Form Input Missing Label (phone)",
                "severity": "critical",
                "wcag_criterion": "3.3.2",
                "description": "Phone input has no associated label.",
                "suggestion": "Add <label for='phone'>Phone</label>.",
                "score_impact": 4.0,
            },
            {
                "title": "No Headings Found",
                "severity": "warning",
                "wcag_criterion": "2.4.6",
                "description": "Page has no heading elements for structure.",
                "suggestion": "Add h1-h6 headings to organize content.",
                "score_impact": 3.0,
            },
            {
                "title": "Zoom/Scaling Disabled",
                "severity": "warning",
                "wcag_criterion": "1.4.4",
                "description": "Viewport prevents zooming (user-scalable=no).",
                "suggestion": "Remove user-scalable=no from viewport meta.",
                "score_impact": 5.0,
            },
            {
                "title": "Empty Link",
                "severity": "warning",
                "wcag_criterion": "2.4.4",
                "description": "A link has no text content.",
                "suggestion": "Add descriptive text to the link.",
                "score_impact": 4.0,
            },
            {
                "title": "Focus Outline Removed",
                "severity": "warning",
                "wcag_criterion": "2.1.1",
                "description": "CSS removes focus outlines with outline:none.",
                "suggestion": "Replace with a visible focus style.",
                "score_impact": 5.0,
            },
            {
                "title": "Duplicate ID: 'content'",
                "severity": "warning",
                "wcag_criterion": "4.1.1",
                "description": "The ID 'content' appears on multiple elements.",
                "suggestion": "Make all IDs unique.",
                "score_impact": 2.0,
            },
            {
                "title": "Iframe Missing Title",
                "severity": "warning",
                "wcag_criterion": "4.1.2",
                "description": "An iframe has no title attribute.",
                "suggestion": "Add title attribute to the iframe.",
                "score_impact": 2.0,
            },
        ],
        "ai_insights": [
            {
                "category": "low_contrast",
                "confidence": 0.92,
                "title": "Low Color Contrast Detected",
                "description": "Multiple text areas have dangerously low contrast ratios.",
            },
            {
                "category": "missing_alt",
                "confidence": 0.88,
                "title": "Missing Alt Text Detected",
                "description": "Several images appear to lack descriptive alt text.",
            },
            {
                "category": "small_text",
                "confidence": 0.71,
                "title": "Small Text Detected",
                "description": "Footer text appears very small (below 12px).",
            },
        ],
    },
    "medium-site": {
        "url": "https://modern-startup.example.com",
        "overall_score": 62.0,
        "grade": "D",
        "description": "Modern startup website — looks good but has accessibility gaps",
        "total_issues": 8,
        "critical_count": 3,
        "warning_count": 5,
        "issues": [
            {
                "title": "Insufficient Color Contrast (3.2:1)",
                "severity": "critical",
                "wcag_criterion": "1.4.3",
                "description": "Subtitle text has a contrast ratio of only 3.2:1 against the gradient background.",
                "suggestion": "Increase contrast to at least 4.5:1. Consider adding a text shadow or background overlay.",
                "score_impact": 4.0,
            },
            {
                "title": "Image Missing Alt Text (team-photo.jpg)",
                "severity": "critical",
                "wcag_criterion": "1.1.1",
                "description": "Team photo has no alt attribute.",
                "suggestion": "Add alt='Team photo showing our 12 team members in the office'.",
                "score_impact": 5.0,
            },
            {
                "title": "Button Missing Accessible Name",
                "severity": "critical",
                "wcag_criterion": "4.1.2",
                "description": "Hamburger menu button has only an icon, no accessible label.",
                "suggestion": "Add aria-label='Open navigation menu' to the button.",
                "score_impact": 4.0,
            },
            {
                "title": "Multiple H1 Headings",
                "severity": "warning",
                "wcag_criterion": "2.4.6",
                "description": "Found 3 <h1> elements. Best practice is one per page.",
                "suggestion": "Keep one <h1> as the main heading, demote others to <h2>.",
                "score_impact": 2.0,
            },
            {
                "title": "Generic Link Text: \"learn more\"",
                "severity": "warning",
                "wcag_criterion": "2.4.4",
                "description": "Multiple links use 'Learn More' without context.",
                "suggestion": "Use 'Learn more about our pricing' instead.",
                "score_impact": 2.0,
            },
            {
                "title": "Missing Skip Navigation Link",
                "severity": "warning",
                "wcag_criterion": "2.4.1",
                "description": "No skip-to-content link found.",
                "suggestion": "Add a skip link as the first focusable element.",
                "score_impact": 3.0,
            },
            {
                "title": "Skipped Heading Level (h1 → h3)",
                "severity": "warning",
                "wcag_criterion": "2.4.6",
                "description": "Heading jumps from h1 to h3, skipping h2.",
                "suggestion": "Use h2 instead of h3.",
                "score_impact": 2.0,
            },
            {
                "title": "Form Input Missing Label (newsletter)",
                "severity": "warning",
                "wcag_criterion": "3.3.2",
                "description": "Newsletter signup input uses only placeholder, no label.",
                "suggestion": "Add a visible <label> element.",
                "score_impact": 4.0,
            },
        ],
        "ai_insights": [
            {
                "category": "low_contrast",
                "confidence": 0.78,
                "title": "Low Color Contrast Detected",
                "description": "Hero section text on gradient background may have insufficient contrast.",
            },
            {
                "category": "small_targets",
                "confidence": 0.55,
                "title": "Small Touch Targets",
                "description": "Social media icon links in the footer appear small.",
            },
        ],
    },
}


@router.get("/demo/{site_id}")
async def get_demo(site_id: str):
    """Get pre-computed demo audit results (Requirement 9).
    
    Available demos:
    - good-site: High-scoring government website (87/100)
    - bad-site: Failing old-design website (34/100)
    - medium-site: Modern startup with gaps (62/100)
    """
    if site_id not in DEMO_RESULTS:
        raise HTTPException(
            status_code=404,
            detail=f"Demo '{site_id}' not found. Available: {list(DEMO_RESULTS.keys())}"
        )
    return DEMO_RESULTS[site_id]
