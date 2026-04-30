"""
Hugging Face AI Insights Service — Calls Hugging Face Inference API for
deep accessibility analysis and design alignment insights.

Uses free Serverless Inference API models.
"""
import json
import logging
import httpx
import asyncio
import re
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

class HFApiService:
    """Service for getting AI-powered accessibility insights via Hugging Face."""

    def __init__(self):
        from backend.config import settings
        self.api_token = settings.HF_TOKEN
        self.model = DEFAULT_MODEL

    @property
    def is_available(self) -> bool:
        return bool(self.api_token and self.api_token.startswith("hf_"))

    async def get_insights(
        self,
        url: str,
        score: float,
        grade: str,
        issues: List[Dict[str, Any]],
        categories: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Get AI-powered insights from Hugging Face."""
        if not self.is_available:
            return {"available": False, "reason": "HF_TOKEN not configured"}

        prompt = self._build_prompt(url, score, grade, issues, categories)

        models_to_try = [
            self.model,
            "Qwen/Qwen2.5-72B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3"
        ]

        for attempt, current_model in enumerate(models_to_try):
            api_url = f"https://api-inference.huggingface.co/models/{current_model}/v1/chat/completions"
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        api_url,
                        headers={
                            "Authorization": f"Bearer {self.api_token}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": current_model,
                            "messages": [
                                {
                                    "role": "system",
                                    "content": (
                                        "You are an expert web accessibility consultant and UI/UX designer. "
                                        "Analyze audit results and provide actionable insights. "
                                        "Always respond with valid JSON only, no markdown."
                                    ),
                                },
                                {"role": "user", "content": prompt},
                            ],
                            "temperature": 0.3,
                            "max_tokens": 1500,
                        },
                    )
                    
                    response.raise_for_status()
                    data = response.json()

                content = data["choices"][0]["message"]["content"]
                
                # Strip thinking tags (used by some reasoning models)
                content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
                
                # Strip markdown code fences if present
                content = content.strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[1]
                if content.endswith("```"):
                    content = content.rsplit("```", 1)[0]
                # Sometimes it outputs ```json
                if content.lower().startswith("json"):
                    content = content[4:]
                content = content.strip()

                parsed = json.loads(content)
                parsed["available"] = True
                parsed["model_used"] = current_model
                logger.info(f"HF insights generated for {url} using {current_model}")
                return parsed

            except httpx.HTTPStatusError as e:
                logger.error(f"HF API error on {current_model}: {e.response.status_code}")
                if attempt < len(models_to_try) - 1:
                    await asyncio.sleep(1.5)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse HF JSON on {current_model}: {e}")
                if attempt < len(models_to_try) - 1:
                    await asyncio.sleep(1.5)
            except Exception as e:
                logger.error(f"HF request failed on {current_model}: {e}")
                if attempt < len(models_to_try) - 1:
                    await asyncio.sleep(1.5)
                    
        # If we exit the loop, all models failed (rate limits, bad json, etc.)
        logger.warning("All HF models failed. Falling back to mock insights.")
        return self._get_mock_insights(url, score, grade, issues)

    def _build_prompt(
        self,
        url: str,
        score: float,
        grade: str,
        issues: List[Dict[str, Any]],
        categories: List[Dict[str, Any]],
    ) -> str:
        """Build the analysis prompt from audit data."""
        critical = [i for i in issues if i.get("severity") == "critical"][:8]
        warnings = [i for i in issues if i.get("severity") == "warning"][:6]

        issues_text = ""
        for i, issue in enumerate(critical, 1):
            issues_text += f"  {i}. [CRITICAL] {issue.get('title','')} — WCAG {issue.get('wcag_criterion','')} — {issue.get('description','')}\n"
        for i, issue in enumerate(warnings, len(critical) + 1):
            issues_text += f"  {i}. [WARNING] {issue.get('title','')} — WCAG {issue.get('wcag_criterion','')} — {issue.get('description','')}\n"

        cat_text = ""
        for cat in categories:
            cat_text += f"  - {cat.get('name','')}: {cat.get('score',0)}/100 ({cat.get('issue_count',0)} issues)\n"

        return f"""Analyze this web accessibility audit and provide expert insights.

URL: {url}
Overall Score: {score}/100 (Grade: {grade})
Total Issues: {len(issues)} (Critical: {len(critical)}, Warnings: {len(warnings)})

Category Breakdown:
{cat_text}

Issues Found:
{issues_text}

Respond with this exact JSON structure:
{{
  "summary": "2-3 sentence overall assessment of the site's accessibility posture",
  "design_issues": [
    {{
      "area": "specific UI area (e.g. Navigation, Hero, Forms, Footer)",
      "problem": "how the design fails accessibility standards",
      "impact": "who is affected and how",
      "fix": "specific design change to resolve it"
    }}
  ],
  "top_fixes": [
    {{
      "priority": 1,
      "title": "short fix title",
      "description": "what to do and why it matters",
      "effort": "low/medium/high",
      "wcag_criteria": "relevant WCAG criterion"
    }}
  ],
  "wcag_priority": {{
    "level_a_gaps": "summary of Level A compliance gaps",
    "level_aa_gaps": "summary of Level AA compliance gaps",
    "recommended_focus": "which WCAG level to prioritize first"
  }},
  "ux_patterns": [
    "specific UX/UI pattern recommendation for better accessibility"
  ],
  "color_typography": {{
    "contrast_assessment": "assessment of color contrast across the site",
    "typography_issues": "any font size, weight, or readability concerns",
    "recommendations": "specific color/typography fixes"
  }},
  "ai_narrative": "friendly 2-3 sentence coaching message for the developer"
}}

Provide 3-5 design_issues, exactly 3 top_fixes, and 3-4 ux_patterns. Be specific and actionable."""

    def _get_mock_insights(self, url: str, score: float, grade: str, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Return fallback mock insights when API fails."""
        critical = [i for i in issues if i.get("severity") == "critical"]
        
        design_issues = []
        if critical:
            for c in critical[:3]:
                design_issues.append({
                    "area": "General UI",
                    "problem": c.get('title', 'Accessibility Issue'),
                    "impact": "Users with disabilities may struggle to interact with this element.",
                    "fix": c.get('suggestion', 'Review WCAG guidelines to fix this issue.')
                })
        else:
            design_issues.append({
                "area": "Overall Design",
                "problem": "Minor accessibility gaps",
                "impact": "Some users may experience slight friction.",
                "fix": "Continue following WCAG best practices."
            })

        top_fixes = []
        for i, c in enumerate((critical if critical else issues)[:3], 1):
            top_fixes.append({
                "priority": i,
                "title": c.get('title', 'Review Accessibility'),
                "description": c.get('description', 'Ensure this element meets accessibility standards.'),
                "effort": "medium",
                "wcag_criteria": c.get('wcag_criterion', 'General')
            })

        return {
            "available": True,
            "model_used": "mock-fallback-due-to-api-error",
            "summary": f"The site {url} scored {score}/100 (Grade {grade}). While some accessibility practices are followed, there are areas requiring attention to ensure inclusive access.",
            "design_issues": design_issues,
            "top_fixes": top_fixes,
            "wcag_priority": {
                "level_a_gaps": "Several Level A criteria may not be met, affecting basic accessibility.",
                "level_aa_gaps": "Level AA compliance needs improvement in color contrast and navigation.",
                "recommended_focus": "Prioritize Level A issues first to establish a baseline of accessibility."
            },
            "ux_patterns": [
                "Ensure all interactive elements have clear visual focus states.",
                "Maintain a consistent and predictable navigation structure.",
                "Use high-contrast color combinations for text and background."
            ],
            "color_typography": {
                "contrast_assessment": "Some areas may have insufficient contrast, making text hard to read.",
                "typography_issues": "Ensure font sizes are legible and scalable.",
                "recommendations": "Use a contrast checker tool to verify all text meets the 4.5:1 ratio."
            },
            "ai_narrative": "Rate limits prevented live AI analysis from Hugging Face, but these general best practices based on your audit results will guide you toward a more accessible design!"
        }

# Singleton
hf_api_service = HFApiService()
