import httpx, json, sys, re

API_KEY = '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'

models = [
    'google/gemma-4-31b-it:free',
    'google/gemma-4-26b-a4b-it:free',
    'qwen/qwen3-coder:free',
    'nvidia/nemotron-3-nano-30b-a3b:free',
]

prompt = """Analyze this web accessibility audit and provide expert insights.

URL: https://example.com
Overall Score: 72/100 (Grade: C)
Total Issues: 3 (Critical: 1, Warnings: 2)

Issues Found:
  1. [CRITICAL] Empty Link -- WCAG 2.4.4 -- A link has no text content.
  2. [WARNING] No Headings Found -- WCAG 2.4.6 -- Page has no heading elements.

Respond with this exact JSON structure:
{
  "summary": "2-3 sentence overall assessment",
  "design_issues": [
    {
      "area": "specific UI area",
      "problem": "how the design fails",
      "impact": "who is affected",
      "fix": "specific fix"
    }
  ],
  "top_fixes": [
    {
      "priority": 1,
      "title": "short fix title",
      "description": "what to do",
      "effort": "low",
      "wcag_criteria": "2.4.4"
    }
  ],
  "ux_patterns": ["recommendation 1"],
  "ai_narrative": "friendly coaching message"
}

Provide 2 design_issues, 2 top_fixes, and 2 ux_patterns. Be specific. Return ONLY valid JSON, no markdown fences, no extra text."""

for model in models:
    print(f'\n=== Testing: {model} ===')
    try:
        r = httpx.post('https://openrouter.ai/api/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {API_KEY}',
                'Content-Type': 'application/json',
                'HTTP-Referer': 'https://accesslens.app',
                'X-Title': 'AccessLens Test',
            },
            json={
                'model': model,
                'messages': [
                    {'role': 'system', 'content': 'You are an expert web accessibility consultant. Always respond with valid JSON only. No markdown, no code fences, no thinking tags.'},
                    {'role': 'user', 'content': prompt}
                ],
                'max_tokens': 1500,
                'temperature': 0.3,
            },
            timeout=60.0
        )
        print(f'Status: {r.status_code}')
        if r.status_code != 200:
            print(f'Error: {r.text[:300]}')
            continue
        
        data = r.json()
        content = data['choices'][0]['message']['content']
        print(f'Content length: {len(content)}')
        print(f'First 200 chars: {repr(content[:200])}')
        
        # Try parsing
        c = content.strip()
        # Strip markdown fences
        if c.startswith('```'):
            c = c.split('\n', 1)[1]
        if c.endswith('```'):
            c = c.rsplit('```', 1)[0]
        # Strip thinking tags
        c = re.sub(r'<think>.*?</think>', '', c, flags=re.DOTALL)
        c = c.strip()
        
        try:
            parsed = json.loads(c)
            print(f'PARSED OK! Keys: {list(parsed.keys())}')
        except json.JSONDecodeError as e:
            print(f'JSON FAILED: {e}')
            print(f'After cleanup first 300: {repr(c[:300])}')
    except Exception as e:
        print(f'Request failed: {e}')
