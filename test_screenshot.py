import asyncio
import base64
from backend.services.page_fetcher import PageFetcher

async def main():
    pf = PageFetcher()
    res = await pf.fetch('https://example.com')
    if res.screenshot_b64:
        with open('test_screenshot.png', 'wb') as f:
            f.write(base64.b64decode(res.screenshot_b64))
    pf.close()

if __name__ == '__main__':
    asyncio.run(main())
