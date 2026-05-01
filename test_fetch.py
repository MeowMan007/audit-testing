import asyncio
from backend.services.page_fetcher import PageFetcher

async def main():
    pf = PageFetcher()
    res = await pf.fetch('https://example.com')
    print('Success:', res.success, 'Error:', res.error, 'Has Screenshot:', bool(res.screenshot_b64))
    pf.close()

if __name__ == '__main__':
    asyncio.run(main())
