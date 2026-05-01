import asyncio
from backend.services.page_fetcher import PageData

async def test():
    page_data = PageData()
    page_data.error = "Selenium error"
    print(page_data.error)
    
asyncio.run(test())
