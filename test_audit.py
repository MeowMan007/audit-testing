import asyncio
import json
import logging
from backend.routers.audit import run_audit
from backend.models.schemas import AuditRequest

logging.basicConfig(level=logging.DEBUG)

async def main():
    req = AuditRequest(url='https://example.com', include_ai=False)
    res = await run_audit(req)
    print("Type of screenshot:", type(res.get('screenshot')))
    
    val = res.get('screenshot')
    if val:
        print("Length:", len(str(val)))
    else:
        print("Screenshot is None or Empty")

if __name__ == "__main__":
    asyncio.run(main())
