import subprocess
import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# Ensure figures dir exists
os.makedirs('figures', exist_ok=True)

print("Starting backend server...")
server_proc = subprocess.Popen(
    ['python', '-m', 'uvicorn', 'backend.main:app', '--host', '127.0.0.1', '--port', '8000'],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

time.sleep(5) # Wait for server to start

try:
    print("Starting Selenium...")
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--window-size=1920,1080")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    frontend_url = "file:///C:/Users/HP/Downloads/nfhds/frontend/index.html"
    print(f"Loading {frontend_url}")
    driver.get(frontend_url)
    time.sleep(2)

    # Take screenshot of the initial dashboard
    driver.save_screenshot('figures/dashboard_initial.png')
    print("Saved figures/dashboard_initial.png")
    
    # Run the demo script directly
    try:
        driver.execute_script("loadDemo('good-site');")
        print("Executed loadDemo('good-site')")
    except Exception as e:
        print("Could not execute demo script:", e)
    
    # Wait for result to load
    time.sleep(5)
    
    # Take screenshot of the results
    driver.save_screenshot('figures/dashboard_results.png')
    print("Saved figures/dashboard_results.png")
    
    driver.quit()
    print("Screenshots captured successfully.")
finally:
    server_proc.terminate()
    print("Server terminated.")
