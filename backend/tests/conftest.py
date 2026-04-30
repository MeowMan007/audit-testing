import pytest
from bs4 import BeautifulSoup
from backend.services.dom_analyzer import DOMData

@pytest.fixture
def sample_html_good():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>Good Site</title>
    </head>
    <body>
        <a href="#content" class="skip-link">Skip to content</a>
        <nav aria-label="Main Navigation">
            <a href="/">Home</a>
        </nav>
        <main id="content">
            <h1>Welcome to Good Site</h1>
            <img src="logo.png" alt="Company Logo">
            <form>
                <label for="email">Email Address</label>
                <input type="email" id="email" name="email">
                <button type="submit">Subscribe</button>
            </form>
            <p style="color: #333333; background-color: #ffffff;">This is high contrast text.</p>
        </main>
    </body>
    </html>
    """

@pytest.fixture
def sample_html_bad():
    return """
    <html>
    <head>
        <!-- No title -->
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
    </head>
    <body>
        <h2>Skipped H1</h2>
        <img src="image1.jpg"> <!-- Missing alt -->
        <a href="more.html">Click here</a> <!-- Generic link text -->
        <div id="duplicate">First duplicate</div>
        <div id="duplicate">Second duplicate</div>
        <input type="text" placeholder="Search"> <!-- Missing label -->
        <button></button> <!-- Missing accessible name -->
        <p style="color: #999999; background-color: #ffffff;">This is low contrast text.</p>
        <iframe src="widget.html"></iframe> <!-- Missing title -->
    </body>
    </html>
    """

@pytest.fixture
def sample_dom_good(sample_html_good):
    from backend.services.dom_analyzer import DOMAnalyzer
    analyzer = DOMAnalyzer()
    return analyzer.analyze(sample_html_good)

@pytest.fixture
def sample_dom_bad(sample_html_bad):
    from backend.services.dom_analyzer import DOMAnalyzer
    analyzer = DOMAnalyzer()
    return analyzer.analyze(sample_html_bad)
