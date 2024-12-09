import pytest
import logging
from playwright.sync_api import Page, expect
from shiny.playwright import controller
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

# Set up logging
logging.basicConfig(
    filename='tests/pytest.log',  # Log output to this file
    level=logging.INFO,  # Set log level to INFO or DEBUG as needed
    format="%(asctime)s - %(levelname)s - %(message)s"  # Log format
)
logger = logging.getLogger(__name__)

app = create_app_fixture("../app.py")


def test_app_loads(page: Page, app: ShinyAppProc):
    """Ensure the app loads and the title is correct."""
    page.goto(app.url)
    logger.info("App loaded successfully")
    assert page.title() == "PRESS ARTICLES EXPLORATION"
    logger.info("Title is correct")


@pytest.mark.usefixtures("page")
def test_file_upload(page: Page, app: ShinyAppProc):
    logger.info("Starting file upload test")
    page.goto(app.url)
    page.get_by_text("UPLOAD ARTICLE").click()
    page.get_by_label("UPLOAD ARTICLE").set_input_files("tests/correct_file.txt")
    page.wait_for_selector("#uploaded_text_content", state="visible", timeout=60000)
    assert "Test Article --" in page.locator("#uploaded_text_header").text_content()
    logger.info("File upload test passed")


@pytest.mark.usefixtures("page")
def test_toggle_view_full_text(page: Page, app: ShinyAppProc):
    logger.info("Testing toggle view full text functionality")
    page.goto(app.url)
    page.get_by_role("button", name="View more").click()
    assert page.text_content("#view_full_text").strip() == "View less"
    page.get_by_role("button", name="View less").click()
    assert page.text_content("#view_full_text").strip() == "View more"
    logger.info("Toggle view full text test passed")


def test_single_module_upload(page: Page, app: ShinyAppProc):
    """Test file upload functionality in the single module."""
    logger.info("Testing single module file upload")
    page.goto(app.url)

    # Navigate to SINGLE module
    page.click("text=SINGLE")

    page.get_by_text("UPLOAD ARTICLE").click()
    page.get_by_label("UPLOAD ARTICLE").set_input_files("tests/correct_file.txt")
    page.wait_for_selector("#uploaded_text_content", state="visible", timeout=60000)
    assert "Test Article --" in page.locator("#uploaded_text_header").text_content()

    page.get_by_role("button", name="View more").click()
    page.wait_for_selector("#uploaded_text_content", state="visible", timeout=60000)
    assert "This is last line" in page.locator("#uploaded_text_content").text_content()
    assert page.text_content("#view_full_text").strip() == "View less"

    page.get_by_role("button", name="View less").click()
    page.wait_for_selector("#uploaded_text_content", state="visible", timeout=60000)
    assert "This is last line" not in page.locator("#uploaded_text_content").text_content()
    assert page.text_content("#view_full_text").strip() == "View more"
    logger.info("Single module file upload test passed")


def test_toggle_buttons(page: Page, app: ShinyAppProc):
    """Test toggle buttons for collapsible sections."""
    logger.info("Testing toggle buttons")
    page.goto(app.url)

    page.get_by_role("tab", name="ALL").click()

    # Testing EDA toggle button
    page.locator("#toggle_eda_button").click()
    page.wait_for_timeout(500)
    assert page.locator("#eda_plots").inner_html().strip() == "<div></div>"
    assert not page.locator("#eda_plots").get_by_role("img").is_visible()

    page.locator("#toggle_eda_button").click()
    page.wait_for_selector("#eda_plots", state="visible", timeout=60000)
    assert page.locator("#eda_plots").inner_html().strip() != "<div></div>"

    # Testing NER toggle button
    page.locator("#toggle_ner_button").click()
    page.wait_for_timeout(5000)
    assert page.locator("#ner_plots").inner_html().strip() == "<div></div>"
    assert not page.locator("#ner_plots").get_by_role("img").is_visible()

    page.locator("#toggle_ner_button").click()
    page.wait_for_selector("#ner_plots", state="visible", timeout=60000)
    assert page.locator("#ner_plots").inner_html().strip() != "<div></div>"

    # Testing Sentiment toggle button
    page.locator("#toggle_sentiment_button").click()
    page.wait_for_timeout(500)
    assert page.locator("#sentiment_plots").inner_html().strip() == "<div></div>"
    assert not page.locator("#sentiment_plots").get_by_role("img").is_visible()

    page.locator("#toggle_sentiment_button").click()
    page.wait_for_selector("#sentiment_plots", state="visible", timeout=60000)
    assert page.locator("#sentiment_plots").inner_html().strip() != "<div></div>"
    logger.info("Toggle buttons test passed")


def test_interactive_plot(page: Page, app: ShinyAppProc):
    """Test an interactive plot's presence and interaction."""
    logger.info("Testing interactive plot")
    page.goto(app.url)

    # Navigate to ALL module
    page.get_by_role("tab", name="ALL").click()
    page.wait_for_timeout(600)

    # Check if plot widget is visible
    plot_widget = page.locator("#sentiment_dist_plot")
    assert plot_widget.is_visible()
    title_element = page.locator('#sentiment_dist_plot > div > div > div > svg:nth-child(3) > g.infolayer > g.g-gtitle > text')
    assert "Gaza before conflict" in title_element.text_content().strip()

    # Simulate changing a filter
    dataset_filter = page.locator("#dataset_filter")
    assert dataset_filter.is_visible()
    dataset_filter.select_option("Ukraine before conflict")
    page.wait_for_timeout(6000)
    page.wait_for_selector("#dataset_filter", state="visible", timeout=60000)
    assert dataset_filter.input_value() == "Ukraine before conflict"
    page.wait_for_selector("#sentiment_dist_plot", state="visible", timeout=60000)
    assert "Ukraine before conflict" in title_element.text_content().strip()
    logger.info("Interactive plot test passed")
