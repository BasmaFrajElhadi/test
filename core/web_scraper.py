import os
import sys
import re
import json
from playwright.sync_api import sync_playwright, Page

# Add project root to sys.path for relative imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

class WebScraper:
    """
    A scraper class to extract detailed information about Egyptian public universities
    from the 'universitiesegypt.com' website using Playwright.
    """

    def __init__(self):
        self.raw_data_folder = os.path.join(project_root, "data", "raw")
        self.universities_data = []
        self.faculties = []
        self.contact_info = []
        self.all_links = []

    def faculties_scraper(self, page: Page):
        """
        Extract faculty information (name and description) for a university.

        Args:
            page (Page): The Playwright page object currently displaying the faculty page.
        """
        self.faculties = []
        faculties_parent = page.locator(".innerListOfUniversities").nth(0)
        faculties_list = faculties_parent.locator("> div")

        for i in range(faculties_list.count()):
            faculty = faculties_list.nth(i)
            faculty_name = faculty.locator("h2").text_content()
            about_faculty = faculty.locator("p").filter(has_text=re.compile(r"\S")).first.text_content()

            self.faculties.append({
                'name': faculty_name, 
                'about': about_faculty})

    def contact_scraper(self, page: Page):
        """
        Extract contact information, social media links, and map URL from the contact page.

        Args:
            page (Page): The Playwright page object currently displaying the contact page.
        """
        self.contact_info = []
        soical_media_contact = []
        contact_parent = page.locator(".newsListDate").nth(0)
        contact_list = contact_parent.locator("> i")  

        for i in range(contact_list.count() - 1):
            contact = contact_list.nth(i)
            contact_name = contact.locator("strong").text_content()
            info = contact.locator("span").text_content()
            self.contact_info.append({'contact_name': contact_name, "contact_info":info})
            
        # social media contact
        social_div = page.locator(".socialIcon").nth(0)
        links = social_div.locator("a")  

        for i in range(links.count()):
            link = links.nth(i)
            href = link.get_attribute("href")
            
            # get platform name from the class of <i> inside <a>
            icon_class = link.locator("i").get_attribute("class")
            platform = icon_class.split()[-1].replace("fa-", "").capitalize()
            self.contact_info.append({'contact_name': platform, "contact_info":href})
            
        # get map 
        map_src  = page.locator("iframe").nth(0).get_attribute("src")
        self.contact_info.append({'contact_name': 'map_src', 'contact_info':map_src})


    def get_all_universities_links(self, page: Page):
        """
        Collect links to all public universities.

        Args:
            page (Page): The Playwright page object displaying the universities list.
        """
        while True:
            list_of_universities = page.locator("ol").nth(2)
            universities = list_of_universities.locator("li")

            for i in range(universities.count()):
                href = universities.nth(i).locator("a").get_attribute("href")
                if href and href not in self.all_links:
                    self.all_links.append(href)

            next_button = page.locator("#ctl00_ctl00_ContentPlaceHolder1_ContentPlaceHolder1_lnkNext")
            if next_button.get_attribute("disabled"):
                break  # Stop if no next page
            next_button.click()
            page.wait_for_load_state("networkidle")

    
    def scrap_public_universities(self):
        """
        Main method to scrape public universities:
        - Collects all public university links.
        - Visits each university and scrapes its information.
        """
        pw = sync_playwright().start()
        browser = pw.chromium.launch(headless=False, slow_mo=5000)
        page = browser.new_page()
        page.goto("https://www.universitiesegypt.com/")
        
         # Extract only public universities
        page.get_by_text("Public Universities").nth(0).click()
        page.wait_for_load_state("networkidle")

        # Gather all university URLs
        self.get_all_universities_links(page)

        # Loop through universities
        for href in self.all_links:
            page.goto(href)
            menu = page.locator(".leftMenu")
            university_name = menu.locator("h1").text_content()
            detailed_data = page.locator(".newsListDate")
            research_centers_availability = detailed_data.locator("i:has-text('Research Centers Availability') span").text_content()
            number_of_students = detailed_data.locator("i:has-text('Number of Students') span").text_content()
            number_of_staff = detailed_data.locator("i:has-text('Number of Staff') span").text_content()
            gender = detailed_data.locator("i:has-text('Gender:') span").text_content()
            university_about = page.locator(".blockAll").text_content()
            rating = page.locator("#ctl00_ctl00_ContentPlaceHolder1_ContentPlaceHolder1_spAvgRating").text_content()


            # Faculties Page
            faculties_item = menu.locator("text=Faculties / Programs")
            faculties_item.click()

            page.wait_for_load_state("networkidle")
            self.faculties_scraper(page)

            # Contact Page
            page.wait_for_load_state("networkidle")
            contact_item = menu.locator("text=Contacts")
            contact_item.click()
            self.contact_scraper(page)

            # Append to dataset
            self.universities_data.append({
                "university_name": university_name,
                "about": university_about,
                "research_centers_availability": research_centers_availability,
                "number_of_students": number_of_students,
                "number_of_staff": number_of_staff,
                "gender": gender,
                "rating": rating,
                "type": "public",
                "faculties": self.faculties,
                "contact_info": self.contact_info
            })

        print(self.universities_data)

        browser.close()
        pw.stop()
        print("Scraping completed successfully") 

    
    def save_data_to_json(self, file_name="raw_universities_data.json"):
        """
        Save the scraped data into a JSON file inside the raw data folder.

        Args:
            file_name (str): Name of the JSON file to create.
        """
        os.makedirs(self.raw_data_folder, exist_ok=True)
        full_path = os.path.join(self.raw_data_folder, file_name)
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(self.universities_data, f, ensure_ascii=False, indent=4)
        print(f"Data saved to {full_path}")



if __name__ == "__main__":
    scraper = WebScraper()
    scraper.scrap_public_universities()
    scraper.save_data_to_json()