import re
import json
import os 
import sys
from langdetect import detect
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import MarianMTModel, MarianTokenizer
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Add project root to sys.path for relative imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)


class TextProcessor():
    """
    TextProcessor handles loading, cleaning, translating, flattening, and chunking 
    of university data for further use in RAG (Retrieval-Augmented Generation) pipelines.
    """

    def __init__(self):
        self.full_path_of_raw_data = os.path.join(project_root, "data" , "raw" , "raw_universities_data.json")
        self.full_path_of_processed_folder = os.path.join(project_root , "data" , "processed")

        # Load MarianMT translation model for Arabic -> English
        model_name = "Helsinki-NLP/opus-mt-ar-en"
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)

        # Data containers
        self.universities_data = []
        self.universities_processed_data = []

    def load_data(self, full_path):
        """
        Load JSON data from a file.

        Args:
            full_path (str): Path to the JSON file.

        Returns:
            list: Loaded JSON raw data.
        """
        with open(full_path, 'r', encoding='utf-8') as file:        
            return json.load(file)
    
    def remove_punctuation(self, text):
        """
        Remove all punctuation from a string, except apostrophes.

        Args:
            text (str): Input string.

        Returns:
            str: Cleaned string.
        """
        return re.sub(r"[^\w\s']", '', text)

    def removing_extra_whitespace(self, text):
        """
        Normalize whitespace by removing extra spaces and newlines.

        Args:
            text (str): Input string.

        Returns:
            str: Normalized string.
        """
        return ' '.join(text.replace('\n', '').split())
    
    def is_arabic(self, text: str) -> bool:
        """
        Check if a string contains Arabic characters.

        Args:
            text (str): Input string.

        Returns:
            bool: True if Arabic characters are found.
        """
        arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
        return arabic_pattern.search(text)

    def translate_if_arabic(self, text: str) -> str:
        """
        Translate Arabic text to English using MarianMT.

        Args:
            text (str): Input string.

        Returns:
            str: Translated string (or original if not Arabic).
        """
        if not self.is_arabic(text):
            return text
        try:
            inputs = self.tokenizer([text], return_tensors="pt", padding=True)
            translated = self.model.generate(**inputs)
            return self.tokenizer.decode(translated[0], skip_special_tokens=True)
        except Exception as e:
            print(f"Translation failed for: {text[:50]}... Error: {e}")
            return text
        
    def normalization(self):
        """
        Load raw data and normalize it:
        - Translate Arabic text to English.
        - Remove punctuation and extra whitespace.
        - Convert text to lowercase.
        """
        self.universities_data  = self.load_data(self.full_path_of_raw_data)
        for university in self.universities_data:
            university['university_name'] = university['university_name'].strip().lower()

            university['about'] = self.translate_if_arabic(university['about'])
            university['about'] = self.removing_extra_whitespace(university['about']).lower()
            university['about'] = self.remove_punctuation(university['about'])
            
            university['research_centers_availability'] = university['research_centers_availability'].lower()
            university['gender'] = university['gender'].lower()
            university['rating'] = self.remove_punctuation(university['rating'])

            for faculty in university['faculties']:
                faculty['name'] = self.translate_if_arabic(faculty['name'])
                faculty['name'] = self.removing_extra_whitespace(faculty['name']).lower()
                faculty['name'] = self.remove_punctuation(faculty['name'])

                faculty['about'] = self.translate_if_arabic(faculty['about'])
                faculty['about'] = self.removing_extra_whitespace(faculty['about']).lower()
                faculty['about'] = self.remove_punctuation(faculty['about'])
                
            for contact in university['contact_info']:
                contact['contact_name'] = self.translate_if_arabic(contact['contact_name'])
                contact['contact_name'] = self.remove_punctuation(contact['contact_name']).lower()
                contact['contact_info'] = self.translate_if_arabic(contact['contact_info'])
                contact['contact_info'] = self.removing_extra_whitespace(contact['contact_info']).lower()

        self.save_data_into_processed_folder(self.universities_data, "processed_universities_data.json")
        
    def flatting_json(self):
        """
        Flatten the university JSON structure into a single text string per university
        and save as individual TXT files for RAG.
        """
        full_path = os.path.join(self.full_path_of_processed_folder, "processed_universities_data.json")
        self.universities_processed_data = self.load_data(full_path)
        flattened = []

        for university in self.universities_processed_data:
            # Flatten faculties
            faculties = [
                f"{f['name']} at {university['university_name']}: {f['about']}"
                for f in university['faculties']
            ]
            faculty_text = "\n".join(faculties)

            # Flatten contact info
            contacts = university['contact_info']
            contact_texts = [f"{c['contact_name']}: {c['contact_info']}" for c in contacts]
            contact_text = "; ".join(contact_texts)

            # Build flattened text
            flattened.append({
                "text": (
                    f"{university['university_name']}: {university['about']}\n\n"
                    f"research centers availability: {university['research_centers_availability']}\n\n"
                    f"number of students: {university['number_of_students']}\n\n"
                    f"number of staff: {university['number_of_staff']}\n\n"
                    f"gender: {university['gender']}\n\n"
                    f"rating: {university['rating']}\n\n"
                    f"type: {university['type']}\n\n"
                    f"{faculty_text}\n\n"
                    f"Contact information for {university['university_name']}: {contact_text}"
                ),
                "metadata": {
                    "university_name": university["university_name"],
                    "source": "https://www.universitiesegypt.com/",
                    "scrapping_date": "28-9-2025",
                    "type": university['type']
                }
            })

            # Save each university as a TXT file
            self.save_university_docs_data(flattened[-1]['text'], university['university_name'])

        # Save the flattened JSON
        self.save_data_into_processed_folder(flattened, "flattened_universities.json")

    def chunking(self):
        """
        Create chunks from flattened university texts for RAG.
        Uses RecursiveCharacterTextSplitter from LangChain.
        """
        full_path = os.path.join(self.full_path_of_processed_folder, "flattened_universities.json")
        universities_data = self.load_data(full_path)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,      
            chunk_overlap=100 
        )

        all_chunks = []

        for university in universities_data:
            chunks = text_splitter.create_documents(
                [university["text"]],
                metadatas=[university["metadata"]]  
            )
            all_chunks.extend(chunks)

        # Save chunks as JSON
        all_chunks_data = [
            {"text": doc.page_content, "metadata": doc.metadata} 
            for doc in all_chunks
        ]
        self.save_data_into_processed_folder(all_chunks_data,"university_docs.json")
        
    def save_university_docs_data(self, university_data: str, university_name: str):
        """
        Save individual university text as a PDF file.

        Args:
            university_data (str): The flattened text of the university.
            university_name (str): Name of the university.
        """
        folder_path = os.path.join(project_root, "data", "docs")
        os.makedirs(folder_path, exist_ok=True)

        # Make a safe filename
        safe_name = university_name.replace(' ', '_')
        filename = f"{safe_name}.pdf"
        full_file_path = os.path.join(folder_path, filename)

        # Create a PDF using ReportLab
        c = canvas.Canvas(full_file_path, pagesize=letter)
        width, height = letter

        # Split long text into lines that fit in the page
        y = height - 50 
        line_height = 14
        max_chars_per_line = 90

        for paragraph in university_data.split("\n"):
            for line in [paragraph[i:i+max_chars_per_line] for i in range(0, len(paragraph), max_chars_per_line)]:
                c.drawString(50, y, line)
                y -= line_height
                if y < 50:  # start new page if we reach bottom
                    c.showPage()
                    y = height - 50

        c.save()


    def save_data_into_processed_folder(self, data, file_name):
        """
        Save data into the processed folder as JSON.

        Args:
            data (any): Data to save (list or dict).
            file_name (str): Filename for the JSON file.
        """
        full_path = os.path.join(self.full_path_of_processed_folder , file_name)
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    text_pr = TextProcessor()
    text_pr.normalization()
    text_pr.flatting_json()
    text_pr.chunking()

