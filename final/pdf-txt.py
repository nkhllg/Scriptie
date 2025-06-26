import fitz  # PyMuPDF
import os
import glob
import re

# List of country names (this is just a partial list, you can expand it)
COUNTRIES = [
    "Singapore", "Australia", "New Zealand", "Japan", "Hong Kong", "South Korea"
]

def extract_country_from_text(text):
    # Try to find a country name from the list of countries in the text
    for country in COUNTRIES:
        if re.search(r'\b' + re.escape(country) + r'\b', text, re.IGNORECASE):
            return country
    return "Unknown"  # Return 'Unknown' if no country is found

def pdf_to_text(pdf_path, txt_path):
    try:
        doc = fitz.open(pdf_path)
        metadata = doc.metadata
        
        text = ""
        for page in doc:
            text += page.get_text()
        
        country = extract_country_from_text(text)
        metadata['country'] = country
        
        with open(txt_path, 'w', encoding='utf-8') as f_out:
            f_out.write(f"Metadata for {pdf_path}: {metadata}\n\n")
            f_out.write(f"Country found in {pdf_path}: {country}\n\n")
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                f_out.write(page_text)
                f_out.write('\n' + '-'*80 + '\n')

        print(f"Text successfully saved to '{txt_path}'")
    except Exception as e:
        print(f"Error with '{pdf_path}': {e}")


if __name__ == "__main__":
    pdf_folder = "pdf/Austalia/Victoria"
    txt_folder = "txt/Australia/Victoria"

    os.makedirs(txt_folder, exist_ok=True)  # ensure output folder exists

    pdf_files = glob.glob(os.path.join(pdf_folder, "*.pdf"))

    for pdf_file in pdf_files:
        filename = os.path.splitext(os.path.basename(pdf_file))[0]
        txt_file = os.path.join(txt_folder, f"{filename}.txt")
        pdf_to_text(pdf_file, txt_file)
