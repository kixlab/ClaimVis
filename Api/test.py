import spacy
from spacy import displacy
from datetime import datetime

def tag_date_time(text: str):
    # Parse date time from the claim
    # Use NLP libraries to extract date from user_claim
    nlp = spacy.load("en_core_web_sm")
    
    doc = nlp(text)

    replaced_text = ""
    for token in doc:
        if token.ent_type_ == "DATE":
            replaced_text += "{date} "
        else:
            replaced_text += token.text + " "

    print(replaced_text.strip())

if __name__ == "__main__":
    tag_date_time("Some people are crazy enough to get out in the winter, especially november and december where it's freezing code outside.")