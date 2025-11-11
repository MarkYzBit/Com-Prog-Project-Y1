import cv2
import pytesseract
from thefuzz import process, fuzz
import matplotlib.pyplot as plt
import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
import os

def preprocess_image_bytes(url, target_dpi_scale = 1.5):
    img = cv2.imread(url)
    if target_dpi_scale > 1.0:
        img = cv2.resize(img, None, fx=target_dpi_scale, fy=target_dpi_scale, interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_contrast = clahe.apply(img_gray)
    img_blur = cv2.bilateralFilter(img_contrast, 12, 75, 75)
    img_final = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)
    return img_final

def ocr_text_from_bytes(processed_image, lang='eng'):
    img_final = preprocess_image_bytes(processed_image)
    if img_final is None:
        return 'Could not preprocess image'
    config = f'--oem 3 --psm 6 -l {lang}'
    text = pytesseract.image_to_string(img_final, config=config)
    return text

def normalize_text(img):
    normalized_text = ocr_text_from_bytes(img).lower().replace('\n', ' ')
    keywords1 = [
        "ingredients:",
        "ingredients list:",
        "inci:",
        "inci list:",
        "ingredients",
        "ingredients list"
    ]
    keywords2 = [
        "water",
        "aqua",
        "water/aqua",
        "water/ aqua",
        "water / aqua",
        "aqua/water",
        "glycerin",
        "butylene glycol",
    ]
    final_start_index = -1
    keyword_found = None
    header_end_index = -1
    for keyword in keywords1:
        index = normalized_text.find(keyword)
        if index != -1:
            header_end_index = index + len(keyword)
            keyword_found = keyword
            break
    search_area = ''
    if header_end_index != -1:
        search_area = normalized_text[header_end_index:]
    else:
        seach_area = normalized_text
    first_ing_index = -1
    for keyword in keywords2:
        index = search_area.find(keyword)
        if index != -1:
            first_ing_index = index
            keyword_found = keyword
            break
    if header_end_index != -1:
        if first_ing_index != -1:
            final_start_index = header_end_index + first_ing_index
        else:
            final_start_index = header_end_index
    else:
        if first_ing_index != -1:
            final_start_index = first_ing_index
        else:
            print('No keyword found')
    tokens = []
    if final_start_index != -1:
        ingredient_blob = normalized_text[final_start_index:]
        end_index = ingredient_blob.find('.')
        if end_index >= int(len(ingredient_blob)/2):
            ingredient_blob = ingredient_blob[:end_index]
        unwanted_char = [':', '[', ']', '(', ')', "'"]
        for char in unwanted_char:
            ingredient_blob = ingredient_blob.replace(char, '').replace('!', 'l')
        raw_list = ingredient_blob.split(', ')
        for item in raw_list:
            cleaned_item = item.strip()
            if cleaned_item:
                tokens.append(cleaned_item)
    else:
        print('No keyword Ingredient')
    return tokens

def fuzzy_match(tokens):
    MASTER_INGREDIENTS = [
        "water", "aqua", "glycerin", "niacinamide", "butylene glycol", "glutathione",
        "oryza sativa (rice) bran extract", "dimethicone", "retinol", "hyaluronic acid",
        "cyclopentasiloxane", "allantoin", "phenoxyethanol", "ethylhexylglycerin",
        "ascorbyl glucoside", "fragrance", 'bht', 'aha', 'bha', 'oleate', 'magnolol',
        "ethylhexylglycerin", "butyrospermum parkii butter", "c13-14 isoparaffin",
        "phytosphingosine", "potassium phosphate", "diisopropyl adipate", "ascorbic acid",
        "ascorbic"
    ]
    corrected_tokens = []
    threshold = 80
    for token in tokens:
        best_match, score = process.extractOne(token, MASTER_INGREDIENTS)
        if score >= threshold:
            corrected_tokens.append(best_match)
        else:
            corrected_tokens.append(token)
    return corrected_tokens

def extract_user_ingredients_from_image(image_path, max_tokens=50):
    tokens = normalize_text(image_path)
    if not tokens:
        return []
    tokens = tokens[:max_tokens]
    ingredients = fuzzy_match(tokens)
    return ingredients

def generate_summary(image_path, csv_path=None):
    tokens = normalize_text(image_path)
    if not tokens:
        return []
    tokens = tokens[:5]
    ingredients = fuzzy_match(tokens)
    user_ingredients = ingredients

    # load and standardize dataset
    if csv_path is None:
        base_dir = os.path.dirname(__file__)
        csv_path = os.path.join(base_dir, 'ingredients.csv')

    df = pd.read_csv(csv_path)
    df['Ingredient'] = df['Ingredient'].astype(str).str.strip().str.lower()
    df = df.drop_duplicates(subset=['Ingredient']).reset_index(drop=True)
    
    #prepare spaCy matcher
    nlp = spacy.load('en_core_web_sm')
    matcher = PhraseMatcher(nlp.vocab, attr='LOWER')

    #create patterns for ingredient names
    patterns = [nlp.make_doc(ing) for ing in df['Ingredient']]
    matcher.add('INGREDIENTS', patterns)

    #match user ingredients to KB
    user_text = ', '.join(user_ingredients).lower()
    doc = nlp(user_text)
    matches = matcher(doc)

    matched_names = set([doc[start:end].text.lower() for match_id, start, end in matches])

    #filter matched ingredients with dataset
    matched_df = df[df['Ingredient'].isin(matched_names)].copy()

    #summarize and recommend
    if not matched_df.empty:
        avg_score = matched_df['Skin_Friendliness_Score'].mean()

        props = ', '.join(matched_df['Properties'].dropna().unique())
        avoid = ', '.join(matched_df['Avoid_With'].dropna().unique())
        rec_with = ', '.join(matched_df['Recommended_With'].dropna().unique())
        func_cats = ', '.join(matched_df['Function_Category'].dropna().unique())

        summary = f"""
🧴 **Skincare Recommendation Summary**
---------------------------------------
✅ Core Ingredients: {', '.join(matched_df['Ingredient'].tolist())}

🧠 Key Properties: {props}

💡 Functions: {func_cats}

🌟 Average Skin Friendliness Score: {avg_score:.2f} / 5

🚫 Avoid Combining With: {avoid}

🤝 Works Well With: {rec_with}

✨ Recommendation:
These ingredients are generally beneficial for {', '.join(matched_df['Skin_Type'].unique())} skin types. 
Ensure you avoid conflicting actives and layer hydration before strong treatments like retinol or acids.
        """
    else:
        summary = "Try Again."

    return summary
