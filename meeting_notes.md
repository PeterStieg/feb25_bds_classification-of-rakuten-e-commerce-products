# feb25_bds_classification-of-rakuten-e-commerce-products

## Meeting Notes: May 5th

1. **Streamlit**
    1. Text preprocessing:
       1. Fix errors on 4. Question marks
       2. Fix errors on 5. Parentheses
    2. Image preprocessing: Add preview for images shown in DataFrame
    3. Large files, e.g. text data `df_text_clean.csv` and image data `vgg16_transfer_model.pth`
        1. Host on Google Cloud Platform
        2. Add model(s) to st.cache_data
        3. Add progress bar

## Meeting Notes: April 22nd

1. **Mandatory task**: Text preprocessing
    1. Save vectorizers in pickle file
    2. @Wilsbert12: Could you please correct these notes and fill in the gaps?
    3. Time estimate:
        1. Not more than 5 hours for this part of project
        2. Not more than 2 hours for training
    3. deepL_result.csv: Temporary file we can probably include in .gitignore @thomas-borer
    4. df_localization.csv: Latest and greatest in terms of localization 

## Meeting Notes: April 17th
IMPORTANT NOTE: We are running very late. Defense is on the week of the 12th of may. We still do not have the complete translation.
**IMPORTANT TASK: GET TRANSLATION READY ASAP!**

1. **Traditional ML:**
    - Problem: Long computing times
    - Solution:
        - Vectorization before pipeline -> save & reuse
        - One set of Hyperparameters per vectorization only (do more if time allows)
        - deepl_results ist die falsche Datei als Input -> richtige Datei?
        - Use resources like Google Colab to access GPU -> speed x10




## Meeting Notes: April 17th

`NOTE: Time left until deadline May 2nd: Two weeks`

1. **Mandatory task**: Text preprocessing
    1. Modelization via vectorization of sentences with TF-IDF, Word2VEc, BoW (Bag of Words)
    2. Modelizzation with two approaches: logistic regression, random forest, XGBoost, etc. and then in a second step with BERT, e.g. CamemBERT or FlauBERT
3. **Optional task**: Image preprocessing
    1. Bounding box: Grayscale images, binary threshold with 235, find contours, optional margins
    2. Standardization: Removing white borders with cv2
    3. Downsampling: Probably most common resolution of data set, 250px or 299px
 3. **Optional task**: Image modelization with three approaches
    1. Classical way:
    2. CNNs: ResNet, EfficientNet
    3. Transformers: ViT (Vision Transformer), DTR (Meta AI)
 4. **Mandatory task**: Voting classifier which combines the result of text and image models  
 5. **Next steps**: Split work into
    1. Text classification
    3. Image preprocessing
    4. Presentation in Streamlit


## Meeting Notes: April 8th

**Peter's questions**

1. Flags:
   - [x] 0,1 instead of True, False

2. Descriptions:
   - [ ] Drop errors, e.g. "<br>Attention !!! Ce produit est un import [...]"

   >Yaniv's feedback 11apr2025: _'yes you should wait to see if it's a minority class or not. We would delete with ease if it's in a majority class, otherwise we should keep it because we need more datas'_

4. Merged text:
   - [x] Length
   - [x] Number of words

5. Cleaned string variables
   - [ ] Similarity function
   - [ ] Drop similar descriptions or wait for under- and oversampling?

---
## Meeting notes: April 3rd
1. [Step 2] // Preprocessing and feature engineering #7
   - [x] 1. Do not substitute accents
   - [x] 2. Do not convert everything to lowercase (yet)
   3. Set flag if...
      - [x] 1. ... description is missing
      - [x] 2. ... string is all upper case
      - [x] 3. ... if text contains encoding error
      - [x] 4. ... if title and description are identical
   8. Do not merge strings if title and description are 95% identical checking:
      1. Lower case vs upper case
      2. Multiple spaces

---
## Meeting notes: April 2nd
1. *GitHub // General understanding & bug fixing*
2. *Presentation of Thomas' progress* on language
4. *Hybrid solution of over- und undersampling* after having defined a threshold
   1. Undersampling: Filter, e.g. best text, length of text, keyword frequency
   2. Oversampling: Generate new text using [ChatGPT API](https://openai.com/api/)
5. *Categories:*
   1. See each product type category as individual category
   2. Transform categories similar to date in car insurance example
6. *Processing of Images* with cv2

---
## Meeting notes: March 27th
1. *Preprocessing // General:* Create PY file for proprecessing
2. *Preprocessing // DataFrame:* Add boolean column for description_available
3. *Preprocessing // DataFrame:* Rename column names according to standards
4. *Preprocessing // Images:* Delete white background, create images with same ratio
5. *DataViz:* Create separate Notebook for Data Visualization

---
## Meeting notes: March 25th
1. **DataFrame:** Emphasis on using _[langdetect](https://pypi.org/project/langdetect/)_ to get probability of language
2. **DataFrame:** Merge _designation_ and _description_ in additional text field
3. **Order of steps:**
   1. Cleaning
   2. Translation
   3. Normalization
4. **Images:**
   1. Delete blank space
   2. Normalize ratio of images
   3. Goal for next week: Set bounding box
5. **Approach:**
  1. Choose one language, e.g. English
  2. Translate other languages via _DeepL_'s API
  3. TF-IDF: Define importance of word (or sentence)
  4. Apply text mining techniques only after, e.g. lemmatization
  5. Define threshold for designation and description via mean, median, etc. to exclude text, which is too long (@RW, @TB: Did I miss anything important?)
  6. Goal is to Word2Vec (CBOW, skip-gram), BERT and Meta's alternative (?)

## Additional Notes from Yaniv:
1. Continue your work on the preprocessing tasks on the images and the text as :
      1. Delete HTML tags and special caracters on the text. Pay attention to some outliers.
      2. Merge description and designation columns to remove missing values
      3. For the translation, we can try to do everything in english thanks to the DeepL API.
      4. For the images, we can work on the ratio of the images as the size of the bouding box around the image. You should find something arount 20% of images below a ratio of 0.8. These images will be zoomed if it's possible.
2. I liked the data viz you have presented but I can propose to do some others about :
   1. Class distribution
   2. images by category, notice that they are oddly made
   3. Occurrence of words by category
   4. Word cloud (module 131 Text Mining) on each category or on the whole column designation+description

---
## Meeting notes: March 18th

**A. To-Dos // Preprocessing**
1. **DataFrame:** Detect languages e.g. FRA, ENG, GER, etc. in designation and description with _[langdetect](https://pypi.org/project/langdetect/)_
2. **DataFrame:** Are there mixed languages in designation and description, e.g. FRA & ENG, FRA & GER, etc.
3. **DataFrame:** Merge product type with product type names if possible with [Liste des catégories (categorymap)](https://global.fr.shopping.rakuten.com/developpeur/liste-des-categories-categorymap/)
4. **DataFrame:** Are there duplicate items, e.g. product id, designation, description?
5. **DataFrame:** Add TF-IDF (Term Frequency-Inverse Document Frequency) as proxy for text quality.
6. **DataFrame:** Are there short _(meaningless)_ text (designations or descriptions)?
7. **DataFrame:** Are there long texts due to keyword spamming meaning duplicated, repetitive text?
8. **DataFrame:** Are there formatting errors and html in the text, e.g. '&uuml;' or à?

**B. Backlog**
- **Images:** Sizes, resolution, ratios, blank images (borders).
- **Plot:** Visualize NaNs per product type.
- **Research:** Reduce vector size of text.

**C. Verschiedenes**
- [Normalform von Datenbanken](https://www.tinohempel.de/info/info/datenbank/normalisierung.htm)

---
## Meeting notes: March 18th

**A. WordCloud**
1. Stopwords filtern
2. Mehrsprachigkeit ggf. beachten

**B. Visualizations:**
1. Fünfte Grafik mit Matrix aus numerischen Werten aufgrund fehlender Daten nicht möglich

**C. Weitere mgl. Fragen & Grafiken:**
1. Verteilung von NaNs in Beschreibung pro Product Type
2. Gibt es fehlende Bilder?
3. Verteilung von Auflösung?
4. Wahrscheinlichkeit, dass fehlende Textinhalten (s. o.) zu fehlenden Bildern bzw. niedrig-auflösenden Grafiken führen?

**D. Verschiedenes**
1. [GitHub Projects](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects): Projektmanagement à la Kanban mit GANTT-Charts
2. [GitHub Codespace](https://github.com/features/codespaces): Programmierumgebung
3. [Formattierungsmöglichkeiten in Markdown](http://markdownguide.org/)

**E. Tasks**
1. Thomas: Wordclouds
2. Robert: Distribution of Product Type Codes (Count plot), Distribution of Title Lengths
3. Peter: Sample images from different product types, Missing images, Image resolution
