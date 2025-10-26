import pandas as pd
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from io import StringIO
import os
!pip install langdetect
from langdetect import detect, DetectorFactory
import nltk

# Install required libraries and download NLTK data
try:
    print("Installing required libraries...")
    # Install pandarallel
    get_ipython().system('pip install pandas openpyxl nltk langdetect pandarallel')
except NameError:
  print("If not in Colab environment, please run 'pip install ...' manually.")

# Fix random seed for deterministic behavior of langdetect
DetectorFactory.seed = 42

# Download NLTK resources
print("Downloading NLTK data...")
try:
    nltk.download('punkt', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    print("NLTK data download complete.")
except Exception as e:
    print(f"NLTK download error: {e}")

try:
    from pandarallel import pandarallel
    # Initialize pandarallel
    pandarallel.initialize(progress_bar=True)
    PANDARALLEL_READY = True
    print("\n[Optimization] Pandarallel initialized successfully. Using parallel processing for language detection.")
except Exception as e:
    # Fallback to sequential processing
    print(f"\n[Optimization] Pandarallel initialization failed: {e}. Using sequential processing.")
    PANDARALLEL_READY = False

# Global variables
COMMENT_COLUMN = 'Comment_Content'
KEYWORD_COLUMN_NAME = 'Keyword'

def load_custom_stopwords(filepath):
    """Load custom stopword list from a file or prompt upload in Colab."""
    # Start with default NLTK English stopwords
    custom_stopwords = set(stopwords.words('english'))

    # Check Colab environment and request upload
    try:
        from google.colab import files
        print("\n[Upload Stopword File] Please upload 'custom_stopwords.txt'.")
        uploaded = files.upload()

        if not uploaded:
            print(f"Warning: No file uploaded. Using default NLTK stopwords ({len(custom_stopwords)} words).")
            return custom_stopwords

        # Use the first uploaded file
        uploaded_filename = list(uploaded.keys())[0]
        print(f"Loading stopwords from uploaded file '{uploaded_filename}'.")

        # Read uploaded file as string and build stopword set
        file_content = uploaded[uploaded_filename].decode('utf-8')
        custom_stopwords = set(line.strip().lower() for line in file_content.splitlines() if line.strip())

    except Exception as e:
        print(f"Error while loading stopwords: {e}. Using default NLTK stopwords.")

    print(f"Loaded {len(custom_stopwords)} custom stopwords successfully.")
    return custom_stopwords

def detect_language(text):
    if not isinstance(text, str):
        return 'too_short'

    clean_text = str(text).strip()

    # Force 'en' for short text (less than 3 words or 10 chars) to prevent detection errors
    if len(clean_text.split()) < 3 or len(clean_text) < 10:
        return 'en'

    try:
        return detect(clean_text)
    except:
        return 'error'  # Return 'error' if detection fails

def preprocess_text(text, custom_stopwords):
    """Preprocess text: lowercase, remove mentions/URLs/hashtags/emojis, punctuation removal, stopword removal, tokenization."""
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove @mentions
    text = re.sub(r'@\w+', '', text)

    # 3. Remove URLs
    text = re.sub(r'https?://\S+', '', text, flags=re.MULTILINE)

    # 4. Remove hashtags
    text = re.sub(r'#\w+', '', text)

    # 5. Remove emojis using Unicode range
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
        "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
        "\U0001F1E0-\U0001F1FF"  # Flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    text = emoji_pattern.sub(r'', text)

    # 6. Remove special characters and punctuation (keep only alphabets and spaces)
    text = re.sub(r'[^\w\s]', '', text)

    # 7. Tokenize text
    tokens = word_tokenize(text)

    # 8. Remove stopwords and short words (length 1)
    processed_tokens = [
        word for word in tokens
        if word not in custom_stopwords and len(word) > 1
    ]

    return " ".join(processed_tokens)

# Download NLTK resources
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('vader_lexicon')
nltk.download('stopwords')

def perform_sentiment_analysis(input_excel_path, stopwords_file_path):
    """
    Load Excel file, perform language detection, preprocessing, and VADER sentiment analysis.
    """
    try:
        from google.colab import files

        # 1. Request data file upload if not found
        if not os.path.exists(input_excel_path):
            print(f"\n[Upload Data File] File '{input_excel_path}' not found. Please upload the Excel file.")
            data_uploaded = files.upload()

            # Verify upload
            if input_excel_path not in data_uploaded:
                # Use first uploaded file if filename differs
                if data_uploaded:
                    uploaded_key = list(data_uploaded.keys())[0]
                    if uploaded_key.endswith('.xlsx') or uploaded_key.endswith('.xlsm'):
                        input_excel_path = uploaded_key
                        print(f"Using uploaded file '{uploaded_key}' as data file.")
                    else:
                        print("Error: Uploaded file is not an Excel file.")
                        return
                else:
                    print("Error: Data file upload canceled.")
                    return

        # 2. Load data
        print(f"\n[Data Loading] Loading all sheets from '{input_excel_path}'...")
        df_dict = pd.read_excel(input_excel_path, engine='openpyxl', sheet_name=None)

        if not df_dict:
            print(f"Error: No sheets found in '{input_excel_path}'.")
            return

        # Merge all sheets into one DataFrame
        df = pd.concat(df_dict.values(), ignore_index=True)

        # Fill merged cells with previous valid values
        df[KEYWORD_COLUMN_NAME] = df[KEYWORD_COLUMN_NAME].fillna(method='ffill')

    except FileNotFoundError:
        print(f"Error: File '{input_excel_path}' not found.")
        return
    except Exception as e:
        print(f"Error while loading data: {e}")
        return

    # Check for required columns
    required_cols = [KEYWORD_COLUMN_NAME, COMMENT_COLUMN]
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Missing required column '{col}'. Columns found: {df.columns.tolist()}")
            return

    print(f"Loaded {len(df)} rows successfully (merged from all sheets).")

    # 3. Language detection
    print("Performing language detection...")

    # Use parallel or sequential processing
    global PANDARALLEL_READY
    if PANDARALLEL_READY:
         df['detected_language'] = df[COMMENT_COLUMN].parallel_apply(detect_language)
    else:
         df['detected_language'] = df[COMMENT_COLUMN].apply(detect_language)

    # Filter English comments only
    df_english = df[df['detected_language'] == 'en'].copy()
    print(f"Filtered {len(df_english)} English ('en') comments successfully.")

    if df_english.empty:
        print("No valid English comments found for analysis.")
        return

    # 4. Load custom stopwords
    custom_stopwords = load_custom_stopwords(stopwords_file_path)

    # 5. Preprocess text
    print("Preprocessing text...")
    df_english['processed_text'] = df_english[COMMENT_COLUMN].apply(lambda x: preprocess_text(x, custom_stopwords))

    # 6. Validate after preprocessing
    df_final = df_english[df_english['processed_text'].str.len() > 0].copy()
    print(f"{len(df_final)} comments ready for final analysis after preprocessing.")

    if df_final.empty:
        print("No valid text remains after preprocessing.")
        return

    # 7. Initialize VADER and calculate sentiment scores
    sid = SentimentIntensityAnalyzer()
    print("Performing VADER sentiment analysis...")
    df_final['scores'] = df_final['processed_text'].apply(sid.polarity_scores)

    df_final['neg'] = df_final['scores'].apply(lambda x: x['neg'])
    df_final['neu'] = df_final['scores'].apply(lambda x: x['neu'])
    df_final['pos'] = df_final['scores'].apply(lambda x: x['pos'])
    df_final['compound'] = df_final['scores'].apply(lambda x: x['compound'])

    # Assign sentiment labels
    def get_sentiment(score):
        if score >= 0.05: return 'Positive'
        elif score <= -0.05: return 'Negative'
        else: return 'Neutral'

    df_final['sentiment'] = df_final['compound'].apply(get_sentiment)

    # 8. Output results
    print("\n" + "="*50)
    print("FINAL ANALYSIS RESULTS")
    print("="*50)

    print("\n--- Overall Sentiment Summary (Compound Score) ---")
    print(df_final['sentiment'].value_counts())

    print("\n--- Average VADER Scores per Keyword ---")

    # Calculate mean VADER scores by Keyword
    score_cols = ['neg', 'neu', 'pos', 'compound']
    keyword_summary = df_final.groupby(KEYWORD_COLUMN_NAME)[score_cols].mean()

    # Add comment count
    keyword_summary['Comment_Count'] = df_final.groupby(KEYWORD_COLUMN_NAME).size()

    # Round to 3 decimal places
    keyword_summary = keyword_summary.round(3)

    # Sort by compound score
    keyword_summary = keyword_summary.sort_values(by='compound', ascending=False)

    # Rename columns
    keyword_summary.columns = ['Avg_Negative_Score', 'Avg_Neutral_Score', 'Avg_Positive_Score', 'Avg_Compound_Score', 'Comment_Count']

    print(keyword_summary)

    print("\n--- Top 5 Positive Comments (with Keyword) ---")
    top_positive = df_final.sort_values(by='compound', ascending=False).head(5)
    print(top_positive[[KEYWORD_COLUMN_NAME, COMMENT_COLUMN, 'compound', 'sentiment']].to_string(index=False))

    print("\n--- Top 5 Negative Comments (with Keyword) ---")
    top_negative = df_final.sort_values(by='compound', ascending=True).head(5)
    print(top_negative[[KEYWORD_COLUMN_NAME, COMMENT_COLUMN, 'compound', 'sentiment']].to_string(index=False))

    # 9. Save results (.xlsx)
    output_path = 'Sentiment_Analysis_Results.xlsx'

    # Save both detailed and summary sheets
    output_cols = [KEYWORD_COLUMN_NAME, COMMENT_COLUMN, 'detected_language', 'processed_text', 'neg', 'neu', 'pos', 'compound', 'sentiment']

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # All comments sheet
        df_final.to_excel(writer, index=False, sheet_name='All_Comments', columns=output_cols)
        # Keyword summary sheet
        keyword_summary.to_excel(writer, sheet_name='Keyword_Summary')

    print(f"\nResults saved to '{output_path}' with 2 sheets (All_Comments, Keyword_Summary).")

    # Automatic Excel download in Colab
    from google.colab import files
    files.download(output_path)

INPUT_FILE = 'Comments.xlsm'
STOPWORDS_FILE = 'custom_stopwords.txt'
perform_sentiment_analysis(INPUT_FILE, STOPWORDS_FILE)
