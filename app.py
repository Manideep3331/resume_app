import streamlit as st
import pandas as pd
pip install scikit-learn
import sklearn
#from sklearn.linear_model import LinearRegression  # Example

model = sklearn.linear_model.LinearRegression() # Accessing using sklearn. prefix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2  # For PDF parsing
import io

# Function to extract text from a PDF file
def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

# Function to process resumes and calculate similarity
def process_resumes(resumes):
    vectorizer = TfidfVectorizer(stop_words='english')  # Remove common English words
    vectors = vectorizer.fit_transform(resumes)
    similarity_matrix = cosine_similarity(vectors)
    return similarity_matrix

st.title("Resume Matcher")

uploaded_files = st.file_uploader("Upload Resumes (PDFs)", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    resumes = []
    filenames = []  # Store filenames for display
    for uploaded_file in uploaded_files:
        text = extract_text_from_pdf(uploaded_file)
        if text:
            resumes.append(text)
            filenames.append(uploaded_file.name)

    if resumes:
        with st.spinner("Processing resumes..."): # Display a spinner while processing
            similarity_matrix = process_resumes(resumes)

        st.write("Similarity Matrix:")
        st.dataframe(pd.DataFrame(similarity_matrix, index=filenames, columns=filenames))

        # Improved display of top matches
        st.subheader("Top Matches:")
        for i, row in enumerate(similarity_matrix):
            # Sort by similarity, exclude self-comparison
            top_matches = sorted(enumerate(row), key=lambda x: x[1], reverse=True)[1:]
            if top_matches: # Check if there are any matches other than self
                st.write(f"**{filenames[i]}**:")
                for match_index, similarity_score in top_matches:
                    st.write(f"- Matched with **{filenames[match_index]}**: {similarity_score:.2f}")
            else:
                st.write(f"**{filenames[i]}**: No other matches found.")
else:
    st.info("Please upload PDF resumes to begin.")
