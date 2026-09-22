import os
import re
import math
import time

import streamlit as st
from nltk.stem import PorterStemmer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Mini Search Engine",
    page_icon="🔎",
    layout="wide"
)


# ============================================================
# INITIAL SETUP
# ============================================================

stemmer = PorterStemmer()
DATA_FOLDER = "data"

# ============================================================
# SESSION STATE
# ============================================================

if "search_history" not in st.session_state:
    st.session_state.search_history = []

if "last_results" not in st.session_state:
    st.session_state.last_results = []

if "last_query" not in st.session_state:
    st.session_state.last_query = ""

if "last_search_time" not in st.session_state:
    st.session_state.last_search_time = 0.0


# ============================================================
# CUSTOM CSS - UI POLISH
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666666;
        margin-bottom: 25px;
    }

    .result-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        margin-bottom: 15px;
        background-color: #ffffff;
    }

    .score {
        font-size: 15px;
        font-weight: 600;
    }

    .small-text {
        color: #666666;
        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🔎 Mini Search Engine</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    A lightweight information retrieval system using
    an inverted index, TF-IDF and cosine similarity.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTION - LOAD DOCUMENTS FROM DATA FOLDER
# ============================================================

def load_documents_from_folder():

    documents = {}

    if not os.path.exists(DATA_FOLDER):

        os.makedirs(DATA_FOLDER)

        return documents


    try:

        files = os.listdir(DATA_FOLDER)

    except Exception as e:

        st.error(
            f"Could not access the data folder: {e}"
        )

        return documents


    for file_name in files:

        if not file_name.lower().endswith(".txt"):

            continue


        file_path = os.path.join(
            DATA_FOLDER,
            file_name
        )


        try:

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as file:

                text = file.read()


            if text.strip():

                documents[file_name] = text

            else:

                st.warning(
                    f"⚠️ {file_name} is empty and "
                    f"was not indexed."
                )


        except UnicodeDecodeError:

            st.warning(
                f"⚠️ {file_name} could not be read "
                f"as UTF-8 text."
            )


        except Exception as e:

            st.warning(
                f"Could not read {file_name}: {e}"
            )


    return documents


# ============================================================
# DOCUMENT UPLOAD
# ============================================================

st.subheader("📤 Add Documents")

st.write(
    "Upload `.txt` files to add them to the search collection."
)

uploaded_files = st.file_uploader(
    "Choose text files",
    type=["txt"],
    accept_multiple_files=True
)


if uploaded_files:

    uploaded_count = 0


    for uploaded_file in uploaded_files:

        try:

            file_bytes = uploaded_file.getvalue()

            text = file_bytes.decode(
                "utf-8"
            )


            # Check for empty document

            if text.strip() == "":

                st.warning(
                    f"⚠️ {uploaded_file.name} is empty "
                    f"and was not added."
                )

                continue


            file_path = os.path.join(
                DATA_FOLDER,
                uploaded_file.name
            )


            # Save uploaded document permanently

            with open(
                file_path,
                "w",
                encoding="utf-8"
            ) as file:

                file.write(text)


            uploaded_count += 1


        except UnicodeDecodeError:

            st.error(
                f"❌ {uploaded_file.name} is not a valid "
                f"UTF-8 text file."
            )


        except Exception as e:

            st.error(
                f"❌ Could not save "
                f"{uploaded_file.name}: {e}"
            )


    if uploaded_count > 0:

        st.success(
            f"✅ {uploaded_count} document(s) "
            f"added successfully."
        )


# ============================================================
# LOAD ALL DOCUMENTS
# ============================================================

documents = load_documents_from_folder()


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text):

    if not isinstance(text, str):

        return []


    tokens = re.findall(
        r"\b\w+\b",
        text.lower()
    )


    return tokens


# ============================================================
# STOP WORDS
# ============================================================

stop_words = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "in",
    "on",
    "at",
    "to",
    "of",
    "for",
    "from",
    "with",
    "by",
    "and",
    "or",
    "but",
    "as",
    "it",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "he",
    "she",
    "we",
    "they",
    "my",
    "your",
    "his",
    "her",
    "our",
    "their",
    "me",
    "him",
    "them",
    "do",
    "does",
    "did",
    "will",
    "would",
    "can",
    "could",
    "should",
    "have",
    "has",
    "had"
}


def remove_stop_words(tokens):

    return [
        token
        for token in tokens
        if token not in stop_words
    ]


# ============================================================
# STEMMING
# ============================================================

def stem_tokens(tokens):

    return [
        stemmer.stem(token)
        for token in tokens
    ]


# ============================================================
# BUILD INDEX
# ============================================================

def build_index(documents):

    tokenized_documents = {}

    processed_documents = {}

    stemmed_documents = {}


    # --------------------------------------------------------
    # TEXT PREPROCESSING
    # --------------------------------------------------------

    for file_name, text in documents.items():

        tokens = tokenize(text)

        tokenized_documents[file_name] = tokens


        filtered_tokens = remove_stop_words(
            tokens
        )

        processed_documents[file_name] = (
            filtered_tokens
        )


        stemmed = stem_tokens(
            filtered_tokens
        )

        stemmed_documents[file_name] = (
            stemmed
        )


    # --------------------------------------------------------
    # INVERTED INDEX
    # --------------------------------------------------------

    inverted_index = {}


    for file_name, tokens in stemmed_documents.items():

        for token in tokens:

            if token not in inverted_index:

                inverted_index[token] = []


            if file_name not in inverted_index[token]:

                inverted_index[token].append(
                    file_name
                )


    # --------------------------------------------------------
    # VOCABULARY
    # --------------------------------------------------------

    vocabulary = set()


    for tokens in stemmed_documents.values():

        vocabulary.update(tokens)


    vocabulary = sorted(
        vocabulary
    )


    # --------------------------------------------------------
    # IDF
    # --------------------------------------------------------

    idf_values = {}

    total_documents = len(
        stemmed_documents
    )


    for word in vocabulary:

        document_count = 0


        for tokens in stemmed_documents.values():

            if word in tokens:

                document_count += 1


        if total_documents == 0:

            idf_values[word] = 0


        else:

            # Smoothed IDF
            idf_values[word] = math.log(
                (total_documents + 1)
                /
                (document_count + 1)
            ) + 1


    # --------------------------------------------------------
    # DOCUMENT TF-IDF VECTORS
    # --------------------------------------------------------

    document_vectors = {}


    for document_name, tokens in stemmed_documents.items():

        vector = []


        for word in vocabulary:

            if len(tokens) == 0:

                tf = 0

            else:

                count = tokens.count(
                    word
                )

                tf = count / len(tokens)


            tfidf = (
                tf
                *
                idf_values.get(
                    word,
                    0
                )
            )


            vector.append(
                tfidf
            )


        document_vectors[
            document_name
        ] = vector


    return (
        tokenized_documents,
        processed_documents,
        stemmed_documents,
        inverted_index,
        vocabulary,
        idf_values,
        document_vectors
    )


# ============================================================
# AUTOMATIC INDEX REBUILDING
# ============================================================

(
    tokenized_documents,
    processed_documents,
    stemmed_documents,
    inverted_index,
    vocabulary,
    idf_values,
    document_vectors
) = build_index(
    documents
)


# ============================================================
# INDEX STATUS
# ============================================================

if len(documents) > 0:

    st.caption(
        f"🔄 Index built successfully for "
        f"**{len(documents)} document(s)**."
    )

else:

    st.warning(
        "⚠️ No documents are currently indexed. "
        "Add `.txt` files to start searching."
    )


# ============================================================
# QUERY PROCESSING
# ============================================================

def process_query(query):

    if not isinstance(query, str):

        return []


    tokens = tokenize(
        query
    )


    tokens = remove_stop_words(
        tokens
    )


    tokens = stem_tokens(
        tokens
    )


    return tokens


# ============================================================
# SEARCH USING INVERTED INDEX
# ============================================================

def search(
    query,
    mode="OR"
):

    query_tokens = process_query(
        query
    )


    if len(query_tokens) == 0:

        return set()


    document_sets = []


    for token in query_tokens:

        if token in inverted_index:

            document_sets.append(
                set(
                    inverted_index[token]
                )
            )


    if len(document_sets) == 0:

        return set()


    # --------------------------------------------------------
    # AND SEARCH
    # --------------------------------------------------------

    if mode == "AND":

        matching_documents = (
            document_sets[0]
        )


        for document_set in document_sets[1:]:

            matching_documents = (
                matching_documents.intersection(
                    document_set
                )
            )


    # --------------------------------------------------------
    # OR SEARCH
    # --------------------------------------------------------

    else:

        matching_documents = set()


        for document_set in document_sets:

            matching_documents.update(
                document_set
            )


    return matching_documents


# ============================================================
# TERM FREQUENCY
# ============================================================

def calculate_tf(
    word,
    tokens
):

    if len(tokens) == 0:

        return 0


    count = tokens.count(
        word
    )


    return count / len(tokens)


# ============================================================
# TF-IDF
# ============================================================

def calculate_tfidf(
    word,
    tokens
):

    tf = calculate_tf(
        word,
        tokens
    )


    idf = idf_values.get(
        word,
        0
    )


    return tf * idf


# ============================================================
# QUERY VECTOR
# ============================================================

def query_vector(query):

    tokens = process_query(
        query
    )


    vector = []


    for word in vocabulary:

        score = (
            calculate_tf(
                word,
                tokens
            )
            *
            idf_values.get(
                word,
                0
            )
        )


        vector.append(
            score
        )


    return vector


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    vector1,
    vector2
):

    if len(vector1) != len(vector2):

        return 0


    dot_product = 0

    magnitude1 = 0

    magnitude2 = 0


    for i in range(
        len(vector1)
    ):

        dot_product += (
            vector1[i]
            *
            vector2[i]
        )


        magnitude1 += (
            vector1[i] ** 2
        )


        magnitude2 += (
            vector2[i] ** 2
        )


    magnitude1 = math.sqrt(
        magnitude1
    )

    magnitude2 = math.sqrt(
        magnitude2
    )


    if (
        magnitude1 == 0
        or
        magnitude2 == 0
    ):

        return 0


    return (
        dot_product
        /
        (
            magnitude1
            *
            magnitude2
        )
    )


# ============================================================
# RANKED SEARCH
# ============================================================

def cosine_ranked_search(
    query,
    mode="OR"
):

    matching_documents = search(
        query,
        mode
    )


    query_vec = query_vector(
        query
    )


    scored_results = []


    for document in matching_documents:

        document_vec = (
            document_vectors.get(
                document,
                []
            )
        )


        score = cosine_similarity(
            query_vec,
            document_vec
        )


        scored_results.append(
            (
                document,
                score
            )
        )


    scored_results.sort(
        key=lambda x: x[1],
        reverse=True
    )


    return scored_results


# ============================================================
# FIND QUERY MATCH IN ORIGINAL TEXT
# ============================================================

def find_query_match(
    word,
    query_words
):

    clean_word = re.sub(
        r"[^\w]",
        "",
        word.lower()
    )


    stemmed_word = stemmer.stem(
        clean_word
    )


    return stemmed_word in query_words


# ============================================================
# QUERY-AWARE SNIPPET
# ============================================================

def get_snippet(
    document_name,
    query
):

    text = documents.get(
        document_name,
        ""
    )


    if text.strip() == "":

        return "No preview available."


    query_words = process_query(
        query
    )


    text_words = text.split()


    # --------------------------------------------------------
    # No query terms
    # --------------------------------------------------------

    if len(query_words) == 0:

        snippet_words = text_words[:40]

        snippet = " ".join(
            snippet_words
        )


        if len(text_words) > 40:

            snippet += " ..."


        return snippet


    # --------------------------------------------------------
    # Find first matching word
    # --------------------------------------------------------

    match_position = None


    for i, word in enumerate(
        text_words
    ):

        if find_query_match(
            word,
            query_words
        ):

            match_position = i

            break


    # --------------------------------------------------------
    # No match
    # --------------------------------------------------------

    if match_position is None:

        snippet_words = text_words[:40]

        snippet = " ".join(
            snippet_words
        )


        if len(text_words) > 40:

            snippet += " ..."


        return snippet


    # --------------------------------------------------------
    # Create context around match
    # --------------------------------------------------------

    start = max(
        0,
        match_position - 12
    )


    end = min(
        len(text_words),
        match_position + 20
    )


    snippet_words = text_words[
        start:end
    ]


    snippet = " ".join(
        snippet_words
    )


    if start > 0:

        snippet = "... " + snippet


    if end < len(text_words):

        snippet += " ..."


    # --------------------------------------------------------
    # Highlight original query words
    # --------------------------------------------------------

    original_query_tokens = tokenize(
        query
    )


    for query_word in original_query_tokens:

        if query_word in stop_words:

            continue


        pattern = re.compile(
            r"\b"
            +
            re.escape(query_word)
            +
            r"\w*",
            re.IGNORECASE
        )


        snippet = pattern.sub(
            lambda match:
                f"**{match.group(0)}**",
            snippet
        )


    return snippet


# ============================================================
# SEARCH QUALITY INFORMATION
# ============================================================

def calculate_query_statistics(
    query,
    results
):

    query_tokens = process_query(
        query
    )


    unique_query_terms = set(
        query_tokens
    )


    matched_terms = 0


    for term in unique_query_terms:

        if term in inverted_index:

            matched_terms += 1


    return {
        "query_terms": len(query_tokens),
        "unique_terms": len(
            unique_query_terms
        ),
        "matched_terms": matched_terms,
        "results": len(results)
    }


# ============================================================
# HOW THE SEARCH ENGINE WORKS
# ============================================================

with st.expander(
    "📖 How does this search engine work?"
):

    st.markdown(
        """
        ### 1️⃣ Text Preprocessing

        Each document goes through:

        **Tokenization → Stop-word removal → Stemming**

        ### 2️⃣ Inverted Index

        Each processed term is mapped to the
        documents in which it appears.

        Example:

        `python → python.txt, machine_learning.txt`

        ### 3️⃣ TF-IDF

        TF-IDF measures the importance of a term
        within a document collection.

        **TF** measures how frequently a term occurs.

        **IDF** reduces the importance of terms that
        occur across many documents.

        ### 4️⃣ Vector Representation

        Documents and queries are represented as
        numerical TF-IDF vectors.

        ### 5️⃣ Cosine Similarity

        The query vector is compared with each
        matching document vector.

        ### 6️⃣ Ranking

        Documents are sorted according to their
        cosine similarity score.

        ### 7️⃣ Search Modes

        **OR:** returns documents containing
        at least one query term.

        **AND:** returns documents containing
        all query terms.
        """
    )


# ============================================================
# DASHBOARD
# ============================================================

st.divider()

st.subheader("📊 Search Engine Dashboard")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "📄 Documents",
        len(documents)
    )


with col2:

    st.metric(
        "🔤 Vocabulary",
        len(vocabulary)
    )


with col3:

    total_terms = sum(
        len(tokens)
        for tokens
        in stemmed_documents.values()
    )


    st.metric(
        "📑 Indexed Terms",
        total_terms
    )


with col4:

    st.metric(
        "🔎 Index Entries",
        len(inverted_index)
    )


# ============================================================
# DOCUMENT COLLECTION
# ============================================================

if len(documents) > 0:

    with st.expander(
        "📚 View Indexed Documents"
    ):

        for document_name, text in documents.items():

            word_count = len(
                tokenize(text)
            )


            st.write(
                f"**📄 {document_name}** — "
                f"{word_count} words"
            )


# ============================================================
# SEARCH SECTION
# ============================================================

st.divider()

st.subheader("🔍 Search")


query = st.text_input(
    "Enter your search query",
    placeholder=(
        "Example: python machine learning"
    ),
    key="search_query"
)


search_mode = st.radio(
    "Search Mode",
    [
        "OR",
        "AND"
    ],
    horizontal=True
)


search_button = st.button(
    "🔍 Search",
    type="primary",
    use_container_width=True
)


# ============================================================
# SEARCH PROCESS
# ============================================================

if search_button:

    # --------------------------------------------------------
    # EMPTY QUERY
    # --------------------------------------------------------

    if query.strip() == "":

        st.warning(
            "⚠️ Please enter a search query."
        )


    # --------------------------------------------------------
    # NO DOCUMENTS
    # --------------------------------------------------------

    elif len(documents) == 0:

        st.error(
            "❌ No documents are available "
            "for searching."
        )


    else:

        processed_query = process_query(
            query
        )


        # ----------------------------------------------------
        # Query contains only stop words
        # ----------------------------------------------------

        if len(processed_query) == 0:

            st.warning(
                "⚠️ Your query contains only "
                "stop words. Try using more "
                "specific terms."
            )


        else:

            # ------------------------------------------------
            # SEARCH TIMER
            # ------------------------------------------------

            start_time = time.perf_counter()


            results = cosine_ranked_search(
                query,
                search_mode
            )


            end_time = time.perf_counter()


            search_time = (
                end_time
                -
                start_time
            )

            # ------------------------------------------------
            # SAVE SEARCH INFORMATION
            # ------------------------------------------------

            st.session_state.last_results = (results)
            st.session_state.last_query = (query)
            st.session_state.last_search_time = (search_time)

            # Add search history

            if (query.strip() not in st.session_state.search_history):
                st.session_state.search_history.append( query.strip())

            # ------------------------------------------------
            # QUERY STATISTICS
            # ------------------------------------------------

            statistics = calculate_query_statistics(query,results)

            # ------------------------------------------------
            # SEARCH STATISTICS
            # ------------------------------------------------

            st.subheader( "📊 Search Statistics")
            stat1, stat2, stat3, stat4 = st.columns( 4 )
            with stat1:
                st.metric( "Documents Found", len(results))
            with stat2:
                st.metric("Query Terms", statistics["unique_terms"])
            with stat3:
                st.metric("Matched Terms", statistics["matched_terms"])
            with stat4:
                st.metric("Search Time", f"{search_time:.5f}s")
            st.divider()

            # ------------------------------------------------
            # NO RESULTS
            # ------------------------------------------------

            if len(results) == 0:
                st.error("❌ No matching documents found.")
                st.info(
                    "Try one of these approaches:\n\n"
                    "• Use fewer words\n"
                    "• Check spelling\n"
                    "• Switch from AND to OR\n"
                    "• Use a more general term"
                )


            # ------------------------------------------------
            # DISPLAY RESULTS
            # ------------------------------------------------

            else:
                st.subheader(f"🔎 Results for: "f"**{query}**")
                st.caption( f"Search mode: {search_mode}  |  "f"{len(results)} result(s)")
                for i, (document,score) in enumerate(results,start=1):

                    # ----------------------------------------
                    # RESULT CARD
                    # ----------------------------------------

                    st.markdown('<div class="result-card">',unsafe_allow_html=True)
                    st.markdown( f"### {i}. 📄 {document}")
                    st.markdown(
                        f'<div class="score">'
                        f"Relevance Score: "
                        f"{score:.4f}"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                    # ----------------------------------------
                    # Score progress bar
                    # ----------------------------------------

                    progress_score = min( max(score, 0.0),1.0)
                    st.progress(progress_score)

                    # ----------------------------------------
                    # Snippet
                    # ----------------------------------------

                    snippet = get_snippet( document, query)
                    st.markdown(snippet)
                    st.markdown("</div>",unsafe_allow_html=True)


# ============================================================
# ANALYTICS
# ============================================================

st.divider()
st.subheader("📈 Search Analytics")
analytics_col1, analytics_col2 = st.columns(2)
with analytics_col1:

    st.write("**Document Size Distribution**" )

    if len(documents) > 0:

        document_sizes = {}

        for name, text in documents.items():

            document_sizes[name] = len( tokenize(text))
        st.bar_chart(document_sizes )


    else:
        st.info( "No document data available.")

with analytics_col2:
    st.write("**Search Result Scores**" )

    last_results = ( st.session_state.last_results)

    if len(last_results) > 0:
        score_data = {}

        for document, score in last_results:

            score_data[document] = score

        st.bar_chart( score_data)

    else:
        st.info(
            "Perform a search to see "
            "relevance scores."
        )

# ============================================================
# SEARCH PERFORMANCE
# ============================================================

if (st.session_state.last_query and len(st.session_state.last_results) > 0):

    st.info(
        f"Last search: "
        f"**{st.session_state.last_query}**  |  "
        f"Time: "
        f"**{st.session_state.last_search_time:.5f} sec**"
    )

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("⚙️ Search Engine")

    # --------------------------------------------------------
    # Basic statistics
    # --------------------------------------------------------

    st.subheader("📊 Index Statistics")
    st.metric( "Documents", len(documents))
    st.metric("Unique Terms",len(vocabulary))
    total_terms = sum(
        len(tokens)
        for tokens
        in stemmed_documents.values()
    )

    st.metric( "Total Terms", total_terms)

    st.divider()

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    st.subheader("🔧 Configuration")
    st.write("📚 **Index:** Inverted Index")
    st.write("📈 **Ranking:** TF-IDF")
    st.write("📐 **Similarity:** Cosine Similarity")
    st.write("🔤 **Processing:** Porter Stemming")
    st.write("🚫 **Stop Words:** Removed")
    st.divider()


    # ========================================================
    # INDEX EXPLORER
    # ========================================================

    st.subheader("🔎 Index Explorer")

    if len(vocabulary) > 0:

        available_letters = sorted(
            set(
                word[0].upper()
                for word in vocabulary
                if word
            )
        )

        selected_letter = st.selectbox("Choose starting letter",available_letters)


        filtered_terms = [
            word
            for word in vocabulary
            if word[0].upper()
            == selected_letter
        ]

        st.write(f"**{len(filtered_terms)} terms found**")


        for word in filtered_terms[:25]:

            document_list = (inverted_index[word])

            st.write(f"**{word}** → "f"{', '.join(document_list)}")

        if len(filtered_terms) > 25:
            st.caption(f"Showing first 25 of "f"{len(filtered_terms)} terms.")


    else:
        st.info("Index is empty.")

    st.divider()

    # ========================================================
    # SEARCH HISTORY
    # ========================================================

    st.subheader( "🕘 Search History")

    if len(st.session_state.search_history) == 0:
        st.write("No searches yet.")


    else:
        for previous_query in reversed(st.session_state.search_history):
            st.write(f"• {previous_query}")

        if st.button("🗑️ Clear History",use_container_width=True):
            st.session_state.search_history = []
            st.rerun()


    st.divider()

    # ========================================================
    # PROJECT INFORMATION
    # ========================================================

    st.subheader("💡 Project Components")
    st.write("✅ Tokenization")
    st.write("✅ Stop-word removal")
    st.write("✅ Stemming")
    st.write("✅ Inverted Index" )
    st.write("✅ TF-IDF")
    st.write( "✅ Cosine Similarity")
    st.write("✅ Ranked Retrieval" )
    st.write( "✅ Query-aware Snippets")
    st.write( "✅ Document Upload")
    st.write("✅ Search Analytics")


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption("Mini Search Engine • Python + Streamlit • ""Information Retrieval Project")