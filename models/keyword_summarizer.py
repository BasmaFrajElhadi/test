from keybert import KeyBERT

class KeywordSummarizer:
    """
    KeywordSummarizer extracts the most relevant keyword or short phrase
    from a given text using the KeyBERT model.

    This class is ideal for generating concise, meaningful titles or session
    names based on the content of longer text inputs — for example, naming
    chat sessions or summarizing topics.

    Attributes:
        kw_model (KeyBERT): The underlying KeyBERT keyword extraction model.
    """
    def __init__(self):
        self.kw_model = KeyBERT()

    def summarize_text(self, text: str, summary_length: int = 5, n_phrases: int = 5):
        """
        Extract the most relevant keyword or phrase from the given text.

        The method uses KeyBERT to identify key phrases (1–n_phrases words) that
        capture the main idea of the text. If no keyword is found, a truncated
        substring of the original text is returned instead.

        Args:
            text (str): The input text to summarize or extract a keyword from.
            summary_length (int, optional): The maximum number of characters
                to return if no keyword is found. Defaults to 5.

        Returns:
            str: The extracted keyword or a shortened fallback string.
        """
        keywords = self.kw_model.extract_keywords(text, keyphrase_ngram_range=(1,n_phrases), stop_words='english', top_n=1)
        return keywords[0][0] if keywords else text[:summary_length]
