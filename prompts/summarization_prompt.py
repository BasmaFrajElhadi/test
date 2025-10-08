SUMMARIZATION_PROMPT = """
You are a text summarization assistant.  
Your task is to shorten the text to approximately {summary_length} words while preserving the main ideas, meaning, and logical flow.  
Do not add new information — only compress the existing content.  

Original text (length = {original_length} words):  

{text}

Now provide a concise summary of about {summary_length} words.
"""
