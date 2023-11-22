import tiktoken

def create_token_dic(text):
    unique_chars = sorted(set(text))
    tok_to_char = {token : character for token, character in enumerate(unique_chars)}
    char_to_tok = {character : token for token, character in enumerate(unique_chars)}
    return tok_to_char, char_to_tok

def tokenize(text, char_to_tok):
    tokenized_text = [char_to_tok[character] for character in text]
    return tokenized_text

def detokenize(text, tok_to_char):
    detokenized_text = [tok_to_char[token] for token in text]
    return detokenized_text

def get_subword_encoding():
    enc = tiktoken.get_encoding("gpt2")
    return enc
