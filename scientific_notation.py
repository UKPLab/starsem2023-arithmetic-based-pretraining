import re

def rebuild_string_in_new_notation(text:str) -> str:
    '''
        Method for rebuilding a passed string in
        scientific notation
        :param text: string to be rebuild
        :return: string in scientific notation
    '''
    splitted = text.split()
    return ' '.join([get_scientific_notation(token) for token in splitted])

def get_scientific_notation(token:str) -> str:
    '''
        Method for changing the representation of a passed string;
        if it is a number, it will be replaced with scientific
        notification (314.1 -> 314 <EXP> 1), see also 
        https://arxiv.org/pdf/2010.05345.pdf (NumBERT)
    '''
       
    return str(token).replace('.', ' <EXP> ') if re.match(r'[(+-]?(\d*\.)\d+', token) else token
    

def convert_text_scientific_notation(text: str, special_tokens: list) -> str:
    '''
        method for converting the output of using the scientific notation.
        in this case, the special tokens are not automatically removed, 
        because needs to be handled separately.
        :param text: passed text to be post-processed
        :param special_tokens: list of special tokens
        :return post-processed string
    '''

    text = re.sub(r' <EXP> ', r'.', text)
    for token in special_tokens:
        text = text.replace(token, ' ')
    text = re.sub(r' +', r' ', text)
    text = ' '.join(text.split())

    return text             
