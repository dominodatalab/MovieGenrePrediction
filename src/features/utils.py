# remove some punctuation
def remove_punctuation(input_string):
    input_string = input_string.replace(',','')
    cleaned_string = input_string.replace('.','')    
    return cleaned_string
