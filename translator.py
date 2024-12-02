from transformers import MarianMTModel, MarianTokenizer
model_name = 'Helsinki-NLP/opus-mt-uk-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate(text):
    # Tokenize the input text
    tokenized_text = tokenizer.prepare_seq2seq_batch([text], return_tensors='pt')

    # Perform the translation
    translation = model.generate(**tokenized_text)

    # Decode the translated text
    translated_text = tokenizer.decode(translation[0], skip_special_tokens=True)

    return translated_text

while True:
    text_to_translate = input("Enter text im Ukrainian to translate into English: ")
    print("-----")
    print(translate(text_to_translate))
