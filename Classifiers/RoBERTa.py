from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/roberta2roberta_L-24_wikisplit")
model = AutoModelForSeq2SeqLM.from_pretrained("google/roberta2roberta_L-24_wikisplit")

long_sentence = """Due to the hurricane, Lobsterfest has been canceled, making Bob very happy about it and he decides to open Bob 's Burgers for customers who were planning on going to Lobsterfest."""

input_ids = tokenizer(tokenizer.bos_token + long_sentence + tokenizer.eos_token, return_tensors="pt").input_ids
output_ids = model.generate(input_ids)[0]
print(tokenizer.decode(output_ids, skip_special_tokens=True))
# should output
# Due to the hurricane, Lobsterfest has been canceled, making Bob very happy about it. He decides to open Bob's Burgers for customers who were planning on going to Lobsterfest. 