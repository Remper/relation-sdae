from os import path
import re

folder = "/Users/remper/Downloads/wiki_text_embeddings"
max = 400000

print("Extracting dictionary")
word_dict = list()
with open(path.join(folder, "word.dict"), "rb") as reader:
    for line in reader:
        word, count = re.match("\((.+),([0-9]+)\)", line.decode("utf-8")).groups()
        word_dict.append([word, int(count)])

word_dict = sorted(word_dict, key=lambda tuple: -tuple[1])

first = True
counter = 0
with open(path.join(folder, "word.filtered.dict"), "wb") as writer:
    for tuple in word_dict:
        if not first:
            writer.write("\n".encode("utf-8"))
        first = False
        writer.write(tuple[0].encode("utf-8"))
        counter += 1
        if counter >= max:
            break
