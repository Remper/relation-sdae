from os import path
import re
import gzip

folder = "/Users/remper/Downloads/wiki_text_embeddings"
print("Extracting dictionary")
first = True
word_dict = set()
with gzip.open(path.join(folder, "row_embedding.tsv.gz"), "r") as reader:
    with open(path.join(folder, "word.dict"), "wb") as writer:
        for line in reader:
            row = line.decode("utf-8").rstrip().split('\t')

            if (re.match("^[0-9]+$", row[0])):
                continue

            if not first:
                writer.write("\n".encode("utf-8"))
            first = False
            word_dict.add(row[0])
            writer.write(row[0].encode("utf-8"))


print("Extracting POS tags")
postags = set()
lines = 0
with gzip.open(path.join(folder, "output.txt.gz"), "r") as reader:
    for line in reader:
        for word in line.decode("utf-8").rstrip().split(' '):
            word_postag = word.split("@")
            if word_postag[0] not in word_dict:
                continue
            if len(word_postag) > 1:
                postags.add(word_postag[1])
        lines += 1
        if lines % 1000000 == 0:
            print("Parsed %dm sentences (%d tags)" % (lines / 1000000, len(postags)))
            with open(path.join(folder, "postag.dict"), "wb") as writer:
                for tag in postags:
                    writer.write("\n".encode("utf-8"))
                    writer.write(tag.encode("utf-8"))
