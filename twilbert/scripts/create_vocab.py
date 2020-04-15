import sentencepiece as spm
import os
import sys


corpus_input = sys.argv[1]
corpus_text = "__corpus.txt"
model_prefix = "m"
model_type = "bpe"
vocab_size = 30000


# Generate one tweet per row file if not exists #
fw = open(corpus_text, "w", encoding="utf8")
c = 0
with open(corpus_input, "r") as fr:
    for line in fr.readlines():
        id_, text, id_reply, reply = line.strip().split("\t")
        fw.write(text.strip() + "\n")
        fw.write(reply.strip() + "\n")
        c += 1
        if c % 1000000 == 0:
            print(c)
fw.close()

spm.SentencePieceTrainer.Train('--input=%s --model_prefix=%s --model_type=%s \
                               --vocab_size=%d' % (corpus_text, model_prefix,
                                                   model_type, vocab_size))
os.remove("m.model")
os.remove(corpus_text)

fr = open("m.vocab", "r", encoding="utf8")
fw = open("vocab", "w", encoding="utf8")


extra_tokens = ["[PAD]", "[UNK]", "[MASK]", "[CLS]", "[SEP]"]
remove_tokens = ["<unk>", "<s>", "</s>"]

for t in extra_tokens:
    fw.write(t + "\n")

for line in fr.readlines():
    w, _ = line.strip().split("\t")
    w = w.replace("▁", "##")
    if w not in remove_tokens:
        fw.write(w + "\n")

fr.close()
fw.close()
os.remove("m.vocab")
