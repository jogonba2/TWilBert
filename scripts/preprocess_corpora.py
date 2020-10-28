import pandas as pd
import csv
import re
import html


fr = open("unique_zz_pairs.tsv", "r", encoding="utf8")
fw = open("prepro_unique_zz_pairs.tsv", "w", encoding="utf8")

regex = re.compile(r"https?://\S+")
regex_mentions = re.compile(r"(@+\w+)")
h = {}
n_reps = 0
n_samples = 0
for line in fr.readlines():
    try:
        id_1, text, id_2, reply = line.strip().split("\t")

        text = regex.sub("url", text).strip()
        reply = regex.sub("url", reply).strip()
        text = regex_mentions.sub("user", text).strip()
        reply = regex_mentions.sub("user", reply).strip()
        text = html.unescape(text)
        reply = html.unescape(reply)
        text = text.replace("…", " ").strip()
        reply = reply.replace("…", " ").strip()
        text = " ".join(text.split())
        reply = " ".join(reply.split())
        stext = text.split()
        sreply = reply.split()
        ltext = len(stext)
        lreply = len(sreply)
        text_word_set = set(stext)
        reply_word_set = set(sreply)
        if (text_word_set in [{"user"}, {"url"}, {"user", "url"}]) and \
           (reply_word_set in [{"user"}, {"url"}, {"user", "url"}]):
           continue
        if ltext <= 2 and lreply <= 2:
            continue
        if (id_1, id_2) in h:
            n_reps += 1
        else:
            h[(id_1, id_2)] = True
            fw.write(id_1 + "\t" + text + "\t" + id_2 + "\t" + reply + "\n")
    except:
        pass
    n_samples += 1
    if n_samples % 100000 == 0:
        print("De %d, %d repetidos" % (n_samples, n_reps))

print("De %d, %d repetidos" % (n_samples, n_reps))

fr.close()
fw.close()
