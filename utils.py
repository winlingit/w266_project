import os
import re
import nltk
import itertools

# strip markdown, new line chars
def preprocess(line):
    strip = ["*", "'''", "\n"]
    
    for s in strip:
        line = line.replace(s, "")
    return line.split(": ")