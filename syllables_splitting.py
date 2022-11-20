from typing import List

vowel_sounds = "А, О, У, Ы, Э, Е, Ё, И, Ю, Я".lower().split(", ")
consonant_sounds = "Б, В, Г, Д, Ж, З, Й, К, Л, М, Н, П, Р, С, Т, Ф, Х, Ц, Ч, Ш, Щ".lower().split(", ")
sonorous_sounds = "р, л, м, н, й".split(", ")


def split(word: str) -> List[str]:
    length = len(word)
    syllables = []
    cur_syllable = ""
    skip = False
    for i, letter in enumerate(word):
        if skip:
            skip = False
            continue
        cur_syllable += letter
        if i == length - 1:
            if letter in consonant_sounds or letter in ["ь", "ъ"]:
                syllables[-1] += cur_syllable
            else:
                syllables.append(cur_syllable)
            return syllables
        if letter in ("ь", "ъ"):
            if len(cur_syllable) == 1:
                syllables[-1] += letter
                cur_syllable = ""
            continue
        if letter in consonant_sounds:
            continue
        if (i + 1 < length and (word[i + 1] in vowel_sounds or word[i + 1] not in sonorous_sounds)
                or i + 2 < length and word[i + 2] in vowel_sounds):
            syllables.append(cur_syllable)
            cur_syllable = ""
            continue
        cur_syllable += word[i + 1]
        syllables.append(cur_syllable)
        cur_syllable = ""
        skip = True
    return syllables


if __name__ == "__main__":
    print(split("водонагреватель"))