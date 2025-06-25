# import fitz
import os
os.environ['PANPHON_ENCODING'] = 'utf-8'

import re
import panphon.distance
from phonemizer import phonemize
from nltk.corpus import words
from collections import Counter
from panphon.featuretable import FeatureTable
import nltk
import pdfplumber
import numpy as np
import pandas as pd
from flask import jsonify
import json
import eng_to_ipa as ipa  # Add this import

# Initialize required objects
ft = FeatureTable()
distancer = panphon.distance.Distance()

filename = "data/backend_results.json"

def read_json_file(filename):
    with open(filename, 'r') as file:
        return json.load(file)


def parse_tables_from_json(data):

    freq_table = []
    freq_table.append(["IPA Variant", "Frequency", "Fraction"])  
    for variant in data["ipa_variants"]:
        row = [variant["ipa"], str(variant["frequency"]), str(variant["fraction"])]
        freq_table.append(row)


    conf_matrix = []
    conf_matrix.append([""] + data["confusion_matrix"]["labels"])  
    for i, row in enumerate(data["confusion_matrix"]["matrix"]):
        conf_row = [data["confusion_matrix"]["labels"][i]] + [str(val) for val in row]
        conf_matrix.append(conf_row)

    return freq_table, conf_matrix




json_data = read_json_file(filename)

freq_table, conf_matrix = parse_tables_from_json(json_data)


print("\nFrequency Count Table:")
for row in freq_table:
    print("|", " | ".join(row), "|")


print("\nConfusion Matrix Table:")
for row in conf_matrix:
    print("|", " | ".join(row), "|")

# Extract IPA words from freq_table (skip header row)
ipa_words = [row[0] for row in freq_table[1:]]  # Skip header row

full_text = ""

# if filename.lower().endswith('.pdf'): 
#     doc = fitz.open(filename)

#     for page in doc:
#         full_text += page.get_text()
#     print("\nFull Text from PDF:")
#     print(full_text)

# try:
#     ft = FeatureTable()
# except UnicodeDecodeError:
#     import os
#     os.environ['PANPHON_ENCODING'] = 'utf-8'
#     ft = FeatureTable()

# print(dir(ft))
# features = ft.segment_to_vector('ŋ')
# print(features)

###INTERNATIONAL ACCEPTANCE
tenet_scores ={}
def add_tenet_score(tenet_scores, tenet_name, ipa_score_dict):
    if tenet_name not in tenet_scores:
        tenet_scores[tenet_name] = {}
    for ipa, score in ipa_score_dict.items():
        try:
            tenet_scores[tenet_name][ipa] = float(score)
        except ValueError:
            continue
def panphon_international_acceptance (variance, standard):
   v= variance.strip("/ ")
   s= standard.strip("/ ")
   dist = distancer.feature_edit_distance(v,s)
   max_len = max(len(v),len(s))
   similarity = 1-(dist/max_len) if max_len > 0 else 0
   similarity = max(0, min(1, similarity))
   return similarity


def evaluate_file_variants(target_word, ipa_words):
    print(target_word, ipa_words)
    standard_ipa = ipa.convert(target_word).strip()
    print(f"Standard IPA for {target_word}: {standard_ipa}")
    results = {}
    for var in ipa_words:
        score = panphon_international_acceptance(var, standard_ipa)
        results[var] = round(score, 3)
    return results
##target word 
# target_word = input("Enter the target word: ").strip()
read_json_file("data/latest_rankings.json")
with open("data/latest_rankings.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Access the word
target_word = data["ipa"]["word"]
print(target_word) 
scores = evaluate_file_variants(target_word,ipa_words)

print("\nInternational acceptance scores for variants from file:")
# for variance, score in scores:
#     print(f"{variance} → {score}")
for variance, score in scores.items():
    print(f"{variance} → {score}")

# tenet_scores['International Acceptance'] = evaluate_file_variants(target_word, ipa_words)
add_tenet_score(tenet_scores, "International Acceptance", scores)

###PHONETIC SIMPLICITY
complex_feature_weights = {
    # 'cont':2,
    # 'strident':2,
    # 'delrel':2.5,
    # 'dorsal':2,
    # 'voice':1.5,
    # 'spreadgl':1.5,
    # 'constrgl':2,
    # 'nasal':1.5,
    # 'labiodental':1.5,
    # 'distributed':1.2
    'syl': 0.5,              # syllabic - simple
    'son': 0.5,              # sonorant - simple
    'cons': 1.0,             # consonantal - basic
    'voice': 1.5,            # voiced - medium
    'sg': 2.0,               # spread glottis (aspiration) - complex
    'cg': 2.5,               # constricted glottis (ejectives, implosives) - complex
    'ant': 1.0,              # anterior place - basic
    'cor': 1.0,              # coronal place - basic
    'distr': 1.2,            # distributed - moderate
    'lat': 1.2,              # lateral - moderate
    'nasal': 1.5,            # nasal - medium
    'strid': 2.0,            # strident - complex (noisy fricatives)
    'latn': 1.2,             # lateral nasal? treated similar to lat
    'delrel': 2.5,           # delayed release (affricates) - complex
    'high': 1.0,             # high vowels/consonants - basic
    'low': 1.0,              # low vowels/consonants - basic
    'back': 1.0,             # back vowels/consonants - basic
    'round': 1.0,            # rounded lips - basic
    'lab': 1.0,              # labial place - basic
    'labiodental': 1.5,      # labiodental - medium
    'dorsal': 2.0,           # dorsal (velar/uvular) - complex
    'pharyngeal': 3.0,       # pharyngeal place - very complex
    'glottal': 2.0,          # glottal place - complex
    'spreadgl': 2.0,         # spread glottis (aspiration) - complex
    'constrictedgl': 2.5,    # constricted glottis (ejectives) - very complex
    # Additional common features if in panphon:
    'strident': 2.0,
    'long': 1.2,
    'stress': 0.8,
    'tone': 1.5,
}

def explain_phonetic_simplicity(ipa_word , weights=complex_feature_weights):
  ipa_word = ipa_word.strip("/ ").replace(":","")
  segments = ft.ipa_segs(ipa_word)
  total_complexity = 0
  total_segments = 0

  for seg in segments:
    if ft.seg_known(seg):
      feats=ft.segment_to_vector(seg)
      print(f"Segment :{seg}")
      segment_complexity=0
      for feat, val in zip(ft.names, feats):
        # print(f"Feature: {feat}, Value:{val}")
        if val == '+' and feat in weights:
          w = weights[feat]
          segment_complexity += w
          print(f"Feature:{feat},Weight:{w}")
          print(f"Total complexity: {segment_complexity}")
      total_complexity += segment_complexity
      total_segments += 1
    else:
      print(f"No data for segment:{seg}")

  if total_segments == 0:
    print("No valid segments ")
    return 0


  avg_complexity = total_complexity / total_segments
  max_possible = sum(weights.values())
  normalized = avg_complexity / max_possible if max_possible else 0
  simplicity_score = 1 - min(1.0, normalized)

  print(f"\n Total Complexity: {round(total_complexity, 3)}")
  print(f" Avg Complexity per Segment: {round(avg_complexity, 3)}")
  print(f"Simplicity Score: {round(simplicity_score, 3)} (1 = simplest)")

  return round(simplicity_score, 3)

def get_phonetic_simplicity_score(ipa_word, weights=complex_feature_weights):
    ipa_word = ipa_word.strip("/ ").replace(":", "")
    segments = ft.ipa_segs(ipa_word)
    total_complexity = 0
    total_segments = 0

    for seg in segments:
        if ft.seg_known(seg):
            feats = ft.segment_to_vector(seg)
            segment_complexity = sum(
                weights[feat] for feat, val in zip(ft.names, feats)
                if val == '+' and feat in weights
            )
            total_complexity += segment_complexity
            total_segments += 1

    if total_segments == 0:
        print(f"{ipa_word} →  No valid segments. Simplicity score = 0.0")
        return 0.0

    avg_complexity = total_complexity / total_segments
    max_possible = sum(weights.values())
    normalized = avg_complexity / max_possible if max_possible else 0
    simplicity_score = round(1 - min(1.0, normalized), 3)

    print(f"{ipa_word} → Simplicity Score: {simplicity_score}")
    return simplicity_score
phonetic_simplicity_scores = {}

for word in ipa_words:
    score = get_phonetic_simplicity_score(word)
    phonetic_simplicity_scores[word] = score
add_tenet_score(tenet_scores, "phonetic_simplicity", phonetic_simplicity_scores)


###FREQUENCY 
# Frequency
def add_tenet_score(tenet_scores, tenet_name, ipa_score_dict):
    if tenet_name not in tenet_scores:
        tenet_scores[tenet_name] = {}
    for ipa, score in ipa_score_dict.items():
        try:
            tenet_scores[tenet_name][ipa] = float(score)
        except ValueError:
            continue
def extract_ipa_and_fraction(freq_table):
    ipa_fraction = {}
    for row in freq_table:
        if len(row) < 3:
            continue  # skip short or malformed rows
        if row[0].lower().startswith('ipa'):
            continue  # skip header
        try:
            ipa = row [0].strip()
            fraction = float(row[2])
            formatted_fraction = f"{fraction:.3f}"
            ipa_fraction[ipa] =fraction
        except ValueError:
            continue  # skip if conversion fails (e.g., dashes)
    return ipa_fraction

ipa_fraction = extract_ipa_and_fraction(freq_table)

for ipa, fraction in ipa_fraction.items():
    print(f"{ipa} → {fraction}")



add_tenet_score(tenet_scores, "Frequency", ipa_fraction)


####CONSTRASTIVENESS
constrastiveness_scores = {}

def compute_contrastiveness(ipa_words):
    contrastiveness_scores = {}
    for word in ipa_words:
        max_distance = 0
        for second_word in ipa_words:
            if word != second_word:
                dist = panphon.distance.Distance().weighted_feature_edit_distance(word.strip("/"), second_word.strip("/"))
                max_distance = max(max_distance, dist)

        max_possible = 10 * max(len(w.strip("/")) for w in ipa_words)
        normalised_distance = min(max_distance / max_possible if max_possible else 0, 1.0)
        contrastiveness_scores[word] = round(normalised_distance, 2)

    return contrastiveness_scores
# add_tenet_score(tenet_scores, "Frequency", ipa_fraction)
contrastiveness_scores = compute_contrastiveness(ipa_words)
add_tenet_score(tenet_scores, "Contrastiveness", contrastiveness_scores)



###paedagogic convenience
ipa_vowel_list = ['iː', 'ɪ', 'e', 'æ', 'ʌ', 'ɑː', 'ɒ', 'ɔː', 'ʊ', 'uː', 'ə', 'ɜː',
                  'eɪ', 'aɪ', 'ɔɪ', 'əʊ', 'aʊ', 'ɪə', 'eə', 'ʊə', 'eɪə', 'aɪə']
ipa_vowels = ''.join([re.escape(v) for v in ipa_vowel_list])

def compute_paedagogic_convenience(ipa):
    """
    Compute paedagogic convenience score for an IPA transcription.
    Based on Ugorji's Preference Grammar (PG) framework.
    Returns a normalized score (0-1) and explanations for penalties/rewards.
    """
    # Clean IPA input
    ipa_clean = ipa.strip("/").replace("ː", "")  # Remove length marks for simplicity
    score = 0
    explanations = []

    # 1. Consonant clusters
    cluster_pattern = r'[^' + ipa_vowels + r'\s]{2,}'
    clusters = re.findall(cluster_pattern, ipa_clean)
    if clusters:
        longest = max(len(c) for c in clusters)
        if longest >= 3:
            score += 2
            explanations.append("Triple+ consonant cluster (+2)")
        else:
            score += 1
            explanations.append("Double consonant cluster (+1)")
        for c in clusters:
            if len(set(c)) > 1:  # Heterorganic if different consonants
                score += 1
                explanations.append("Heterorganic cluster (+1)")
                break

    # 2. Syllable count (fixed: count vowels/diphthongs as syllable nuclei)
    syllable_pattern = '|'.join([re.escape(v) for v in ipa_vowel_list])
    syllables = len(re.findall(syllable_pattern, ipa_clean))
    if syllables >= 3:
        score += 1
        explanations.append("Trisyllabic or longer (+1)")

    # 3. Place & Manner Variegation
    ft = FeatureTable()
    segments = ft.ipa_segs(ipa_clean)
    places = set()
    manners = set()

    for seg in segments:
        if ft.seg_known(seg):
            vec = ft.segment_to_vector(seg)
            feature_dict = dict(zip(ft.names, vec))
            
            if feature_dict.get('labial') == '+': places.add('labial')
            if feature_dict.get('coronal') == '+': places.add('coronal')
            if feature_dict.get('dorsal') == '+': places.add('dorsal')
            if feature_dict.get('pharyngeal') == '+': places.add('pharyngeal')
            if feature_dict.get('nasal') == '+': manners.add('nasal')
            if feature_dict.get('approximant') == '+': manners.add('approximant')
            if feature_dict.get('trill') == '+': manners.add('trill')
            if feature_dict.get('continuant') == '+' and feature_dict.get('strident') == '+': manners.add('fricative')
            if feature_dict.get('stop') == '+' or feature_dict.get('closed') == '+': manners.add('stop')

    if len(places) > 1:
        score += 1
        explanations.append("Place variegation (+1)")
    if len(manners) > 1:
        score += 1
        explanations.append("Manner variegation (+1)")

    # Normalize score (max possible score is 6: 2 for triple cluster, 1 for heterorganic,
    # 1 for syllables, 1 for place, 1 for manner)
    max_score = 6
    normalized_score = 1 - (score / max_score) if max_score > 0 else 1.0
    normalized_score = round(normalized_score, 3)  # Match output precision (e.g., 0.948)

    return normalized_score, explanations


paedagogic_convenience_scores = {}
for word in ipa_words:
    score, explanations = compute_paedagogic_convenience(word)
    paedagogic_convenience_scores[word] = score

# Add to tenet_scores
add_tenet_score(tenet_scores, "Paedagogic Convenience", paedagogic_convenience_scores)


###DISAMBIGUITY 
disambiguity_scores = {}
def compute_disambiguity(ipa_words, conf_matrix):
   
    labels = [row[0] for row in conf_matrix[1:]]  
    matrix = np.array([[float(val) for val in row[1:]] for row in conf_matrix[1:]])
    try:
        matrix = np.array([[float(val) for val in row[1:]] for row in conf_matrix[1:]])
        print(f"Confusion Matrix:\n{matrix}")
    except ValueError as e:
        print(f"Error parsing confusion matrix: {e}")
        matrix = np.zeros((len(labels), len(labels)))
    for word in ipa_words:
        if word in labels:
            idx = labels.index(word)
            total = np.sum(matrix[idx])  
            correct = matrix[idx, idx] if total > 0 else 0 
            score = correct / total if total > 0 else 0.0  
        else:
            
            distances = []
            for other_word in ipa_words:
                if word != other_word:
                    dist = distancer.weighted_feature_edit_distance(word.strip("/"), other_word.strip("/"))
                    distances.append(dist)
            max_possible = 10 * max(len(w.strip("/")) for w in ipa_words)
            avg_distance = np.mean(distances) if distances else 0
            score = avg_distance / max_possible if max_possible > 0 else 0.0

        normalized_score = round(min(score, 1.0), 3)
        print(f"{word} → Disambiguity Score: {normalized_score}")
        disambiguity_scores[word] = normalized_score

    return disambiguity_scores
disambiguity_scores = compute_disambiguity(ipa_words, conf_matrix)
add_tenet_score(tenet_scores, "Disambiguity", disambiguity_scores)




###CREATION OF TABLE
import pandas as pd
def build_score_table(tenet_scores):
    """Builds a DataFrame from the tenet_scores dictionary."""
    all_variants = set()
    for scores in tenet_scores.values():
        all_variants.update(scores.keys())
    all_variants = sorted(all_variants)

    data = {}
    for tenet, scores in tenet_scores.items():
        col = []
        for variant in all_variants:
            score = scores.get(variant, None)
            col.append(score)
        data[tenet] = col

    # df = pd.DataFrame(data, index=all_variants)
    df = pd.DataFrame.from_dict(tenet_scores)

    return df
for tenet in tenet_scores:
    if isinstance(tenet_scores[tenet], list):
        tenet_scores[tenet] = dict(tenet_scores[tenet])


for tenet in tenet_scores:
    tenet_scores[tenet] = {k: float(v) for k, v in tenet_scores[tenet].items()}

# Build and display the score-based operability table
score_table = build_score_table(tenet_scores)

print("\n Operability Table (Raw Scores):")
# Set display options to show all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(score_table)
output_filename = "operability_table.csv"
score_table.to_csv(output_filename, encoding='utf-8')

# === AHP CALCULATION ===

with open ("data/latest_rankings.json","r") as f :
    data = json.load(f)

full_name ={
     "IA": "International Acceptance",
    "PS": "Phonetic Simplicity",
    "PC": "Paedagogic Convenience",
    "CO": "Contrastiveness",
    "F": "Frequency",
    "DI": "Disambiguity"
}
user_input = {full_name[k]: v for k, v in data["sliders"].items()}
def build_comparison_matrix(priorities):
    keys = list(priorities.keys())
    size = len(keys)
    matrix = np.ones((size, size))
    for i in range(size):
        for j in range(size):
            matrix[i][j] = priorities[keys[i]] / priorities[keys[j]]
    return matrix, keys

def calculate_priority_vector(matrix):
    col_sum = np.sum(matrix, axis=0)
    normalized = matrix / col_sum
    priority_vector = np.mean(normalized, axis=1)
    return priority_vector

def calculate_consistency(matrix, weights):
    n = matrix.shape[0]
    weighted_sum = np.dot(matrix, weights)
    lambda_max = np.sum(weighted_sum / weights) / n
    CI = (lambda_max - n) / (n - 1)
    RI_dict = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
               6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    RI = RI_dict.get(n, 1.49)
    CR = CI / RI if RI != 0 else 0
    return round(CR, 4), round(lambda_max, 4)

def ahp_analysis(user_input):
    matrix, keys = build_comparison_matrix(user_input)
    weights = calculate_priority_vector(matrix)
    CR, lambda_max = calculate_consistency(matrix, weights)
    return {
        "tenets": keys,
        "weights": dict(zip(keys, weights)),
        "consistency_ratio": CR,
        "lambda_max": lambda_max
    }


result = ahp_analysis(user_input)
print("\n AHP Weights (based on user priority):")
for tenet, weight in result["weights"].items():
    print(f"{tenet}: {weight:.4f}")






score_table = build_score_table(tenet_scores)
score_table.columns = [col.lower().replace(" ", "_") for col in score_table.columns]
print("\n Columns in Operability Table Before AHP:", list(score_table.columns))

result = ahp_analysis(user_input)
print("\n AHP Weights (based on user priority):")
for tenet, weight in result["weights"].items():
    print(f"{tenet}: {weight:.4f}")
print(f"\nConsistency Ratio: {result['consistency_ratio']} (λ max: {result['lambda_max']})")

for phoneme in tenet_scores:
    tenet_scores[phoneme] = {k: float(v) for k, v in tenet_scores[phoneme].items()}


print("\nOperability Table (Raw Scores):")
print(score_table)

# score_table.to_csv("operability_table.csv", encoding='utf-8')
# print("\n Operability table saved as: operability_table.csv")

# APPPLICATION OF THE WEIGHTS ON TBLE

def apply_ahp_to_operability(score_table: pd.DataFrame, weights: dict):

     score_table.columns = [col.lower().replace(" ", "_") for col in score_table.columns]
     weight_map = {k.lower().replace(" ", "_"): v for k, v in weights.items()}


     common_columns = [col for col in score_table.columns if col in weight_map]


     score_table['Weighted Score'] = score_table[common_columns].apply(
            lambda row: sum(row[col] * weight_map[col] for col in common_columns), axis=1
        ).round(4)
     

    
     return score_table.sort_values(by='Weighted Score', ascending=False)


#running the AHP
result = ahp_analysis(user_input)
final_table = apply_ahp_to_operability(score_table, result["weights"])
print(final_table)

best_transcription = final_table.index[0]
print(f"The preferred grammar pronunciation is {best_transcription}")




# print("\n Weighted Operability Table (Higher = Better Adherence):")
# print(final_table)

# final_table.to_csv("weighted_operability_table.csv", encoding='utf-8')
# print("\n Weighted operability table saved as: weighted_operability_table.csv")

