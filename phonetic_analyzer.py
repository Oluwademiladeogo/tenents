import os
os.environ['PANPHON_ENCODING'] = 'utf-8'

import re
import panphon.distance
from phonemizer import phonemize
from nltk.corpus import words
from collections import Counter
from panphon.featuretable import FeatureTable
import nltk
import numpy as np
import pandas as pd
import eng_to_ipa as ipa
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from config import Config

@dataclass
class AnalysisRequest:
    word: str
    ipa_variants: List[Dict[str, Any]]
    confusion_matrix: Dict[str, Any]
    sliders: Dict[str, int]

@dataclass
class AnalysisResult:
    target_word: str
    best_transcription: str
    final_table: Dict[str, Dict[str, float]]
    operability_table: pd.DataFrame
    ahp_weights: Dict[str, float]
    consistency_ratio: float

class PhoneticAnalyzer:
    def __init__(self):
        self.ft = FeatureTable()
        self.distancer = panphon.distance.Distance()
        self.complex_feature_weights = Config.get_feature_weights()
        
        # IPA vowel list for paedagogic convenience
        self.ipa_vowel_list = Config.get_ipa_vowel_list()
        self.ipa_vowels = ''.join([re.escape(v) for v in self.ipa_vowel_list])
        
    def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """Main analysis method that processes the request and returns results."""
        # Extract IPA words from variants
        ipa_words = [variant["ipa"] for variant in request.ipa_variants]
        
        # Parse confusion matrix
        freq_table, conf_matrix = self._parse_tables_from_json(request)
        
        # Calculate all tenet scores
        tenet_scores = self._calculate_all_tenet_scores(request.word, ipa_words, conf_matrix, freq_table)
        
        # Build operability table
        score_table = self._build_score_table(tenet_scores)
        
        # Apply AHP analysis
        ahp_result = self._apply_ahp_analysis(request.sliders)
        
        # Apply weights to get final results
        final_table = self._apply_ahp_to_operability(score_table, ahp_result["weights"])
        
        best_transcription = final_table.index[0]
        
        return AnalysisResult(
            target_word=request.word,
            best_transcription=best_transcription,
            final_table=self._dataframe_to_dict(final_table),
            operability_table=score_table,
            ahp_weights=ahp_result["weights"],
            consistency_ratio=ahp_result["consistency_ratio"]
        )
    
    def _parse_tables_from_json(self, request: AnalysisRequest) -> Tuple[List, List]:
        """Parse frequency table and confusion matrix from request data."""
        # Frequency table
        freq_table = [["IPA Variant", "Frequency", "Fraction"]]
        for variant in request.ipa_variants:
            row = [variant["ipa"], str(variant["frequency"]), str(variant["fraction"])]
            freq_table.append(row)
        
        # Confusion matrix
        conf_matrix = [[""] + request.confusion_matrix["labels"]]
        for i, row in enumerate(request.confusion_matrix["matrix"]):
            conf_row = [request.confusion_matrix["labels"][i]] + [str(val) for val in row]
            conf_matrix.append(conf_row)
        
        return freq_table, conf_matrix
    
    def _calculate_all_tenet_scores(self, target_word: str, ipa_words: List[str], 
                                  conf_matrix: List, freq_table: List) -> Dict[str, Dict[str, float]]:
        """Calculate scores for all tenets."""
        tenet_scores = {}
        
        # International Acceptance
        ia_scores = self._evaluate_international_acceptance(target_word, ipa_words)
        tenet_scores["International Acceptance"] = ia_scores
        
        # Phonetic Simplicity
        ps_scores = {word: self._get_phonetic_simplicity_score(word) for word in ipa_words}
        tenet_scores["Phonetic Simplicity"] = ps_scores
        
        # Frequency
        freq_scores = self._extract_ipa_and_fraction(freq_table)
        tenet_scores["Frequency"] = freq_scores
        
        # Contrastiveness
        contrast_scores = self._compute_contrastiveness(ipa_words)
        tenet_scores["Contrastiveness"] = contrast_scores
        
        # Paedagogic Convenience
        pc_scores = {word: self._compute_paedagogic_convenience(word)[0] for word in ipa_words}
        tenet_scores["Paedagogic Convenience"] = pc_scores
        
        # Disambiguity
        disamb_scores = self._compute_disambiguity(ipa_words, conf_matrix)
        tenet_scores["Disambiguity"] = disamb_scores
        
        return tenet_scores
    
    def _evaluate_international_acceptance(self, target_word: str, ipa_words: List[str]) -> Dict[str, float]:
        """Calculate international acceptance scores."""
        standard_ipa = ipa.convert(target_word).strip()
        results = {}
        for var in ipa_words:
            score = self._panphon_international_acceptance(var, standard_ipa)
            results[var] = round(score, 3)
        return results
    
    def _panphon_international_acceptance(self, variance: str, standard: str) -> float:
        """Calculate panphon international acceptance score."""
        v = variance.strip("/ ")
        s = standard.strip("/ ")
        dist = self.distancer.feature_edit_distance(v, s)
        max_len = max(len(v), len(s))
        similarity = 1 - (dist / max_len) if max_len > 0 else 0
        return max(0, min(1, similarity))
    
    def _get_phonetic_simplicity_score(self, ipa_word: str) -> float:
        """Calculate phonetic simplicity score for an IPA word."""
        ipa_word = ipa_word.strip("/ ").replace(":", "")
        segments = self.ft.ipa_segs(ipa_word)
        total_complexity = 0
        total_segments = 0

        for seg in segments:
            if self.ft.seg_known(seg):
                feats = self.ft.segment_to_vector(seg)
                segment_complexity = sum(
                    self.complex_feature_weights[feat] for feat, val in zip(self.ft.names, feats)
                    if val == '+' and feat in self.complex_feature_weights
                )
                total_complexity += segment_complexity
                total_segments += 1

        if total_segments == 0:
            return 0.0

        avg_complexity = total_complexity / total_segments
        max_possible = sum(self.complex_feature_weights.values())
        normalized = avg_complexity / max_possible if max_possible else 0
        simplicity_score = round(1 - min(1.0, normalized), 3)

        return simplicity_score
    
    def _extract_ipa_and_fraction(self, freq_table: List) -> Dict[str, float]:
        """Extract IPA and fraction from frequency table."""
        ipa_fraction = {}
        for row in freq_table:
            if len(row) < 3:
                continue  # skip short or malformed rows
            if row[0].lower().startswith('ipa'):
                continue  # skip header
            try:
                ipa = row[0].strip()
                fraction = float(row[2])
                ipa_fraction[ipa] = fraction
            except ValueError:
                continue  # skip if conversion fails
        return ipa_fraction
    
    def _compute_contrastiveness(self, ipa_words: List[str]) -> Dict[str, float]:
        """Compute contrastiveness scores for IPA words."""
        contrastiveness_scores = {}
        for word in ipa_words:
            max_distance = 0
            for second_word in ipa_words:
                if word != second_word:
                    dist = self.distancer.weighted_feature_edit_distance(
                        word.strip("/"), second_word.strip("/")
                    )
                    max_distance = max(max_distance, dist)

            max_possible = 10 * max(len(w.strip("/")) for w in ipa_words)
            normalised_distance = min(max_distance / max_possible if max_possible else 0, 1.0)
            contrastiveness_scores[word] = round(normalised_distance, 2)

        return contrastiveness_scores
    
    def _compute_paedagogic_convenience(self, ipa: str) -> Tuple[float, List[str]]:
        """Compute paedagogic convenience score for an IPA transcription."""
        # Clean IPA input
        ipa_clean = ipa.strip("/").replace("ː", "")  # Remove length marks for simplicity
        score = 0
        explanations = []

        # 1. Consonant clusters
        cluster_pattern = r'[^' + self.ipa_vowels + r'\s]{2,}'
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
        syllable_pattern = '|'.join([re.escape(v) for v in self.ipa_vowel_list])
        syllables = len(re.findall(syllable_pattern, ipa_clean))
        if syllables >= 3:
            score += 1
            explanations.append("Trisyllabic or longer (+1)")

        # 3. Place & Manner Variegation
        segments = self.ft.ipa_segs(ipa_clean)
        places = set()
        manners = set()

        for seg in segments:
            if self.ft.seg_known(seg):
                vec = self.ft.segment_to_vector(seg)
                feature_dict = dict(zip(self.ft.names, vec))
                
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

        # Normalize score (max possible score is 6)
        max_score = 6
        normalized_score = 1 - (score / max_score) if max_score > 0 else 1.0
        normalized_score = round(normalized_score, 3)

        return normalized_score, explanations
    
    def _compute_disambiguity(self, ipa_words: List[str], conf_matrix: List) -> Dict[str, float]:
        """Compute disambiguity scores for IPA words."""
        disambiguity_scores = {}
        
        labels = [row[0] for row in conf_matrix[1:]]
        try:
            matrix = np.array([[float(val) for val in row[1:]] for row in conf_matrix[1:]])
        except ValueError:
            matrix = np.zeros((len(labels), len(labels)))
            
        for word in ipa_words:
            if word in labels:
                idx = labels.index(word)
                total = np.sum(matrix[idx])
                correct = matrix[idx, idx] if total > 0 else 0
                score = correct / total if total > 0 else 0.0
            else:
                # Fallback to distance-based calculation
                distances = []
                for other_word in ipa_words:
                    if word != other_word:
                        dist = self.distancer.weighted_feature_edit_distance(
                            word.strip("/"), other_word.strip("/")
                        )
                        distances.append(dist)
                max_possible = 10 * max(len(w.strip("/")) for w in ipa_words)
                avg_distance = np.mean(distances) if distances else 0
                score = avg_distance / max_possible if max_possible > 0 else 0.0

            normalized_score = round(min(score, 1.0), 3)
            disambiguity_scores[word] = normalized_score

        return disambiguity_scores
    
    def _build_score_table(self, tenet_scores: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Build a DataFrame from the tenet_scores dictionary."""
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

        df = pd.DataFrame.from_dict(tenet_scores)
        return df
    
    def _apply_ahp_analysis(self, sliders: Dict[str, int]) -> Dict[str, Any]:
        """Apply AHP analysis to user slider preferences."""
        full_name = {
            "IA": "International Acceptance",
            "PS": "Phonetic Simplicity",
            "PC": "Paedagogic Convenience",
            "CO": "Contrastiveness",
            "F": "Frequency",
            "DI": "Disambiguity"
        }
        
        user_input = {full_name[k]: v for k, v in sliders.items()}
        
        matrix, keys = self._build_comparison_matrix(user_input)
        weights = self._calculate_priority_vector(matrix)
        CR, lambda_max = self._calculate_consistency(matrix, weights)
        
        return {
            "tenets": keys,
            "weights": dict(zip(keys, weights)),
            "consistency_ratio": CR,
            "lambda_max": lambda_max
        }
    
    def _build_comparison_matrix(self, priorities: Dict[str, int]) -> Tuple[np.ndarray, List[str]]:
        """Build comparison matrix from priorities."""
        keys = list(priorities.keys())
        size = len(keys)
        matrix = np.ones((size, size))
        for i in range(size):
            for j in range(size):
                matrix[i][j] = priorities[keys[i]] / priorities[keys[j]]
        return matrix, keys
    
    def _calculate_priority_vector(self, matrix: np.ndarray) -> np.ndarray:
        """Calculate priority vector from comparison matrix."""
        col_sum = np.sum(matrix, axis=0)
        normalized = matrix / col_sum
        priority_vector = np.mean(normalized, axis=1)
        return priority_vector
    
    def _calculate_consistency(self, matrix: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
        """Calculate consistency ratio and lambda max."""
        n = matrix.shape[0]
        weighted_sum = np.dot(matrix, weights)
        lambda_max = np.sum(weighted_sum / weights) / n
        CI = (lambda_max - n) / (n - 1)
        
        RI_dict = Config.get_ahp_ri_values()
        RI = RI_dict.get(n, 1.49)
        CR = CI / RI if RI != 0 else 0
        return round(CR, 4), round(lambda_max, 4)
    
    def _apply_ahp_to_operability(self, score_table: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
        """Apply AHP weights to operability table."""
        score_table.columns = [col.lower().replace(" ", "_") for col in score_table.columns]
        weight_map = {k.lower().replace(" ", "_"): v for k, v in weights.items()}

        common_columns = [col for col in score_table.columns if col in weight_map]

        score_table['Weighted Score'] = score_table[common_columns].apply(
            lambda row: sum(row[col] * weight_map[col] for col in common_columns), axis=1
        ).round(4)

        return score_table.sort_values(by='Weighted Score', ascending=False)
    
    def _dataframe_to_dict(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Convert DataFrame to dictionary format for JSON response."""
        result = {}
        for index, row in df.iterrows():
            result[index] = {}
            for column in df.columns:
                if pd.notna(row[column]):
                    result[index][column] = float(row[column])
        return result
