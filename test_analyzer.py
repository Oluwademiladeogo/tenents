#!/usr/bin/env python3
"""
Test script for the PhoneticAnalyzer to verify the refactored implementation works correctly.
"""

import json
from phonetic_analyzer import PhoneticAnalyzer, AnalysisRequest

def test_analyzer():
    """Test the phonetic analyzer with sample data."""
    
    # Sample request data (similar to what would be sent in a POST request)
    sample_request = {
        "word": "way",
        "ipa_variants": [
            {
                "ipa": "dˈeɪ",
                "frequency": 1.0,
                "fraction": 0.5
            },
            {
                "ipa": "wˈeɪ",
                "frequency": 1.0,
                "fraction": 0.5
            }
        ],
        "confusion_matrix": {
            "labels": ["dˈeɪ", "wˈeɪ"],
            "matrix": [
                [1.0, 0.75],
                [0.75, 1.0]
            ]
        },
        "sliders": {
            "IA": 6,
            "DI": 5,
            "CO": 4,
            "PC": 3,
            "PS": 2,
            "F": 1
        }
    }
    
    try:
        # Initialize analyzer
        analyzer = PhoneticAnalyzer()
        
        # Create analysis request
        request = AnalysisRequest(
            word=sample_request["word"],
            ipa_variants=sample_request["ipa_variants"],
            confusion_matrix=sample_request["confusion_matrix"],
            sliders=sample_request["sliders"]
        )
        
        # Perform analysis
        print("Starting analysis...")
        result = analyzer.analyze(request)
        
        # Display results
        print(f"\n=== Analysis Results ===")
        print(f"Target Word: {result.target_word}")
        print(f"Best Transcription: {result.best_transcription}")
        print(f"Consistency Ratio: {result.consistency_ratio}")
        
        print(f"\n=== AHP Weights ===")
        for tenet, weight in result.ahp_weights.items():
            print(f"{tenet}: {weight:.4f}")
        
        print(f"\n=== Final Table ===")
        for ipa, scores in result.final_table.items():
            print(f"{ipa}: {scores}")
        
        print(f"\n✅ Analysis completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_analyzer()
    exit(0 if success else 1) 