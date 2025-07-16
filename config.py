import os
from typing import Dict, Any

class Config:
    """Configuration settings for the Tenets API."""
    
    # Analysis settings
    COMPLEX_FEATURE_WEIGHTS: Dict[str, float] = {
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
        'strident': 2.0,
        'long': 1.2,
        'stress': 0.8,
        'tone': 1.5,
    }
    
    # AHP settings
    AHP_RI_VALUES: Dict[int, float] = {
        1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
        6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49
    }
    
    # IPA vowel list for paedagogic convenience
    IPA_VOWEL_LIST: list = [
        'iː', 'ɪ', 'e', 'æ', 'ʌ', 'ɑː', 'ɒ', 'ɔː', 'ʊ', 'uː', 'ə', 'ɜː',
        'eɪ', 'aɪ', 'ɔɪ', 'əʊ', 'aʊ', 'ɪə', 'eə', 'ʊə', 'eɪə', 'aɪə'
    ]
    
    # Environment settings
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "5000"))
    
    # API settings
    API_TITLE: str = "Tenets API"
    API_DESCRIPTION: str = "API for phonetic analysis and rankings"
    API_VERSION: str = "1.0.0"
    
    # CORS settings
    CORS_ORIGINS: list = ["*"]  
    CORS_CREDENTIALS: bool = True
    CORS_METHODS: list = ["*"]
    CORS_HEADERS: list = ["*"]
    
    @classmethod
    def get_feature_weights(cls) -> Dict[str, float]:
        """Get a copy of the feature weights dictionary."""
        return cls.COMPLEX_FEATURE_WEIGHTS.copy()
    
    @classmethod
    def get_ahp_ri_values(cls) -> Dict[int, float]:
        """Get a copy of the AHP RI values dictionary."""
        return cls.AHP_RI_VALUES.copy()
    
    @classmethod
    def get_ipa_vowel_list(cls) -> list:
        """Get a copy of the IPA vowel list."""
        return cls.IPA_VOWEL_LIST.copy() 