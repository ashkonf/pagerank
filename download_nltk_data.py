import logging
import sys

import nltk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    logger.info("NLTK data downloaded successfully")
except Exception as e:
    logger.error("Error downloading NLTK data: %s", e)
    sys.exit(1)
