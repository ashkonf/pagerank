import logging
import sys

import nltk

logger = logging.getLogger(__name__)


def main() -> None:
    try:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        nltk.download("averaged_perceptron_tagger", quiet=True)
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        logger.info("NLTK data downloaded successfully")
    except Exception as e:
        logger.exception("Error downloading NLTK data: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
