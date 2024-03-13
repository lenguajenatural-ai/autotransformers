from dataclasses import dataclass, field
from typing import Dict, Any
try:
    import nlpaug.augmenter.char as nac
    import nlpaug.augmenter.word as naw
    import nlpaug.augmenter.sentence as nas

    class_translator = {
        "ocr": nac.OcrAug,
        "contextual_w_e": naw.ContextualWordEmbsAug,
        "synonym": naw.SynonymAug,
        "backtranslation": naw.BackTranslationAug,
        "contextual_s_e": nas.ContextualWordEmbsForSentenceAug,
        "abstractive_summ": nas.AbstSummAug,
    }
except:
    print("Warning: Your current setup does not support data augmentation.")


@dataclass
class NLPAugConfig:
    """
    Configuration for augmenters.

    Parameters
    ----------
    name : str
        Name of the data augmentation technique. Possible values currently are `ocr` (for OCR augmentation), `contextual_w_e`
        for Contextual Word Embedding augmentation, `synonym`, `backtranslation`, `contextual_s_e` for Contextual Word Embeddings for Sentence Augmentation,
        `abstractive_summ`. If using a custom augmenter class this can be a random name.
    augmenter_cls: Any
        An optional augmenter class, from `nlpaug` library. Can be used instead of using an identifier name
        for loading the class (see param `name` of this class).
    proportion : float
        Proportion of data augmentation.
    aug_kwargs : Dict
        Arguments for the data augmentation class. See https://github.com/makcedward/nlpaug/blob/master/example/textual_augmenter.ipynb
    """

    name: str = field(metadata={"help": "Name of the data augmentation technique. If using a custom augmenter class this can be a random name."})
    augmenter_cls: Any = field(
        default=None,
        metadata={"help": "An optional augmenter class, from `nlpaug` library. Can be used instead of using an identifier name for loading the class (see param `name` of this class)."}
    )
    proportion: float = field(
        default=0.1, metadata={"help": "proportion of data augmentation"}
    )
    aug_kwargs: Dict = field(
        default=None,
        metadata={
            "help": "Arguments for the data augmentation class. See https://github.com/makcedward/nlpaug/blob/master/example/textual_augmenter.ipynb"
        },
    )
