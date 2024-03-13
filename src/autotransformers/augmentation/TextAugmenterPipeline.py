try:
    from .augmenter_config import class_translator
except:
    print("Warning: Your current setup does not support data augmentation.")
from tqdm import tqdm
import numpy as np


class NLPAugPipeline:
    """
    Augment text data, with various forms of augmenting. It uses `nlpaug` in the background.

    The configuration of the augmentation pipeline is done with `autotransformers.augmentation.augmenter_config.NLPAugConfig`.
    NLPAugPipeline receives a list of configs of that type, where each config defines a type
    of augmentation technique to use, as well as the proportion of the train dataset that is
    to be augmented.

    Parameters
    ----------
    steps: List[autotransformers.augmentation.augmenter_config.NLPAugConfig]
        List of steps. Each step must be a NLPAugConfig instance.
    text_field: str
        Name of the field in the dataset where texts are.
    """

    def __init__(self, steps, text_field: str = "text"):
        self.text_field = text_field
        self.pipeline = {
            i: {
                "augmenter": class_translator[config.name](**config.aug_kwargs) if config.augmenter_cls is None else config.augmenter_cls(**config.aug_kwargs),
                "prop": config.proportion,
            }
            for i, config in enumerate(steps)
        }

    def augment(self, samples):
        """
        Augment data for datasets samples following the configuration defined at init.

        Parameters
        ----------
        samples:
            Samples from a datasets.Dataset

        Returns
        -------
        samples:
            Samples from a datasets.Dataset but processed.
        """
        fields = [k for k in samples.keys()]
        new_samples = {field: [] for field in fields}
        for augmenter in tqdm(
            self.pipeline, desc="Iterating over data augmentation methods..."
        ):
            samples_selection_idxs = np.random.choice(
                range(len(samples[fields[0]])),
                size=int(self.pipeline[augmenter]["prop"] * len(samples[fields[0]])),
                replace=False,
            )
            texts_augment = [
                samples[self.text_field][idx] for idx in samples_selection_idxs
            ]
            augmented_texts = self.pipeline[augmenter]["augmenter"].augment(
                texts_augment
            )
            for example_idx, augmented_example in zip(
                samples_selection_idxs, augmented_texts
            ):
                for field in fields:
                    if field == self.text_field:
                        new_samples[field].append(augmented_example)
                    else:
                        new_samples[field].append(samples[field][example_idx])
        for field in tqdm(fields, desc="Updating samples batch with augmented data..."):
            samples[field].extend(new_samples[field])
        return samples
