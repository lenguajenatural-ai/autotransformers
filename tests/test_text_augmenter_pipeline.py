from autotransformers.augmentation import NLPAugPipeline, NLPAugConfig
from datasets import load_dataset


def test_aug_pipeline():
    """Test for text augmenter pipeline, test if it augments quantity of data."""
    dataset = load_dataset("avacaondata/wnli_tests")
    dataset = dataset["train"]
    steps = [
        NLPAugConfig(
            name="contextual_w_e",
            aug_kwargs={
                "model_path": "CenIA/albert-tiny-spanish",
                "action": "insert",
                "device": "cpu",
            },
        ),
    ]
    aug_pipeline = NLPAugPipeline(steps=steps, text_field="sentence1")
    augmented_dataset = dataset.map(aug_pipeline.augment, batched=True)
    assert len(augmented_dataset[:]["sentence1"]) > len(
        dataset[:]["sentence1"]
    ), "The dataset was not augmented."
