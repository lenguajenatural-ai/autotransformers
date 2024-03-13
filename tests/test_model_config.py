from autotransformers import ModelConfig
from transformers import MarianMTModel, Seq2SeqTrainer


def test_model_config():
    """Test that model config saves some parameters correctly."""

    def tokenize_dataset(examples):
        return examples

    def hp_space(trial):
        return trial

    marianmt_config = {
        "max_length_summary": 512,
        "n_trials": 1,
        "save_dir": "prueba_marianmt_savedir",
        "random_init_trials": 1,
        "name": "Helsinki-NLP/opus-mt-en-es",
        "save_name": "testname",
        "hp_space": hp_space,
        "num_beams": 4,
        "trainer_cls_summarization": Seq2SeqTrainer,
        "model_cls_summarization": MarianMTModel,
        "custom_tokenization_func": tokenize_dataset,
        "only_test": False,
    }

    marianmt_config = ModelConfig(**marianmt_config)
    assert (
        marianmt_config.save_name == "testname"
    ), f"The name should be testname and is {marianmt_config.save_name}"
    assert (
        marianmt_config.num_beams == 4
    ), f"Number of beams should be 4 and is {marianmt_config.num_beams}"
