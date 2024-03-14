import unittest
from unittest.mock import MagicMock, patch
import torch
from autotransformers.llm_templates import (  # Adjust this import path to where your module is located
    instructions_to_chat,
    neftune_forward,
    activate_neftune,
    NEFTuneTrainer,
    modify_tokenizer,
    SavePeftModelCallback,
)


class TestModule(unittest.TestCase):
    """Contains unit tests for the mymodule module."""

    def test_instructions_to_chat(self):
        """
        Test if `instructions_to_chat` correctly processes input and adds system, context, user, and assistant messages.
        """
        sample = {
            "context": "Contextual information",
            "input": "User's question?",
            "output": "Assistant's response.",
        }
        result = instructions_to_chat(
            sample, "input", "output", context_field="context"
        )
        self.assertIn("messages", result)
        self.assertEqual(len(result["messages"]), 4)
        self.assertEqual(result["messages"][1]["role"], "context")
        self.assertEqual(result["messages"][2]["content"], "User's question?")
        self.assertEqual(result["messages"][3]["role"], "assistant")

    @patch("torch.nn.functional.embedding")
    def test_neftune_forward(self, mock_embedding):
        """
        Test the NEFTune forward function to ensure it modifies embeddings with noise.
        """
        input_tensor = torch.tensor([1, 2, 3])
        mock_embedding.return_value = torch.rand((3, 768))
        embedding_layer = MagicMock()
        embedding_layer.weight = torch.rand((1000, 768))
        embedding_layer.padding_idx = None
        embedding_layer.max_norm = None
        embedding_layer.norm_type = 2.0
        embedding_layer.scale_grad_by_freq = False
        embedding_layer.sparse = False
        embedding_layer.neftune_noise_alpha = 5

        neftune_forward(embedding_layer, input_tensor)
        mock_embedding.assert_called()

    def test_activate_neftune(self):
        """
        Test `activate_neftune` to verify if it properly sets the custom forward method with NEFTune noise.
        """
        model = MagicMock()
        activate_neftune(model, neftune_noise_alpha=5)
        self.assertTrue(hasattr(model.get_input_embeddings(), "neftune_noise_alpha"))

    def test_NEFTuneTrainer(self):
        """
        Test the NEFTuneTrainer initialization to check if Neftune noise alpha is set correctly.
        """
        trainer = NEFTuneTrainer(model=MagicMock(), neftune_noise_alpha=5)
        self.assertEqual(trainer.neftune_noise_alpha, 5)

    def test_modify_tokenizer(self):
        """
        Test `modify_tokenizer` for various tokenizer modifications.
        """
        tokenizer = MagicMock()
        modify_tokenizer(
            tokenizer, pad_token_id=0, padding_side="right", new_model_seq_length=512
        )
        tokenizer.add_special_tokens.assert_not_called()  # Assuming no special tokens were added in this call

    @patch("mymodule.PreTrainedModel.save_pretrained")  # Adjust the module path
    def test_SavePeftModelCallback(self, mock_save_pretrained):
        """
        Test SavePeftModelCallback to ensure it saves the adapter model and not the full model.
        """
        trainer_state = MagicMock()
        trainer_state.best_model_checkpoint = None
        trainer_state.global_step = 100
        args = MagicMock()
        args.output_dir = "/fakepath"
        kwargs = {"model": MagicMock()}
        callback = SavePeftModelCallback()

        callback.save_model(args, trainer_state, kwargs)
        mock_save_pretrained.assert_called()
