class SkipMix:
    """
    Simple class to skip mix of dataset and model.

    Two properties: dataset to skip and model to skip.

    Parameters
    ----------
    dataset_name: str
        Name of the dataset, as in alias parameter of DatasetConfig.
    model_name: str
        Name of the model, as in save_name parameter of ModelConfig.
    """

    def __init__(self, dataset_name: str, model_name: str):
        self.dataset_name = dataset_name
        self.model_name = model_name
