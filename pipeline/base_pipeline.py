
from pathlib import Path


class BasePipeline():
    def __init__(self, console_messages:bool =True):
        self.console_messages = console_messages
        self.dataset = None
        self.train_dataset = None
        self.test_dataset = None
        self.matrix_states = None
        self.dim_feature = None
        self.num_features = None

    def _console_message(self, message:str) -> None:
        """Prints a message to the console with a prefix.

        Args:
            message (str): Message to print.
        """
        if self.console_messages:
            print(f"\033[45mXAIForge: \033[0m{message}")
    
    def _get_path(self, file_name:str, directory:str, extension:str) -> Path:

        name = Path(file_name).stem
        path = Path(directory).joinpath(name).with_suffix(extension)

        if not path.exists():
            raise FileNotFoundError(f"File {path} not found.")
        return path