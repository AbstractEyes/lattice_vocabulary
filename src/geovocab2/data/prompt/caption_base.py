import abc


class CaptionBase(abc.ABC):
    """
    Base class for caption synthesis prompts.
    """

    def __init__(self,
                 name: str = "caption_base",
                 uid: str = "prompt.caption_base"):
        self.name = name
        self.uid = uid

    def generate(self, *args, **kwargs) -> str:
        """
        Generate a caption based on provided arguments.

        Must be implemented by subclasses.

        Returns:
            Generated caption as a string.
        """
        pass

    def generate_batch(self, *args, **kwargs) -> list[str]:
        """
        Generate a batch of captions based on provided arguments.

        Must be implemented by subclasses.

        Returns:
            List of generated captions.
        """
        pass

    def __getitem__(self, key):
        pass