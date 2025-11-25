import abc


class CaptionSynthBase(abc.ABC):
    """
    Base class for caption synthesis prompts.
    """

    def __init__(self,
                 name: str = "caption_synth_base",
                 uid: str = "prompt.caption_synth_base"):
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

    def __getitem__(self, key):
        pass