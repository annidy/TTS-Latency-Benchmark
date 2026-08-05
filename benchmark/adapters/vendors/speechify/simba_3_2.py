from ...base import ModelCapabilities
from ...registry import registry
from .base import SpeechifyBase


@registry.decorator("speechify.simba-3.2")
class SpeechifySimba32(SpeechifyBase):
    capabilities = ModelCapabilities(transports=("http",))

    @property
    def model_id(self) -> str:
        return str(self.setting("model.id", "simba-3.2"))
