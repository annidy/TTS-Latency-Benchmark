from ...base import ModelCapabilities
from ...registry import registry
from .base import ElevenLabsBase


@registry.decorator("elevenlabs.v3")
class ElevenLabsV3(ElevenLabsBase):
    capabilities = ModelCapabilities(transports=("http",))

    @property
    def model_id(self) -> str:
        return str(self.setting("model.id", "eleven_v3"))
