from ...base import ModelCapabilities
from ...registry import registry
from .base import InworldBase


@registry.decorator("inworld.tts-2")
class InworldTTS2(InworldBase):
    capabilities = ModelCapabilities(
        transports=("http", "websocket"),
        websocket_persistent=True,
    )

    @property
    def model_id(self) -> str:
        return str(self.setting("model.id", "inworld-tts-2"))
