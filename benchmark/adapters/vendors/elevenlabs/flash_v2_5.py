from ...base import ModelCapabilities
from ...registry import registry
from .base import ElevenLabsBase


@registry.decorator("elevenlabs.flash-v2.5")
class ElevenLabsFlashV25(ElevenLabsBase):
    capabilities = ModelCapabilities(
        transports=("http", "websocket"),
        websocket_persistent=True,
        supports_incremental_text=True,
    )

    @property
    def model_id(self) -> str:
        return str(self.setting("model.id", "eleven_flash_v2_5"))
