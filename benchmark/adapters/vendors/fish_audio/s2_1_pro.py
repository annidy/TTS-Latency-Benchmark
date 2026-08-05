from ...base import ModelCapabilities
from ...registry import registry
from .base import FishAudioBase


@registry.decorator("fish-audio.s2.1-pro")
class FishAudioS21Pro(FishAudioBase):
    capabilities = ModelCapabilities(
        transports=("http", "websocket"),
        websocket_persistent=True,
        supports_incremental_text=True,
    )

    @property
    def model_id(self) -> str:
        return str(self.setting("model.id", "s2.1-pro-free"))
