from ...base import ModelCapabilities
from ...registry import registry
from .base import BreezeBlueBase


@registry.decorator("breezeblue.breeze-tts-2")
class BreezeBlueTTS2(BreezeBlueBase):
    capabilities = ModelCapabilities(
        transports=("http", "websocket"),
        websocket_persistent=True,
        supports_incremental_text=True,
        provides_server_timing=True,
    )

    @property
    def model_id(self) -> str:
        return str(self.setting("model.id", "breeze-tts-2"))
