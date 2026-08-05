from ...base import ModelCapabilities
from ...registry import registry
from .base import XAIBase


@registry.decorator("xai.tts")
class XAITTS(XAIBase):
    capabilities = ModelCapabilities(
        transports=("http", "websocket"),
        websocket_persistent=True,
        supports_incremental_text=True,
    )

    @property
    def model_id(self) -> str:
        return str(self.setting("model.id", "xai-tts"))
