from ...base import ModelCapabilities
from ...registry import registry
from .base import AsyncVoiceBase


@registry.decorator("async.flash-v1.5")
class AsyncFlashV15(AsyncVoiceBase):
    capabilities = ModelCapabilities(
        transports=("http", "websocket"),
        websocket_persistent=True,
        supports_incremental_text=True,
    )

    @property
    def model_id(self) -> str:
        return str(self.setting("model.id", "async_flash_v1.5"))
