from ...base import ModelCapabilities
from ...registry import registry
from .base import CartesiaBase


@registry.decorator("cartesia.sonic-3.5")
class CartesiaSonic35(CartesiaBase):
    capabilities = ModelCapabilities(
        transports=("http", "websocket"),
        websocket_persistent=True,
    )

    @property
    def model_id(self) -> str:
        return str(self.setting("model.id", "sonic-3.5-2026-05-04"))
