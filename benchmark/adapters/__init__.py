from .base import (
    AdapterProtocol,
    ModelAdapter,
    ModelCapabilities,
    MultiTransportModelAdapter,
)
from .registry import AdapterRegistry, registry

# Import the deliberately small supported vendor set for registration side effects.
from .vendors.async_voice import flash_v1_5 as _async_flash_v1_5  # noqa: F401
from .vendors.breezeblue import breeze_tts_2 as _breeze_tts_2  # noqa: F401
from .vendors.cartesia import sonic_3_5 as _sonic_3_5  # noqa: F401
from .vendors.elevenlabs import flash_v2_5 as _elevenlabs_flash_v2_5  # noqa: F401
from .vendors.elevenlabs import v3 as _elevenlabs_v3  # noqa: F401
from .vendors.fish_audio import s2_1_pro as _fish_s2_1_pro  # noqa: F401
from .vendors.inworld import tts_2 as _inworld_tts_2  # noqa: F401
from .vendors.speechify import simba_3_2 as _speechify_simba_3_2  # noqa: F401
from .vendors.xai import tts as _xai_tts  # noqa: F401

__all__ = [
    "AdapterProtocol",
    "AdapterRegistry",
    "ModelAdapter",
    "ModelCapabilities",
    "MultiTransportModelAdapter",
    "registry",
]
