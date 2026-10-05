"""Entities subscribe to updates only once Home Assistant adds them (#47)."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from custom_components.album_slideshow import sensor, text

_ENTRY = SimpleNamespace(entry_id="entry", title="Album")


def _store():
    store = SimpleNamespace(listeners=[], hidden_photo_ids=set(), pair_divider_color="#ffffff")
    store.add_listener = store.listeners.append
    return store


class _Coordinator:
    provider = "local_folder"

    def __init__(self):
        self.data = {}
        self.listeners = []
        self.store = _store()

    def async_add_listener(self, callback):
        self.listeners.append(callback)
        return lambda: self.listeners.remove(callback)


def test_disabled_sensors_never_subscribe():
    # Home Assistant constructs disabled entities but never adds them.
    coordinator = _Coordinator()
    for cls in (
        sensor.AlbumCountSensor,
        sensor.AlbumTitleSensor,
        sensor.CacheUsageSensor,
        sensor.HiddenPhotoCountSensor,
        sensor.EnrichmentProgressSensor,
    ):
        cls(_ENTRY, coordinator)
    assert coordinator.listeners == []
    assert coordinator.store.listeners == []


def test_sensor_subscribes_when_added_and_unsubscribes_when_removed():
    coordinator = _Coordinator()
    hidden = sensor.HiddenPhotoCountSensor(_ENTRY, coordinator)
    asyncio.run(hidden.async_added_to_hass())
    assert coordinator.listeners == [hidden.async_write_ha_state]
    assert coordinator.store.listeners == [hidden.async_write_ha_state]

    for remove in hidden._on_remove:
        remove()
    assert coordinator.listeners == []


def test_disabled_text_entity_never_subscribes():
    store = _store()
    entity = text.PairDividerColorText(_ENTRY, store)
    assert store.listeners == []

    asyncio.run(entity.async_added_to_hass())
    assert store.listeners == [entity.async_write_ha_state]
