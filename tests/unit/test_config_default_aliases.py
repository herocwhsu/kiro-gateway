"""Pins the default alias and hide lists: the old default alias auto-kiro was listed in /v1/models
but never applied on the request path, so clients that chose it got HTTP 400."""
from kiro import config


def test_no_default_model_alias():
    assert config.MODEL_ALIASES == {}


def test_auto_is_not_hidden_from_the_model_list():
    assert "auto" not in config.HIDDEN_FROM_LIST


def test_the_models_list_shows_auto_and_no_auto_kiro():
    from kiro.cache import ModelInfoCache
    from kiro.model_resolver import ModelResolver

    cache = ModelInfoCache()
    for m in config.FALLBACK_MODELS:
        cache._cache[m["modelId"]] = m  # fill without the async update path
    resolver = ModelResolver(cache, hidden_models=config.HIDDEN_MODELS, aliases=config.MODEL_ALIASES, hidden_from_list=config.HIDDEN_FROM_LIST)
    models = resolver.get_available_models()
    assert "auto" in models
    assert "auto-kiro" not in models
