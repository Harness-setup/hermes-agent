"""Regression tests for _emit_voice_state_hook (tui_gateway/server.py).

wake.detected/voice.status/TTS-playback events only reach the JSON-RPC
transport straight to the connected UI client -- no plugin hook exists for
any of them. _emit_voice_state_hook is new, additive instrumentation so
backend plugins (e.g. pebble-signal) can reflect listening/speaking on a
visual indicator.
"""
from pathlib import Path

import yaml

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tui_gateway import server


def _make_capture_plugin(hermes_home: Path, sink: Path) -> None:
    plugin_dir = hermes_home / "plugins" / "capture_voice_state"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump({"name": "capture_voice_state", "version": "0.1.0"}), encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(
        "def register(ctx):\n"
        f"    def _capture(**kw):\n"
        f"        open({str(sink)!r}, 'a').write(kw.get('state', '') + '\\n')\n"
        f"    ctx.register_hook('voice_state_changed', _capture)\n",
        encoding="utf-8",
    )
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": ["capture_voice_state"]}}), encoding="utf-8",
    )


def _load_plugins():
    import hermes_cli.plugins as plugins_mod
    plugins_mod._plugin_manager = plugins_mod.PluginManager()
    plugins_mod._plugin_manager.discover_and_load()


def test_emit_voice_state_hook_fires_with_correct_state(tmp_path, monkeypatch):
    # NOTE: named "voice_hook_home", not "hermes_test" -- tests/conftest.py's
    # autouse `_hermetic_environment` fixture already creates and mkdir()s
    # `tmp_path / "hermes_test"` as the sandboxed HERMES_HOME for every test,
    # so reusing that name here collides with FileExistsError regardless of
    # this test's own logic.
    hermes_home = tmp_path / "voice_hook_home"
    hermes_home.mkdir()
    sink = hermes_home / "captured.log"
    _make_capture_plugin(hermes_home, sink)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    token = set_hermes_home_override(hermes_home)
    try:
        _load_plugins()
        server._emit_voice_state_hook("listening")
        server._emit_voice_state_hook("speaking")
        server._emit_voice_state_hook("idle")
    finally:
        reset_hermes_home_override(token)

    assert sink.read_text().splitlines() == ["listening", "speaking", "idle"]


def test_emit_voice_state_hook_never_raises_even_if_invoke_hook_fails(monkeypatch):
    import hermes_cli.plugins as plugins_mod

    def _boom(*a, **k):
        raise RuntimeError("plugin manager unavailable")

    monkeypatch.setattr(plugins_mod, "invoke_hook", _boom)
    server._emit_voice_state_hook("speaking")  # must not raise
