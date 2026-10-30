from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from textual.actions import SkipAction
from textual.geometry import Offset
from textual.selection import Selection
from textual.widgets import Input, OptionList, Static

from codeworld.config import FactorySettings
from codeworld.console.config_tui import (
    ConfigurationBrowser,
    ConfigurationEditorApp,
    ConfigurationSearch,
    SaveConfirmation,
    TerminalInput,
    ValueEditor,
    launch_configuration_editor,
)
from codeworld.console.configuration import RunConfigurationSession
from codeworld.console.models import CliInvocation, load_run_invocation, write_run_manifest
from codeworld.run_layout import RunLayout
from codeworld.state import StateStore


def _session(tmp_path: Path) -> RunConfigurationSession:
    (tmp_path / "configs").mkdir()
    config = tmp_path / "configs" / "build.yaml"
    config.write_text("concurrency: 4\n", encoding="utf-8")
    settings = FactorySettings.from_files((config,))
    run_directory = tmp_path / "outputs" / "build-test"
    write_run_manifest(
        CliInvocation(
            command="build",
            project_root=tmp_path,
            config_files=(config,),
            output_directory=run_directory,
            repository_source="owner/repo",
        ),
        "run-123",
    )
    state = StateStore(RunLayout(run_directory).state)
    state.create_run("run-123", "build", settings.model_dump(mode="json"))
    state.interrupt_run("run-123", "stopped")
    state.close()
    return RunConfigurationSession(tmp_path, run_directory)


def test_terminal_input_leaves_native_paste_to_terminal() -> None:
    editor = TerminalInput()

    with pytest.raises(SkipAction):
        editor.action_paste()


def test_configuration_tui_leaves_mouse_selection_to_the_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session(tmp_path)
    observed: dict[str, bool] = {}

    def run_without_a_terminal(
        _app: ConfigurationEditorApp,
        *,
        mouse: bool,
    ) -> None:
        observed["mouse"] = mouse

    monkeypatch.setattr(ConfigurationEditorApp, "run", run_without_a_terminal)

    assert launch_configuration_editor(tmp_path, session.run_directory) == 0
    assert observed == {"mouse": False}


@pytest.mark.asyncio
async def test_configuration_tui_supports_hierarchy_edit_modal_search_and_back(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path)
    revised_config = tmp_path / "configs" / "revised.yaml"
    revised_config.write_text("concurrency: 8\n", encoding="utf-8")
    with session:
        app = ConfigurationEditorApp(session)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            browser = cast(ConfigurationBrowser, app.screen)
            assert isinstance(browser, ConfigurationBrowser)
            assert browser.query_one("#config-options", OptionList).option_count == 3

            await pilot.press("enter")
            await pilot.pause()
            assert browser.group is not None and browser.group.id == "execution"

            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, ValueEditor)
            value_input = app.screen.query_one("#value-input", Input)
            assert value_input.has_focus

            value_input.selection = type(value_input.selection)(0, len(value_input.value))
            expected_clipboard = value_input.value
            await pilot.press("ctrl+c")
            await pilot.pause()
            assert app.clipboard == expected_clipboard
            assert value_input.selection.is_empty
            assert isinstance(app.screen, ValueEditor)

            help_text = app.screen.query_one("#value-help", Static)
            app.screen.selections = {help_text: Selection.from_offsets(Offset(0, 0), Offset(10, 0))}
            await pilot.press("super+c")
            await pilot.pause()
            assert app.clipboard == "Enter appl"
            assert not app.screen.selections
            assert isinstance(app.screen, ValueEditor)

            value_input.value = '["configs/revised.yaml"]'
            await pilot.press("enter")
            await pilot.pause()
            assert app.screen is browser
            assert session.pending_change_count == 1

            await pilot.press("escape")
            await pilot.pause()
            assert browser.group is None

            options = browser.query_one("#config-options", OptionList)
            options.highlighted = 2
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, SaveConfirmation)

            await pilot.press("escape")
            await pilot.pause()
            assert app.screen is browser
            assert session.pending_change_count == 1

            options = browser.query_one("#config-options", OptionList)
            options.highlighted = 2
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, SaveConfirmation)
            await pilot.press("down", "enter")
            await pilot.pause()
            assert app.screen is browser
            assert session.pending_change_count == 0
            assert load_run_invocation(tmp_path, session.run_directory).config_files == (
                revised_config,
            )
            assert app.saved_changes == 1

            options = browser.query_one("#config-options", OptionList)
            options.highlighted = 1
            await pilot.press("enter")
            await pilot.pause()
            search = cast(ConfigurationSearch, app.screen)
            assert isinstance(search, ConfigurationSearch)
            query = search.query_one("#search-query", Input)
            assert query.has_focus

            await pilot.press("l", "i", "m", "i", "t")
            await pilot.pause()
            assert search.results
            assert any("limit" in result.label for result in search.results)

            await pilot.press("escape")
            await pilot.pause()
            assert app.screen is browser
