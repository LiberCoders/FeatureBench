"""Textual YAML-path and execution-input editor for one durable run."""

from __future__ import annotations

import os
from pathlib import Path
from typing import ClassVar, cast

# Native terminal selection must survive pressing Command in VS Code. Textual's
# enhanced keyboard protocol reports modifier-only key presses, which redraws the
# alternate screen and clears that selection before Command+C can copy it.
os.environ["TEXTUAL_DISABLE_KITTY_KEY"] = "1"

from rich.text import Text
from textual.actions import SkipAction
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, Vertical
from textual.screen import ModalScreen, Screen
from textual.widgets import Input, Label, OptionList, Static
from textual.widgets.option_list import Option

from codeworld.console.configuration import (
    CONFIGURATION_GROUPS,
    ConfigPath,
    ConfigurationEntry,
    ConfigurationGroup,
    RunConfigurationSession,
    SearchResult,
    format_config_path,
)


class TerminalInput(Input):
    """Input using Textual selection/copy while preserving terminal-native paste."""

    def action_paste(self) -> None:
        raise SkipAction()


class ValueEditor(ModalScreen[str | None]):
    """Single-value YAML editor that applies a validated draft change."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", show=False, priority=True),
    ]

    def __init__(self, *, title: str, value: str) -> None:
        super().__init__()
        self.editor_title = title
        self.value = value

    def compose(self) -> ComposeResult:
        with Container(id="value-dialog"):
            yield Label(self.editor_title, id="value-title")
            editor = TerminalInput(
                value=self.value,
                id="value-input",
                select_on_focus=False,
            )
            editor.cursor_blink = False
            yield editor
            yield Static(
                "Enter apply · Esc cancel · Ctrl/Command+C copy",
                id="value-help",
            )

    def on_mount(self) -> None:
        editor = self.query_one("#value-input", Input)
        editor.focus()
        editor.cursor_position = len(editor.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    def action_cancel(self) -> None:
        self.dismiss(None)


class SaveConfirmation(ModalScreen[bool]):
    """Explicit confirmation before the complete validated draft is persisted."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", show=False, priority=True),
    ]

    def __init__(self, change_count: int) -> None:
        super().__init__()
        self.change_count = change_count

    def compose(self) -> ComposeResult:
        with Container(id="save-dialog"):
            yield Label("Save run inputs?", id="save-title")
            yield Static(
                f"{self.change_count} pending change(s)",
                id="save-message",
            )
            yield OptionList(
                Option("No", id="no"),
                Option("Yes", id="yes"),
                id="save-options",
                compact=True,
            )
            yield Static("↑↓ navigate · Enter confirm · Esc cancel", id="save-help")

    def on_mount(self) -> None:
        options = self.query_one("#save-options", OptionList)
        options.highlighted = 0
        options.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option_id == "yes")

    def action_cancel(self) -> None:
        self.dismiss(False)


def _entry_prompt(entry: ConfigurationEntry) -> Text:
    prompt = Text()
    prompt.append("▸ " if entry.is_container else "  ", style="#7487e8")
    prompt.append(entry.name, style="bold #e2e7ff" if entry.is_container else "#cfd7f3")
    prompt.append("    ")
    prompt.append(entry.rendered_value, style="#7f8bad")
    return prompt


class ConfigurationBrowser(Screen[None]):
    """Hierarchical navigation for a run's YAML paths and execution inputs."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "back", show=False, priority=True),
    ]

    def __init__(self, session: RunConfigurationSession) -> None:
        super().__init__()
        self.session = session
        self.group: ConfigurationGroup | None = None
        self.paths: list[ConfigPath] = []
        self.items: list[ConfigurationGroup | ConfigurationEntry | str] = []
        self.editing_path: ConfigPath | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="config-shell"):
            yield Static("", id="config-title")
            yield Static("", id="config-breadcrumb")
            yield OptionList(id="config-options", compact=True)
            yield Static("", id="config-status")
            yield Static(
                "↑↓ navigate · Enter open/edit · Esc back · Ctrl+C exit",
                id="config-help",
            )

    def on_mount(self) -> None:
        self.query_one("#config-status", Static).display = False
        self._refresh()
        self.query_one("#config-options", OptionList).focus()

    def on_screen_resume(self) -> None:
        self._refresh()
        self.query_one("#config-options", OptionList).focus()

    @property
    def config_app(self) -> ConfigurationEditorApp:
        return cast(ConfigurationEditorApp, self.app)  # pyright: ignore[reportUnknownMemberType]

    def _current_entries(self) -> tuple[ConfigurationEntry, ...]:
        if self.group is None:
            return ()
        if self.paths:
            return self.session.children(self.paths[-1])
        return self.session.group_entries(self.group.id)

    def _refresh(self) -> None:
        title = Text()
        title.append("FeatureBench run inputs", style="bold #8da0ff")
        title.append(f"  {self.session.run_directory.name}", style="#dce2fa")
        title.append(f"  ·  {self.session.status}", style="#7f8bad")
        if self.session.pending_change_count:
            title.append(
                f"  ·  {self.session.pending_change_count} unsaved",
                style="bold #e8b86d",
            )
        self.query_one("#config-title", Static).update(title)

        options = self.query_one("#config-options", OptionList)
        options.clear_options()
        self.items.clear()
        if self.group is None:
            breadcrumb = "Run inputs"
            for group in CONFIGURATION_GROUPS:
                self.items.append(group)
                options.add_option(Option(f"▸ {group.label}", id=str(len(self.items) - 1)))
            self.items.append("search")
            options.add_option(Option("⌕ Search", id=str(len(self.items) - 1)))
            self.items.append("save")
            save_prompt = Text()
            save_prompt.append("✓ Save", style="bold #72d6b0")
            save_prompt.append(
                f"    {self.session.pending_change_count} pending change(s)",
                style="#7f8bad",
            )
            options.add_option(Option(save_prompt, id=str(len(self.items) - 1)))
        else:
            suffix = " / " + format_config_path(self.paths[-1]) if self.paths else ""
            breadcrumb = f"Run inputs / {self.group.label}{suffix}"
            entries = self._current_entries()
            for entry in entries:
                self.items.append(entry)
                options.add_option(Option(_entry_prompt(entry), id=str(len(self.items) - 1)))
            if not entries:
                options.add_option(Option("No run input fields", id="empty", disabled=True))
        self.query_one("#config-breadcrumb", Static).update(breadcrumb)
        if options.option_count:
            options.highlighted = 0

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_id == "empty":
            return
        try:
            item = self.items[event.option_index]
        except IndexError:
            return
        if isinstance(item, ConfigurationGroup):
            self.group = item
            self.paths.clear()
            self._refresh()
            return
        if item == "search":
            self.config_app.push_screen(ConfigurationSearch(self.session))
            return
        if item == "save":
            self._request_save()
            return
        if not isinstance(item, ConfigurationEntry):
            return
        if item.is_container:
            self.paths.append(item.path)
            self._refresh()
            return
        self._edit(
            item.path, f"{self.group.label if self.group else ''}.{format_config_path(item.path)}"
        )

    def _edit(self, path: ConfigPath, title: str) -> None:
        self.editing_path = path
        current = self.session.entry(path)
        self.config_app.push_screen(
            ValueEditor(title=title, value=current.rendered_value),
            self._apply_value,
        )

    def _apply_value(self, value: str | None) -> None:
        path = self.editing_path
        self.editing_path = None
        if path is None or value is None:
            self.query_one("#config-options", OptionList).focus()
            return
        status = self.query_one("#config-status", Static)
        try:
            result = self.session.edit(path, value)
        except Exception as exc:
            status.update(f"[bold red]Not applied:[/] {exc}")
            status.display = True
        else:
            status.update(
                f"[bold #e8b86d]Draft updated[/] {format_config_path(path)} = {result.new_value!r}"
            )
            status.display = True
            self._refresh()
        self.query_one("#config-options", OptionList).focus()

    def _request_save(self) -> None:
        status = self.query_one("#config-status", Static)
        if not self.session.pending_change_count:
            status.update("[yellow]No pending run input changes.[/]")
            status.display = True
            return
        self.config_app.push_screen(
            SaveConfirmation(self.session.pending_change_count),
            self._confirm_save,
        )

    def _confirm_save(self, confirmed: bool | None) -> None:
        status = self.query_one("#config-status", Static)
        if not confirmed:
            status.update("[yellow]Save cancelled.[/]")
            status.display = True
            self.query_one("#config-options", OptionList).focus()
            return
        try:
            result = self.session.save()
        except Exception as exc:
            status.update(f"[bold red]Not saved:[/] {exc}")
            status.display = True
        else:
            self.config_app.saved_changes += result.count
            if result.warning:
                status.update(f"[yellow]{result.warning}[/]")
            else:
                status.update(f"[bold #72d6b0]Saved[/] {result.count} run input change(s)")
            status.display = True
            self._refresh()
        self.query_one("#config-options", OptionList).focus()

    def action_back(self) -> None:
        if self.paths:
            self.paths.pop()
            self._refresh()
            return
        if self.group is not None:
            self.group = None
            self._refresh()


class ConfigurationSearch(Screen[None]):
    """Cross-owner fuzzy search that opens the same validated field editor."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "previous_result", show=False, priority=True),
        Binding("down", "next_result", show=False, priority=True),
        Binding("enter", "edit_result", show=False, priority=True),
        Binding("escape", "back", show=False, priority=True),
    ]

    def __init__(self, session: RunConfigurationSession) -> None:
        super().__init__()
        self.session = session
        self.results: tuple[SearchResult, ...] = ()
        self.editing_path: ConfigPath | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="search-shell"):
            yield Static("Search", id="search-title")
            query = TerminalInput(
                placeholder="Search…",
                id="search-query",
                select_on_focus=False,
            )
            query.cursor_blink = False
            yield query
            yield OptionList(id="search-results", compact=True)
            yield Static("", id="search-status")
            yield Static(
                "↑↓ navigate · Enter edit · Esc back · Ctrl+C exit",
                id="search-help",
            )

    def on_mount(self) -> None:
        self.query_one("#search-status", Static).display = False
        self.query_one("#search-query", Input).focus()
        self._refresh_results("")

    @property
    def config_app(self) -> ConfigurationEditorApp:
        return cast(ConfigurationEditorApp, self.app)  # pyright: ignore[reportUnknownMemberType]

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "search-query":
            self._refresh_results(event.value)

    def _refresh_results(self, query: str) -> None:
        self.results = self.session.search(query)
        options = self.query_one("#search-results", OptionList)
        options.clear_options()
        for index, result in enumerate(self.results):
            prompt = Text()
            prompt.append(result.label, style="#cfd7f3")
            prompt.append("    ")
            prompt.append(result.entry.rendered_value, style="#7f8bad")
            options.add_option(Option(prompt, id=str(index)))
        if not self.results:
            options.add_option(Option("No matching fields", id="empty", disabled=True))
        if options.option_count:
            options.highlighted = 0

    def _move(self, delta: int) -> None:
        options = self.query_one("#search-results", OptionList)
        if not self.results:
            return
        current = options.highlighted or 0
        options.highlighted = max(0, min(len(self.results) - 1, current + delta))

    def action_previous_result(self) -> None:
        self._move(-1)

    def action_next_result(self) -> None:
        self._move(1)

    def action_edit_result(self) -> None:
        options = self.query_one("#search-results", OptionList)
        index = options.highlighted
        if index is None or not 0 <= index < len(self.results):
            return
        result = self.results[index]
        self.editing_path = result.entry.path
        self.config_app.push_screen(
            ValueEditor(title=result.label, value=result.entry.rendered_value),
            self._apply_value,
        )

    def _apply_value(self, value: str | None) -> None:
        path = self.editing_path
        self.editing_path = None
        if path is None or value is None:
            self.query_one("#search-query", Input).focus()
            return
        status = self.query_one("#search-status", Static)
        try:
            result = self.session.edit(path, value)
        except Exception as exc:
            status.update(f"[bold red]Not applied:[/] {exc}")
            status.display = True
        else:
            status.update(
                f"[bold #e8b86d]Draft updated[/] {format_config_path(path)} = {result.new_value!r}"
            )
            status.display = True
            query = self.query_one("#search-query", Input).value
            self._refresh_results(query)
        self.query_one("#search-query", Input).focus()

    def action_back(self) -> None:
        self.config_app.pop_screen()


class ConfigurationEditorApp(App[None]):
    """Full-screen, keyboard-first editor whose execution core has no UI dependency."""

    TITLE = "FeatureBench Run Inputs"
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("ctrl+c", "copy_or_exit", show=False, priority=True),
        Binding("super+c", "copy_selection", show=False, priority=True),
    ]

    CSS = """
    Screen {
        background: #080b14;
        color: #dfe5ff;
    }
    #config-shell, #search-shell {
        width: 100%;
        height: 100%;
        padding: 1 2;
    }
    #config-title, #search-title {
        height: 2;
        text-style: bold;
    }
    #config-breadcrumb {
        height: 2;
        color: #8491b6;
    }
    #config-options, #search-results {
        width: 100%;
        height: 1fr;
        border: round #2d375c;
        background: #0b1020;
        padding: 1;
    }
    OptionList > .option-list--option-highlighted {
        background: #202a4a;
        color: #f0f2ff;
    }
    #config-status, #search-status {
        height: auto;
        min-height: 2;
        max-height: 5;
        padding: 1 0 0 0;
    }
    #config-help, #search-help {
        height: 2;
        content-align: center middle;
        color: #7783a3;
    }
    #search-query, #value-input {
        width: 100%;
        height: 3;
        border: round #53639d;
        background: #151c31;
        color: #edf0ff;
    }
    #search-query:focus, #value-input:focus {
        border: round #7d72ff;
    }
    ValueEditor, SaveConfirmation {
        align: center middle;
        background: rgba(4, 7, 15, 0.82);
    }
    #value-dialog, #save-dialog {
        width: 120;
        max-width: 94%;
        height: auto;
        padding: 1 2;
        border: round #39436d;
        background: #101629;
    }
    #value-title, #save-title {
        height: 2;
        text-style: bold;
    }
    #value-help {
        height: 2;
        margin-top: 1;
        color: #7d89aa;
    }
    #save-message {
        height: 3;
        color: #aeb9da;
    }
    #save-options {
        width: 100%;
        height: 6;
        border: round #53639d;
        background: #151c31;
    }
    #save-help {
        height: 2;
        content-align: center middle;
        color: #7d89aa;
    }
    """

    def __init__(self, session: RunConfigurationSession) -> None:
        super().__init__()
        self.session = session
        self.saved_changes = 0

    def on_mount(self) -> None:
        self.push_screen(ConfigurationBrowser(self.session))

    def _copy_current_selection(self) -> bool:
        """Copy either an Input selection or the current screen text selection."""
        focused = self.focused
        if isinstance(focused, Input) and focused.selected_text:
            self.copy_to_clipboard(focused.selected_text)
            focused.selection = type(focused.selection).cursor(focused.cursor_position)
            return True

        selected_text = self.screen.get_selected_text()
        if selected_text:
            self.copy_to_clipboard(selected_text)
            self.screen.clear_selection()
            return True
        return False

    def action_copy_or_exit(self) -> None:
        """Copy on Windows/Linux when selected; otherwise preserve Ctrl+C exit."""
        if not self._copy_current_selection():
            self.exit()

    def action_copy_selection(self) -> None:
        """Copy a macOS Command+C selection without assigning it exit semantics."""
        if not self._copy_current_selection():
            raise SkipAction()


def launch_configuration_editor(project_root: Path, run_directory: Path) -> int:
    """Own the run lock and launch the terminal editor, returning saved field count."""
    with RunConfigurationSession(project_root, run_directory) as session:
        app = ConfigurationEditorApp(session)
        # Navigation is intentionally keyboard-first. Leaving mouse reporting disabled
        # gives VS Code Terminal ownership of drag selection and its native clipboard.
        app.run(mouse=False)
        return app.saved_changes
