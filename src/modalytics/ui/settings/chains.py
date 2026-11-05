"""
Settings › Processing › Chains: your processing chains (browse, add, edit, remove, update profiles).

    ┌ Chain          Steps                        Makes             Measured  In profiles             ┐
    │ Denoised OD    Median filter → OD map       slides → OD map   yes       Colonic tissue (differs)│
    │ Smooth phase   Gaussian blur                phase → phase     no        —                       │
    │ [New ▾] [Edit…] [Duplicate] [Delete] [Update in profiles…] [Folder]                                │
    └────────────────────────────────────────────────────────────────────────────────────────────────────┘

The same library as the chain editor's "My chains" (Process tab): YAML files in
your library folder (``processing.chains.ChainLibrary``). Edit opens the chain
editor on a sample image of the chain's input type.

Profiles keep their own copy of a chain, so editing a chain here never changes
a profile's results by itself: *Update in profiles…* lists the profiles holding
a chain of the same name and replaces the copies you pick (each keeps its own
*measured* choice). Projects keep their own copy of their profile in turn.
"""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from modalytics.core.model import ACQUIRED_TYPES, IMAGE_TYPES
from modalytics.processing.chains import (Chain, ChainLibrary, input_types, profiles_using,
                                          update_profiles)
from modalytics.processing.registry import ProcessingLibrary
from modalytics.ui.settings.widgets import clear_tree, list_tree, muted, open_in_system

PLURAL = {"wsi": "slides", "phase": "phase maps", "brightfield": "brightfield images"}
SHORT = {"wsi": "slides", "phase": "phase", "brightfield": "brightfield"}


def _short(t: str) -> str:
    return IMAGE_TYPES.get(t, t).split(" (")[0]


class UpdateProfilesDialog(QtWidgets.QDialog):
    """Pick the profiles whose copy of a chain is replaced by the library's."""

    def __init__(self, chain: Chain, using: list, library: ProcessingLibrary, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Update “{chain.name}” in profiles")
        self.setMinimumWidth(520)
        col = QtWidgets.QVBoxLayout(self)
        col.addWidget(muted(f"Your chain: {chain.summary(library)}\n\nProfiles holding a chain called "
                            f"“{chain.name}” (their own copy). Ticked ones get yours; each keeps its own "
                            f"“measured” choice. Projects already made keep their own copy of their profile."))
        self.boxes = []
        for pid, pname, image_type, theirs in using:
            same = theirs.same_steps(chain, library)
            box = QtWidgets.QCheckBox(f"{pname} · {PLURAL.get(image_type, image_type)} · "
                                      + ("the same already" if same else f"differs: {theirs.summary(library)}"))
            box.setChecked(not same)
            box.setEnabled(not same)
            self.boxes.append((box, (pid, image_type)))
            col.addWidget(box)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok
                                             | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Update")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        col.addWidget(buttons)

    def targets(self) -> list[tuple[str, str]]:
        return [t for box, t in self.boxes if box.isChecked() and box.isEnabled()]


class ChainsPage(QtWidgets.QWidget):
    """Your chain library (see the module docstring).

    Args:
        library: The processing steps.

    Signals:
        profiles_changed(): profiles were updated with a chain.
    """

    profiles_changed = QtCore.Signal()

    def __init__(self, library: ProcessingLibrary | None = None, parent=None) -> None:
        super().__init__(parent)
        from modalytics.profiles.library import ProfileLibrary
        self.library = library or ProcessingLibrary()
        self.chains = ChainLibrary()
        self.profiles = ProfileLibrary()
        col = QtWidgets.QVBoxLayout(self)
        col.addWidget(muted("Your processing chains: made here or in the chain editor (Process › Processing "
                            "chains…), applied from the Process tab or added to profiles. A profile keeps its own "
                            "copy: Update in profiles… replaces it when you want."))
        self.tree = list_tree(["Chain", "Steps", "Makes", "Measured", "In profiles"])
        self.tree.setColumnWidth(0, 180)
        self.tree.setColumnWidth(1, 300)
        self.tree.setColumnWidth(2, 240)
        self.tree.setColumnWidth(3, 80)
        self.tree.itemDoubleClicked.connect(lambda *_: self._edit())
        col.addWidget(self.tree, 1)
        row = QtWidgets.QHBoxLayout()
        self.new_btn = QtWidgets.QPushButton("New ▾")
        self.new_btn.clicked.connect(self._new_menu)
        row.addWidget(self.new_btn)
        self.button_btn = QtWidgets.QPushButton("Button…")
        self.button_btn.setToolTip("The label and icon of the chain's button on the Process tab (when a profile puts "
                                   "it in the toolbar)")
        self.update_btn = QtWidgets.QPushButton("Update in profiles…")
        self.update_btn.setToolTip("Replace the copies of the selected chain in the profiles that hold it")
        for b, slot in ((QtWidgets.QPushButton("Edit…"), self._edit),
                        (QtWidgets.QPushButton("Duplicate"), self._duplicate),
                        (QtWidgets.QPushButton("Delete"), self._delete),
                        (self.button_btn, self._button),
                        (self.update_btn, self._update_profiles),
                        (QtWidgets.QPushButton("Folder"), lambda: open_in_system(self.chains.folder))):
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch(1)
        col.addLayout(row)
        self.reload()

    # ------------------------------------------------------------ the list
    def reload(self, select: str | None = None) -> None:
        """Re-read the chains (keeps the selection, or selects ``select``)."""
        select = select or self.selected_name()
        clear_tree(self.tree)
        chosen = None
        for chain in self.chains.chains():
            kinds = input_types(chain, self.library)
            by_output: dict[str, list[str]] = {}                 # "slides, brightfield → OD map"
            for t in kinds:
                by_output.setdefault(_short(chain.output_type(t, self.library)), []).append(SHORT[t])
            makes = "; ".join(f"{', '.join(ins)} → {out}" for out, ins in by_output.items()) \
                or "⚠ " + "; ".join(chain.problems(ACQUIRED_TYPES[0], self.library))
            using = profiles_using(chain.name, self.profiles)
            where = ", ".join(sorted({f"{name}" + ("" if c.same_steps(chain, self.library) else " (differs)")
                                      for _pid, name, _t, c in using})) or "—"
            item = QtWidgets.QTreeWidgetItem(self.tree, [chain.name, chain.summary(self.library), makes,
                                                         "yes" if chain.measure else "no", where])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, chain.name)
            from modalytics.ui.ribbon import icon
            item.setIcon(0, icon(chain.button_icon, self.palette().color(self.foregroundRole()).name()))
            item.setToolTip(0, f"Button: “{chain.button_label}”")
            item.setToolTip(4, "Profiles holding a chain of this name (their own copy)")
            if chain.name == select:
                chosen = item
        if chosen is None and self.tree.topLevelItemCount():
            chosen = self.tree.topLevelItem(0)
        self.tree.setCurrentItem(chosen)

    def selected_name(self) -> str | None:
        item = self.tree.currentItem() if hasattr(self, "tree") else None
        return item.data(0, QtCore.Qt.ItemDataRole.UserRole) if item else None

    def _selected(self) -> Chain | None:
        name = self.selected_name()
        return self.chains.get(name) if name else None

    # ------------------------------------------------------------ actions
    def _editor(self, chain: Chain, image_type: str) -> Chain | None:
        """The chain editor on a sample image of ``image_type``; the chain if OK."""
        from modalytics.ui.chain_editor import ChainEditor
        dlg = ChainEditor(None, chain, for_type=image_type, library=self.library, parent=self)
        return dlg.chain if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted else None

    def _new_menu(self) -> None:
        menu = QtWidgets.QMenu(self)
        for t in ACQUIRED_TYPES:
            menu.addAction(f"For {PLURAL[t]}…", lambda t=t: self._new(t))
        menu.exec(self.new_btn.mapToGlobal(self.new_btn.rect().bottomLeft()))

    def _new(self, image_type: str) -> None:
        chain = self._editor(Chain("New chain", []), image_type)
        if chain is None or not chain.steps:
            return
        if self.chains.get(chain.name) and QtWidgets.QMessageBox.question(
                self, "New chain", f"A chain called “{chain.name}” exists. Replace it?") != \
                QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self.chains.save(chain)
        self.reload(chain.name)

    def _edit(self) -> None:
        old = self._selected()
        if old is None:
            return
        kinds = input_types(old, self.library)
        chain = self._editor(old, kinds[0] if kinds else "phase")
        if chain is None:
            return
        chain.short, chain.icon = chain.short or old.short, chain.icon or old.icon   # its button stays as it was
        if chain.name != old.name:
            if self.chains.get(chain.name) and QtWidgets.QMessageBox.question(
                    self, "Edit chain", f"A chain called “{chain.name}” exists. Replace it?") != \
                    QtWidgets.QMessageBox.StandardButton.Yes:
                return
            self.chains.delete(old.name)
        self.chains.save(chain)
        self.reload(chain.name)
        if any(not c.same_steps(chain, self.library) for *_x, c in profiles_using(chain.name, self.profiles)):
            if QtWidgets.QMessageBox.question(
                    self, "Edit chain", f"Profiles hold an older copy of “{chain.name}”. Update them now?") == \
                    QtWidgets.QMessageBox.StandardButton.Yes:
                self._update_profiles()

    def _duplicate(self) -> None:
        chain = self._selected()
        if chain is None:
            return
        names = {c.name for c in self.chains.chains()}
        n = 2
        while f"{chain.name} ({n})" in names:
            n += 1
        copy = chain.copy()
        copy.name = f"{chain.name} ({n})"
        self.chains.save(copy)
        self.reload(copy.name)

    def _delete(self) -> None:
        chain = self._selected()
        if chain is None:
            return
        using = profiles_using(chain.name, self.profiles)
        extra = ("\n\nProfiles keep their own copy: " + ", ".join(sorted({n for _p, n, _t, _c in using}))) \
            if using else ""
        if QtWidgets.QMessageBox.question(self, "Delete chain", f"Delete “{chain.name}” from your chains?{extra}") \
                == QtWidgets.QMessageBox.StandardButton.Yes:
            self.chains.delete(chain.name)
            self.reload()

    def _button(self) -> None:
        """The label and icon of the selected chain's toolbar button."""
        chain = self._selected()
        if chain is None:
            return
        dlg = ButtonDialog(chain.name, chain.short, chain.icon, self)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            chain.short, chain.icon = dlg.values()
            self.chains.save(chain)
            self.reload(chain.name)

    def _update_profiles(self) -> None:
        chain = self._selected()
        if chain is None:
            return
        using = profiles_using(chain.name, self.profiles)
        if not using:
            QtWidgets.QMessageBox.information(self, "Update in profiles",
                                              f"No profile holds a chain called “{chain.name}”.")
            return
        dlg = UpdateProfilesDialog(chain, using, self.library, self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted or not dlg.targets():
            return
        n = update_profiles(chain, dlg.targets(), self.profiles)
        self.reload(chain.name)
        self.profiles_changed.emit()
        QtWidgets.QMessageBox.information(self, "Update in profiles", f"Updated {n} cop{'y' if n == 1 else 'ies'} "
                                          f"of “{chain.name}”.")


class ButtonDialog(QtWidgets.QDialog):
    """A chain's toolbar button: its label (empty: the chain's name) and icon (empty: a chain icon)."""

    def __init__(self, name: str, short: str, icon_name: str, parent=None) -> None:
        super().__init__(parent)
        from modalytics.ui.icon_picker import IconField
        self.setWindowTitle(f"Button of “{name}”")
        self.setMinimumWidth(460)
        form = QtWidgets.QFormLayout(self)
        form.addRow(muted("Shown on the Process tab when a profile puts the chain in its toolbar (Settings › "
                          "Profiles › Processing steps)."))
        self.short = QtWidgets.QLineEdit(short)
        self.short.setPlaceholderText(name)
        self.icon = IconField(hint="link")
        self.icon.edit.setPlaceholderText("Default: a chain icon")
        self.icon.set_icon(icon_name)
        form.addRow("Button label", self.short)
        form.addRow("Icon", self.icon)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok
                                             | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._ok)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _ok(self) -> None:
        from modalytics import icons
        name = self.icon.icon_name()
        if name and not icons.allowed(name):
            QtWidgets.QMessageBox.warning(self, "Icon", f"'{name}' is not an icon of the set: Choose… picks one.")
            return
        self.accept()

    def values(self) -> tuple[str, str]:
        return self.short.text().strip(), self.icon.icon_name()
