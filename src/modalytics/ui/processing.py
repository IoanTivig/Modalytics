"""
Processing an image from the UI: chains, the Process and Show menus.

    Process ›  Processing chains…                      (the chain editor, on the selected image)
               My chains › Denoised OD, …              (your saved chains)
               Quick step › Make › OD map…             (one step, only for looking)
                            Filter › Gaussian blur…
    Show    ›  ● Original   ○ OD map (OD) · measured   ○ Gaussian blur
               Edit chain ›  …     Remove view ›  …

A chain applied to an image adds a *view* of it (``core.model.ImageView``: same
pixel grid, placement and masks; computed when needed) and shows it, as one
undo step. Show switches what the canvas draws. Views marked *measured* feed
the parameters of their image type: the profile's (standard) ones the plain
results, others extra results labelled with the chain's name (stage 6); the
others are only for looking.
"""

from __future__ import annotations

from PySide6 import QtWidgets

from modalytics.core.commands import Command
from modalytics.core.model import IMAGE_TYPES
from modalytics.processing.chains import Chain, ChainLibrary
from modalytics.processing.registry import ProcessingLibrary, StepSpec
from modalytics.ui.settings.widgets import muted, setting_widget


class StepDialog(QtWidgets.QDialog):
    """A single step's options, before it is applied to an image (Quick step)."""

    def __init__(self, spec: StepSpec, image, values: dict | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{spec.name} — {image.name}")
        self.setMinimumWidth(440)
        self.values = spec.resolve(values)
        form = QtWidgets.QFormLayout(self)
        made = "keeps the image type" if spec.is_filter else \
            f"makes a {IMAGE_TYPES.get(spec.output_type, spec.output_type).split(' (')[0].lower()} view"
        form.addRow(muted(f"{spec.description}\n\nIt {made} (same size, placement and masks; the original stays), "
                          f"for looking. To measure it, or to combine steps: Process › Processing chains…"))
        for o in spec.options:
            form.addRow(o.label, setting_widget(o, self.values[o.key],
                                                lambda v, k=o.key: self.values.__setitem__(k, v)))
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok
                                             | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setText("Apply")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)


def by_hand(image, view):
    """A view made by hand, as it starts: a filter (the image's own type) is left out of the parameters until
    ticked in Layers, so it never changes the results silently; a view of another type (an OD map) keeps its
    chain's choice."""
    if view.image_type == image.modality:
        view.chain = dict(view.chain, measure=False)
    return view


def apply_chain(parent, project, undo, image, chain: Chain, replace=None) -> bool:
    """Add (or, with ``replace``, update) a view of ``image`` made by ``chain``, and show it: one undo step.

    Returns:
        bool: True if done.
    """
    from modalytics.processing.run import make_chain_view
    try:
        view = make_chain_view(image, chain.copy(), standard=bool(replace is not None and replace.standard))
    except Exception as exc:
        QtWidgets.QMessageBox.warning(parent, chain.name, f"{image.name}: {exc}")
        return False
    if replace is not None:                              # an edited chain keeps its parameters tick (Layers)
        view.chain = dict(view.chain, measure=replace.measure)
    else:
        by_hand(image, view)
    before = image.shown
    if replace is None:
        undo.push(Command(f"{chain.name} of {image.name}",
                          lambda: project.add_view(image, view, show=True),
                          lambda: (project.remove_view(image, view), project.show_view(image, before))))
    else:
        index = image.views.index(replace)

        def do():
            project.remove_view(image, replace)
            project.add_view(image, view, show=True, index=index)

        def undo_():
            project.remove_view(image, view)
            project.add_view(image, replace, show=before is replace, index=index)
        undo.push(Command(f"Edit {chain.name} of {image.name}", do, undo_))
    return True


def apply_chain_to_all(parent, project, undo, image_type: str, chain: Chain) -> int:
    """Add a view made by ``chain`` to every image of ``image_type`` (one undo step). Returns how many."""
    from modalytics.processing.run import make_chain_view
    made, failed = [], []
    for image in [i for i in project.images if i.modality == image_type]:
        try:
            made.append((image, by_hand(image, make_chain_view(image, chain.copy()))))
        except Exception as exc:
            failed.append(f"{image.name}: {exc}")
    if failed:
        QtWidgets.QMessageBox.warning(parent, chain.name, "\n".join(failed))
    if made:
        undo.push(Command(f"{chain.name} on {len(made)} image(s)",
                          lambda: [project.add_view(i, v) for i, v in made],
                          lambda: [project.remove_view(i, v) for i, v in made]))
    return len(made)


def apply_step(parent, project, undo, image, spec: StepSpec, values: dict | None = None, ask: bool = True) -> bool:
    """Quick step: a view made by one step, only for looking (one undo step). Returns True if added."""
    if ask and spec.options:
        dlg = StepDialog(spec, image, values, parent)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return False
        values = dlg.values
    chain = Chain(spec.name, [{"step": spec.uid, "options": spec.resolve(values)}], measure=False)
    return apply_chain(parent, project, undo, image, chain)


def remove_view(project, undo, image, view) -> None:
    """Remove a view (one undo step)."""
    index, shown = image.views.index(view), image.shown is view
    undo.push(Command(f"Remove {view.name} of {image.name}", lambda: project.remove_view(image, view),
                      lambda: project.add_view(image, view, show=shown, index=index)))


def open_chain_editor(parent, project, undo, image, view=None) -> None:
    """The chain editor on an image (editing ``view``'s chain if given)."""
    from modalytics.ui.chain_editor import ChainEditor
    chain = Chain.from_dict(view.chain) if view is not None else None
    ChainEditor(image, chain, project, undo, view=view, parent=parent).exec()


def view_label(view) -> str:
    """How a view is named in Layers and menus, e.g. "OD map (OD) · measured"."""
    text = f"{view.name} ({view.unit})" if view.unit else view.name
    return text + (" · measured" if view.measure else "")


def offered(project, item_id: str) -> bool:
    """Whether the project's profile offers a step or chain ("chain:<name>") at all (not hidden)."""
    profile = getattr(project, "profile", None)
    return profile is None or profile.step_place(item_id) != "hidden"


def fill_step_menu(menu: QtWidgets.QMenu, parent, project, undo, images: list,
                   library: ProcessingLibrary | None = None) -> bool:
    """Quick steps the images can take (those the profile offers), one submenu per category (Correction,
    Denoise…); several images: each image that can take the step gets it. Returns False when there is none."""
    library = library or ProcessingLibrary()
    by_title: dict = {}
    seen: set = set()
    for t in dict.fromkeys(i.modality for i in images):       # the steps any of the images can take, once each
        for title, group in library.grouped(t):
            fresh = [s for s in group if s.uid not in seen and offered(project, s.uid)]
            seen |= {s.uid for s in fresh}
            by_title.setdefault(title, []).extend(fresh)
    for title, group in by_title.items():
        if not group:
            continue
        g = menu.addMenu(title)
        for spec in group:
            act = g.addAction(spec.name + ("…" if spec.options else ""),
                              lambda s=spec: run_step(parent, project, undo, images, s))
            act.setToolTip(spec.description)
        g.setToolTipsVisible(True)
    return bool(seen)


def fill_chain_menu(menu: QtWidgets.QMenu, parent, project, undo, images: list,
                    library: ProcessingLibrary | None = None) -> bool:
    """Your saved chains that fit the images and the profile offers (several: each image that can take the chain
    gets it). Returns False when there is none (the menu then says how to make one)."""
    library = library or ProcessingLibrary()
    types = {i.modality for i in images}
    mine = [c for c in ChainLibrary().chains()
            if offered(project, f"chain:{c.name}") and any(not c.problems(t, library) for t in types)]
    for chain in mine:
        if len(images) == 1:
            act = menu.addAction(chain.name, lambda c=chain: apply_chain(parent, project, undo, images[0], c))
        else:
            act = menu.addAction(chain.name, lambda c=chain: apply_chain_to_images(parent, project, undo, images, c))
        act.setToolTip(chain.summary(library) + (" · measured" if chain.measure else ""))
    menu.setToolTipsVisible(True)
    if not mine:
        menu.addAction("No saved chain fits: make one in the Chain editor").setEnabled(False)
    return bool(mine)


def run_step(parent, project, undo, images: list, spec: StepSpec) -> int:
    """One quick step on the images (its options asked once); returns how many got the view."""
    if len(images) == 1:
        return int(apply_step(parent, project, undo, images[0], spec))
    return apply_step_to_images(parent, project, undo, images, spec)


def fill_process_menu(menu: QtWidgets.QMenu, parent, project, undo, image,
                      library: ProcessingLibrary | None = None) -> None:
    """Processing chains…, your chains, and quick single steps for the image."""
    library = library or ProcessingLibrary()
    menu.addAction("Processing chains…", lambda: open_chain_editor(parent, project, undo, image))
    sub = menu.addMenu("My chains")
    sub.setEnabled(fill_chain_menu(sub, parent, project, undo, [image], library))
    quick = menu.addMenu("Quick step")
    quick.setEnabled(fill_step_menu(quick, parent, project, undo, [image], library))


def fill_show_menu(menu: QtWidgets.QMenu, project, undo, image, parent=None) -> None:
    """Original + views (the shown one checked); Edit chain and Remove view for each view."""
    for view in [None, *image.views]:
        act = menu.addAction("Original" if view is None else view_label(view),
                             lambda v=view: project.show_view(image, v))
        act.setCheckable(True)
        act.setChecked(image.shown is view)
    menu.addSeparator()
    if image.views:
        edit = menu.addMenu("Edit chain")
        rm = menu.addMenu("Remove view")
        for view in image.views:
            edit.addAction(view.name, lambda v=view: open_chain_editor(parent, project, undo, image, v))
            rm.addAction(view.name, lambda v=view: remove_view(project, undo, image, v))
    else:
        menu.addAction("No views yet: Process › makes one").setEnabled(False)


# ---------------------------------------------------------------- several images at once
def apply_chain_to_images(parent, project, undo, images: list, chain: Chain, show: bool = True) -> int:
    """Add a view made by ``chain`` to each of ``images`` that can take it, shown: one undo step.

    Images of a type the chain cannot take are left out (and said); returns how many got the view.
    """
    from modalytics.processing.run import make_chain_view
    library = ProcessingLibrary()
    made, failed, skipped = [], [], []
    for image in images:
        if chain.problems(image.modality, library):
            skipped.append(image.name)
            continue
        try:
            made.append((image, image.shown, by_hand(image, make_chain_view(image, chain.copy()))))
        except Exception as exc:                                   # one image failing does not stop the others
            failed.append(f"{image.name}: {exc}")
    notes = failed + ([f"Left out (it does not take their image type): {', '.join(skipped)}"] if skipped else [])
    if notes:
        QtWidgets.QMessageBox.warning(parent, chain.name, "\n".join(notes))
    if made:
        undo.push(Command(f"{chain.name} on {len(made)} image(s)",
                          lambda: [project.add_view(i, v, show=show) for i, _b, v in made],
                          lambda: [(project.remove_view(i, v), project.show_view(i, b)) for i, b, v in made]))
    return len(made)


def apply_step_to_images(parent, project, undo, images: list, spec: StepSpec) -> int:
    """Quick step on several images: its options asked once (on the first image it takes), then one view each."""
    usable = [i for i in images if spec.accepts(i.modality)]
    if not usable:
        QtWidgets.QMessageBox.information(parent, spec.name, f"{spec.name} does not take these images' types.")
        return 0
    values = None
    if spec.options:
        dlg = StepDialog(spec, usable[0], None, parent)
        dlg.setWindowTitle(f"{spec.name} — {len(usable)} image(s)")
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return 0
        values = dlg.values
    chain = Chain(spec.name, [{"step": spec.uid, "options": spec.resolve(values)}], measure=False)
    return apply_chain_to_images(parent, project, undo, images, chain)


def fill_process_menu_many(menu: QtWidgets.QMenu, parent, project, undo, images: list,
                           library: ProcessingLibrary | None = None) -> None:
    """Process several images: your chains and the quick steps, each applied to every image that can take it."""
    library = library or ProcessingLibrary()
    sub = menu.addMenu("My chains")
    sub.setEnabled(fill_chain_menu(sub, parent, project, undo, images, library))
    quick = menu.addMenu("Quick step")
    quick.setEnabled(fill_step_menu(quick, parent, project, undo, images, library))
