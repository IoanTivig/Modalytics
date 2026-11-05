"""
Hierarchical layer list: images (top of the list = drawn on top) with, under each,
its annotation, the models' suggestions and its processed versions in folders.

    ▣ sample01                          👁   the image (the eye: shown; crossed out: hidden)
       ▾ Annotations                    👁   its masks (the folder's eye: all of them)
           ● Gland                      👁
           ● Lumen                      👁
       ▾ Suggestions (12)               👁   a model's suggestions, waiting for review
           ● Gland (10)                 👁   each group's (its eye: drawn or not)
           ● Lumen (2)                  👁
       ▾ Processed
           ◉ Original                   Σ    what is drawn: the original or one of its views (click to show
           ○ OD map (OD)                Σ    it); Σ: included in the parameters (click: in or out)
           ○ Gaussian blur              σ

A folder appears only when it has something in it (Processed: once the image has a
view). Parameters of a type are measured on every included version of it: the plain
results from one (the profile's view, else the original, else the first view), the
others labelled with their name (``parameters.engine``). A filter applied by hand
starts left out, so it never changes the results silently.

* Ctrl / Shift-click selects several layers (``Project.select_many``); the
  right-click menu then acts on all of them: show / hide, frame in view, move to
  the top or bottom, remove — and what the main window adds (process, segment).
* Images are dragged up or down to change the drawing order (several at once
  keep their order). Masks and views stay under their image (a mask goes to
  another image with Transfer to…).
* The eye at the right of a row shows or hides it; a hidden layer's name is dimmed.

The tree is a *view* of ``Project``: it is rebuilt when the structure
changes and edits go back through ``Project`` methods.
"""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from modalytics.core.model import ImageNode, MaskNode, Project, SuggestionNode

MODALITY_COLORS = {"wsi": "#d16ba5", "phase": "#4cc9f0", "brightfield": "#f4a261", "od": "#c77dff",
                   "opd": "#80ed99", "dry_mass": "#57cc99", "height": "#48cae4", "ri": "#90e0ef",
                   "stain": "#b5179e", "feature": "#adb5bd"}
MODALITY_LABELS = {"wsi": "WSI", "phase": "Phase", "brightfield": "Brightfield", "od": "OD map",
                   "opd": "OPD map", "dry_mass": "Dry-mass map", "height": "Height map", "ri": "RI map",
                   "stain": "Stain map", "feature": "Feature map"}
VIEW_ROLE = QtCore.Qt.ItemDataRole.UserRole + 1       # on view rows: (image id, view id or "" for the original)
FOLDER = QtCore.Qt.ItemDataRole.UserRole + 2          # on folder rows: (image id, "annotations" | "processed")
SUGG_GROUP = QtCore.Qt.ItemDataRole.UserRole + 3      # on a suggestions group row: (image id, group)
EYE = 1                                               # the column of the show / hide eye (versions: the Σ tick)
EYE_PX = 18                                           # the icons' size (eyes, Σ, swatches)


def _swatch(color: str, round_: bool = False) -> QtGui.QIcon:
    """Small coloured icon for a modality or a mask colour."""
    pm = QtGui.QPixmap(16, 16)
    pm.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    p.setBrush(QtGui.QColor(color))
    p.setPen(QtCore.Qt.PenStyle.NoPen)
    p.drawEllipse(1, 1, 14, 14) if round_ else p.drawRoundedRect(1, 1, 14, 14, 4, 4)
    p.end()
    return QtGui.QIcon(pm)


class LayerTree(QtWidgets.QTreeWidget):
    """The tree, where images (top-level rows) are dragged up and down to restack them.

    Signals:
        reordered(list): the image ids, top of the list first, after a drop.
    """

    reordered = QtCore.Signal(list)
    side_clicked = QtCore.Signal(object)       # the right column of a row that is not a layer (folder, version…)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("LayerTree")
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)
        self.setDropIndicatorShown(True)

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt)
        """A click on the eye of a folder, a group of suggestions or the Σ of a version acts at once, without
        selecting the row (selecting a version would draw it)."""
        item = self.itemAt(event.position().toPoint())
        special = item is not None and any(item.data(0, role) for role in (VIEW_ROLE, FOLDER, SUGG_GROUP))
        if special and event.button() == QtCore.Qt.MouseButton.RightButton:
            event.accept()                               # its menu comes (customContextMenuRequested), no selection
            return
        if (special and self.columnAt(int(event.position().x())) == EYE
                and event.button() == QtCore.Qt.MouseButton.LeftButton):
            self.side_clicked.emit(item)
            event.accept()
            return
        super().mousePressEvent(event)

    def _moving(self) -> list:
        """The image rows being dragged (only images move; none if a mask or view row is among the selection)."""
        rows = self.selectedItems()
        if not rows or any(r.parent() is not None for r in rows):
            return []
        return sorted(rows, key=self.indexOfTopLevelItem)

    def dragMoveEvent(self, event) -> None:  # noqa: N802 (Qt)
        if self._moving():
            event.setDropAction(QtCore.Qt.DropAction.MoveAction)
            event.accept()
            super().dragMoveEvent(event)
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # noqa: N802 (Qt)
        moving = self._moving()
        event.setDropAction(QtCore.Qt.DropAction.IgnoreAction)       # the project is reordered; the tree rebuilt
        event.accept()
        if not moving:
            return
        self.drop_at(moving, self.itemAt(event.position().toPoint()), event.position().y())

    def drop_at(self, moving: list, target, y: float) -> None:
        """Put the dragged image rows before or after the image row ``target`` (a child row counts as its image;
        None: at the end) and say the new order."""
        moving = sorted(moving, key=self.indexOfTopLevelItem)          # they keep their order in the list
        while target is not None and target.parent() is not None:
            target = target.parent()
        rest = [self.topLevelItem(i) for i in range(self.topLevelItemCount())
                if self.topLevelItem(i) not in moving]
        if target is None:
            at = len(rest)
        elif target in moving:
            return
        else:
            rect = self.visualItemRect(target)
            at = rest.index(target) + (1 if y > rect.center().y() else 0)
        order = rest[:at] + moving + rest[at:]
        self.reordered.emit([it.data(0, QtCore.Qt.ItemDataRole.UserRole) for it in order])


class LayerPanel(QtWidgets.QWidget):
    """Tree of images and masks with show / hide eyes (see the module docstring).

    Args:
        project: The project to show.
        fit_requested: Callback(node or list of nodes) to frame them in the canvas.
        extend_menu: Callback(menu, nodes) adding the main window's actions (process, segment) to the
            right-click menu of the selected layers.
        undo: The undo stack (including a version in the parameters, or not, is one undo step).
    """

    def __init__(self, project: Project, fit_requested=None, parent=None, extend_menu=None, undo=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.project = project
        self.fit_requested = fit_requested
        self.extend_menu = extend_menu
        self._items: dict[str, QtWidgets.QTreeWidgetItem] = {}
        self._folders: dict[tuple, QtWidgets.QTreeWidgetItem] = {}     # (image id, kind) → folder row
        self._closed: set[tuple] = set()                 # rows the user closed: (image id, kind), kind "image"…
        self._updating = False
        self._eyes = None
        self._sigmas = None
        self.undo = undo

        # 1. Tree: names (indented) and, at the right, the eye
        self.tree = LayerTree()
        self.tree.setColumnCount(2)
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(18)                     # room for the arrows
        self.tree.setIconSize(QtCore.QSize(EYE_PX, EYE_PX))
        header = self.tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(EYE, QtWidgets.QHeaderView.ResizeMode.Fixed)
        header.resizeSection(EYE, EYE_PX + 14)
        self.tree.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._context_menu)
        self.tree.itemSelectionChanged.connect(self._tree_selection)
        self.tree.itemClicked.connect(self._clicked)
        self.tree.itemDoubleClicked.connect(lambda item, col: col != EYE and self.fit_requested and
                                            self._node(item) is not None and self.fit_requested(self._node(item)))
        self.tree.reordered.connect(self._reordered)
        self.tree.side_clicked.connect(self._side_clicked)
        self.tree.itemCollapsed.connect(lambda it: self._open_state(it, False))
        self.tree.itemExpanded.connect(lambda it: self._open_state(it, True))
        hint = QtWidgets.QLabel("Drag images to restack them · Ctrl/Shift-click selects several")
        hint.setObjectName("Muted")
        hint.setWordWrap(True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self.tree)
        layout.addWidget(hint)

        project.structure_changed.connect(self.rebuild)
        project.node_changed.connect(self._node_changed)
        project.selection_changed.connect(self._model_selection)
        self.rebuild()

    # 0. The eye icons (small, in the theme's text colours)
    def _eye(self, visible: bool) -> QtGui.QIcon:
        if self._eyes is None:
            import qtawesome as qta
            pal = self.palette()
            on = pal.color(QtGui.QPalette.ColorRole.Text)
            off = pal.color(QtGui.QPalette.ColorRole.PlaceholderText)
            self._eyes = (qta.icon("mdi6.eye-off-outline", color=off), qta.icon("mdi6.eye-outline", color=on))
        return self._eyes[bool(visible)]

    def _sigma(self, included: bool) -> QtGui.QIcon:
        """A version's parameters tick: Σ in the text colour (included), faint (left out)."""
        if self._sigmas is None:
            import qtawesome as qta
            pal = self.palette()
            self._sigmas = (qta.icon("mdi6.sigma", color=pal.color(QtGui.QPalette.ColorRole.PlaceholderText)),
                            qta.icon("mdi6.sigma", color=pal.color(QtGui.QPalette.ColorRole.Highlight)))
        return self._sigmas[bool(included)]

    def changeEvent(self, event) -> None:  # noqa: N802 (Qt)
        if event.type() == QtCore.QEvent.Type.PaletteChange:       # the theme changed: eyes in its colours
            self._eyes = self._sigmas = None
            QtCore.QTimer.singleShot(0, self.rebuild)
        super().changeEvent(event)

    # 1. Model → tree
    def rebuild(self) -> None:
        """Rebuild the tree from the project (top image first)."""
        self._updating = True
        self.tree.clear()
        self._items.clear()
        self._folders.clear()
        for img in reversed(self.project.images):
            item = self._make_item(img, MODALITY_COLORS.get(img.modality, "#999"))
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsDragEnabled)
            font = item.font(0)
            font.setBold(True)                           # the image: its folders and layers are under it
            item.setFont(0, font)
            self.tree.addTopLevelItem(item)
            # 1. Annotations: the masks
            if img.masks:
                folder = self._folder(item, img, "annotations", "Annotations",
                                      "The annotation: the objects drawn by hand, per class. The eye shows or hides "
                                      "all of them.")
                for mask in img.masks:
                    r, g, b, _ = mask.color
                    folder.addChild(self._make_item(mask, QtGui.QColor.fromRgbF(r, g, b).name(), round_=True))
                self._show_folder_eye(img)
            # 2. Suggestions: the models' proposals waiting for review, per group
            sugg = img.suggestions
            if sugg is not None:
                child = self._make_item(sugg, "#9aa0a6", round_=True)
                child.setIcon(0, QtGui.QIcon())              # a folder, like Annotations and Processed
                item.addChild(child)
                self._suggestion_text(child, sugg)
                self._suggestion_groups(child, sugg)
                child.setExpanded((img.id, "suggestions") not in self._closed)
            # 3. Processed: the original and its views (what is drawn; what the parameters use)
            if img.views:
                folder = self._folder(item, img, "processed", "Processed",
                                      "The versions of the image: ◉ the one drawn (click a row to draw it); Σ: "
                                      "included in the parameters (click it: in or out). The plain results come "
                                      "from one version of each type, the others' are labelled with their name.")
                for view in [None, *img.views]:
                    folder.addChild(self._version_row(img, view))
            item.setExpanded((img.id, "image") not in self._closed)
        self._updating = False
        self._model_selection(self.project.selected)

    def _folder(self, parent, img, kind: str, name: str, tip: str) -> QtWidgets.QTreeWidgetItem:
        """A folder row under an image (not a layer: it groups them)."""
        row = QtWidgets.QTreeWidgetItem([name, ""])
        row.setData(0, FOLDER, (img.id, kind))
        row.setToolTip(0, tip)
        row.setFlags(row.flags() & ~QtCore.Qt.ItemFlag.ItemIsDragEnabled & ~QtCore.Qt.ItemFlag.ItemIsDropEnabled
                     & ~QtCore.Qt.ItemFlag.ItemIsSelectable)
        font = row.font(0)
        font.setItalic(True)                             # a folder (Annotations, Suggestions, Processed)
        row.setFont(0, font)
        parent.addChild(row)
        row.setExpanded((img.id, kind) not in self._closed)
        self._folders[(img.id, kind)] = row
        return row

    def _show_folder_eye(self, img) -> None:
        """The Annotations folder's eye: open while any of the image's masks is shown."""
        row = self._folders.get((img.id, "annotations"))
        if row is not None:
            shown = any(m.visible for m in img.masks)
            row.setIcon(EYE, self._eye(shown))
            row.setToolTip(EYE, "Click to hide all the masks" if shown else "Click to show all the masks")

    def _version_row(self, img, view) -> QtWidgets.QTreeWidgetItem:
        """A version of the image: ◉ when drawn; its Σ tick (included in the parameters)."""
        from modalytics.ui.processing import view_label
        name = "Original" if view is None else view_label(view).replace(" · measured", "")
        row = QtWidgets.QTreeWidgetItem([("◉ " if img.shown is view else "○ ") + name, ""])
        row.setData(0, VIEW_ROLE, (img.id, view.id if view else ""))
        row.setFlags(row.flags() & ~QtCore.Qt.ItemFlag.ItemIsDragEnabled & ~QtCore.Qt.ItemFlag.ItemIsDropEnabled)
        included = img.original_measured if view is None else view.measure
        kind = img.modality if view is None else view.image_type
        plain = img.measured_view(kind)
        role = ("" if not included else
                " — the plain results" if (plain is view) or (plain is None and view is None) else
                f" — results labelled “{name.split(' (')[0]}”")
        row.setIcon(EYE, self._sigma(included))
        row.setToolTip(EYE, ("Included in the parameters" + role + ": click to leave it out") if included else
                       "Left out of the parameters: click to include it")
        if view is not None:
            from modalytics.processing.chains import Chain
            steps = Chain.from_dict(view.chain).summary(self._steps_library())
            row.setToolTip(0, f"{steps}\n" + ("In the parameters" + role + (" (the profile's)" if view.standard
                                                                               else "") if included else
                                              "Only for looking (Σ: include it in the parameters)")
                           + "\nClick to draw it; masks are shared")
        else:
            row.setToolTip(0, "The image as imported. Click to draw it."
                           + ("\nIn the parameters" + role if included else "\nLeft out of the parameters"))
        if not included:
            row.setForeground(0, self.palette().color(QtGui.QPalette.ColorRole.PlaceholderText))
        return row

    def _suggestion_groups(self, item, node) -> None:
        """The suggestions' groups under the Suggestions row, each with its count and eye."""
        item.takeChildren()
        colors = self._class_colors()
        hidden = set(node.hidden_groups)
        for group, n in node.groups():
            row = QtWidgets.QTreeWidgetItem([f"{group} ({n})", ""])
            row.setData(0, SUGG_GROUP, (node.parent.id if node.parent else "", group))
            r, g, b, _a = colors.get((group, ""), (0.6, 0.6, 0.6, 1))
            row.setIcon(0, _swatch(QtGui.QColor.fromRgbF(r, g, b).name(), round_=True))
            row.setFlags(row.flags() & ~QtCore.Qt.ItemFlag.ItemIsDragEnabled & ~QtCore.Qt.ItemFlag.ItemIsDropEnabled
                         & ~QtCore.Qt.ItemFlag.ItemIsSelectable)
            row.setToolTip(0, f"{n} suggested {group} object(s), waiting for review")
            self._show_visible(row, group not in hidden)
            item.addChild(row)

    def _class_colors(self) -> dict:
        """{(group, subgroup): rgba}: the masks', else the profile's, else the project's class colours."""
        labels = self.project.labels
        out = {(g, s): labels.color(g, s) for g, s in labels.choices()}
        prof = self.project.profile
        if prof is not None:
            out.update({(g, s): prof.classes.color(g, s) for g, s in prof.classes.choices()})
        out.update({(m.group, m.subgroup): m.color for img in self.project.images for m in img.masks})
        return out

    def _open_state(self, item, opened: bool) -> None:
        """Remember which rows the user closed (kept when the tree is rebuilt)."""
        if self._updating:
            return
        folder = item.data(0, FOLDER)
        node = self._node(item)
        key = tuple(folder) if folder else (node.parent.id, "suggestions") if isinstance(node, SuggestionNode) \
            and node.parent is not None else (node.id, "image") if isinstance(node, ImageNode) else None
        if key is not None:
            (self._closed.discard if opened else self._closed.add)(key)

    def _make_item(self, node, color: str, round_: bool = False) -> QtWidgets.QTreeWidgetItem:
        item = QtWidgets.QTreeWidgetItem([node.name, ""])
        item.setData(0, QtCore.Qt.ItemDataRole.UserRole, node.id)
        item.setIcon(0, _swatch(color, round_))
        item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsDragEnabled & ~QtCore.Qt.ItemFlag.ItemIsDropEnabled)
        self._show_visible(item, node.visible)
        if isinstance(node, ImageNode):
            self._mark_missing(item, node)
        self._items[node.id] = item
        return item

    def _show_visible(self, item, visible: bool) -> None:
        """The eye, and the name dimmed when hidden."""
        item.setIcon(EYE, self._eye(visible))
        item.setToolTip(EYE, "Shown: click to hide" if visible else "Hidden: click to show")
        pal = self.palette()
        item.setForeground(0, pal.color(QtGui.QPalette.ColorRole.Text if visible
                                        else QtGui.QPalette.ColorRole.PlaceholderText))

    @staticmethod
    def _mark_missing(item, node) -> None:
        """Tooltip; an image whose file was not found is shown in italics, the tooltip says why."""
        from modalytics.io.sources import is_missing
        gone = is_missing(node)
        font = item.font(0)
        font.setItalic(gone)
        item.setFont(0, font)
        item.setToolTip(0, (f"⚠ File not found: {node.source.path} (Properties › Relink…)\n" if gone else "")
                        + f"{MODALITY_LABELS.get(node.modality, node.modality)} · {node.pixel_size_um[0]:.4g} µm/px")

    def _steps_library(self):
        """The processing steps (read once; for the chains' descriptions)."""
        if getattr(self, "_steps", None) is None:
            from modalytics.processing.registry import ProcessingLibrary
            self._steps = ProcessingLibrary()
        return self._steps

    def _node(self, item):
        return self.project.find(item.data(0, QtCore.Qt.ItemDataRole.UserRole)) if item else None

    @staticmethod
    def _suggestion_text(item, node) -> None:
        n = len(node.items)
        item.setText(0, f"Suggestions ({n})" if n else "Suggestions (all reviewed)")
        font = item.font(0)
        font.setItalic(True)                             # a folder, like Annotations and Processed
        item.setFont(0, font)
        item.setToolTip(0, "Objects a model proposes, waiting for review: accept, change the grade or delete each "
                           "(Review tool, or Properties), or all of them, or a group's (right-click). They are not "
                           "part of the annotation and are not measured unless Measure › Include suggestions is on.")

    def _node_changed(self, node, field: str) -> None:
        item = self._items.get(node.id)
        if item is not None and field == "suggestions":
            self._updating = True
            self._suggestion_text(item, node)
            self._suggestion_groups(item, node)
            item.setExpanded(node.parent is None or (node.parent.id, "suggestions") not in self._closed)
            self._updating = False
            return
        if item is not None and field in ("source", "views"):
            QtCore.QTimer.singleShot(0, self.rebuild)     # views / the shown one changed (not mid-signal)
            return
        if item is None or field not in ("name", "visible"):
            return
        self._updating = True
        if not isinstance(node, SuggestionNode):
            item.setText(0, node.name)
        self._show_visible(item, node.visible)
        if isinstance(node, MaskNode) and node.parent is not None:
            self._show_folder_eye(node.parent)
        self._updating = False

    def _model_selection(self, _node=None) -> None:
        self._updating = True
        self.tree.clearSelection()
        chosen = self.project.selection or ([self.project.selected] if self.project.selected else [])
        for n in chosen:
            item = self._items.get(n.id)
            if item is not None:
                item.setSelected(True)
        primary = self._items.get(self.project.selected.id) if self.project.selected else None
        if primary is not None:
            self.tree.setCurrentItem(primary, 0, QtCore.QItemSelectionModel.SelectionFlag.NoUpdate)
            self.tree.scrollToItem(primary)
        self._updating = False

    # 2. Tree → model
    def _clicked(self, item, column: int) -> None:
        """A click on the eye shows or hides that layer (all the selected ones, if it is among them)."""
        if column != EYE or self._updating:
            return
        node = self._node(item)
        if node is None:
            return
        selected = self._selected_nodes()
        targets = selected if item.isSelected() and len(selected) > 1 else [node]
        visible = not node.visible
        for n in targets:
            self.project.set_visible(n, visible)

    def _side_clicked(self, item) -> None:
        """The eye of a folder or a group of suggestions, or a version's Σ (included in the parameters)."""
        folder, group, version = item.data(0, FOLDER), item.data(0, SUGG_GROUP), item.data(0, VIEW_ROLE)
        if folder and folder[1] == "annotations":
            image = self.project.find(folder[0])
            if image is not None:
                shown = not any(m.visible for m in image.masks)
                for m in image.masks:
                    self.project.set_visible(m, shown)
        elif group:
            image = self.project.find(group[0])
            node = image.suggestions if image is not None else None
            if node is not None:
                self.project.set_suggestions_shown(node, group[1], group[1] in node.hidden_groups)
        elif version:
            image = self.project.find(version[0])
            if image is None:
                return
            view = next((v for v in image.views if v.id == version[1]), None)
            if version[1] and view is None:
                return
            was = image.original_measured if view is None else view.measure
            before = image.metadata.get("measure_original")
            name = "the original" if view is None else view.name
            from modalytics.core.commands import Command

            def undo_():
                if view is None and before is None:          # back to "as the profile says"
                    image.metadata.pop("measure_original", None)
                    self.project.node_changed.emit(image, "views")
                else:
                    self.project.set_measured(image, view, was)
            command = Command(f"{'Leave out' if was else 'Include'} {name} of {image.name} "
                              f"{'from' if was else 'in'} the parameters",
                              lambda: self.project.set_measured(image, view, not was), undo_)
            if self.undo is not None:
                self.undo.push(command)
            else:
                command.do()

    def _selected_nodes(self) -> list:
        return [n for n in (self._node(it) for it in self.tree.selectedItems()) if n is not None]

    def _tree_selection(self) -> None:
        if self._updating:
            return
        items = self.tree.selectedItems()
        view_row = items[0].data(0, VIEW_ROLE) if len(items) == 1 else None
        if view_row:                                     # a view row: show that version, select its image
            image = self.project.find(view_row[0])
            if image is not None:
                self.project.select(image)
                self.project.show_view(image, next((v for v in image.views if v.id == view_row[1]), None))
            return
        nodes = self._selected_nodes()
        current = self._node(self.tree.currentItem())
        self.project.select_many(nodes, current if current in nodes else None)

    def _reordered(self, ids: list) -> None:
        """A drop: the images in the order of the list (top first) → the drawing order (bottom first)."""
        by_id = {img.id: img for img in self.project.images}
        order = [by_id[i] for i in reversed(ids) if i in by_id]
        if len(order) == len(self.project.images):
            keep, primary = list(self.project.selection), self.project.selected
            self.project.reorder_images(order)
            self.project.select_many(keep, primary)

    def _move_to(self, images: list, top: bool) -> None:
        rest = [i for i in self.project.images if i not in images]
        moving = [i for i in self.project.images if i in images]
        self.project.reorder_images(rest + moving if top else moving + rest)

    def _context_menu(self, pos) -> None:
        item = self.tree.itemAt(pos)
        if item is None:
            return
        node = self._node(item)
        special = self._special_menu(item, node)
        if special is not None:
            special.exec(self.tree.viewport().mapToGlobal(pos))
            return
        if node is None:
            return
        if not item.isSelected():                        # right-click outside the selection: that layer alone
            self.project.select(node)
        nodes = self._selected_nodes() or [node]
        images = [n for n in nodes if isinstance(n, ImageNode)]
        menu = QtWidgets.QMenu(self)
        many = len(nodes) > 1
        if self.fit_requested:
            menu.addAction(f"Frame {len(nodes)} layers in view" if many else "Frame in view",
                           lambda: self.fit_requested(nodes if many else node))
        hide = any(n.visible for n in nodes)
        menu.addAction(("Hide" if hide else "Show") + (f" {len(nodes)} layers" if many else ""),
                       lambda: [self.project.set_visible(n, not hide) for n in nodes])
        if images:
            menu.addAction("Move to the top", lambda: self._move_to(images, True))
            menu.addAction("Move to the bottom", lambda: self._move_to(images, False))
        if self.extend_menu is not None:
            menu.addSeparator()
            self.extend_menu(menu, nodes)
        if not many and isinstance(node, MaskNode):
            # Transfer / duplicate keep the world position (see Project.transfer_mask)
            menu.addSeparator()
            others = [i for i in self.project.images if i is not node.parent]
            move_menu = menu.addMenu("Transfer to")
            copy_menu = menu.addMenu("Duplicate to")
            for img in reversed(others):
                move_menu.addAction(img.name, lambda i=img: self.project.transfer_mask(node, i))
                copy_menu.addAction(img.name, lambda i=img: self.project.transfer_mask(node, i, duplicate=True))
            move_menu.setEnabled(bool(others))
            copy_menu.setEnabled(bool(others))
        menu.addSeparator()
        menu.addAction(f"Remove {len(nodes)} layers…" if many else "Remove", lambda: self.remove(nodes))
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def _special_menu(self, item, node):
        """The menu of a version row, a group of suggestions or the Suggestions folder (None: the layers' menu)."""
        version, group = item.data(0, VIEW_ROLE), item.data(0, SUGG_GROUP)
        if version:
            image = self.project.find(version[0])
            if image is None:
                return None
            view = next((v for v in image.views if v.id == version[1]), None)
            if version[1] and view is None:
                return None
            menu = QtWidgets.QMenu(self)
            name = "the original" if view is None else view.name
            if image.shown is not view:
                menu.addAction(f"Draw {name}", lambda: (self.project.select(image),
                                                        self.project.show_view(image, view)))
            included = image.original_measured if view is None else view.measure
            menu.addAction("Leave out of the parameters" if included else "Include in the parameters",
                           lambda: self._side_clicked(item))
            if view is not None:
                from modalytics.ui import processing as proc
                menu.addSeparator()
                menu.addAction("Edit chain…", lambda: proc.open_chain_editor(self, self.project, self.undo, image,
                                                                              view))
                menu.addAction(f"Remove {view.name}", lambda: self._remove_view(image, view))
            return menu
        if group or isinstance(node, SuggestionNode):
            if group:
                image = self.project.find(group[0])
                node = image.suggestions if image is not None else None
            if node is None:
                return None
            items = [it for it in node.items if not group or it.group == group[1]]
            what = f"{group[1]} suggestions" if group else "suggestions"
            menu = QtWidgets.QMenu(self)
            accept = menu.addAction(f"Accept all {len(items)} {what}", lambda: self._review_all(node, items, True))
            delete = menu.addAction(f"Delete all {len(items)} {what}", lambda: self._review_all(node, items, False))
            accept.setEnabled(bool(items))
            delete.setEnabled(bool(items))
            menu.addSeparator()
            if group:
                shown = group[1] not in node.hidden_groups
                menu.addAction("Hide them" if shown else "Show them",
                               lambda: self.project.set_suggestions_shown(node, group[1], not shown))
            else:
                if self.fit_requested:
                    menu.addAction("Frame in view", lambda: self.fit_requested(node))
                menu.addAction("Hide" if node.visible else "Show",
                               lambda: self.project.set_visible(node, not node.visible))
            return menu
        return None

    def _review_all(self, node, items: list, accept: bool) -> None:
        """Accept (into the annotation, each with its suggested grade) or delete suggestions: one undo step."""
        from modalytics.services import suggest
        if not items:
            return
        command = suggest.accept_command(self.project, node, items) if accept else \
            suggest.reject_command(self.project, node, items)
        if self.undo is not None:
            self.undo.push(command)
        else:
            command.do()

    def _remove_view(self, image, view) -> None:
        """Remove a processed version (one undo step; the original is drawn again if it was the one drawn)."""
        from modalytics.ui import processing as proc
        if self.undo is not None:
            proc.remove_view(self.project, self.undo, image, view)
        else:
            self.project.remove_view(image, view)

    def remove(self, nodes: list) -> None:
        """Remove layers (asks first for several, or for an image: its masks go with it)."""
        images = [n for n in nodes if isinstance(n, ImageNode)]
        if len(nodes) > 1 or images:
            what = ", ".join(n.name for n in nodes[:5]) + (" …" if len(nodes) > 5 else "")
            ask = QtWidgets.QMessageBox.question(
                self, "Remove layers", f"Remove {len(nodes)} layer(s): {what}?"
                + ("\n\nAn image is removed with its masks and views." if images else ""))
            if ask != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        for n in images:
            if n in self.project.images:
                self.project.remove(n)
        for n in nodes:
            if not isinstance(n, ImageNode) and n.parent is not None and n.parent in self.project.images:
                self.project.remove(n)
