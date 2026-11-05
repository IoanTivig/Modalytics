"""
Dark and light themes from one set of design tokens.

Every colour used by the UI *and* the canvas comes from ``TOKENS``; the Qt
style sheet is generated from them, so a new theme is one more dict.

Look: deep slate surfaces with a single teal accent (not napari's neutral
grey), high-contrast text, thin borders, 6 px radii.
"""

from __future__ import annotations

from PySide6 import QtGui, QtWidgets

TOKENS: dict[str, dict[str, str]] = {
    "dark": {
        "bg": "#11151b",            # window background
        "surface": "#171c24",       # panels
        "surface_alt": "#1e2530",   # inputs, hovered rows
        "border": "#2a3341",
        "text": "#e3e8ef",
        "text_muted": "#8b97a8",
        "accent": "#2fb3a3",        # teal
        "accent_text": "#06221f",
        "selection": "#1f4e4a",
        "danger": "#e5675f",
        "warning": "#e0a33a",       # amber: something to check
        "canvas_bg": "#0a0d11",
        "minimap_bg": "#05070a",
        "minimap_border": "#9aa6b8",
    },
    "light": {
        "bg": "#eef1f5",
        "surface": "#fbfcfd",
        "surface_alt": "#ffffff",
        "border": "#d3d9e2",
        "text": "#17202b",
        "text_muted": "#5a6677",
        "accent": "#0e8a7d",
        "accent_text": "#ffffff",
        "selection": "#cdeee9",
        "danger": "#c4423a",
        "warning": "#a8670a",
        "canvas_bg": "#dfe4ea",
        "minimap_bg": "#c9d0d9",
        "minimap_border": "#4a5566",
    },
}


def branch_arrows(t: dict[str, str]) -> tuple[str, str]:
    """(closed, open) arrow images in the theme's muted text colour (written once): the trees' branches, the
    drop-down lists and the menu buttons.

    A style sheet that gives a branch a background (hovered, selected) stops Qt drawing its arrow there, and a styled
    drop-down list gets Windows' old arrow, so the arrows are images the style sheet names.
    """
    import qtawesome as qta

    from modalytics import paths
    folder = paths.user_dir("cache") / "theme"
    folder.mkdir(parents=True, exist_ok=True)
    out = []
    for name, icon in (("closed", "mdi6.chevron-right"), ("open", "mdi6.chevron-down")):
        path = folder / f"branch_{name}_{t['text_muted'].lstrip('#')}.png"
        if not path.exists():
            qta.icon(icon, color=t["text_muted"]).pixmap(32, 32).save(str(path))
        out.append(path.as_posix())
    return out[0], out[1]


def stylesheet(t: dict[str, str], arrows: tuple[str, str] | None = None) -> str:
    """Build the application style sheet from theme tokens.

    Args:
        t: One entry of ``TOKENS``.
        arrows: The arrow images (``branch_arrows``): trees, drop-down lists, menu buttons; None: Qt's own.

    Returns:
        str: Qt style sheet.
    """
    return f"""
    QWidget {{ background: {t['bg']}; color: {t['text']}; font-size: 10pt; }}
    QLabel {{ background: transparent; }}
    QMainWindow::separator {{ background: {t['bg']}; width: 6px; height: 6px; }}
    QMainWindow::separator:hover {{ background: {t['accent']}; }}
    QDockWidget {{ titlebar-close-icon: none; }}
    QDockWidget::title {{ background: {t['surface']}; padding: 8px 10px; border-bottom: 1px solid {t['border']};
                          font-weight: 600; text-transform: uppercase; }}
    QDockWidget > QWidget {{ background: {t['surface']}; }}
    #Panel, #Panel QWidget {{ background: {t['surface']}; }}
    QLabel#Muted {{ color: {t['text_muted']}; }}
    QLabel#SectionTitle {{ color: {t['text_muted']}; font-size: 8.5pt; font-weight: 600; padding-top: 8px; }}
    QLabel#AppTitle {{ font-size: 12pt; font-weight: 700; color: {t['accent']}; padding: 0 12px 0 6px; }}
    QToolBar {{ background: {t['surface']}; border: none; border-bottom: 1px solid {t['border']}; spacing: 4px; padding: 4px; }}
    QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 6px; padding: 5px 10px; }}
    QToolButton:hover {{ background: {t['surface_alt']}; border-color: {t['border']}; }}
    QToolButton:checked {{ background: {t['accent']}; color: {t['accent_text']}; }}
    QToolButton:disabled {{ color: {t['text_muted']}; }}
    QToolButton#Step {{ padding: 5px 12px; }}
    #Ribbon, #RibbonPage, #RibbonBody, #RibbonGroup {{ background: {t['surface']}; }}
    QScrollArea#RibbonScroll, QScrollArea#RibbonScroll > QWidget {{ background: {t['surface']}; border: none; }}
    QScrollArea#RibbonScroll QScrollBar:horizontal {{ height: 8px; background: transparent; }}
    QScrollArea#RibbonScroll QScrollBar::handle:horizontal {{ background: {t['border']}; border-radius: 3px;
        min-width: 40px; }}
    QScrollArea#RibbonScroll QScrollBar::add-line, QScrollArea#RibbonScroll QScrollBar::sub-line {{ width: 0; }}
    #Ribbon {{ border-bottom: 1px solid {t['border']}; }}
    #Panel QLabel, #Ribbon QLabel {{ background: transparent; }}
    QToolBar#RibbonBar {{ padding: 0; border-bottom: 1px solid {t['border']}; }}
    QTabBar#RibbonTabs {{ background: transparent; }}
    QTabBar#RibbonTabs::tab {{ background: transparent; color: {t['text_muted']}; padding: 6px 14px;
        border: none; border-bottom: 2px solid transparent; font-weight: 600; }}
    QTabBar#RibbonTabs::tab:selected {{ color: {t['text']}; border-bottom: 2px solid {t['accent']}; }}
    QTabBar#RibbonTabs::tab:hover {{ color: {t['text']}; }}
    QLabel#RibbonGroupTitle {{ color: {t['text_muted']}; font-size: 8pt; padding: 1px 0 2px 0; }}
    QFrame#RibbonSeparator {{ color: {t['border']}; }}
    QFrame#RibbonCornerSeparator {{ color: {t['border']}; }}
    QToolButton#RibbonCorner {{ padding: 4px; border-radius: 6px; border: 1px solid transparent; }}
    QToolButton#RibbonCorner:hover {{ background: {t['selection']}; }}
    QToolButton#RibbonCorner:checked {{ background: transparent; border: 1px solid {t['accent']}; }}
    QToolButton#RibbonButton {{ padding: 4px 8px; font-size: 9pt; }}   /* width: RibbonButton */
    QToolButton#RibbonButton:checked {{ background: {t['selection']}; color: {t['text']};
        border: 1px solid {t['accent']}; }}
    QTreeWidget {{ background: {t['surface']}; border: none; outline: 0; }}
    QTreeWidget::item {{ padding: 4px 2px; border-radius: 4px; }}
    QTreeWidget::item:hover {{ background: {t['surface_alt']}; }}
    QTreeWidget::item:selected {{ background: {t['selection']}; color: {t['text']}; }}
    QTreeWidget#ListTree, QTreeWidget#LayerTree {{ show-decoration-selected: 1; }}
    QTreeWidget#LayerTree::item {{ border-radius: 0; padding: 3px 2px; }}
    QTreeWidget#LayerTree::branch:selected {{ background: {t['selection']}; }}
    QTreeWidget#LayerTree::branch:hover {{ background: {t['surface_alt']}; }}
    QTreeWidget#ListTree::item {{ border-radius: 0; }}
    QTreeWidget#ListTree::branch:selected {{ background: {t['selection']}; }}
    QTreeWidget#ListTree::branch:hover {{ background: {t['surface_alt']}; }}
    QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {{ background: {t['surface_alt']}; border: 1px solid {t['border']};
        border-radius: 6px; padding: 3px 6px; selection-background-color: {t['accent']}; }}
    QLineEdit:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {t['accent']}; }}
    QComboBox:disabled, QLineEdit:disabled, QLineEdit:read-only, QSpinBox:disabled, QDoubleSpinBox:disabled {{
        color: {t['text_muted']}; background: {t['surface']}; border-color: {t['surface_alt']}; }}
    QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{ width: 0px; border: none; }}
    QPushButton {{ background: {t['surface_alt']}; border: 1px solid {t['border']}; border-radius: 6px; padding: 4px 10px; }}
    QPushButton:hover {{ border-color: {t['accent']}; }}
    QPushButton:disabled {{ color: {t['text_muted']}; border-color: {t['surface_alt']}; background: {t['surface']}; }}
    QSlider::groove:horizontal {{ height: 4px; background: {t['border']}; border-radius: 2px; }}
    QSlider::sub-page:horizontal {{ background: {t['accent']}; border-radius: 2px; }}
    QSlider::handle:horizontal {{ background: {t['text']}; width: 12px; height: 12px; margin: -5px 0; border-radius: 6px; }}
    QMenu {{ background: {t['surface']}; border: 1px solid {t['border']}; padding: 4px; }}
    QMenu::item {{ padding: 5px 18px; border-radius: 4px; }}
    QMenu::item:selected {{ background: {t['selection']}; }}
    QMenu::separator {{ height: 1px; background: {t['border']}; margin: 4px 6px; }}
    QMenuBar {{ background: {t['surface']}; }}
    QMenuBar::item:selected {{ background: {t['surface_alt']}; }}
    QStatusBar {{ background: {t['surface']}; border-top: 1px solid {t['border']}; color: {t['text_muted']}; }}
    QStatusBar QLabel {{ background: transparent; color: {t['text_muted']}; padding: 0 8px; }}
    QScrollBar:vertical {{ background: transparent; width: 10px; }}
    QScrollBar::handle:vertical {{ background: {t['border']}; border-radius: 5px; min-height: 24px; }}
    QScrollBar:horizontal {{ background: transparent; height: 10px; }}
    QScrollBar::handle:horizontal {{ background: {t['border']}; border-radius: 5px; min-width: 24px; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
    QToolTip {{ background: {t['surface_alt']}; color: {t['text']}; border: 1px solid {t['border']}; }}
    QPlainTextEdit, QTextEdit {{ background: {t['surface_alt']}; border: 1px solid {t['border']}; border-radius: 6px; }}
    QListWidget, QTableWidget {{ background: {t['surface']}; border: 1px solid {t['border']}; border-radius: 6px;
        outline: 0; }}
    QListWidget::item {{ padding: 4px 6px; border-radius: 4px; }}
    QListWidget::item:hover {{ background: {t['surface_alt']}; }}
    QListWidget::item:selected {{ background: {t['selection']}; color: {t['text']}; }}
    QHeaderView::section {{ background: {t['surface_alt']}; color: {t['text_muted']}; border: none;
        border-bottom: 1px solid {t['border']}; padding: 4px 6px; }}
    QTableWidget {{ gridline-color: {t['border']}; }}
    QListWidget#SettingsNav {{ border: none; background: {t['surface']}; font-size: 10.5pt; }}
    QListWidget#SettingsNav::item {{ padding: 9px 12px; }}
    QToolButton#StartNav {{ font-size: 12pt; padding: 8px 18px; border: 1px solid {t['border']}; border-radius: 8px; }}
    QToolButton#StartNav:checked {{ background: {t['accent']}; color: {t['accent_text']}; border-color: {t['accent']}; }}
    QToolButton#StartNav:disabled {{ color: {t['text_muted']}; border-style: dashed; }}
    QListWidget#ProfileCards {{ border: none; background: transparent; }}
    QListWidget#ProfileCards::item {{ padding: 12px 14px; margin: 3px 0; border: 1px solid {t['border']};
        border-radius: 8px; background: {t['surface']}; }}
    QListWidget#ProfileCards::item:selected {{ border-color: {t['accent']}; background: {t['selection']}; }}
    """ + _arrow_rules(arrows)


def _arrow_rules(arrows: tuple[str, str] | None) -> str:
    """The arrows drawn from the theme's images: the trees' branches, the drop-down lists', the menu buttons'.

    Last in the sheet, so they win over the boxes' own padding. Without images: Qt's own (Windows-like) arrows.
    """
    if not arrows:
        return ""
    closed, opened = arrows
    rules = []
    for tree in ("QTreeWidget#LayerTree", "QTreeWidget#ListTree"):
        rules += [f"{tree}::branch:has-children:closed {{ image: url({closed}); }}",
                  f"{tree}::branch:has-children:open {{ image: url({opened}); }}"]
    rules += ["QComboBox { padding-right: 24px; }",
              "QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: center right; width: 22px; "
              "border: none; background: transparent; }",
              f"QComboBox::down-arrow {{ image: url({opened}); width: 16px; height: 16px; }}",
              f"QToolButton::menu-indicator {{ image: url({opened}); width: 10px; height: 10px; }}",
              "QToolButton::menu-button { border: none; width: 16px; }",
              f"QToolButton::menu-arrow {{ image: url({opened}); width: 12px; height: 12px; }}"]
    return "\n".join(rules) + "\n"


def apply_theme(app: QtWidgets.QApplication, name: str) -> dict[str, str]:
    """Apply a theme to the whole application.

    Args:
        app: The application.
        name: "dark" or "light".

    Returns:
        dict: The theme tokens (for the canvas and custom widgets).
    """
    t = TOKENS[name]
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    for role, key in ((QtGui.QPalette.ColorRole.Window, "bg"), (QtGui.QPalette.ColorRole.Base, "surface_alt"),
                      (QtGui.QPalette.ColorRole.AlternateBase, "surface"),     # every other table row
                      (QtGui.QPalette.ColorRole.Text, "text"), (QtGui.QPalette.ColorRole.WindowText, "text"),
                      (QtGui.QPalette.ColorRole.Button, "surface_alt"), (QtGui.QPalette.ColorRole.ButtonText, "text"),
                      (QtGui.QPalette.ColorRole.Highlight, "accent"),
                      (QtGui.QPalette.ColorRole.HighlightedText, "accent_text")):
        pal.setColor(role, QtGui.QColor(t[key]))
    app.setPalette(pal)
    try:
        arrows = branch_arrows(t)
    except Exception:                                    # noqa: BLE001 (no user folder: Qt's own arrows)
        arrows = None
    app.setStyleSheet(stylesheet(t, arrows))
    return t
