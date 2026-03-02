"""
main.py — MotionGuard application entry point.

IMPORTANT: QT_QPA_PLATFORM must be set BEFORE any Qt or OpenCV imports.
This also prevents ti_motion_detect_v4.py from overriding it to "offscreen".
"""

import os
import sys
from pathlib import Path

# --- Must be first: force Qt platform before any imports ---
os.environ.setdefault("QT_QPA_PLATFORM", "windows")

# --- Add app/ directory to sys.path so all subpackages resolve ---
_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

# --- Standard library imports ---
import logging
import signal
import traceback

# --- PySide6 ---
from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtGui import QIcon
from PySide6.QtCore import Qt

# --- Application modules ---
from utils.paths import logo_path
from utils.logging_setup import setup_logging, update_log_level
from storage.db import init_db
from storage.migrations import run_migrations
from storage.repositories import get_setting
from ui.main_window import MainWindow

# ---------------------------------------------------------------------------
# Stylesheet — military-grade, white background, simple black text
# ---------------------------------------------------------------------------

STYLESHEET = """
QMainWindow, QDialog, QWidget {
    background-color: #FFFFFF;
    color: #000000;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 13px;
}

QToolBar {
    background: #F0F0F0;
    border-bottom: 1px solid #CCCCCC;
    spacing: 4px;
    padding: 2px;
}

QMenuBar {
    background: #F0F0F0;
    border-bottom: 1px solid #CCCCCC;
}

QMenuBar::item:selected {
    background: #000000;
    color: #FFFFFF;
}

QMenu {
    background: #FFFFFF;
    border: 1px solid #CCCCCC;
}

QMenu::item:selected {
    background: #000000;
    color: #FFFFFF;
}

QPushButton {
    background: #E8E8E8;
    border: 1px solid #AAAAAA;
    padding: 4px 10px;
    min-height: 22px;
}

QPushButton:hover {
    background: #D0D0D0;
    border: 1px solid #888888;
}

QPushButton:pressed {
    background: #B0B0B0;
}

QPushButton:disabled {
    background: #F0F0F0;
    color: #AAAAAA;
    border: 1px solid #CCCCCC;
}

QPushButton:default {
    border: 2px solid #333333;
}

QTreeWidget {
    border: 1px solid #CCCCCC;
    background: #FFFFFF;
}

QTreeWidget::item:selected {
    background: #000000;
    color: #FFFFFF;
}

QTreeWidget::item:hover {
    background: #E8E8E8;
}

QListWidget {
    border: 1px solid #CCCCCC;
    background: #FFFFFF;
}

QListWidget::item:selected {
    background: #000000;
    color: #FFFFFF;
}

QTableWidget {
    border: 1px solid #CCCCCC;
    gridline-color: #EEEEEE;
    background: #FFFFFF;
}

QTableWidget::item:selected {
    background: #000000;
    color: #FFFFFF;
}

QHeaderView::section {
    background: #F0F0F0;
    border: 1px solid #CCCCCC;
    padding: 3px 6px;
    font-weight: bold;
}

QGroupBox {
    border: 1px solid #CCCCCC;
    margin-top: 8px;
    padding-top: 4px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    font-weight: bold;
}

QTabWidget::pane {
    border: 1px solid #CCCCCC;
}

QTabBar::tab {
    background: #F0F0F0;
    border: 1px solid #CCCCCC;
    padding: 5px 12px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background: #FFFFFF;
    border-bottom: none;
    font-weight: bold;
}

QTabBar::tab:hover {
    background: #E0E0E0;
}

QLineEdit, QTextEdit, QPlainTextEdit {
    border: 1px solid #CCCCCC;
    background: #FFFFFF;
    padding: 3px;
}

QLineEdit:focus, QTextEdit:focus {
    border: 1px solid #666666;
}

QSpinBox, QDoubleSpinBox, QComboBox {
    border: 1px solid #CCCCCC;
    background: #FFFFFF;
    padding: 2px;
    min-height: 22px;
}

QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 16px;
    border-left: 1px solid #CCCCCC;
    border-bottom: 1px solid #CCCCCC;
    background: #F0F0F0;
}

QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {
    background: #D8D8D8;
}

QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed {
    background: #B8B8B8;
}

QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 16px;
    border-left: 1px solid #CCCCCC;
    border-top: 1px solid #CCCCCC;
    background: #F0F0F0;
}

QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background: #D8D8D8;
}

QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {
    background: #B8B8B8;
}

QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    image: none;
    width: 0;
    height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 5px solid #444444;
}

QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    image: none;
    width: 0;
    height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #444444;
}

QComboBox::drop-down {
    border-left: 1px solid #CCCCCC;
}

QSlider::groove:horizontal {
    height: 4px;
    background: #CCCCCC;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #333333;
    border: 1px solid #555555;
    width: 12px;
    height: 12px;
    border-radius: 6px;
    margin: -4px 0;
}

QScrollBar:vertical {
    background: #F0F0F0;
    width: 12px;
}

QScrollBar::handle:vertical {
    background: #AAAAAA;
    border-radius: 4px;
    min-height: 20px;
}

QStatusBar {
    border-top: 1px solid #CCCCCC;
    background: #F8F8F8;
    font-size: 11px;
}

QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid #AAAAAA;
    background: #FFFFFF;
}

QCheckBox::indicator:checked {
    background: #333333;
    border: 1px solid #000000;
}

QSplitter::handle {
    background: #CCCCCC;
    width: 2px;
    height: 2px;
}
"""


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------

_exception_dialog_shown = False   # guard against re-entrant dialog loops


def _handle_exception(exc_type, exc_value, exc_tb) -> None:
    global _exception_dialog_shown
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return
    log = logging.getLogger("motionguard.crash")
    msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    log.critical("Uncaught exception:\n%s", msg)
    # Only show one dialog at a time; repeated errors only go to the log
    if _exception_dialog_shown:
        return
    _exception_dialog_shown = True
    try:
        QMessageBox.critical(
            None,
            "MotionGuard — Unexpected Error",
            f"An unexpected error occurred:\n\n{exc_value}\n\n"
            "See the log file for details.",
        )
    except Exception:
        pass
    finally:
        _exception_dialog_shown = False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    # Install global exception handler
    sys.excepthook = _handle_exception

    # Initialize logging (level updated from DB after DB is ready)
    setup_logging("INFO")
    log = logging.getLogger("motionguard")
    log.info("=" * 60)
    log.info("MotionGuard starting  Python %s", sys.version.split()[0])
    log.info("=" * 60)

    # Initialize and migrate database
    try:
        init_db()
        run_migrations()
    except Exception as exc:
        log.critical("Database initialization failed: %s", exc)
        # Can't show QMessageBox before QApplication
        print(f"FATAL: Database error: {exc}", file=sys.stderr)
        return 1

    # Apply saved log level
    update_log_level(get_setting("log_level", "INFO"))

    # Create QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("MotionGuard")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("TI")
    app.setQuitOnLastWindowClosed(False)  # stay alive in tray

    # Apply stylesheet
    app.setStyleSheet(STYLESHEET)

    # Set window icon
    icon_p = logo_path()
    if icon_p.exists():
        app.setWindowIcon(QIcon(str(icon_p)))

    # Create and show main window
    window = MainWindow()

    start_min = get_setting("start_minimized", "0") == "1"
    if not start_min:
        window.show()

    # Allow Ctrl+C in the terminal to trigger a clean shutdown.
    # Qt on Windows captures SIGINT and never forwards it to Python unless we
    # install our own handler.  A 200 ms QTimer wakes the event loop regularly
    # so the signal handler can actually execute.
    from PySide6.QtCore import QTimer
    signal.signal(signal.SIGINT, lambda *_: window._quit())
    _sig_poll = QTimer()
    _sig_poll.start(200)
    _sig_poll.timeout.connect(lambda: None)  # keeps Python's signal machinery alive

    log.info("Application event loop starting")
    exit_code = app.exec()
    log.info("Application exiting with code %d", exit_code)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
