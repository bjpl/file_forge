"""Processing Queue View."""
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QPushButton,
    QProgressBar, QTextEdit, QGroupBox, QSplitter
)
from PySide6.QtCore import Qt, Signal

from ..viewmodels import ProcessingViewModel
from ..state import AppState


class ProcessingView(QWidget):
    """Processing queue view for batch file processing.

    Features:
    - Queue management (add/remove files)
    - Start/pause/cancel controls
    - Progress tracking with ETA
    - Real-time log display
    """

    processing_started = Signal()
    processing_complete = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._viewmodel = ProcessingViewModel()
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Setup processing UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # Header
        header_layout = QHBoxLayout()
        header = QLabel("Processing Queue")
        header.setStyleSheet("font-size: 24px; font-weight: 600;")
        header_layout.addWidget(header)
        header_layout.addStretch()

        # Add files button
        add_btn = QPushButton("+ Add Files")
        add_btn.clicked.connect(self._add_files)
        add_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078D4;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #106EBE;
            }
        """)
        header_layout.addWidget(add_btn)

        layout.addLayout(header_layout)

        # Main content splitter
        splitter = QSplitter(Qt.Vertical)

        # Queue section
        queue_group = QGroupBox("Files in Queue")
        queue_layout = QVBoxLayout(queue_group)

        self._queue_list = QListWidget()
        self._queue_list.setSelectionMode(QListWidget.ExtendedSelection)
        queue_layout.addWidget(self._queue_list)

        # Queue controls
        queue_controls = QHBoxLayout()
        remove_btn = QPushButton("Remove Selected")
        remove_btn.clicked.connect(self._remove_selected)
        queue_controls.addWidget(remove_btn)

        clear_btn = QPushButton("Clear Queue")
        clear_btn.clicked.connect(self._clear_queue)
        queue_controls.addWidget(clear_btn)
        queue_controls.addStretch()

        self._queue_count = QLabel("0 files")
        queue_controls.addWidget(self._queue_count)

        queue_layout.addLayout(queue_controls)
        splitter.addWidget(queue_group)

        # Progress section
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)

        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setMinimum(0)
        self._progress_bar.setMaximum(100)
        self._progress_bar.setValue(0)
        self._progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #E1E1E1;
                border-radius: 4px;
                text-align: center;
                height: 24px;
            }
            QProgressBar::chunk {
                background-color: #0078D4;
                border-radius: 3px;
            }
        """)
        progress_layout.addWidget(self._progress_bar)

        # Progress info
        info_layout = QHBoxLayout()
        self._status_label = QLabel("Ready")
        info_layout.addWidget(self._status_label)
        info_layout.addStretch()
        self._eta_label = QLabel("")
        info_layout.addWidget(self._eta_label)
        progress_layout.addLayout(info_layout)

        # Control buttons
        controls_layout = QHBoxLayout()

        self._start_btn = QPushButton("▶ Start")
        self._start_btn.clicked.connect(self._start_processing)
        self._start_btn.setStyleSheet("""
            QPushButton {
                background-color: #107C10;
                color: white;
                border: none;
                padding: 10px 24px;
                border-radius: 4px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #0E6E0E;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
            }
        """)
        controls_layout.addWidget(self._start_btn)

        self._pause_btn = QPushButton("⏸ Pause")
        self._pause_btn.clicked.connect(self._pause_processing)
        self._pause_btn.setEnabled(False)
        controls_layout.addWidget(self._pause_btn)

        self._cancel_btn = QPushButton("⏹ Cancel")
        self._cancel_btn.clicked.connect(self._cancel_processing)
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #D83B01;
                color: white;
                border: none;
                padding: 10px 24px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #B83301;
            }
            QPushButton:disabled {
                background-color: #CCCCCC;
            }
        """)
        controls_layout.addWidget(self._cancel_btn)

        controls_layout.addStretch()
        progress_layout.addLayout(controls_layout)

        splitter.addWidget(progress_group)

        # Log section
        log_group = QGroupBox("Processing Log")
        log_layout = QVBoxLayout(log_group)

        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setStyleSheet("""
            QTextEdit {
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 12px;
                background-color: #1E1E1E;
                color: #D4D4D4;
                border: 1px solid #333;
            }
        """)
        log_layout.addWidget(self._log_text)

        splitter.addWidget(log_group)

        splitter.setSizes([200, 150, 200])
        layout.addWidget(splitter)

    def _connect_signals(self):
        """Connect viewmodel signals."""
        self._viewmodel.queue_changed.connect(self._update_queue_display)
        self._viewmodel.progress_changed.connect(self._update_progress)
        self._viewmodel.log_added.connect(self._add_log_entry)
        self._viewmodel.processing_complete.connect(self._on_complete)

        # Connect to app state for selected files
        AppState.instance().files_selected.connect(self._on_files_selected)

    def _on_files_selected(self, files: list):
        """Handle files selected from app state."""
        paths = [Path(f) if isinstance(f, str) else f for f in files]
        self._viewmodel.add_to_queue(paths)

    def _add_files(self):
        """Add files dialog."""
        from PySide6.QtWidgets import QFileDialog
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Files to Process",
            str(Path.home()),
            "All Files (*.*)"
        )
        if files:
            paths = [Path(f) for f in files]
            self._viewmodel.add_to_queue(paths)

    def _remove_selected(self):
        """Remove selected items from queue."""
        for item in self._queue_list.selectedItems():
            path = item.data(Qt.UserRole)
            if path:
                self._viewmodel.remove_from_queue(Path(path))

    def _clear_queue(self):
        """Clear processing queue."""
        self._viewmodel.clear_queue()

    def _update_queue_display(self):
        """Update queue list display."""
        self._queue_list.clear()

        for file_path in self._viewmodel.queue:
            item = QListWidgetItem(f"📄 {file_path.name}")
            item.setData(Qt.UserRole, str(file_path))
            item.setToolTip(str(file_path))
            self._queue_list.addItem(item)

        count = len(self._viewmodel.queue)
        self._queue_count.setText(f"{count} file{'s' if count != 1 else ''}")
        self._start_btn.setEnabled(count > 0 and not self._viewmodel.is_running)

    def _start_processing(self):
        """Start processing."""
        self._viewmodel.start()
        self._start_btn.setEnabled(False)
        self._pause_btn.setEnabled(True)
        self._cancel_btn.setEnabled(True)
        self._status_label.setText("Processing...")
        self.processing_started.emit()

    def _pause_processing(self):
        """Pause/resume processing."""
        if self._viewmodel.is_paused:
            self._viewmodel.resume()
            self._pause_btn.setText("⏸ Pause")
            self._status_label.setText("Processing...")
        else:
            self._viewmodel.pause()
            self._pause_btn.setText("▶ Resume")
            self._status_label.setText("Paused")

    def _cancel_processing(self):
        """Cancel processing."""
        self._viewmodel.cancel()
        self._start_btn.setEnabled(True)
        self._pause_btn.setEnabled(False)
        self._cancel_btn.setEnabled(False)
        self._status_label.setText("Cancelled")

    def _update_progress(self, value: int):
        """Update progress bar."""
        self._progress_bar.setValue(value)

        # Update ETA
        eta = self._viewmodel.eta_seconds
        if eta > 0:
            minutes, seconds = divmod(eta, 60)
            self._eta_label.setText(f"ETA: {minutes}m {seconds}s")
        else:
            self._eta_label.setText("")

    def _add_log_entry(self, entry: str):
        """Add entry to log."""
        self._log_text.append(entry)
        # Auto-scroll to bottom
        scrollbar = self._log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_complete(self, results: dict):
        """Handle processing completion."""
        self._start_btn.setEnabled(True)
        self._pause_btn.setEnabled(False)
        self._cancel_btn.setEnabled(False)
        self._status_label.setText("Complete")
        self._eta_label.setText("")
        self.processing_complete.emit(results)

    def refresh(self):
        """Refresh view."""
        self._update_queue_display()
