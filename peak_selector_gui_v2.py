#!/usr/bin/env python3
"""
IronLPeaks — Fe L-edge Peak Selector GUI v2
IronLPeaks: Publication-quality peak analysis tool for Fe L-edge XANES spectroscopy.
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter

# Import our modules
try:
    from fe_data_loader_v2 import FeDataLoaderV2
except ImportError:
    FeDataLoaderV2 = None

try:
    from arctan_baseline_Fe2p_v2 import ArctanBaselineFe2pV2
except ImportError:
    ArctanBaselineFe2pV2 = None

try:
    from van_aken_fe_quantification import VanAkenFeQuantifier
except ImportError:
    VanAkenFeQuantifier = None

try:
    from extract_athena_spectra import load_prj_spectra
except ImportError:
    load_prj_spectra = None

# Sun Valley theme not used — keep native look consistent with CarbonKPeaks/SulfurKPeaks
_HAS_SV_TTK = False

# ---------------------------------------------------------------------------
# Publication-quality matplotlib defaults
# ---------------------------------------------------------------------------
_PUB_RC = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'axes.linewidth': 1.2,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'legend.fontsize': 9,
    'legend.framealpha': 0.8,
}
plt.rcParams.update(_PUB_RC)

# UI accent colors (used in labels and plot markers)
_ACCENT = '#2962ff'
_SUCCESS = '#2e7d32'
_ERROR = '#c62828'
_SECTION_FG = '#37474f'

# Fe L-edge physical constants (Von der Heyden 2017, Cressey 1993)
_FE_L3_MIN_SEPARATION = 0.8    # minimum resolvable L3 splitting (eV)
_FE_L3_MAX_SEPARATION = 3.5    # maximum physical L3 splitting (eV)
_SG_POLYORDER = 3              # Savitzky-Golay polynomial order

# Supported spectrum file extensions
_SPECTRUM_FILETYPES = [
    ("All Spectrum Files", "*.csv *.nor *.txt *.dat *.xas *.prj"),
    ("CSV files", "*.csv"),
    ("Athena Project", "*.prj"),
    ("Normalized files", "*.nor"),
    ("Text files", "*.txt"),
    ("DAT files", "*.dat"),
    ("XAS files", "*.xas"),
    ("All files", "*.*"),
]
_SPECTRUM_EXTENSIONS = ('.csv', '.nor', '.txt', '.dat', '.xas')


class PeakSelectorGUI:
    """Main application class for the Fe L-edge Peak Selector."""

    def __init__(self, root, csv_path=None, report_id=None,
                 spectra_dir=None, selected_files=None):
        self.root = root
        self.csv_path = csv_path
        self.report_id = report_id or "test"
        self.spectra_dir = spectra_dir
        self.selected_files = selected_files

        # Data storage
        self.df = None
        self.current_sample_index = 0
        self.current_spectrum_data = None
        self.current_smoothed_data = None
        self.current_baseline = None
        self.current_baseline_fitter = None
        self.selected_peaks = []
        self.peak_candidates = np.array([])
        self.current_peak_candidate_idx = 0
        self.original_spectrum_data = None
        self.current_vanaken = None
        self._athena_spectra = {}  # sample_name -> {energy, intensity}
        self._session_save_path = None  # path for Ctrl+S quick-save
        self._missing_spectrum_warned = set()  # one-time missing-file warnings

        # Track whether arctan param fields were explicitly edited by user
        self._baseline_params_are_manual = False

        # Guard flag: prevents on_sample_select from firing during
        # programmatic listbox updates (populate_sample_list, etc.)
        self._updating_listbox = False

        # Peak detection window
        self.window_start = 705.0
        self.window_end = 718.0
        self.dragging_window = False
        self.drag_start_x = None

        # Remember last-used directory for file dialogs
        self._last_dir = spectra_dir or os.getcwd()

        # Resolve icon path (next to this script)
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        self._icon_path = os.path.join(_script_dir, 'ironlpeaks.ico')
        if not os.path.exists(self._icon_path):
            self._icon_path = os.path.join(_script_dir, 'iron_peaks.ico')

        # Set icon immediately so it shows during load
        self._set_icon()

        # Load data only if explicitly provided
        if self.csv_path is not None:
            self.load_csv()
        elif self.selected_files:
            self.auto_discover_samples()
        else:
            # Start with empty session — user adds files via File menu
            self.df = pd.DataFrame(columns=[
                'sample_name', 'sample_group',
                'peak1_energy', 'peak2_energy', 'peak1_intensity',
                'peak2_intensity', 'delta_ev', 'intensity_ratio',
                'include_in_report', 'smoothing_sigma',
                'baseline_offset', 'baseline_slope',
                'baseline_step', 'baseline_width', 'baseline_edge',
                'baseline_edge_l2',
                'vanaken_fe3', 'vanaken_ratio',
                'vanaken_p', 'vanaken_il3',
                'vanaken_il2', 'vanaken_l3_lo',
                'vanaken_l3_hi', 'vanaken_l2_lo',
                'vanaken_l2_hi', 'vanaken_l3_centroid'])
            self.sample_file_paths = {}

        # Build GUI
        self._apply_theme()
        self.setup_gui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    # ------------------------------------------------------------------
    # Theme / styling
    # ------------------------------------------------------------------
    def _apply_theme(self):
        """Apply default ttk theme (consistent with CarbonKPeaks/SulfurKPeaks)."""
        pass

    def _set_icon(self):
        """Set the window icon (called early so it shows during load)."""
        if os.path.exists(self._icon_path):
            try:
                self.root.iconbitmap(self._icon_path)
            except tk.TclError:
                pass

    # ------------------------------------------------------------------
    # GUI setup
    # ------------------------------------------------------------------
    def setup_gui(self):
        """Build the main GUI layout."""
        self.root.title("IronLPeaks \u2014 Fe L-edge Peak Selector")

        # Responsive window sizing
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        ww = min(int(sw * 0.9), 1600)
        wh = min(int(sh * 0.9), 960)
        xp = (sw - ww) // 2
        yp = (sh - wh) // 2
        self.root.geometry(f"{ww}x{wh}+{xp}+{yp}")
        self.root.minsize(1000, 600)

        # Menu bar
        self._build_menu()

        # Main container
        outer = ttk.Frame(self.root)
        outer.pack(fill=tk.BOTH, expand=True)

        # Notebook (tabs)
        self.notebook = ttk.Notebook(outer)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=(4, 0))

        # Peak Selection tab
        peak_tab = ttk.Frame(self.notebook)
        self.notebook.add(peak_tab, text="Peak Selection")
        self._build_peak_selection_tab(peak_tab)

        # Baseline tab
        self._build_baseline_tab()

        # Fe Quantification tab (van Aken)
        self._build_vanaken_tab()

        # Visualization tab (includes peak analysis)
        self._build_visualization_tab()

        # Status bar at bottom
        self._status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(outer, textvariable=self._status_var,
                               relief=tk.SUNKEN, anchor=tk.W, padding=(6, 2))
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        # Keyboard shortcuts
        self.root.bind('<Left>', lambda e: None if isinstance(e.widget, (tk.Entry, ttk.Entry, ttk.Combobox)) else self.prev_sample())
        self.root.bind('<Right>', lambda e: None if isinstance(e.widget, (tk.Entry, ttk.Entry, ttk.Combobox)) else self.next_sample())
        self.root.bind('<Control-o>', lambda e: self.open_spectrum_files())
        self.root.bind('<Control-O>', lambda e: self.open_spectrum_files())
        self.root.bind('<Control-s>', lambda e: self.save_session_csv())
        self.root.bind('<Control-S>', lambda e: self.save_session_csv())
        self.root.bind('<Control-Shift-s>', lambda e: self.export_peak_data())
        self.root.bind('<Control-Shift-S>', lambda e: self.export_peak_data())

        # Load first sample
        if hasattr(self, 'df') and self.df is not None and len(self.df) > 0:
            self.load_current_sample()

    def _build_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Add Files\u2026",
                              accelerator="Ctrl+O",
                              command=self.open_spectrum_files)
        file_menu.add_command(label="Add Folder\u2026",
                              command=self.open_spectrum_folder)
        file_menu.add_command(label="Open Athena Project\u2026",
                              command=self.open_athena_project)
        file_menu.add_separator()
        file_menu.add_command(label="Remove Selected\u2026",
                              command=self.remove_selected_files)
        file_menu.add_command(label="Clear All Samples",
                              command=self.new_session)
        file_menu.add_separator()
        file_menu.add_command(label="Save Session\u2026",
                              accelerator="Ctrl+S",
                              command=self.save_session_csv)
        file_menu.add_command(label="Load Session\u2026",
                              command=self.load_session_csv)
        file_menu.add_separator()
        file_menu.add_command(label="Export All Data\u2026",
                              command=self.export_all_data)
        file_menu.add_command(label="Export Peak Data\u2026",
                              accelerator="Ctrl+Shift+S",
                              command=self.export_peak_data)
        file_menu.add_separator()
        file_menu.add_command(label="Save & Exit", command=self.save_and_exit)
        file_menu.add_command(label="Exit", command=self.root.quit)

    # ------------------------------------------------------------------
    # Peak Selection Tab
    # ------------------------------------------------------------------
    def _build_peak_selection_tab(self, parent):
        """Build the peak-selection tab with sidebar + plot area."""
        paned = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        self.main_paned = paned

        # LEFT: sidebar
        sidebar_outer = ttk.Frame(paned, width=320)
        paned.add(sidebar_outer, weight=0)
        self._build_sidebar(sidebar_outer)

        # RIGHT: toolbar + plots
        right = ttk.Frame(paned)
        paned.add(right, weight=1)
        self._build_toolbar(right)
        self._build_plot_area(right)

    # ---- Sidebar ----
    def _build_sidebar(self, container):
        canvas = tk.Canvas(container, width=300, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        sidebar = ttk.Frame(canvas)

        sidebar.bind("<Configure>",
                     lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        self._sidebar_canvas_win = canvas.create_window((0, 0), window=sidebar, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Resize the inner sidebar frame to match the canvas width when the panel is resized
        def _on_canvas_resize(event):
            canvas.itemconfigure(self._sidebar_canvas_win, width=event.width)
        canvas.bind('<Configure>', _on_canvas_resize)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Bug 7 fix: bind mousewheel only when cursor is over the sidebar canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bind_wheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_wheel(event):
            canvas.unbind_all("<MouseWheel>")

        canvas.bind("<Enter>", _bind_wheel)
        canvas.bind("<Leave>", _unbind_wheel)

        # -- File I/O section (prominent, at top) --
        file_frame = ttk.Frame(sidebar)
        file_frame.pack(fill=tk.X, padx=4, pady=(6, 2))
        ttk.Button(file_frame, text="Add Files\u2026",
                   command=self.open_spectrum_files).pack(
                       side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(file_frame, text="Add Folder\u2026",
                   command=self.open_spectrum_folder).pack(
                       side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(file_frame, text="Remove",
                   command=self.remove_selected_files).pack(
                       side=tk.LEFT, fill=tk.X, expand=True)

        # -- Samples section --
        self._section_header(sidebar, "Samples")

        listbox_frame = ttk.Frame(sidebar)
        listbox_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        self.sample_listbox = tk.Listbox(listbox_frame, height=15,
                                         selectmode=tk.EXTENDED,
                                         exportselection=False,
                                         font=('TkDefaultFont', 9),
                                         activestyle='none',
                                         relief=tk.FLAT, bd=1,
                                         highlightthickness=1,
                                         highlightcolor=_ACCENT)
        lb_scroll = ttk.Scrollbar(listbox_frame, orient='vertical',
                                  command=self.sample_listbox.yview)
        lb_xscroll = ttk.Scrollbar(listbox_frame, orient='horizontal',
                                   command=self.sample_listbox.xview)
        self.sample_listbox.configure(yscrollcommand=lb_scroll.set,
                                      xscrollcommand=lb_xscroll.set)
        lb_xscroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.sample_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        lb_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.sample_listbox.bind('<<ListboxSelect>>', self.on_sample_select)

        ttk.Label(sidebar, text="Ctrl+Click / Shift+Click for multi-select",
                  font=('TkDefaultFont', 7), foreground='gray').pack(pady=(0, 4))

        nav_frame = ttk.Frame(sidebar)
        nav_frame.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(nav_frame, text="\u25C0 Previous",
                   command=self.prev_sample).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(nav_frame, text="Next \u25B6",
                   command=self.next_sample).pack(side=tk.LEFT)

        self.sample_info_label = ttk.Label(sidebar, text="Sample: 0/0 | Selected: 0",
                                           font=('TkDefaultFont', 9))
        self.sample_info_label.pack(pady=(0, 6))

        self.include_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(sidebar, text="Include in report",
                        variable=self.include_var,
                        command=self.on_include_change).pack(pady=(0, 10))

        # -- Detection Window --
        self._section_header(sidebar, "Detection Window (eV)")

        win_frame = ttk.Frame(sidebar)
        win_frame.pack(fill=tk.X, pady=(0, 8), padx=4)
        ttk.Label(win_frame, text="Start:").grid(row=0, column=0, sticky='w')
        self.window_start_var = tk.DoubleVar(value=self.window_start)
        e1 = ttk.Entry(win_frame, textvariable=self.window_start_var, width=8)
        e1.grid(row=0, column=1, padx=(4, 0))
        e1.bind('<Return>', self.on_window_change)

        ttk.Label(win_frame, text="End:").grid(row=1, column=0, sticky='w')
        self.window_end_var = tk.DoubleVar(value=self.window_end)
        e2 = ttk.Entry(win_frame, textvariable=self.window_end_var, width=8)
        e2.grid(row=1, column=1, padx=(4, 0))
        e2.bind('<Return>', self.on_window_change)

        # -- Smoothing --
        self._section_header(sidebar, "Peak Detection & Smoothing")

        sm_frame = ttk.Frame(sidebar)
        sm_frame.pack(fill=tk.X, pady=(0, 8), padx=4)

        ttk.Label(sm_frame, text="Smoothing \u03C3:").grid(row=0, column=0, sticky='w')
        self.smoothing_var = tk.DoubleVar(value=0.0)
        se = ttk.Entry(sm_frame, textvariable=self.smoothing_var, width=8)
        se.grid(row=0, column=1, padx=(4, 0))
        se.bind('<Return>', self.apply_smoothing)
        se.bind('<FocusOut>', self.apply_smoothing)
        ttk.Button(sm_frame, text="Smooth",
                   command=self.apply_smoothing).grid(row=0, column=2, padx=(4, 0))

        ttk.Label(sm_frame, text="Sensitivity (0\u20131):").grid(row=1, column=0, sticky='w')
        self.min_peak_height_var = tk.DoubleVar(value=0.1)
        he = ttk.Entry(sm_frame, textvariable=self.min_peak_height_var, width=8)
        he.grid(row=1, column=1, padx=(4, 0))
        he.bind('<Return>', self.on_smoothing_change)
        he.bind('<FocusOut>', self.on_smoothing_change)

        self.show_smoothed_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(sm_frame, text="Show Smoothed Overlay",
                        variable=self.show_smoothed_var,
                        command=self.on_show_smoothed_toggle).grid(
                            row=2, column=0, columnspan=3, sticky='w', pady=(4, 0))

        ttk.Button(sm_frame, text="Reset Smoothing",
                   command=self.reset_to_original).grid(
                       row=3, column=0, columnspan=3, pady=(4, 0), sticky='ew')
        ttk.Button(sm_frame, text="Export Spectrum Data\u2026",
                   command=self.save_smoothed_spectrum).grid(
                       row=4, column=0, columnspan=3, pady=(4, 0), sticky='ew')

        # -- Sample Group --
        self._section_header(sidebar, "Sample Group")

        cls_frame = ttk.Frame(sidebar)
        cls_frame.pack(fill=tk.X, pady=(0, 8), padx=4)

        ttk.Label(cls_frame,
                  text="Type a new group or pick an existing one:",
                  foreground='gray', font=('TkDefaultFont', 8)
                  ).pack(anchor='w')
        self.sample_group_var = tk.StringVar(value="")
        self.sample_group_combo = ttk.Combobox(
            cls_frame, textvariable=self.sample_group_var,
            values=[], width=22)
        self.sample_group_combo.pack(fill=tk.X, pady=(2, 0))
        self.sample_group_combo.bind('<Return>', self.on_classification_change)
        self.sample_group_combo.bind(
            '<<ComboboxSelected>>', self.on_classification_change)

        batch_frame = ttk.Frame(cls_frame)
        batch_frame.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(batch_frame, text="Apply to Selected",
                   command=self.apply_classification_to_selected
                   ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(batch_frame, text="Clear",
                   command=self.clear_classification
                   ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        ttk.Button(cls_frame, text="Rename Selected\u2026",
                   command=self.rename_selected_samples
                   ).pack(fill=tk.X, pady=(4, 0))

        # -- Peak Analysis Info --
        self._section_header(sidebar, "Peak Analysis")

        self.peak_analysis_label = ttk.Label(sidebar,
                                             text="Select 2 peaks to see analysis",
                                             foreground="gray", wraplength=260)
        self.peak_analysis_label.pack(anchor='w', padx=4, pady=(0, 8))

        # -- Peak Actions --
        self._section_header(sidebar, "Peak Actions")

        pk_btn_frame = ttk.Frame(sidebar)
        pk_btn_frame.pack(fill=tk.X, padx=4, pady=(0, 4))
        ttk.Button(pk_btn_frame, text="Detect Peaks",
                   command=self.detect_peaks_selected).pack(
                       side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(pk_btn_frame, text="Clear Peaks",
                   command=self.clear_peaks_selected).pack(
                       side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
        ttk.Button(sidebar, text="Auto-Select Peaks (Selected)",
                   command=self.auto_select_peaks_selected).pack(fill=tk.X, padx=4, pady=(0, 8))

        # -- Export section --
        self._section_header(sidebar, "Export")

        exp_frame = ttk.Frame(sidebar)
        exp_frame.pack(fill=tk.X, padx=4, pady=(0, 4))
        ttk.Button(exp_frame, text="Export All (Ctrl+S)",
                   command=self.export_all_data).pack(
                       side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(exp_frame, text="Export Peaks",
                   command=self.export_peak_data).pack(
                       side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
        ttk.Button(sidebar, text="Save & Exit",
                   command=self.save_and_exit).pack(fill=tk.X, padx=4, pady=(4, 4))

    def _section_header(self, parent, text):
        """Draw a subtle section header with a separator line."""
        sep = ttk.Separator(parent, orient=tk.HORIZONTAL)
        sep.pack(fill=tk.X, pady=(8, 2), padx=4)
        lbl = ttk.Label(parent, text=text,
                        font=('TkDefaultFont', 9, 'bold'),
                        foreground=_SECTION_FG)
        lbl.pack(anchor='w', padx=4, pady=(0, 4))

    # ---- Toolbar (baseline controls) ----
    def _build_toolbar(self, parent):
        """Horizontal toolbar strip above the plots for baseline controls."""
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill=tk.X, padx=2, pady=(2, 0))

        ttk.Button(toolbar, text="Fit Baseline",
                   command=self.fit_baseline).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(toolbar, text="Re-fit",
                   command=self._refit_baseline).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(toolbar, text="Reset",
                   command=self.reset_baseline_params).pack(side=tk.LEFT, padx=(0, 8))

        self.subtract_baseline_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(toolbar, text="Subtract from Plots",
                        variable=self.subtract_baseline_var,
                        command=self.on_subtract_baseline_toggle).pack(side=tk.LEFT, padx=(0, 8))

        self.baseline_status_label = ttk.Label(toolbar, text="No baseline",
                                               foreground="gray")
        self.baseline_status_label.pack(side=tk.LEFT, padx=(0, 10))

        # Advanced toggle
        self.show_baseline_params = tk.BooleanVar(value=False)
        ttk.Checkbutton(toolbar, text="Show Advanced \u25BC",
                        variable=self.show_baseline_params,
                        command=self.toggle_baseline_params).pack(side=tk.RIGHT)

        # Advanced params frame (initially hidden)
        self.advanced_baseline_frame = ttk.Frame(parent)

        pf = ttk.Frame(self.advanced_baseline_frame)
        pf.pack(fill=tk.X, pady=(4, 0), padx=4)

        ttk.Label(pf, text="Penalty:").grid(row=0, column=0, sticky='w')
        self.penalty_weight_var = tk.DoubleVar(value=200.0)
        pe = ttk.Entry(pf, textvariable=self.penalty_weight_var, width=8)
        pe.grid(row=0, column=1, padx=2)
        pe.bind('<Return>', self.on_parameter_change)
        pe.bind('<FocusOut>', self.on_parameter_change)

        ttk.Label(pf, text="Max Refit:").grid(row=0, column=2, sticky='w', padx=(8, 0))
        self.max_refit_var = tk.IntVar(value=10)
        me = ttk.Entry(pf, textvariable=self.max_refit_var, width=6)
        me.grid(row=0, column=3, padx=2)
        me.bind('<Return>', self.on_parameter_change)
        me.bind('<FocusOut>', self.on_parameter_change)

        self.iterative_refit_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(pf, text="Iterative Refit",
                        variable=self.iterative_refit_var,
                        command=self.on_parameter_change).grid(row=0, column=4, padx=(8, 0))

        ttk.Label(pf, text="Arctan Parameters:",
                  font=('TkDefaultFont', 8, 'bold')).grid(
                      row=1, column=0, columnspan=5, sticky='w', pady=(6, 2))

        # Arctan param entries — mark user edits via _on_arctan_field_edit
        ttk.Label(pf, text="Offset:").grid(row=2, column=0, sticky='e')
        self.offset_var = tk.StringVar(value="auto")
        oe = ttk.Entry(pf, textvariable=self.offset_var, width=9)
        oe.grid(row=2, column=1, padx=2)
        oe.bind('<Key>', self._on_arctan_field_edit)

        ttk.Label(pf, text="Slope:").grid(row=2, column=2, sticky='e', padx=(6, 0))
        self.slope_var = tk.StringVar(value="auto")
        sle = ttk.Entry(pf, textvariable=self.slope_var, width=9)
        sle.grid(row=2, column=3, padx=2)
        sle.bind('<Key>', self._on_arctan_field_edit)

        ttk.Label(pf, text="Step:").grid(row=2, column=4, sticky='e', padx=(6, 0))
        self.step_height_var = tk.StringVar(value="auto")
        ste = ttk.Entry(pf, textvariable=self.step_height_var, width=9)
        ste.grid(row=2, column=5, padx=2)
        ste.bind('<Key>', self._on_arctan_field_edit)

        ttk.Label(pf, text="Width:").grid(row=3, column=0, sticky='e')
        self.edge_width_var = tk.StringVar(value="auto")
        we = ttk.Entry(pf, textvariable=self.edge_width_var, width=9)
        we.grid(row=3, column=1, padx=2)
        we.bind('<Key>', self._on_arctan_field_edit)

        ttk.Label(pf, text="L3 Edge:").grid(row=3, column=2, sticky='e', padx=(6, 0))
        self.edge_position_var = tk.StringVar(value="auto")
        epe = ttk.Entry(pf, textvariable=self.edge_position_var, width=9)
        epe.grid(row=3, column=3, padx=2)
        epe.bind('<Key>', self._on_arctan_field_edit)

        ttk.Label(pf, text="L2 Edge:").grid(row=3, column=4, sticky='e', padx=(6, 0))
        self.edge_l2_position_var = tk.StringVar(value="auto")
        el2e = ttk.Entry(pf, textvariable=self.edge_l2_position_var, width=9)
        el2e.grid(row=3, column=5, padx=2)
        el2e.bind('<Key>', self._on_arctan_field_edit)

        ttk.Label(pf, text="Region (eV):").grid(row=4, column=0, sticky='w', pady=(6, 0))
        rf = ttk.Frame(pf)
        rf.grid(row=4, column=1, columnspan=5, sticky='w', pady=(6, 0))
        self.baseline_start_var = tk.StringVar(value="")
        ttk.Entry(rf, textvariable=self.baseline_start_var, width=8).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Label(rf, text="to").pack(side=tk.LEFT)
        self.baseline_end_var = tk.StringVar(value="")
        ttk.Entry(rf, textvariable=self.baseline_end_var, width=8).pack(side=tk.LEFT, padx=(4, 4))
        ttk.Label(rf, text="(blank=full)", font=('TkDefaultFont', 7)).pack(side=tk.LEFT)

    def _on_arctan_field_edit(self, event=None):
        """Mark that the user has manually edited an arctan param field."""
        self._baseline_params_are_manual = True

    # ---- Plot area ----
    def _build_plot_area(self, parent):
        """Create the three-subplot figure."""
        plot_frame = ttk.Frame(parent)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        self.fig = Figure(figsize=(14, 12), constrained_layout=True)
        gs = self.fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.7])
        self.ax_full = self.fig.add_subplot(gs[0, :])    # full width
        self.ax_l3 = self.fig.add_subplot(gs[1, 0])       # left — peak clicking
        self.ax_l2 = self.fig.add_subplot(gs[1, 1])       # right — baseline check
        self.ax_deriv = self.fig.add_subplot(gs[2, :])    # full width

        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()

        # Mouse events for peak selection
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_motion)

        # Populate sample list
        self.populate_sample_list()

        # Show empty-state welcome message if no data loaded
        if self.df is None or len(self.df) == 0:
            self._show_empty_state()

    # ------------------------------------------------------------------
    # Analysis Tab
    # ------------------------------------------------------------------
    def _build_analysis_tab(self):
        analysis_tab = ttk.Frame(self.notebook)
        self.notebook.add(analysis_tab, text="Analysis")

        table_frame = ttk.LabelFrame(analysis_tab, text="Peak Analysis Data", padding=5)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(5, 2))

        columns = ('Sample', 'Group',
                   'Peak1 (eV)', 'Peak2 (eV)', '\u0394 eV', 'Intensity Ratio')
        self.peak_tree = ttk.Treeview(table_frame, columns=columns,
                                      show='headings', height=10)
        for col in columns:
            self.peak_tree.heading(col, text=col)
            self.peak_tree.column(col, width=100)

        tree_scroll = ttk.Scrollbar(table_frame, orient='vertical',
                                    command=self.peak_tree.yview)
        self.peak_tree.configure(yscrollcommand=tree_scroll.set)
        self.peak_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        plot_frame = ttk.LabelFrame(analysis_tab, text="Peak Analysis Scatter Plot",
                                    padding=5)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(2, 5))

        self.analysis_fig = Figure(figsize=(10, 6), constrained_layout=True)
        self.analysis_ax = self.analysis_fig.add_subplot(111)
        self.analysis_canvas = FigureCanvasTkAgg(self.analysis_fig, plot_frame)
        self.analysis_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.analysis_annot = self.analysis_ax.annotate(
            "", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9, edgecolor='gray'),
            arrowprops=dict(arrowstyle="->"))
        self.analysis_annot.set_visible(False)
        self.analysis_canvas.mpl_connect("motion_notify_event", self.on_analysis_hover)

        btn_frame = ttk.Frame(analysis_tab)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(btn_frame, text="Export Peak Data\u2026",
                   command=self.export_peak_data).pack(side=tk.RIGHT)

    # ------------------------------------------------------------------
    # Baseline Tab
    # ------------------------------------------------------------------
    def _build_baseline_tab(self):
        """Build a dedicated baseline inspection / adjustment tab."""
        bl_tab = ttk.Frame(self.notebook)
        self.notebook.add(bl_tab, text="Baseline")

        # PanedWindow: left sidebar + right plot area
        paned = ttk.PanedWindow(bl_tab, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # ── LEFT SIDEBAR ──
        sidebar_outer = ttk.Frame(paned, width=300)
        paned.add(sidebar_outer, weight=0)

        bl_canvas = tk.Canvas(sidebar_outer, width=280, highlightthickness=0)
        bl_scroll = ttk.Scrollbar(sidebar_outer, orient="vertical",
                                  command=bl_canvas.yview)
        sidebar = ttk.Frame(bl_canvas)

        sidebar.bind("<Configure>",
                     lambda e: bl_canvas.configure(
                         scrollregion=bl_canvas.bbox("all")))
        self._bl_sidebar_win = bl_canvas.create_window(
            (0, 0), window=sidebar, anchor="nw")
        bl_canvas.configure(yscrollcommand=bl_scroll.set)

        def _bl_canvas_resize(event):
            bl_canvas.itemconfigure(self._bl_sidebar_win, width=event.width)
        bl_canvas.bind('<Configure>', _bl_canvas_resize)

        bl_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        bl_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Mousewheel scoping
        def _bl_mousewheel(event):
            bl_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        def _bl_bind_wheel(event):
            bl_canvas.bind_all("<MouseWheel>", _bl_mousewheel)
        def _bl_unbind_wheel(event):
            bl_canvas.unbind_all("<MouseWheel>")
        bl_canvas.bind("<Enter>", _bl_bind_wheel)
        bl_canvas.bind("<Leave>", _bl_unbind_wheel)

        # -- Navigation --
        self._section_header(sidebar, "Navigation")

        bl_nav = ttk.Frame(sidebar)
        bl_nav.pack(fill=tk.X, padx=4, pady=(0, 4))
        ttk.Button(bl_nav, text="\u25c0 Previous",
                   command=self.prev_sample).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(bl_nav, text="Next \u25b6",
                   command=self.next_sample).pack(side=tk.LEFT)

        self.baseline_sample_label = ttk.Label(
            sidebar, text="Sample: 0/0",
            font=('TkDefaultFont', 9))
        self.baseline_sample_label.pack(pady=(0, 6))

        # -- Baseline Parameters (editable, shared StringVars) --
        self._section_header(sidebar, "Arctan Parameters")

        bp = ttk.Frame(sidebar)
        bp.pack(fill=tk.X, padx=4, pady=(0, 6))

        for i, (name, var_name) in enumerate([
            ('Offset', 'offset_var'), ('Slope', 'slope_var'),
            ('Step', 'step_height_var'), ('Width', 'edge_width_var'),
            ('L3 Edge', 'edge_position_var'), ('L2 Edge', 'edge_l2_position_var'),
        ]):
            ttk.Label(bp, text=f"{name}:", width=9, anchor='e',
                      font=('TkDefaultFont', 8, 'bold')).grid(
                          row=i, column=0, sticky='e', pady=1)
            var = getattr(self, var_name)
            entry = ttk.Entry(bp, textvariable=var, width=12)
            entry.grid(row=i, column=1, sticky='w', padx=(4, 0), pady=1)
            entry.bind('<Key>', self._on_arctan_field_edit)

        # -- Fitting Options (editable) --
        self._section_header(sidebar, "Fitting Options")

        opt_frame = ttk.Frame(sidebar)
        opt_frame.pack(fill=tk.X, padx=4, pady=(0, 6))

        ttk.Label(opt_frame, text="Penalty:", width=9, anchor='e',
                  font=('TkDefaultFont', 8, 'bold')).grid(row=0, column=0, sticky='e')
        ttk.Entry(opt_frame, textvariable=self.penalty_weight_var,
                  width=10).grid(row=0, column=1, sticky='w', padx=(4, 0))

        ttk.Label(opt_frame, text="Max Refit:", width=9, anchor='e',
                  font=('TkDefaultFont', 8, 'bold')).grid(row=1, column=0, sticky='e')
        ttk.Entry(opt_frame, textvariable=self.max_refit_var,
                  width=10).grid(row=1, column=1, sticky='w', padx=(4, 0))

        ttk.Checkbutton(opt_frame, text="Iterative Refit",
                        variable=self.iterative_refit_var).grid(
                            row=2, column=0, columnspan=2, sticky='w', pady=(4, 0))

        # -- Baseline Region --
        self._section_header(sidebar, "Baseline Region (eV)")

        reg_frame = ttk.Frame(sidebar)
        reg_frame.pack(fill=tk.X, padx=4, pady=(0, 6))

        ttk.Label(reg_frame, text="Start:").grid(row=0, column=0, sticky='e')
        ttk.Entry(reg_frame, textvariable=self.baseline_start_var,
                  width=10).grid(row=0, column=1, padx=(4, 0))
        ttk.Label(reg_frame, text="End:").grid(row=1, column=0, sticky='e')
        ttk.Entry(reg_frame, textvariable=self.baseline_end_var,
                  width=10).grid(row=1, column=1, padx=(4, 0))
        ttk.Label(reg_frame, text="(blank = full range)",
                  font=('TkDefaultFont', 7), foreground='gray').grid(
                      row=2, column=0, columnspan=2, sticky='w')

        # -- Baseline Status --
        self._section_header(sidebar, "Baseline Status")

        self._bl_status_lbl = ttk.Label(sidebar, text="No baseline fitted",
                                        foreground='gray',
                                        font=('TkDefaultFont', 9))
        self._bl_status_lbl.pack(anchor='w', padx=4, pady=(0, 6))

        # -- Action Buttons --
        self._section_header(sidebar, "Actions")

        ttk.Button(sidebar, text="Auto Fit Baseline",
                   command=self._bl_auto_fit).pack(
                       fill=tk.X, padx=4, pady=(0, 3))
        ttk.Button(sidebar, text="Re-fit (Reset & Fit)",
                   command=self._refit_baseline).pack(
                       fill=tk.X, padx=4, pady=(0, 3))
        ttk.Button(sidebar, text="Fit & Next \u25b6",
                   command=self._bl_fit_and_next).pack(
                       fill=tk.X, padx=4, pady=(0, 3))
        ttk.Button(sidebar, text="Reset Baseline",
                   command=self.reset_baseline_params).pack(
                       fill=tk.X, padx=4, pady=(0, 3))

        # ── RIGHT PANEL: dual subplot figure ──
        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        plot_frame = ttk.Frame(right)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.bl_fig = Figure(figsize=(10, 8), constrained_layout=True)
        bl_gs = self.bl_fig.add_gridspec(2, 1, height_ratios=[1, 1])
        self.bl_ax_top = self.bl_fig.add_subplot(bl_gs[0])
        self.bl_ax_bot = self.bl_fig.add_subplot(bl_gs[1])

        self.bl_canvas = FigureCanvasTkAgg(self.bl_fig, plot_frame)
        self.bl_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        bl_toolbar = NavigationToolbar2Tk(self.bl_canvas, plot_frame)
        bl_toolbar.update()

    def _bl_auto_fit(self):
        """Fit baseline on current sample and update the Baseline tab."""
        self.fit_baseline()
        self._redraw_baseline_tab()

    def _bl_fit_and_next(self):
        """Fit baseline, then advance to the next sample."""
        self.fit_baseline()
        self.next_sample()

    def _redraw_baseline_tab(self):
        """Redraw the Baseline tab's dual subplot display."""
        if not hasattr(self, 'bl_ax_top'):
            return
        self.bl_ax_top.clear()
        self.bl_ax_bot.clear()

        if self.current_spectrum_data is None:
            self.bl_ax_top.text(0.5, 0.5, "No spectrum loaded",
                               ha='center', va='center',
                               transform=self.bl_ax_top.transAxes,
                               fontsize=12, color='gray')
            self.bl_canvas.draw()
            return

        act_energy, act_intensity = self._get_active_spectrum()
        raw_energy = self.current_spectrum_data['energy']
        raw_intensity = self.current_spectrum_data['intensity']
        is_smoothed = self.current_smoothed_data is not None

        sample_name = ""
        if self.df is not None and len(self.df) > 0:
            sample_name = self.df.iloc[self.current_sample_index].get(
                'sample_name', f'Sample_{self.current_sample_index}')

        # ── TOP: Spectrum + Baseline Overlay ──
        self.bl_ax_top.plot(act_energy, act_intensity,
                            color='#1565c0' if is_smoothed else 'black',
                            linewidth=1.2,
                            label='Smoothed' if is_smoothed else 'Spectrum')
        if is_smoothed:
            self.bl_ax_top.plot(raw_energy, raw_intensity,
                                color='black', linewidth=0.6, alpha=0.3,
                                label='Raw')

        bl_status = "No baseline fitted"
        if self.current_baseline is not None:
            self.bl_ax_top.plot(act_energy, self.current_baseline,
                                color='#d32f2f', linestyle='--', linewidth=1.5,
                                label='Baseline')
            bl_status = "Baseline fitted"

        self.bl_ax_top.set_xlabel('Energy (eV)')
        self.bl_ax_top.set_ylabel('Intensity (a.u.)')
        self.bl_ax_top.set_title(f'Spectrum + Baseline \u2014 {sample_name}')
        self.bl_ax_top.legend(loc='upper right', fontsize=8, framealpha=0.7)
        self.bl_ax_top.grid(True, alpha=0.15, which='major')

        # ── BOTTOM: Corrected Spectrum with negative region shading ──
        if self.current_baseline is not None:
            corrected = act_intensity - self.current_baseline
            self.bl_ax_bot.plot(act_energy, corrected, color='#2e7d32',
                                linewidth=1.2, label='Corrected')
            self.bl_ax_bot.axhline(0, color='gray', linestyle='--',
                                   linewidth=0.8, alpha=0.6)

            # Shade negative regions
            neg_mask = corrected < 0
            if np.any(neg_mask):
                self.bl_ax_bot.fill_between(
                    act_energy, corrected, 0,
                    where=neg_mask, alpha=0.25, color='#d32f2f',
                    label='Negative (overshoot)')
                bl_status += f" \u2014 {np.sum(neg_mask)} pts negative"

            # van Aken integration windows
            l3p_mask = (act_energy >= 708.5) & (act_energy <= 710.5)
            if np.any(l3p_mask):
                self.bl_ax_bot.fill_between(
                    act_energy[l3p_mask], 0, corrected[l3p_mask],
                    alpha=0.15, color='#e65100',
                    label="L3\u2032 (708.5\u2013710.5 eV)")
            l2p_mask = (act_energy >= 719.7) & (act_energy <= 721.7)
            if np.any(l2p_mask):
                self.bl_ax_bot.fill_between(
                    act_energy[l2p_mask], 0, corrected[l2p_mask],
                    alpha=0.15, color='#1565c0',
                    label="L2\u2032 (719.7\u2013721.7 eV)")

            self.bl_ax_bot.legend(loc='upper right', fontsize=7, framealpha=0.7)
        else:
            self.bl_ax_bot.text(0.5, 0.5,
                                "Fit baseline to see corrected spectrum",
                                ha='center', va='center',
                                transform=self.bl_ax_bot.transAxes,
                                fontsize=11, color='gray')

        self.bl_ax_bot.set_xlabel('Energy (eV)')
        self.bl_ax_bot.set_ylabel('Intensity (a.u.)')
        self.bl_ax_bot.set_title('Baseline-corrected Spectrum')
        self.bl_ax_bot.grid(True, alpha=0.15, which='major')

        # Update sidebar status
        self._bl_status_lbl.config(
            text=bl_status,
            foreground=_SUCCESS if self.current_baseline is not None else 'gray')

        # Update baseline sample label
        if hasattr(self, 'baseline_sample_label'):
            total = len(self.df) if self.df is not None else 0
            current_num = self.current_sample_index + 1
            self.baseline_sample_label.config(
                text=f"[{current_num}/{total}]  {sample_name}")

        self.bl_canvas.draw()

    # ------------------------------------------------------------------
    # van Aken Fe³⁺/ΣFe Tab
    # ------------------------------------------------------------------
    def _build_vanaken_tab(self):
        """Build the van Aken & Liebscher (2002) Fe3+/SFe quantification tab."""
        vanaken_tab = ttk.Frame(self.notebook)
        self.notebook.add(vanaken_tab, text="Fe Quantification")

        # PanedWindow: sidebar (left) + plot area (right)
        paned = ttk.PanedWindow(vanaken_tab, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # ── LEFT SIDEBAR ──
        sidebar_outer = ttk.Frame(paned, width=300)
        paned.add(sidebar_outer, weight=0)

        sidebar_canvas = tk.Canvas(sidebar_outer, width=280, highlightthickness=0)
        sidebar_scroll = ttk.Scrollbar(sidebar_outer, orient="vertical",
                                       command=sidebar_canvas.yview)
        sidebar = ttk.Frame(sidebar_canvas)

        sidebar.bind("<Configure>",
                     lambda e: sidebar_canvas.configure(
                         scrollregion=sidebar_canvas.bbox("all")))
        self._vanaken_sidebar_win = sidebar_canvas.create_window(
            (0, 0), window=sidebar, anchor="nw")
        sidebar_canvas.configure(yscrollcommand=sidebar_scroll.set)

        def _on_va_canvas_resize(event):
            sidebar_canvas.itemconfigure(self._vanaken_sidebar_win, width=event.width)
        sidebar_canvas.bind('<Configure>', _on_va_canvas_resize)

        sidebar_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        sidebar_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Mousewheel scoping
        def _va_mousewheel(event):
            sidebar_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        def _va_bind_wheel(event):
            sidebar_canvas.bind_all("<MouseWheel>", _va_mousewheel)
        def _va_unbind_wheel(event):
            sidebar_canvas.unbind_all("<MouseWheel>")
        sidebar_canvas.bind("<Enter>", _va_bind_wheel)
        sidebar_canvas.bind("<Leave>", _va_unbind_wheel)

        # -- Sample List --
        self._section_header(sidebar, "Samples")

        va_list_frame = ttk.Frame(sidebar)
        va_list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        self.vanaken_sample_listbox = tk.Listbox(
            va_list_frame, height=15, selectmode=tk.BROWSE,
            exportselection=False, font=('TkDefaultFont', 9),
            activestyle='none', relief=tk.FLAT, bd=1,
            highlightthickness=1, highlightcolor=_ACCENT)
        va_lb_scroll = ttk.Scrollbar(va_list_frame, orient='vertical',
                                     command=self.vanaken_sample_listbox.yview)
        va_lb_xscroll = ttk.Scrollbar(va_list_frame, orient='horizontal',
                                      command=self.vanaken_sample_listbox.xview)
        self.vanaken_sample_listbox.configure(
            yscrollcommand=va_lb_scroll.set,
            xscrollcommand=va_lb_xscroll.set)
        va_lb_xscroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.vanaken_sample_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        va_lb_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.vanaken_sample_listbox.bind('<<ListboxSelect>>',
                                         self._on_vanaken_sample_select)

        # -- Navigation --
        va_nav = ttk.Frame(sidebar)
        va_nav.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(va_nav, text="\u25c0 Previous",
                   command=self.prev_sample).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(va_nav, text="Next \u25b6",
                   command=self.next_sample).pack(side=tk.LEFT)

        self.vanaken_sample_label = ttk.Label(
            sidebar, text="Sample: 0/0",
            font=('TkDefaultFont', 9))
        self.vanaken_sample_label.pack(pady=(0, 6))

        # -- Sample Info Panel --
        self._section_header(sidebar, "Sample Info")

        info_frame = ttk.Frame(sidebar)
        info_frame.pack(fill=tk.X, padx=4, pady=(0, 6))

        self._va_info_labels = {}
        for field in ('Name', 'Baseline', 'Fe\u00b3\u207a/\u03a3Fe',
                      'L3 centroid', "I(L3')/I(L2')", 'p value'):
            row_f = ttk.Frame(info_frame)
            row_f.pack(fill=tk.X, pady=1)
            ttk.Label(row_f, text=f"{field}:", width=12,
                      anchor='e', font=('TkDefaultFont', 8, 'bold')).pack(
                          side=tk.LEFT, padx=(0, 4))
            val_lbl = ttk.Label(row_f, text="\u2014",
                                font=('TkDefaultFont', 8))
            val_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self._va_info_labels[field] = val_lbl

        # -- Include checkbox --
        self._section_header(sidebar, "Actions")

        ttk.Button(sidebar, text="Compute Fe\u00b3\u207a/\u03a3Fe",
                   command=self.compute_vanaken).pack(
                       fill=tk.X, padx=4, pady=(0, 3))
        ttk.Button(sidebar, text="Compute & Next \u25b6",
                   command=self.compute_vanaken_and_next).pack(
                       fill=tk.X, padx=4, pady=(0, 3))
        ttk.Button(sidebar, text="Batch All Selected",
                   command=self.batch_vanaken).pack(
                       fill=tk.X, padx=4, pady=(0, 3))
        ttk.Button(sidebar, text="Clear Result",
                   command=self.clear_vanaken_results).pack(
                       fill=tk.X, padx=4, pady=(0, 3))

        # ── RIGHT PANEL ──
        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        # -- Controls row (integration windows) --
        ctrl_frame = ttk.LabelFrame(right,
                                    text="van Aken & Liebscher (2002) Integration Windows",
                                    padding=5)
        ctrl_frame.pack(fill=tk.X, padx=5, pady=(5, 2))

        ttk.Label(ctrl_frame, text="L3' window:").pack(side=tk.LEFT, padx=(0, 2))
        self.vanaken_l3_lo_var = tk.DoubleVar(value=708.5)
        ttk.Entry(ctrl_frame, textvariable=self.vanaken_l3_lo_var,
                  width=7).pack(side=tk.LEFT, padx=(0, 1))
        ttk.Label(ctrl_frame, text="\u2013").pack(side=tk.LEFT)
        self.vanaken_l3_hi_var = tk.DoubleVar(value=710.5)
        ttk.Entry(ctrl_frame, textvariable=self.vanaken_l3_hi_var,
                  width=7).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Label(ctrl_frame, text="eV").pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(ctrl_frame, text="L2' window:").pack(side=tk.LEFT, padx=(0, 2))
        self.vanaken_l2_lo_var = tk.DoubleVar(value=719.7)
        ttk.Entry(ctrl_frame, textvariable=self.vanaken_l2_lo_var,
                  width=7).pack(side=tk.LEFT, padx=(0, 1))
        ttk.Label(ctrl_frame, text="\u2013").pack(side=tk.LEFT)
        self.vanaken_l2_hi_var = tk.DoubleVar(value=721.7)
        ttk.Entry(ctrl_frame, textvariable=self.vanaken_l2_hi_var,
                  width=7).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Label(ctrl_frame, text="eV").pack(side=tk.LEFT, padx=(0, 10))

        # -- Plot area --
        plot_frame = ttk.Frame(right)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(2, 2))

        self.vanaken_fig = Figure(figsize=(10, 5), constrained_layout=True)
        self.vanaken_ax = self.vanaken_fig.add_subplot(111)
        self.vanaken_canvas = FigureCanvasTkAgg(self.vanaken_fig, plot_frame)
        self.vanaken_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        vanaken_toolbar = NavigationToolbar2Tk(self.vanaken_canvas, plot_frame)
        vanaken_toolbar.update()

        # -- Results label --
        result_frame = ttk.LabelFrame(right, text="van Aken Results", padding=5)
        result_frame.pack(fill=tk.X, padx=5, pady=(2, 5))

        self.vanaken_result_label = ttk.Label(
            result_frame,
            text="No van Aken computation. Select a sample and click "
            "'Compute Fe\u00b3\u207a/\u03a3Fe' to begin.",
            foreground="gray", wraplength=800, justify=tk.LEFT)
        self.vanaken_result_label.pack(anchor='w', fill=tk.X)

    def _refresh_vanaken_sample_list(self):
        """Refresh the van Aken tab sample listbox with status indicators."""
        if not hasattr(self, 'vanaken_sample_listbox'):
            return
        lb = self.vanaken_sample_listbox

        saved_yview = lb.yview()[0]
        lb.delete(0, tk.END)
        if self.df is None:
            return

        for idx, row in self.df.iterrows():
            name = row.get('sample_name', row.get('Sample', f'Sample_{idx}'))
            fe3 = row.get('vanaken_fe3', np.nan)
            has_bl = not pd.isna(row.get('baseline_offset', np.nan))

            if pd.notna(fe3):
                status = f"[{fe3:.3f}]"
            elif has_bl:
                status = "[BL]"
            else:
                status = "[--]"

            lb.insert(tk.END, f"{status} {name}")

            # Color coding
            if pd.notna(fe3):
                lb.itemconfig(idx, fg=_SUCCESS)
            elif idx % 2 == 1:
                lb.itemconfig(idx, bg='#f5f5f5')

        # Restore selection
        if len(self.df) > 0 and self.current_sample_index < len(self.df):
            lb.selection_clear(0, tk.END)
            lb.selection_set(self.current_sample_index)
            lb.yview_moveto(saved_yview)
            lb.see(self.current_sample_index)

    def _on_vanaken_sample_select(self, event):
        """Handle sample selection in the van Aken listbox."""
        if self._updating_listbox:
            return
        sel = self.vanaken_sample_listbox.curselection()
        if sel:
            self.current_sample_index = sel[0]
            # Sync Peak Selection listbox
            self.sample_listbox.selection_clear(0, tk.END)
            self.sample_listbox.selection_set(self.current_sample_index)
            self.sample_listbox.see(self.current_sample_index)
            self.update_sample_info()
            self.load_current_sample()

    def _update_vanaken_info_panel(self):
        """Update the sample info labels in the van Aken sidebar."""
        if not hasattr(self, '_va_info_labels'):
            return

        labels = self._va_info_labels
        # Reset all
        for lbl in labels.values():
            lbl.config(text="\u2014")

        if self.df is None or len(self.df) == 0:
            return

        row = self.df.iloc[self.current_sample_index]
        name = row.get('sample_name', f'Sample_{self.current_sample_index}')
        labels['Name'].config(text=name)

        # Baseline status
        has_bl = not pd.isna(row.get('baseline_offset', np.nan))
        labels['Baseline'].config(
            text="Fitted" if has_bl else "Not fitted",
            foreground=_SUCCESS if has_bl else 'gray')

        # Van Aken results
        fe3 = row.get('vanaken_fe3', np.nan)
        if pd.notna(fe3):
            labels['Fe\u00b3\u207a/\u03a3Fe'].config(
                text=f"{fe3:.3f}", foreground=_SUCCESS)
        else:
            labels['Fe\u00b3\u207a/\u03a3Fe'].config(
                text="Not computed", foreground='gray')

        centroid = row.get('vanaken_l3_centroid', np.nan)
        if pd.notna(centroid):
            labels['L3 centroid'].config(text=f"{centroid:.2f} eV")

        ratio = row.get('vanaken_ratio', np.nan)
        if pd.notna(ratio):
            labels["I(L3')/I(L2')"].config(text=f"{ratio:.3f}")

        p_val = row.get('vanaken_p', np.nan)
        if pd.notna(p_val):
            labels['p value'].config(text=f"{p_val:.4f}")

    # ------------------------------------------------------------------
    # Visualization Tab
    # ------------------------------------------------------------------
    def _build_visualization_tab(self):
        """Build the Visualization tab with summary figure selector."""
        vis_tab = ttk.Frame(self.notebook)
        self.notebook.add(vis_tab, text="Visualization")

        # Controls row
        ctrl_frame = ttk.Frame(vis_tab, padding=5)
        ctrl_frame.pack(fill=tk.X, padx=5, pady=(5, 2))

        ttk.Label(ctrl_frame, text="Figure type:",
                  font=('TkDefaultFont', 9, 'bold')).pack(side=tk.LEFT, padx=(0, 4))

        self._vis_figure_types = [
            "Fe3+/SFe Box Plot by Group",
            "Fe3+/SFe Mean +/- Std Bar",
            "Fe Speciation Stacked Bar",
            "Intensity Ratio vs Peak Splitting",
            "L3 Centroid by Group",
            "All Corrected Spectra Overlay",
            "Combined Summary (4 panels)",
            "Summary Statistics Table",
            "Peak Analysis Scatter",
            "Peak Analysis Table",
        ]
        self._vis_type_var = tk.StringVar(value=self._vis_figure_types[0])
        vis_combo = ttk.Combobox(ctrl_frame, textvariable=self._vis_type_var,
                                 values=self._vis_figure_types, width=35,
                                 state='readonly')
        vis_combo.pack(side=tk.LEFT, padx=(0, 8))

        self._vis_report_only_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctrl_frame, text="Report samples only",
                        variable=self._vis_report_only_var).pack(
                            side=tk.LEFT, padx=(0, 8))

        ttk.Button(ctrl_frame, text="Generate",
                   command=self._vis_generate).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(ctrl_frame, text="Export PNG\u2026",
                   command=self._vis_export).pack(side=tk.LEFT, padx=(0, 4))

        # Plot area
        plot_frame = ttk.Frame(vis_tab)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(2, 5))

        self.vis_fig = Figure(figsize=(12, 8), constrained_layout=True)
        self.vis_ax = self.vis_fig.add_subplot(111)
        self.vis_canvas = FigureCanvasTkAgg(self.vis_fig, plot_frame)
        self.vis_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        vis_toolbar = NavigationToolbar2Tk(self.vis_canvas, plot_frame)
        vis_toolbar.update()

        # Initial message
        self.vis_ax.text(0.5, 0.5,
                         "Select a figure type and click Generate",
                         ha='center', va='center',
                         transform=self.vis_ax.transAxes,
                         fontsize=14, color='gray')
        self.vis_canvas.draw()

    def _vis_get_data(self):
        """Get DataFrame filtered by report-only flag, grouped by sample_group."""
        if self.df is None or len(self.df) == 0:
            return None
        df = self.df.copy()
        if self._vis_report_only_var.get():
            df = df[df.get('include_in_report', True) != False]
        if len(df) == 0:
            return None
        # Ensure sample_group column
        if 'sample_group' not in df.columns or df['sample_group'].isna().all():
            df['sample_group'] = 'All'
        df['sample_group'] = df['sample_group'].fillna('Ungrouped')
        return df

    def _vis_generate(self):
        """Generate the selected figure type."""
        fig_type = self._vis_type_var.get()
        df = self._vis_get_data()
        if df is None:
            messagebox.showwarning("No Data", "No data available for visualization.")
            return

        self.vis_fig.clear()

        try:
            if fig_type == self._vis_figure_types[0]:
                self._vis_boxplot(df)
            elif fig_type == self._vis_figure_types[1]:
                self._vis_mean_bar(df)
            elif fig_type == self._vis_figure_types[2]:
                self._vis_stacked_bar(df)
            elif fig_type == self._vis_figure_types[3]:
                self._vis_scatter_ratio(df)
            elif fig_type == self._vis_figure_types[4]:
                self._vis_centroid(df)
            elif fig_type == self._vis_figure_types[5]:
                self._vis_spectra_overlay(df)
            elif fig_type == self._vis_figure_types[6]:
                self._vis_combined(df)
            elif fig_type == self._vis_figure_types[7]:
                self._vis_table(df)
            elif fig_type == self._vis_figure_types[8]:
                self._vis_peak_scatter(df)
            elif fig_type == self._vis_figure_types[9]:
                self._vis_peak_table(df)
        except Exception as e:
            ax = self.vis_fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Error generating figure:\n{e}",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=11, color='red')

        self.vis_canvas.draw()
        self._set_status(f"Generated: {fig_type}")

    def _vis_export(self):
        """Export the current visualization as PNG."""
        fp = filedialog.asksaveasfilename(
            defaultextension='.png',
            filetypes=[('PNG', '*.png'), ('PDF', '*.pdf'), ('SVG', '*.svg')],
            initialdir=getattr(self, '_last_dir', None))
        if not fp:
            return
        self._last_dir = os.path.dirname(fp)
        self.vis_fig.savefig(fp, dpi=200, bbox_inches='tight')
        self._set_status(f"Exported: {os.path.basename(fp)}")

    def _vis_get_groups(self, df):
        """Return ordered groups and a color map."""
        groups = list(df['sample_group'].unique())
        cmap = plt.cm.tab10
        colors = {g: cmap(i / max(len(groups), 1)) for i, g in enumerate(groups)}
        return groups, colors

    def _vis_boxplot(self, df):
        """Box plot of Fe3+/SFe by group."""
        ax = self.vis_fig.add_subplot(111)
        groups, colors = self._vis_get_groups(df)
        grouped = df.groupby('sample_group')
        present = [g for g in groups if g in grouped.groups]
        box_data = [grouped.get_group(g)['vanaken_fe3'].dropna().values
                    for g in present]
        if not any(len(d) > 0 for d in box_data):
            ax.text(0.5, 0.5, "No Fe3+/SFe data to plot",
                    ha='center', va='center', transform=ax.transAxes, color='gray')
            return
        bp = ax.boxplot(box_data, tick_labels=present, patch_artist=True, widths=0.6)
        for patch, g in zip(bp['boxes'], present):
            patch.set_facecolor(colors[g])
            patch.set_alpha(0.7)
        ax.set_ylabel(r'Fe$^{3+}$/$\Sigma$Fe')
        ax.set_title(r'Fe$^{3+}$/$\Sigma$Fe Distribution by Group')
        ax.axhline(0.5, color='gray', ls='--', lw=0.8, alpha=0.5)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.15, axis='y')

    def _vis_mean_bar(self, df):
        """Mean +/- std bar chart."""
        ax = self.vis_fig.add_subplot(111)
        groups, colors = self._vis_get_groups(df)
        grouped = df.groupby('sample_group')
        present = [g for g in groups if g in grouped.groups]

        means = [np.nanmean(grouped.get_group(g)['vanaken_fe3'].values)
                 for g in present]
        stds = [np.nanstd(grouped.get_group(g)['vanaken_fe3'].values, ddof=1)
                for g in present]
        x = np.arange(len(present))
        ax.bar(x, means, yerr=stds, capsize=4,
               color=[colors[g] for g in present],
               edgecolor='black', linewidth=0.5, alpha=0.8)
        for i, g in enumerate(present):
            n = grouped.get_group(g)['vanaken_fe3'].notna().sum()
            ypos = min(means[i] + stds[i] + 0.02, 0.95)
            ax.text(i, ypos, f'n={n}', ha='center', va='bottom', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(present, fontsize=9)
        ax.set_ylabel(r'Fe$^{3+}$/$\Sigma$Fe')
        ax.set_title(r'Mean Fe$^{3+}$/$\Sigma$Fe $\pm$ Std Dev')
        ax.set_ylim(0, 1.15)
        ax.axhline(0.5, color='gray', ls='--', lw=0.8, alpha=0.5)
        ax.grid(True, alpha=0.15, axis='y')

    def _vis_stacked_bar(self, df):
        """Stacked bar: Fe2+ vs Fe3+ with individual data points."""
        ax = self.vis_fig.add_subplot(111)
        groups, colors = self._vis_get_groups(df)
        grouped = df.groupby('sample_group')
        present = [g for g in groups if g in grouped.groups]
        x = np.arange(len(present))

        means = [np.nanmean(grouped.get_group(g)['vanaken_fe3'].values)
                 for g in present]
        mean_fe3 = [m * 100 for m in means]
        mean_fe2 = [100 - v for v in mean_fe3]
        width = 0.6

        ax.bar(x, mean_fe2, width, label=r'Fe$^{2+}$ (mean %)',
               color='#4472C4', edgecolor='black', linewidth=0.5, alpha=0.7)
        ax.bar(x, mean_fe3, width, bottom=mean_fe2,
               label=r'Fe$^{3+}$ (mean %)',
               color='#ED7D31', edgecolor='black', linewidth=0.5, alpha=0.7)

        rng = np.random.default_rng(42)
        for i, g in enumerate(present):
            fe3_vals = grouped.get_group(g)['vanaken_fe3'].dropna().values * 100
            jitter = rng.uniform(-0.15, 0.15, size=len(fe3_vals))
            ax.scatter(i + jitter, fe3_vals, color='black', s=20, zorder=5,
                       edgecolors='white', linewidth=0.4, alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(present, fontsize=9)
        ax.set_ylabel('Percentage (%)')
        ax.set_ylim(0, 105)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_title('Fe Speciation by Group')
        ax.grid(True, alpha=0.15, axis='y')

    def _vis_scatter_ratio(self, df):
        """Intensity Ratio vs Peak Splitting scatter."""
        ax = self.vis_fig.add_subplot(111)
        groups, colors = self._vis_get_groups(df)
        grouped = df.groupby('sample_group')
        present = [g for g in groups if g in grouped.groups]
        for g in present:
            gd = grouped.get_group(g)
            ax.scatter(gd['intensity_ratio'], gd['delta_ev'],
                       color=colors[g], label=g, edgecolors='black',
                       linewidth=0.3, s=50, alpha=0.8)
        ax.set_xlabel('Intensity Ratio (Peak 1 / Peak 2)')
        ax.set_ylabel(r'$\Delta$E (Peak 2 $-$ Peak 1) (eV)')
        ax.set_title(r'L$_3$ Intensity Ratio vs Peak Splitting')
        ax.axvline(1.0, color='gray', ls='--', lw=0.8, alpha=0.5)
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.15)

    def _vis_centroid(self, df):
        """L3 centroid by group."""
        ax = self.vis_fig.add_subplot(111)
        groups, colors = self._vis_get_groups(df)
        grouped = df.groupby('sample_group')
        present = [g for g in groups if g in grouped.groups]
        box_data = [grouped.get_group(g)['vanaken_l3_centroid'].dropna().values
                    for g in present]
        if not any(len(d) > 0 for d in box_data):
            ax.text(0.5, 0.5, "No L3 centroid data",
                    ha='center', va='center', transform=ax.transAxes, color='gray')
            return
        bp = ax.boxplot(box_data, tick_labels=present, patch_artist=True, widths=0.6)
        for patch, g in zip(bp['boxes'], present):
            patch.set_facecolor(colors[g])
            patch.set_alpha(0.7)
        ax.set_ylabel('L3 Centroid Energy (eV)')
        ax.set_title('L3 Centroid Distribution by Group')
        ax.grid(True, alpha=0.15, axis='y')

    def _vis_spectra_overlay(self, df):
        """Overlay all corrected spectra colored by group."""
        ax = self.vis_fig.add_subplot(111)
        groups, colors = self._vis_get_groups(df)
        plotted = 0
        for _, row in df.iterrows():
            name = row.get('sample_name', '')
            group = row.get('sample_group', 'All')
            # Try to load and plot spectrum
            if hasattr(self, '_athena_spectra') and name in self._athena_spectra:
                spec = self._athena_spectra[name]
                energy, intensity = spec['energy'], spec['intensity']
                # Apply baseline correction if stored
                bl_offset = row.get('baseline_offset')
                if not pd.isna(bl_offset):
                    from arctan_baseline_Fe2p_v2 import ArctanBaselineFe2pV2
                    params = [row.get(c, 0) for c in
                              ('baseline_offset', 'baseline_slope', 'baseline_step',
                               'baseline_width', 'baseline_edge')]
                    bl_l2 = row.get('baseline_edge_l2', 719.5)
                    if pd.isna(bl_l2):
                        bl_l2 = 719.5
                    params.append(bl_l2)
                    baseline = ArctanBaselineFe2pV2.double_arctan_baseline(
                        energy, *params)
                    intensity = intensity - baseline
                ax.plot(energy, intensity, color=colors[group],
                        linewidth=0.8, alpha=0.6,
                        label=group if plotted == 0 or group not in [
                            ax.get_legend_handles_labels()[1]] else None)
                plotted += 1
        if plotted == 0:
            ax.text(0.5, 0.5, "No spectra available for overlay\n"
                    "(requires Athena import)",
                    ha='center', va='center', transform=ax.transAxes, color='gray')
        else:
            # De-duplicate legend
            handles, labels = ax.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            ax.legend(unique.values(), unique.keys(), fontsize=7)
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Intensity (a.u.)')
        ax.set_title('All Corrected Spectra')
        ax.grid(True, alpha=0.15)

    def _vis_combined(self, df):
        """4-panel combined summary."""
        import matplotlib.gridspec as gridspec
        gs = self.vis_fig.add_gridspec(2, 2, hspace=0.30, wspace=0.25)
        groups, colors = self._vis_get_groups(df)
        grouped = df.groupby('sample_group')
        present = [g for g in groups if g in grouped.groups]
        x = np.arange(len(present))

        # (0,0) Box plot
        ax = self.vis_fig.add_subplot(gs[0, 0])
        box_data = [grouped.get_group(g)['vanaken_fe3'].dropna().values
                    for g in present]
        bp = ax.boxplot(box_data, tick_labels=present, patch_artist=True, widths=0.6)
        for patch, g in zip(bp['boxes'], present):
            patch.set_facecolor(colors[g])
            patch.set_alpha(0.7)
        ax.set_ylabel(r'Fe$^{3+}$/$\Sigma$Fe')
        ax.set_title(r'(a) Fe$^{3+}$/$\Sigma$Fe by Group', fontsize=10)
        ax.axhline(0.5, color='gray', ls='--', lw=0.8, alpha=0.5)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis='x', labelsize=8)
        ax.grid(True, alpha=0.15, axis='y')

        # (0,1) Mean +/- Std
        ax = self.vis_fig.add_subplot(gs[0, 1])
        means = [np.nanmean(grouped.get_group(g)['vanaken_fe3'].values)
                 for g in present]
        stds = [np.nanstd(grouped.get_group(g)['vanaken_fe3'].values, ddof=1)
                for g in present]
        ax.bar(x, means, yerr=stds, capsize=4,
               color=[colors[g] for g in present],
               edgecolor='black', linewidth=0.5, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(present, fontsize=8)
        ax.set_ylabel(r'Fe$^{3+}$/$\Sigma$Fe')
        ax.set_title(r'(b) Mean $\pm$ Std Dev', fontsize=10)
        ax.set_ylim(0, 1.15)
        ax.axhline(0.5, color='gray', ls='--', lw=0.8, alpha=0.5)
        ax.grid(True, alpha=0.15, axis='y')

        # (1,0) Stacked bar
        ax = self.vis_fig.add_subplot(gs[1, 0])
        mean_fe3 = [m * 100 for m in means]
        mean_fe2 = [100 - v for v in mean_fe3]
        ax.bar(x, mean_fe2, 0.6, label=r'Fe$^{2+}$',
               color='#4472C4', edgecolor='black', linewidth=0.5, alpha=0.7)
        ax.bar(x, mean_fe3, 0.6, bottom=mean_fe2, label=r'Fe$^{3+}$',
               color='#ED7D31', edgecolor='black', linewidth=0.5, alpha=0.7)
        rng = np.random.default_rng(42)
        for i, g in enumerate(present):
            vals = grouped.get_group(g)['vanaken_fe3'].dropna().values * 100
            jitter = rng.uniform(-0.15, 0.15, size=len(vals))
            ax.scatter(i + jitter, vals, color='black', s=15, zorder=5,
                       edgecolors='white', linewidth=0.3, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(present, fontsize=8)
        ax.set_ylabel('Percentage (%)')
        ax.set_ylim(0, 105)
        ax.legend(fontsize=7)
        ax.set_title('(c) Fe Speciation', fontsize=10)

        # (1,1) Scatter: ratio vs delta_ev
        ax = self.vis_fig.add_subplot(gs[1, 1])
        for g in present:
            gd = grouped.get_group(g)
            ax.scatter(gd['intensity_ratio'], gd['delta_ev'],
                       color=colors[g], label=g, edgecolors='black',
                       linewidth=0.3, s=40, alpha=0.8)
        ax.set_xlabel('Intensity Ratio')
        ax.set_ylabel(r'$\Delta$E (eV)')
        ax.set_title(r'(d) Ratio vs $\Delta$E', fontsize=10)
        ax.axvline(1.0, color='gray', ls='--', lw=0.8, alpha=0.5)
        ax.legend(fontsize=6, ncol=2, loc='best')
        ax.grid(True, alpha=0.15)

    def _vis_table(self, df):
        """Summary statistics table."""
        ax = self.vis_fig.add_subplot(111)
        ax.axis('off')
        groups, _ = self._vis_get_groups(df)
        grouped = df.groupby('sample_group')
        present = [g for g in groups if g in grouped.groups]

        table_data = []
        for g in present:
            gd = grouped.get_group(g)
            nvalid = gd['vanaken_fe3'].notna().sum()
            ntotal = len(gd)
            n_label = f"{nvalid}" if nvalid == ntotal else f"{nvalid}/{ntotal}"
            fe3_vals = gd['vanaken_fe3'].dropna()
            table_data.append([
                g, n_label,
                f"{np.nanmean(gd['vanaken_fe3']):.3f}" if nvalid > 0 else "-",
                f"{np.nanstd(gd['vanaken_fe3'], ddof=1):.3f}" if nvalid > 1 else "-",
                f"{fe3_vals.min():.3f}" if nvalid > 0 else "-",
                f"{fe3_vals.max():.3f}" if nvalid > 0 else "-",
                f"{np.nanmean(gd['vanaken_l3_centroid']):.2f}" if gd['vanaken_l3_centroid'].notna().any() else "-",
                f"{np.nanmean(gd['delta_ev']):.2f}" if 'delta_ev' in gd and gd['delta_ev'].notna().any() else "-",
                f"{np.nanmean(gd['intensity_ratio']):.2f}" if 'intensity_ratio' in gd and gd['intensity_ratio'].notna().any() else "-",
            ])

        col_labels = ['Group', 'n',
                      'Fe3+/SFe\nmean', 'Fe3+/SFe\nstd',
                      'Fe3+/SFe\nmin', 'Fe3+/SFe\nmax',
                      'L3 centroid\n(eV)', 'dE\n(eV)', 'I ratio']
        table = ax.table(cellText=table_data, colLabels=col_labels,
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.5)
        for j in range(len(col_labels)):
            table[0, j].set_facecolor('#4472C4')
            table[0, j].set_text_props(color='white', fontweight='bold')
        for i in range(len(table_data)):
            for j in range(len(col_labels)):
                if i % 2 == 0:
                    table[i + 1, j].set_facecolor('#D9E2F3')
        ax.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=15)

    def _vis_peak_scatter(self, df):
        """Peak analysis scatter: intensity ratio vs peak splitting by group."""
        ax = self.vis_fig.add_subplot(111)
        valid = df.dropna(subset=['delta_ev', 'intensity_ratio'])
        if len(valid) == 0:
            ax.text(0.5, 0.5,
                    'No peak data available\nSelect 2 peaks for each sample',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, color='gray')
            ax.set_xlabel('Intensity Ratio (Low E / High E)')
            ax.set_ylabel(r'$\Delta$E (Peak Separation)')
            ax.set_title(r'Fe L$_3$ Peak Analysis')
            return

        groups, colors = self._vis_get_groups(valid)
        grouped = valid.groupby('sample_group')
        for g in groups:
            if g not in grouped.groups:
                continue
            gd = grouped.get_group(g)
            ax.scatter(gd['intensity_ratio'], gd['delta_ev'],
                       color=colors[g], label=g, edgecolors='black',
                       linewidth=0.8, s=100, alpha=0.8)
        ax.set_xlabel('Intensity Ratio (Low E / High E)')
        ax.set_ylabel(r'$\Delta$E (Peak Separation, eV)')
        ax.set_title(r'Fe L$_3$ Peak Analysis (Von der Heyden)')
        ax.grid(True, alpha=0.15, which='major')
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(handles, labels, loc='best', fontsize=7, framealpha=0.8)

    def _vis_peak_table(self, df):
        """Peak analysis data table."""
        ax = self.vis_fig.add_subplot(111)
        ax.axis('off')

        valid = df.dropna(subset=['peak1_energy'])
        if len(valid) == 0:
            ax.text(0.5, 0.5,
                    'No peak data available\nSelect 2 peaks for each sample',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, color='gray')
            return

        table_data = []
        for _, row in valid.iterrows():
            sn = row.get('sample_name', '')
            gr = row.get('sample_group', '')
            p1 = row.get('peak1_energy', np.nan)
            p2 = row.get('peak2_energy', np.nan)
            de = row.get('delta_ev', np.nan)
            ir = row.get('intensity_ratio', np.nan)
            table_data.append([
                sn, gr,
                f"{p1:.3f}" if pd.notna(p1) else "-",
                f"{p2:.3f}" if pd.notna(p2) else "-",
                f"{de:.3f}" if pd.notna(de) else "-",
                f"{ir:.3f}" if pd.notna(ir) else "-",
            ])

        col_labels = ['Sample', 'Group',
                      'Peak 1\n(eV)', 'Peak 2\n(eV)',
                      r'$\Delta$E' + '\n(eV)', 'Intensity\nRatio']
        table = ax.table(cellText=table_data, colLabels=col_labels,
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.4)
        for j in range(len(col_labels)):
            table[0, j].set_facecolor('#4472C4')
            table[0, j].set_text_props(color='white', fontweight='bold')
        for i in range(len(table_data)):
            for j in range(len(col_labels)):
                if i % 2 == 0:
                    table[i + 1, j].set_facecolor('#D9E2F3')
        ax.set_title('Peak Analysis Data', fontsize=12, fontweight='bold', pad=15)

    # ------------------------------------------------------------------
    # Status bar helper
    # ------------------------------------------------------------------
    def _set_status(self, msg):
        self._status_var.set(msg)

    def _show_empty_state(self):
        """Draw a welcome/empty-state message on the plot area."""
        for ax in (self.ax_full, self.ax_l3, self.ax_l2, self.ax_deriv):
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
        self.ax_full.text(
            0.5, 0.5,
            "IronLPeaks \u2014 Fe L-edge Peak Selector\n\n"
            "Open spectrum files to get started\n"
            "File \u2192 Open Files   |   Ctrl+O\n"
            "File \u2192 Open Folder  |   Drag & drop coming soon",
            ha='center', va='center',
            transform=self.ax_full.transAxes,
            fontsize=14, color='#78909c',
            linespacing=1.8)
        self.ax_l3.text(
            0.5, 0.5,
            "Supported formats: .csv  .nor  .txt  .dat  .xas",
            ha='center', va='center',
            transform=self.ax_l3.transAxes,
            fontsize=11, color='#90a4ae')
        self.ax_l2.text(
            0.5, 0.5,
            "L\u2082 baseline quality check",
            ha='center', va='center',
            transform=self.ax_l2.transAxes,
            fontsize=11, color='#90a4ae')
        self.ax_deriv.text(
            0.5, 0.5,
            "Or load a previous session: File \u2192 Load Session",
            ha='center', va='center',
            transform=self.ax_deriv.transAxes,
            fontsize=11, color='#90a4ae')
        self.canvas.draw()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def auto_discover_samples(self):
        """Auto-discover sample names and build DataFrame."""
        sample_names = []
        self.sample_file_paths = {}

        if self.selected_files:
            for filepath in self.selected_files:
                basename = os.path.basename(filepath)
                sample_name = basename
                while True:
                    name, ext = os.path.splitext(sample_name)
                    if ext.lower() in _SPECTRUM_EXTENSIONS:
                        sample_name = name
                    else:
                        break
                # Disambiguate same-stem files (e.g. sampleA.csv + sampleA.txt)
                unique_name = sample_name
                n = 2
                while unique_name in self.sample_file_paths:
                    unique_name = f"{sample_name}_{n}"
                    n += 1
                sample_names.append(unique_name)
                self.sample_file_paths[unique_name] = filepath
        else:
            if self.spectra_dir:
                try:
                    for fname in os.listdir(self.spectra_dir):
                        if fname.lower().endswith(_SPECTRUM_EXTENSIONS):
                            sample_names.append(os.path.splitext(fname)[0])
                    sample_names = sorted(set(sample_names))
                except Exception:
                    pass

        self.df = pd.DataFrame({
            'sample_name': sample_names,
            'sample_group': "",
            'peak1_energy': np.nan,
            'peak2_energy': np.nan,
            'peak1_intensity': np.nan,
            'peak2_intensity': np.nan,
            'delta_ev': np.nan,
            'intensity_ratio': np.nan,
            'include_in_report': True,
            'smoothing_sigma': 0.0,
            'baseline_offset': np.nan,
            'baseline_slope': np.nan,
            'baseline_step': np.nan,
            'baseline_width': np.nan,
            'baseline_edge': np.nan,
            'baseline_edge_l2': np.nan,
            'vanaken_fe3': np.nan,
            'vanaken_ratio': np.nan,
            'vanaken_p': np.nan,
            'vanaken_il3': np.nan,
            'vanaken_il2': np.nan,
            'vanaken_l3_lo': np.nan,
            'vanaken_l3_hi': np.nan,
            'vanaken_l2_lo': np.nan,
            'vanaken_l2_hi': np.nan,
            'vanaken_l3_centroid': np.nan,
        })

    def load_csv(self):
        """Load the CSV file containing sample data."""
        try:
            self.df = pd.read_csv(self.csv_path)
            for col, default in [('include_in_report', True),
                                 ('sample_group', ""),
                                 ('peak1_energy', np.nan),
                                 ('peak2_energy', np.nan),
                                 ('peak1_intensity', np.nan),
                                 ('peak2_intensity', np.nan),
                                 ('delta_ev', np.nan),
                                 ('intensity_ratio', np.nan),
                                 ('smoothing_sigma', 0.0),
                                 ('baseline_offset', np.nan),
                                 ('baseline_slope', np.nan),
                                 ('baseline_step', np.nan),
                                 ('baseline_width', np.nan),
                                 ('baseline_edge', np.nan),
                                 ('baseline_edge_l2', np.nan),
                                 ('vanaken_fe3', np.nan),
                                 ('vanaken_ratio', np.nan),
                                 ('vanaken_p', np.nan),
                                 ('vanaken_il3', np.nan),
                                 ('vanaken_il2', np.nan),
                                 ('vanaken_l3_lo', np.nan),
                                 ('vanaken_l3_hi', np.nan),
                                 ('vanaken_l2_lo', np.nan),
                                 ('vanaken_l2_hi', np.nan),
                                 ('vanaken_l3_centroid', np.nan)]:
                if col not in self.df.columns:
                    self.df[col] = default
            # Drop legacy columns that are no longer used
            for legacy in ('fe_oxidation_state', 'vanaken_saturation'):
                if legacy in self.df.columns:
                    self.df = self.df.drop(columns=[legacy])
            # Session round-trip: empty sample_group must stay empty, not
            # become NaN (which renders as a spurious 'nan' group)
            self.df['sample_group'] = (
                self.df['sample_group'].fillna('').replace('nan', ''))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV: {e}")
            self.df = pd.DataFrame()

    def populate_sample_list(self, maintain_selection=True):
        """Populate the sample listbox with sample names.

        Sets _updating_listbox guard so that the <<ListboxSelect>> event
        doesn't trigger on_sample_select → load_current_sample during rebuild.
        """
        if not hasattr(self, 'sample_listbox'):
            return

        self._updating_listbox = True
        try:
            saved_index = self.current_sample_index if maintain_selection else 0
            saved_yview = self.sample_listbox.yview()[0] if maintain_selection else 0

            self.sample_listbox.delete(0, tk.END)
            if self.df is None:
                return
            for idx, row in self.df.iterrows():
                sample_name = row.get('sample_name', row.get('Sample', f"Sample_{idx}"))
                status = "\u2713" if row.get('include_in_report', True) else "\u2717"
                sample_group = row.get('sample_group', '')
                cls_info = f" ({sample_group})" if sample_group else ""
                display = f"{status} {sample_name}{cls_info}"
                self.sample_listbox.insert(tk.END, display)

                # Alternating row colors for readability
                if idx % 2 == 1:
                    self.sample_listbox.itemconfig(idx, bg='#f5f5f5')

            if len(self.df) > 0:
                if maintain_selection and saved_index < len(self.df):
                    self.current_sample_index = saved_index
                    self.sample_listbox.selection_clear(0, tk.END)
                    self.sample_listbox.selection_set(saved_index)
                    self.sample_listbox.yview_moveto(saved_yview)
                    self.sample_listbox.see(saved_index)
                else:
                    self.sample_listbox.selection_set(0)
                    self.current_sample_index = 0
                self.update_sample_info()
            self._refresh_vanaken_sample_list()
        finally:
            self._updating_listbox = False

    def update_sample_info(self):
        """Update the sample info label and classification controls."""
        total = len(self.df)
        current_num = self.current_sample_index + 1
        selected_indices = self.sample_listbox.curselection()
        self.sample_info_label.config(
            text=f"Sample: {current_num}/{total} | Selected: {len(selected_indices)}")

        if total > 0:
            row = self.df.iloc[self.current_sample_index]
            self.include_var.set(bool(row.get('include_in_report', True)))
            if hasattr(self, 'sample_group_var'):
                self.sample_group_var.set(row.get('sample_group', ''))

            # Update van Aken tab navigation label + sidebar
            if hasattr(self, 'vanaken_sample_label'):
                name = row.get('sample_name', f'Sample {current_num}')
                fe3 = row.get('vanaken_fe3', np.nan)
                if pd.notna(fe3):
                    fe3_str = f"  |  Fe\u00b3\u207a/\u03a3Fe = {fe3:.3f}"
                else:
                    fe3_str = ""
                self.vanaken_sample_label.config(
                    text=f"[{current_num}/{total}]  {name}{fe3_str}")

            # Sync van Aken listbox selection
            if hasattr(self, 'vanaken_sample_listbox'):
                self.vanaken_sample_listbox.selection_clear(0, tk.END)
                if self.current_sample_index < self.vanaken_sample_listbox.size():
                    self.vanaken_sample_listbox.selection_set(
                        self.current_sample_index)
                    self.vanaken_sample_listbox.see(self.current_sample_index)

            self._update_vanaken_info_panel()

    # ------------------------------------------------------------------
    # Spectrum file I/O
    # ------------------------------------------------------------------
    def read_csv_spectrum_file(self, filepath):
        """Read spectrum data from a CSV/text file.

        Prefers the loader's name-based column detection (handles headers
        like 'energy,normalized_mu' and headerless files); falls back to the
        positional two-column parser below for anything it cannot read.
        """
        if FeDataLoaderV2 is not None:
            try:
                energy, intensity, _ = FeDataLoaderV2().load_spectrum_file(
                    str(filepath))
                return energy, intensity
            except Exception:
                pass
        energy_list = []
        intensity_list = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                parts = line.replace(',', ' ').split()
                if len(parts) >= 2:
                    try:
                        energy_list.append(float(parts[0]))
                        intensity_list.append(float(parts[1]))
                    except ValueError:
                        continue
        if not energy_list:
            raise ValueError("No valid data points found in file")
        # Sort ascending (descending beamline exports break np.interp) and
        # drop duplicate energies so the grid is strictly increasing
        energy = np.array(energy_list)
        intensity = np.array(intensity_list)
        order = np.argsort(energy, kind='stable')
        energy = energy[order]
        intensity = intensity[order]
        keep = np.concatenate(([True], np.diff(energy) > 0))
        return energy[keep], intensity[keep]

    def load_spectrum_data(self, row):
        """Load spectrum data for a given row."""
        # Bug 8 fix: use 'sample_name' not 'sample_id'
        sample_id = row.get('sample_name', row.get('Sample', f'Sample_{self.current_sample_index}'))

        # 0. In-memory Athena spectra
        if hasattr(self, '_athena_spectra') and sample_id in self._athena_spectra:
            spec = self._athena_spectra[sample_id]
            self._store_spectrum(spec['energy'], spec['intensity'])
            return

        # 1. Direct file path mapping
        if hasattr(self, 'sample_file_paths') and sample_id in self.sample_file_paths:
            filepath = self.sample_file_paths[sample_id]
            try:
                energy, intensity = self.read_csv_spectrum_file(filepath)
                self._store_spectrum(energy, intensity)
                return
            except Exception as e:
                # Surface the parse error instead of failing silently
                self._set_status(
                    f"Failed to parse {os.path.basename(str(filepath))}: {e}")

        # 2. Spectra dir — Bug 1 fix: no .csv.csv fallback
        if self.spectra_dir:
            for ext in (*_SPECTRUM_EXTENSIONS, ''):
                filepath = os.path.join(self.spectra_dir, f"{sample_id}{ext}")
                if os.path.exists(filepath):
                    try:
                        energy, intensity = self.read_csv_spectrum_file(filepath)
                        self._store_spectrum(energy, intensity)
                        return
                    except Exception:
                        continue

        # 3. No spectrum available \u2014 never fabricate data. Mark the sample
        # as unavailable; consumers must skip it gracefully.
        self.current_spectrum_data = None
        self.original_spectrum_data = None
        self.current_smoothed_data = None
        self._set_status(
            f"Spectrum file not found for '{sample_id}' \u2014 "
            f"re-add it via File \u2192 Open Files")

    def _store_spectrum(self, energy, intensity):
        energy = np.asarray(energy)
        intensity = np.asarray(intensity)
        # Guarantee ascending energy for np.interp/trapezoid consumers
        if energy.size > 1 and np.any(np.diff(energy) < 0):
            order = np.argsort(energy, kind='stable')
            energy = energy[order]
            intensity = intensity[order]
        self.current_spectrum_data = {'energy': energy, 'intensity': intensity}
        self.original_spectrum_data = {'energy': energy.copy(),
                                       'intensity': intensity.copy()}

    def _restore_row_smoothing(self, row):
        """Rebuild per-sample smoothing from the row's stored sigma.

        Used by batch loops so each iteration analyzes the sample being
        processed, never a stale smoothed spectrum from the displayed sample.
        """
        sigma = row.get('smoothing_sigma', 0.0)
        sigma = 0.0 if pd.isna(sigma) else float(sigma)
        if sigma > 0 and self.original_spectrum_data is not None:
            self.current_smoothed_data = {
                'energy': self.original_spectrum_data['energy'].copy(),
                'intensity': gaussian_filter1d(
                    self.original_spectrum_data['intensity'], sigma=sigma),
            }
        else:
            self.current_smoothed_data = None

    # ------------------------------------------------------------------
    # Active spectrum helper
    # ------------------------------------------------------------------
    def _get_active_spectrum(self):
        """Return the working spectrum: smoothed if smoothing is active, else raw.

        Smoothing is considered active when smoothed data exists (sigma > 0),
        regardless of the 'Show Smoothed Overlay' checkbox — that checkbox
        only controls the plot overlay, not the analysis pipeline.
        """
        if self.current_smoothed_data is not None:
            return (self.current_smoothed_data['energy'],
                    self.current_smoothed_data['intensity'])
        if self.current_spectrum_data is not None:
            return (self.current_spectrum_data['energy'],
                    self.current_spectrum_data['intensity'])
        return None, None

    # ------------------------------------------------------------------
    # Peak detection
    # ------------------------------------------------------------------
    def _compute_sg_derivatives(self, energy, intensity):
        """Compute Savitzky-Golay first and second derivatives.

        Adaptive window length targets ~1.2 eV (below min peak separation).
        Falls back to np.gradient for very sparse data (< 5 points).
        Returns (first_deriv, second_deriv) in physical units (per eV).
        """
        n = len(energy)
        if n < 2:
            # np.gradient needs >= 2 points — return flat derivatives
            z = np.zeros(n, dtype=float)
            return z, z
        if n < 5:
            dx = np.mean(np.diff(energy))
            if dx <= 0:
                dx = 1.0
            d1 = np.gradient(intensity, dx)
            d2 = np.gradient(d1, dx)
            return d1, d2

        dx = np.mean(np.diff(energy))
        if dx <= 0:
            dx = 1.0

        # Target SG window = 1.2 eV
        target_window_ev = 1.2
        win_pts = int(round(target_window_ev / dx))
        # Must be odd and >= polyorder + 2
        if win_pts < _SG_POLYORDER + 2:
            win_pts = _SG_POLYORDER + 2
        if win_pts % 2 == 0:
            win_pts += 1
        if win_pts > n:
            win_pts = n if n % 2 == 1 else n - 1
        if win_pts < _SG_POLYORDER + 2:
            # Too few points for SG — fall back
            d1 = np.gradient(intensity, dx)
            d2 = np.gradient(d1, dx)
            return d1, d2

        d1 = savgol_filter(intensity, window_length=win_pts,
                           polyorder=_SG_POLYORDER, deriv=1, delta=dx)
        d2 = savgol_filter(intensity, window_length=win_pts,
                           polyorder=_SG_POLYORDER, deriv=2, delta=dx)
        return d1, d2

    def find_peak_candidates(self):
        """Find potential peak positions using first-derivative zero crossings.

        Algorithm (Von der Heyden 2017 / WinXAS 'Search Min-Max'):
        1. Subtract baseline (if present) from active spectrum
        2. Window to detection region
        3. Compute SG 1st and 2nd derivatives
        4. Find zero crossings of dI/dE where sign goes + → − (local max)
        5. Verify d²I/dE² < 0 at each crossing (concave-down check)
        6. Interpolate crossing energy for sub-grid precision
        7. Filter by sensitivity threshold (fraction of intensity range)
        8. Enforce minimum separation (_FE_L3_MIN_SEPARATION)
        """
        energy, intensity = self._get_active_spectrum()
        if energy is None:
            self.peak_candidates = np.array([])
            return

        if self.current_baseline is not None:
            intensity = intensity - self.current_baseline

        window_mask = (energy >= self.window_start) & (energy <= self.window_end)
        w_energy = energy[window_mask]
        w_intensity = intensity[window_mask]

        if len(w_energy) < 3:
            self.peak_candidates = np.array([])
            return

        d1, d2 = self._compute_sg_derivatives(w_energy, w_intensity)

        # Find zero crossings of d1: positive → negative = local maximum
        candidates = []
        for i in range(len(d1) - 1):
            if d1[i] >= 0 and d1[i + 1] < 0:
                # Verify concave-down (d2 < 0) at this location
                d2_here = 0.5 * (d2[i] + d2[i + 1])
                if d2_here >= 0:
                    continue
                # Linear interpolation for sub-grid crossing energy
                if d1[i] - d1[i + 1] != 0:
                    frac = d1[i] / (d1[i] - d1[i + 1])
                else:
                    frac = 0.5
                cross_e = w_energy[i] + frac * (w_energy[i + 1] - w_energy[i])
                cross_i = np.interp(cross_e, w_energy, w_intensity)
                candidates.append((cross_e, cross_i))

        if not candidates:
            self.peak_candidates = np.array([])
            self.current_peak_candidate_idx = 0
            return

        # Sensitivity filter: min_peak_height_var is fraction (0–1) of the
        # spectrum's intensity range in the detection window. Normalizing
        # against the spectrum (not the candidate set) avoids the degenerate
        # case where two genuine peaks would always lose the smaller one.
        sens = self.min_peak_height_var.get() if hasattr(self, 'min_peak_height_var') else 0.1
        sens = max(0.0, min(1.0, sens))
        spec_max = float(np.max(w_intensity))
        spec_min = float(np.min(w_intensity))
        spec_range = spec_max - spec_min
        if spec_range > 0:
            threshold = spec_min + sens * spec_range
            candidates = [c for c in candidates if c[1] >= threshold]

        if not candidates:
            self.peak_candidates = np.array([])
            self.current_peak_candidate_idx = 0
            return

        # Enforce minimum separation: when two candidates are closer than
        # _FE_L3_MIN_SEPARATION, keep the taller one
        candidates.sort(key=lambda c: c[0])  # sort by energy
        filtered = [candidates[0]]
        for c in candidates[1:]:
            if c[0] - filtered[-1][0] < _FE_L3_MIN_SEPARATION:
                # Keep the taller candidate
                if c[1] > filtered[-1][1]:
                    filtered[-1] = c
            else:
                filtered.append(c)

        self.peak_candidates = np.array([c[0] for c in filtered])
        self.current_peak_candidate_idx = 0

    # ------------------------------------------------------------------
    # Baseline fitting
    # ------------------------------------------------------------------
    def fit_baseline(self):
        """Fit baseline using ArctanBaselineFe2pV2.

        Uses the active spectrum (smoothed if smoothing is on, else raw)
        so the baseline tracks the working data, not just the raw signal.
        """
        energy, intensity = self._get_active_spectrum()
        if energy is None:
            messagebox.showwarning("Warning", "No spectrum data loaded")
            return
        if ArctanBaselineFe2pV2 is None:
            messagebox.showwarning("Warning", "ArctanBaselineFe2pV2 not available")
            return

        try:

            baseline_start_str = self.baseline_start_var.get().strip()
            baseline_end_str = self.baseline_end_var.get().strip()

            if not baseline_start_str or not baseline_end_str:
                baseline_start = energy.min()
                baseline_end = energy.max()
            else:
                try:
                    baseline_start = float(baseline_start_str)
                    baseline_end = float(baseline_end_str)
                except ValueError:
                    messagebox.showerror("Error",
                                        "Baseline start/end must be numbers or blank")
                    return

            if baseline_start >= baseline_end:
                messagebox.showerror("Error",
                                    "Baseline start must be less than baseline end")
                return

            mask = (energy >= baseline_start) & (energy <= baseline_end)
            if not np.any(mask):
                messagebox.showerror("Error",
                                    f"No data in baseline region "
                                    f"{baseline_start}-{baseline_end} eV")
                return

            energy_bl = energy[mask]
            intensity_bl = intensity[mask]

            penalty_weight = self.penalty_weight_var.get()
            enable_iterative = self.iterative_refit_var.get()
            max_attempts = self.max_refit_var.get()

            # Bug 6 fix: only use manual params when the user explicitly edited them
            manual_params = None
            if self._baseline_params_are_manual:
                try:
                    vals = [self.offset_var, self.slope_var, self.step_height_var,
                            self.edge_width_var, self.edge_position_var,
                            self.edge_l2_position_var]
                    strs = [v.get().strip().lower() for v in vals]
                    if not all(s == "auto" for s in strs):
                        parsed = [float(s) if s != "auto" else None for s in strs]
                        if any(p is not None for p in parsed):
                            manual_params = parsed
                except ValueError:
                    manual_params = None

            self.current_baseline_fitter = ArctanBaselineFe2pV2(
                energy_bl, intensity_bl, verbose=False,
                penalty_weight=penalty_weight,
                enable_iterative_refit=enable_iterative,
                max_refit_attempts=max_attempts)

            bl_result = self.current_baseline_fitter.fit_baseline()
            fit_ok = bool(bl_result and bl_result.get('success'))

            if not fit_ok and not (manual_params
                                   and all(p is not None for p in manual_params)):
                # Arctan fit failed and no full set of manual params to fall
                # back on — never silently adopt the linear fallback baseline
                self.current_baseline = None
                self.current_baseline_fitter = None
                err = (bl_result.get('error', 'fit error')
                       if bl_result else 'fit error')
                self.baseline_status_label.config(text="✗ Fit failed",
                                                  foreground=_ERROR)
                self._set_status(
                    f"Baseline fit failed ({err}) — no baseline applied")
                self.find_peak_candidates()
                self.redraw_plots()
                self._redraw_baseline_tab()
                return

            if manual_params:
                offset, slope, step, width, edge_pos, edge_l2 = manual_params
                fitted = self.current_baseline_fitter.baseline_params
                offset = offset if offset is not None else fitted[0]
                slope = slope if slope is not None else fitted[1]
                step = step if step is not None else fitted[2]
                width = width if width is not None else fitted[3]
                edge_pos = edge_pos if edge_pos is not None else fitted[4]
                edge_l2 = edge_l2 if edge_l2 is not None else (fitted[5] if len(fitted) > 5 else 719.5)
                manual_bl = ArctanBaselineFe2pV2.double_arctan_baseline(
                    energy_bl, offset, slope, step, width, edge_pos, edge_l2)
                self.current_baseline_fitter.baseline = manual_bl
                self.current_baseline_fitter.baseline_params = [
                    offset, slope, step, width, edge_pos, edge_l2]

            # Interpolate to full energy range
            self.current_baseline = np.interp(
                energy, energy_bl, self.current_baseline_fitter.baseline)

            self.baseline_status_label.config(text="\u2713 Fitted",
                                              foreground=_SUCCESS)
            self._set_status("Baseline fitted successfully")

            # Display fitted params (informational — don't set manual flag)
            params = self.current_baseline_fitter.baseline_params
            if params is not None:
                self._baseline_params_are_manual = False  # reset flag
                self.offset_var.set(f"{params[0]:.6f}")
                self.slope_var.set(f"{params[1]:.6f}")
                self.step_height_var.set(f"{params[2]:.6f}")
                self.edge_width_var.set(f"{params[3]:.6f}")
                self.edge_position_var.set(f"{params[4]:.3f}")
                self.edge_l2_position_var.set(f"{params[5]:.3f}" if len(params) > 5 else "719.500")

                if len(self.df) > 0:
                    idx = self.current_sample_index
                    self.df.at[idx, 'baseline_offset'] = params[0]
                    self.df.at[idx, 'baseline_slope'] = params[1]
                    self.df.at[idx, 'baseline_step'] = params[2]
                    self.df.at[idx, 'baseline_width'] = params[3]
                    self.df.at[idx, 'baseline_edge'] = params[4]
                    self.df.at[idx, 'baseline_edge_l2'] = params[5] if len(params) > 5 else 719.5

            # Recalculate peak candidates and redraw
            self.find_peak_candidates()
            self.redraw_plots()
            self._redraw_baseline_tab()

        except Exception as e:
            # Never leave a stale/fallback baseline in place after a failure
            self.current_baseline = None
            self.current_baseline_fitter = None
            self.baseline_status_label.config(text="\u2717 Failed",
                                              foreground=_ERROR)
            messagebox.showerror("Error", f"Failed to fit baseline: {e}")

    def _refit_baseline(self):
        """Force a fresh auto-fit (reset params to auto first)."""
        self._baseline_params_are_manual = False
        self.offset_var.set("auto")
        self.slope_var.set("auto")
        self.step_height_var.set("auto")
        self.edge_width_var.set("auto")
        self.edge_position_var.set("auto")
        self.edge_l2_position_var.set("auto")
        self.fit_baseline()

    def reset_baseline_params(self):
        """Bug 2 fix: Reset clears baseline entirely — does NOT re-fit."""
        self.penalty_weight_var.set(200.0)
        self.max_refit_var.set(10)
        self.iterative_refit_var.set(True)
        self._baseline_params_are_manual = False
        self.offset_var.set("auto")
        self.slope_var.set("auto")
        self.step_height_var.set("auto")
        self.edge_width_var.set("auto")
        self.edge_position_var.set("auto")
        self.edge_l2_position_var.set("auto")

        # Clear baseline
        self.current_baseline = None
        self.current_baseline_fitter = None
        self.baseline_status_label.config(text="No baseline", foreground="gray")
        self._set_status("Baseline cleared — click Fit Baseline or Re-fit to recalculate")
        self.redraw_plots()
        self._redraw_baseline_tab()

        # Redraw without baseline
        self.find_peak_candidates()
        self.redraw_plots()

    def toggle_baseline_params(self):
        if self.show_baseline_params.get():
            self.advanced_baseline_frame.pack(fill=tk.X, pady=(2, 0))
        else:
            self.advanced_baseline_frame.pack_forget()

    def on_parameter_change(self, event=None):
        try:
            penalty = self.penalty_weight_var.get()
            max_attempts = self.max_refit_var.get()
            if penalty < 0 or max_attempts < 1:
                return
        except tk.TclError:
            return
        self.fit_baseline()

    def on_subtract_baseline_toggle(self):
        self.redraw_plots()

    # ------------------------------------------------------------------
    # Plot drawing — publication quality
    # ------------------------------------------------------------------
    def redraw_plots(self):
        """Redraw all three spectrum plots with publication-quality styling.

        Uses the *active* spectrum (smoothed when smoothing is on) as the
        primary working data for the L3 and derivative plots.  The top
        full-spectrum plot always shows raw data so the user can see the
        effect of smoothing.
        """
        if self.current_spectrum_data is None:
            return

        raw_energy = self.current_spectrum_data['energy']
        raw_intensity = self.current_spectrum_data['intensity']

        # Active data = smoothed if available, else raw
        act_energy, act_intensity = self._get_active_spectrum()
        is_smoothed = self.current_smoothed_data is not None

        self.ax_full.clear()
        self.ax_l3.clear()
        self.ax_l2.clear()
        self.ax_deriv.clear()

        sample_name = "Unknown Sample"
        if self.df is not None and len(self.df) > 0:
            sample_name = self.df.iloc[self.current_sample_index].get(
                'sample_name',
                self.df.iloc[self.current_sample_index].get(
                    'Sample', f'Sample_{self.current_sample_index}'))

        # ---- PLOT 1 (TOP): Full spectrum ----
        self.ax_full.plot(raw_energy, raw_intensity, color='black',
                          linewidth=1.0, alpha=0.6 if is_smoothed else 1.0,
                          label='Raw Spectrum')
        if is_smoothed:
            self.ax_full.plot(act_energy, act_intensity, color='#1565c0',
                              linewidth=1.5, label='Smoothed')
        if self.current_baseline is not None:
            self.ax_full.plot(act_energy, self.current_baseline, color='#d32f2f',
                              linestyle='--', linewidth=1.2, label='Baseline')
            corrected = act_intensity - self.current_baseline
            self.ax_full.plot(act_energy, corrected, color='#2e7d32',
                              linewidth=1.5, label='Corrected')
        # Detection window shading
        self.ax_full.axvspan(self.window_start, self.window_end,
                             alpha=0.10, color='#9e9e9e',
                             label=f'Window ({self.window_start:.0f}\u2013'
                                   f'{self.window_end:.0f} eV)')
        self.ax_full.set_xlabel('Energy (eV)')
        self.ax_full.set_ylabel('Intensity (a.u.)')
        title_suffix = f' (\u03c3={self.smoothing_var.get():.1f})' if is_smoothed else ''
        self.ax_full.set_title(
            f'Full Fe L-edge Spectrum \u2014 {sample_name}{title_suffix}')
        self.ax_full.legend(loc='upper right', fontsize=8, framealpha=0.7)
        self.ax_full.grid(True, alpha=0.15, which='major')

        # ---- PLOT 2 (MIDDLE): L3 region ----
        emin, emax = act_energy.min(), act_energy.max()
        l3_start = max(700, emin)
        l3_end = min(720, emax)
        l3_mask = (act_energy >= l3_start) & (act_energy <= l3_end)
        l3_energy = act_energy[l3_mask]

        if self.current_baseline is not None and self.subtract_baseline_var.get():
            corrected = act_intensity - self.current_baseline
            l3_plot = corrected[l3_mask]
            l3_label = 'Corrected (Smoothed)' if is_smoothed else 'Corrected'
            l3_color = '#2e7d32'
        else:
            l3_plot = act_intensity[l3_mask]
            l3_label = 'Smoothed' if is_smoothed else 'Raw'
            l3_color = '#1565c0' if is_smoothed else 'black'
        self.ax_l3.plot(l3_energy, l3_plot, color=l3_color, linewidth=1.5,
                        label=l3_label)

        # Show raw overlay when smoothing is active and checkbox is on
        if is_smoothed and self.show_smoothed_var.get():
            raw_l3_mask = (raw_energy >= l3_start) & (raw_energy <= l3_end)
            raw_l3_e = raw_energy[raw_l3_mask]
            raw_l3_i = raw_intensity[raw_l3_mask]
            if self.current_baseline is not None and self.subtract_baseline_var.get():
                raw_bl = np.interp(raw_energy, act_energy, self.current_baseline)
                raw_l3_i = raw_l3_i - raw_bl[raw_l3_mask]
            self.ax_l3.plot(raw_l3_e, raw_l3_i, color='black', linewidth=0.7,
                            alpha=0.35, label='Raw')

        # Detection window
        self.ax_l3.axvspan(self.window_start, self.window_end,
                           alpha=0.08, color='#9e9e9e')

        # Peak candidates — small red inverted triangles so user sees what's selectable
        if len(self.peak_candidates) > 0:
            for pk_e in self.peak_candidates:
                if l3_start <= pk_e <= l3_end:
                    pk_i = np.interp(pk_e, l3_energy, l3_plot)
                    self.ax_l3.plot(pk_e, pk_i, 'v', color='#d32f2f',
                                   markersize=6, markerfacecolor='none',
                                   markeredgewidth=1.0, alpha=0.7)

        # Selected peaks — filled circles with drop lines
        if self.selected_peaks:
            for i, pk_e in enumerate(self.selected_peaks):
                if self.window_start <= pk_e <= self.window_end:
                    pk_i = np.interp(pk_e, l3_energy, l3_plot)
                    self.ax_l3.plot(pk_e, pk_i, 'o', color=_SUCCESS,
                                   markersize=8, markeredgecolor='black',
                                   markeredgewidth=0.8, zorder=5)
                    self.ax_l3.vlines(pk_e, ymin=self.ax_l3.get_ylim()[0]
                                     if self.ax_l3.get_ylim()[0] < pk_i else 0,
                                     ymax=pk_i, colors=_SUCCESS,
                                     linestyles=':', linewidth=0.8, alpha=0.6)
                    self.ax_l3.annotate(
                        f'{pk_e:.2f}', (pk_e, pk_i),
                        xytext=(5, 8), textcoords='offset points',
                        fontsize=8, ha='left', color=_SUCCESS, weight='bold')

        # van Aken integration-window overlay
        self._draw_vanaken_overlay(l3_energy, l3_plot)

        self.ax_l3.set_xlabel('Energy (eV)')
        self.ax_l3.set_ylabel('Intensity (a.u.)')
        self.ax_l3.set_title('L\u2083 Region \u2014 Peak Selection')
        self.ax_l3.legend(loc='upper right', fontsize=8, framealpha=0.7)
        self.ax_l3.grid(True, alpha=0.15, which='major')
        self.ax_l3.set_xlim(self.window_start, self.window_end)

        # ---- PLOT 2b (MIDDLE-RIGHT): L2 region — baseline quality check ----
        l2_start = max(715, emin)
        l2_end = min(735, emax)
        l2_mask = (act_energy >= l2_start) & (act_energy <= l2_end)
        l2_energy = act_energy[l2_mask]
        l2_raw = act_intensity[l2_mask]

        if len(l2_energy) > 0:
            # Plot spectrum (smoothed or raw)
            spec_label = 'Smoothed' if is_smoothed else 'Spectrum'
            spec_color = '#1565c0' if is_smoothed else 'black'
            self.ax_l2.plot(l2_energy, l2_raw, color=spec_color,
                            linewidth=1.2, label=spec_label)

            # Raw overlay if smoothing active
            if is_smoothed and self.show_smoothed_var.get():
                raw_l2_mask = (raw_energy >= l2_start) & (raw_energy <= l2_end)
                self.ax_l2.plot(raw_energy[raw_l2_mask],
                                raw_intensity[raw_l2_mask],
                                color='black', linewidth=0.7, alpha=0.35,
                                label='Raw')

            # Baseline overlay
            if self.current_baseline is not None:
                bl_l2 = self.current_baseline[l2_mask]
                self.ax_l2.plot(l2_energy, bl_l2, color='#d32f2f',
                                linestyle='--', linewidth=1.2, label='Baseline')

                # Corrected spectrum
                corrected_l2 = l2_raw - bl_l2
                self.ax_l2.plot(l2_energy, corrected_l2, color='#2e7d32',
                                linewidth=1.5, label='Corrected')

                # Shade negative regions in red (baseline overshoot)
                neg_mask = corrected_l2 < 0
                if np.any(neg_mask):
                    self.ax_l2.fill_between(
                        l2_energy, corrected_l2, 0,
                        where=neg_mask, alpha=0.20, color='#d32f2f',
                        label='Negative (overshoot)')

                # van Aken L2' integration window (719.7-721.7 eV)
                l2p_lo, l2p_hi = 719.7, 721.7
                l2p_mask = (l2_energy >= l2p_lo) & (l2_energy <= l2p_hi)
                if np.any(l2p_mask):
                    self.ax_l2.fill_between(
                        l2_energy[l2p_mask], 0, corrected_l2[l2p_mask],
                        alpha=0.20, color='#1565c0',
                        label=f"L2\u2032 ({l2p_lo}\u2013{l2p_hi} eV)")

        self.ax_l2.set_xlabel('Energy (eV)')
        self.ax_l2.set_ylabel('Intensity (a.u.)')
        self.ax_l2.set_title('L\u2082 Region \u2014 Baseline Check')
        self.ax_l2.legend(loc='upper right', fontsize=7, framealpha=0.7)
        self.ax_l2.grid(True, alpha=0.15, which='major')
        self.ax_l2.set_xlim(l2_start, l2_end)

        # ---- PLOT 3 (BOTTOM): First Derivative — Search Min-Max ----
        # WinXAS-style: peaks are where dI/dE crosses zero (+ → −).
        # Only the first derivative is shown; the d²I/dE² < 0 check
        # is used internally but not plotted to avoid visual clutter.
        window_mask = (act_energy >= self.window_start) & (act_energy <= self.window_end)
        w_energy = act_energy[window_mask]
        w_intensity = act_intensity[window_mask]

        if len(w_energy) > 0:
            w_for_deriv = w_intensity
            if self.current_baseline is not None and self.subtract_baseline_var.get():
                w_for_deriv = (act_intensity - self.current_baseline)[window_mask]

            first_deriv, _second_deriv = self._compute_sg_derivatives(
                w_energy, w_for_deriv)
            self.ax_deriv.plot(w_energy, first_deriv, color='#1565c0',
                               linewidth=1.5, label='dI/dE (SG)')
            self.ax_deriv.axhline(y=0, color='gray', linestyle='--',
                                  linewidth=0.8, alpha=0.6,
                                  label='Zero line (peak = crossing)')

            # Show raw derivative as faded overlay when smoothed
            if is_smoothed and self.show_smoothed_var.get():
                raw_w_mask = (raw_energy >= self.window_start) & (raw_energy <= self.window_end)
                raw_w_e = raw_energy[raw_w_mask]
                raw_w_i = raw_intensity[raw_w_mask]
                if self.current_baseline is not None and self.subtract_baseline_var.get():
                    raw_bl = np.interp(raw_energy, act_energy, self.current_baseline)
                    raw_w_i = raw_w_i - raw_bl[raw_w_mask]
                if len(raw_w_e) >= 3:
                    raw_d1, _ = self._compute_sg_derivatives(
                        raw_w_e, raw_w_i)
                    self.ax_deriv.plot(raw_w_e, raw_d1,
                                      color='black', linewidth=0.7,
                                      alpha=0.3, label='Raw dI/dE')

            # Peak candidate markers — vertical lines at zero crossings
            if len(self.peak_candidates) > 0:
                for pk_e in self.peak_candidates:
                    if self.window_start <= pk_e <= self.window_end:
                        self.ax_deriv.axvline(x=pk_e, color='#d32f2f',
                                              linestyle=':', linewidth=0.9,
                                              alpha=0.6)
                        # Small dot at the zero crossing
                        self.ax_deriv.plot(pk_e, 0, 'o', color='#d32f2f',
                                          markersize=5, markerfacecolor='#d32f2f',
                                          zorder=4)

            # Selected peaks — highlighted zero crossings
            if self.selected_peaks:
                for pk_e in self.selected_peaks:
                    if self.window_start <= pk_e <= self.window_end:
                        self.ax_deriv.axvline(x=pk_e, color=_SUCCESS,
                                              linestyle='-', linewidth=1.2,
                                              alpha=0.5)
                        self.ax_deriv.plot(pk_e, 0, 'o', color=_SUCCESS,
                                          markersize=8, markeredgecolor='black',
                                          markeredgewidth=0.8, zorder=5)

        self.ax_deriv.set_xlabel('Energy (eV)')
        self.ax_deriv.set_ylabel('dI/dE (a.u.)')
        self.ax_deriv.set_title('First Derivative \u2014 Search Min-Max')
        self.ax_deriv.legend(loc='upper right', fontsize=8, framealpha=0.7)
        self.ax_deriv.grid(True, alpha=0.15, which='major')
        self.ax_deriv.set_xlim(self.window_start, self.window_end)

        self.canvas.draw()

    def _draw_vanaken_overlay(self, l3_energy, l3_plot):
        """Overlay van Aken L3' integration window on the L3 plot."""
        if VanAkenFeQuantifier is None:
            return

        quantifier = self.current_vanaken
        fe3 = None
        l3_lo = l3_hi = None
        centroid = None

        if quantifier is not None and quantifier.fe3 is not None:
            fe3 = quantifier.fe3
            l3_lo, l3_hi = quantifier.l3_window
            centroid = quantifier.l3_centroid
        elif self.df is not None and len(self.df) > 0:
            row = self.df.iloc[self.current_sample_index]
            fe3_val = row.get('vanaken_fe3')
            if not pd.isna(fe3_val):
                fe3 = fe3_val
                l3_lo = row.get('vanaken_l3_lo')
                l3_hi = row.get('vanaken_l3_hi')
                centroid = row.get('vanaken_l3_centroid')
                if pd.isna(l3_lo) or pd.isna(l3_hi):
                    l3_lo = l3_hi = None

        # Shade L3' integration window
        if l3_lo is not None and l3_hi is not None:
            l3_mask = (l3_energy >= l3_lo) & (l3_energy <= l3_hi)
            if np.any(l3_mask):
                self.ax_l3.fill_between(
                    l3_energy[l3_mask], 0, l3_plot[l3_mask],
                    alpha=0.20, color='#e65100',
                    label=f"L3' ({l3_lo:.1f}\u2013{l3_hi:.1f} eV)")

        # Draw L3 centroid line
        if centroid is not None and not pd.isna(centroid):
            if l3_energy.min() <= centroid <= l3_energy.max():
                self.ax_l3.axvline(x=centroid, color='#2e7d32', linestyle='--',
                                   linewidth=1.0, alpha=0.7,
                                   label=f'L3 centroid ({centroid:.2f} eV)')

        if fe3 is not None and not np.isnan(fe3):
            cen_str = f", centroid={centroid:.2f} eV" if centroid is not None and not np.isnan(centroid) else ""
            self.ax_l3.text(
                0.02, 0.95,
                f"van Aken: Fe\u00b3\u207a/\u03a3Fe = {fe3:.3f}{cen_str}",
                transform=self.ax_l3.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='lightyellow', alpha=0.9,
                          edgecolor='gray'))

    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------
    def on_mouse_press(self, event):
        if event.button != 1 or event.xdata is None:
            return
        if event.inaxes not in (self.ax_l3, self.ax_deriv):
            return

        clicked_energy = event.xdata

        # Unselect if clicking near existing peak
        for sp in list(self.selected_peaks):
            if abs(sp - clicked_energy) < 0.5:
                self.selected_peaks.remove(sp)
                if len(self.df) > 0:
                    idx = self.current_sample_index
                    for col in ('peak1_energy', 'peak2_energy', 'peak1_intensity',
                                'peak2_intensity', 'delta_ev', 'intensity_ratio'):
                        self.df.at[idx, col] = np.nan
                self.update_peak_analysis_display()
                if hasattr(self, 'peak_tree'):
                    self.refresh_peak_analysis()
                if hasattr(self, 'analysis_ax'):
                    self.update_analysis_plot()
                self.redraw_plots()
                return

        # Select new peak (max 2)
        if len(self.selected_peaks) >= 2:
            self._set_status("Max 2 peaks. Click existing peak to unselect.")
            return

        # WinXAS-style: only allow selecting detected peak candidates
        if len(self.peak_candidates) == 0:
            self._set_status("No peak candidates detected — adjust window or sensitivity")
            return

        distances = np.abs(self.peak_candidates - clicked_energy)
        closest = np.argmin(distances)
        if distances[closest] > 0.4:
            self._set_status(
                f"No candidate within 0.4 eV of click — "
                f"click nearer to a candidate marker (\u25BD)")
            return

        selected_energy = self.peak_candidates[closest]
        self.selected_peaks.append(selected_energy)
        self._set_status(f"Selected peak at {selected_energy:.2f} eV "
                         f"({len(self.selected_peaks)}/2)")

        if len(self.selected_peaks) == 2:
            self.auto_calculate_peaks()
            if hasattr(self, 'analysis_ax'):
                self.update_analysis_plot()

        self.update_peak_analysis_display()
        if hasattr(self, 'peak_tree'):
            self.refresh_peak_analysis()
        self.redraw_plots()

    def on_mouse_release(self, event):
        if self.dragging_window and event.inaxes == self.ax_l3 and event.button == 1:
            self.dragging_window = False
            if self.drag_start_x is not None and event.xdata is not None:
                s = min(self.drag_start_x, event.xdata)
                e = max(self.drag_start_x, event.xdata)
                self.window_start = s
                self.window_end = e
                self.window_start_var.set(s)
                self.window_end_var.set(e)
                self.detect_peaks()
                self.redraw_plots()

    def on_mouse_motion(self, event):
        pass

    # ------------------------------------------------------------------
    # Sample navigation
    # ------------------------------------------------------------------
    def on_sample_select(self, event):
        if self._updating_listbox:
            return
        selection = self.sample_listbox.curselection()
        if selection:
            self.current_sample_index = selection[0]
            self.update_sample_info()
            self.load_current_sample()

    def prev_sample(self):
        if self.df is None or len(self.df) == 0:
            return
        self.current_sample_index = (self.current_sample_index - 1) % len(self.df)
        self.sample_listbox.selection_clear(0, tk.END)
        self.sample_listbox.selection_set(self.current_sample_index)
        self.sample_listbox.see(self.current_sample_index)
        self.update_sample_info()
        self.load_current_sample()

    def next_sample(self):
        if self.df is None or len(self.df) == 0:
            return
        self.current_sample_index = (self.current_sample_index + 1) % len(self.df)
        self.sample_listbox.selection_clear(0, tk.END)
        self.sample_listbox.selection_set(self.current_sample_index)
        self.sample_listbox.see(self.current_sample_index)
        self.update_sample_info()
        self.load_current_sample()

    def load_current_sample(self):
        """Load spectrum data for current sample."""
        if self.df is None or len(self.df) == 0:
            return
        row = self.df.iloc[self.current_sample_index]
        # Bug 8 fix: use 'sample_name'
        sample_id = row.get('sample_name',
                            row.get('Sample', f'Sample_{self.current_sample_index}'))

        self._set_status(f"Loading: {sample_id}")
        self.load_spectrum_data(row)

        # Spectrum unavailable — show it clearly and do NOT auto-fit or
        # overwrite this sample's stored results
        if self.current_spectrum_data is None:
            self.current_baseline = None
            self.current_baseline_fitter = None
            self.current_smoothed_data = None
            self.selected_peaks = []
            self.peak_candidates = np.array([])
            self.current_vanaken = None
            if hasattr(self, 'baseline_status_label'):
                self.baseline_status_label.config(text="No baseline",
                                                  foreground="gray")
            for ax in (self.ax_full, self.ax_l3, self.ax_l2, self.ax_deriv):
                ax.clear()
            self.ax_full.text(
                0.5, 0.5,
                f"Spectrum unavailable for '{sample_id}'\n"
                f"Re-add the file via File → Open Files",
                ha='center', va='center',
                transform=self.ax_full.transAxes,
                fontsize=12, color='#c62828')
            self.canvas.draw()
            # Reset smoothing state and refresh the Baseline tab so it does
            # not keep rendering the previously displayed sample
            self.smoothing_var.set(0.0)
            self.show_smoothed_var.set(False)
            self._redraw_baseline_tab()
            self._update_vanaken_display()
            self.redraw_vanaken_plot()
            self.update_peak_analysis_display()
            msg = (f"Spectrum file not found for '{sample_id}' — "
                   f"re-add it via File → Open Files")
            if sample_id not in self._missing_spectrum_warned:
                self._missing_spectrum_warned.add(sample_id)
                messagebox.showwarning("Spectrum Unavailable", msg)
            self._set_status(msg)
            return

        # Reset baseline
        self.current_baseline = None
        self.current_baseline_fitter = None

        # Restore smoothing from stored per-sample sigma
        stored_sigma = row.get('smoothing_sigma', 0.0)
        if pd.isna(stored_sigma):
            stored_sigma = 0.0
        stored_sigma = float(stored_sigma)
        self.smoothing_var.set(stored_sigma)

        if stored_sigma > 0 and self.original_spectrum_data is not None:
            smoothed = gaussian_filter1d(
                self.original_spectrum_data['intensity'], sigma=stored_sigma)
            self.current_smoothed_data = {
                'energy': self.original_spectrum_data['energy'].copy(),
                'intensity': smoothed,
            }
            self.show_smoothed_var.set(True)
        else:
            self.current_smoothed_data = None
            self.show_smoothed_var.set(False)

        # Restore selected peaks
        self.selected_peaks = []
        peak1 = row.get('peak1_energy')
        peak2 = row.get('peak2_energy')
        if not pd.isna(peak1):
            self.selected_peaks.append(peak1)
        if not pd.isna(peak2):
            self.selected_peaks.append(peak2)

        # Reset baseline fitting params
        self.penalty_weight_var.set(200.0)
        self.max_refit_var.set(10)
        self.iterative_refit_var.set(True)

        # Restore baseline params from stored data
        bl_offset = row.get('baseline_offset')
        bl_slope = row.get('baseline_slope')
        bl_step = row.get('baseline_step')
        bl_width = row.get('baseline_width')
        bl_edge = row.get('baseline_edge')
        bl_edge_l2 = row.get('baseline_edge_l2', np.nan)

        # Stored params are authoritative (they produced this sample's stored
        # results and may be manually tuned): reconstruct the baseline from
        # them instead of auto-refitting over them. Auto-fit only when none
        # are stored.
        self._baseline_params_are_manual = not pd.isna(bl_offset)
        self.offset_var.set(f"{bl_offset:.6f}" if not pd.isna(bl_offset) else "auto")
        self.slope_var.set(f"{bl_slope:.6f}" if not pd.isna(bl_slope) else "auto")
        self.step_height_var.set(f"{bl_step:.6f}" if not pd.isna(bl_step) else "auto")
        self.edge_width_var.set(f"{bl_width:.6f}" if not pd.isna(bl_width) else "auto")
        self.edge_position_var.set(f"{bl_edge:.3f}" if not pd.isna(bl_edge) else "auto")
        self.edge_l2_position_var.set(f"{bl_edge_l2:.3f}" if not pd.isna(bl_edge_l2) else "auto")

        # Auto-adjust detection window if out of range
        if self.current_spectrum_data is not None:
            en = self.current_spectrum_data['energy']
            if self.window_end < en.min() or self.window_start > en.max():
                self.window_start = max(705.0, en.min())
                self.window_end = min(712.0, en.max())
                self.window_start_var.set(self.window_start)
                self.window_end_var.set(self.window_end)

        # fit_baseline uses _get_active_spectrum() which will use the
        # smoothed data we just restored. It also calls
        # find_peak_candidates + redraw_plots.
        self.fit_baseline()

        # Restore van Aken results from DataFrame
        self.current_vanaken = None
        self._update_vanaken_display()
        self.redraw_vanaken_plot()

        self.update_peak_analysis_display()
        self._set_status(f"Loaded: {sample_id}")

    # ------------------------------------------------------------------
    # Smoothing
    # ------------------------------------------------------------------
    def apply_smoothing(self, event=None):
        if self.original_spectrum_data is None:
            return
        try:
            sigma = self.smoothing_var.get()
        except tk.TclError:
            sigma = 0.0
            self.smoothing_var.set(0.0)

        if sigma <= 0:
            self.current_smoothed_data = None
            self.show_smoothed_var.set(False)
        else:
            smoothed = gaussian_filter1d(self.original_spectrum_data['intensity'],
                                         sigma=sigma)
            self.current_smoothed_data = {
                'energy': self.original_spectrum_data['energy'].copy(),
                'intensity': smoothed,
            }
            self.show_smoothed_var.set(True)

        # Store sigma per-sample so it persists across navigation
        if self.df is not None and len(self.df) > 0:
            self.df.at[self.current_sample_index, 'smoothing_sigma'] = sigma

        # Clear existing selected peaks — smoothing changes the landscape,
        # so old selections are no longer reliable. Clear the stored values
        # too so the user starts fresh with the new candidates.
        self.selected_peaks = []
        if self.df is not None and len(self.df) > 0:
            idx = self.current_sample_index
            for col in ('peak1_energy', 'peak2_energy', 'peak1_intensity',
                        'peak2_intensity', 'delta_ev', 'intensity_ratio'):
                self.df.at[idx, col] = np.nan

        # Re-fit baseline on the (now smoothed) active data, which also
        # recalculates peak candidates and redraws.
        self.fit_baseline()
        self.update_peak_analysis_display()
        self._set_status(
            f"Smoothing \u03c3={sigma:.1f} applied — "
            f"{len(self.peak_candidates)} peak candidate(s) found"
            if sigma > 0 else "Smoothing removed")

    def reset_to_original(self):
        self.current_smoothed_data = None
        self.smoothing_var.set(0.0)
        self.show_smoothed_var.set(False)
        if self.df is not None and len(self.df) > 0:
            self.df.at[self.current_sample_index, 'smoothing_sigma'] = 0.0
        # Clear selected peaks — data landscape changed
        self.selected_peaks = []
        if self.df is not None and len(self.df) > 0:
            idx = self.current_sample_index
            for col in ('peak1_energy', 'peak2_energy', 'peak1_intensity',
                        'peak2_intensity', 'delta_ev', 'intensity_ratio'):
                self.df.at[idx, col] = np.nan
        # Re-fit baseline on raw data, recalculate peaks
        self.fit_baseline()
        self.update_peak_analysis_display()

    def on_show_smoothed_toggle(self):
        if self.show_smoothed_var.get() and self.current_smoothed_data is None:
            self.show_smoothed_var.set(False)
            return
        self.find_peak_candidates()
        self.redraw_plots()

    def save_smoothed_spectrum(self):
        """Export spectrum data (raw or smoothed) for current, selected, or all samples.

        Prompts user to choose scope: current sample, selected samples, or all.
        If multiple samples, saves each as a separate CSV file in a chosen directory.
        """
        if self.df is None or len(self.df) == 0:
            messagebox.showwarning("No Data", "No samples loaded.")
            return

        # Determine selected indices
        sel_indices = self._get_selected_indices()
        # Build choice dialog
        choices = ["Current sample only"]
        if len(sel_indices) > 1:
            choices.append(f"Selected samples ({len(sel_indices)})")
        choices.append(f"All samples ({len(self.df)})")

        if len(choices) == 1:
            scope = 'current'
        else:
            choice_win = tk.Toplevel(self.root)
            choice_win.title("Export Spectrum Data")
            choice_win.transient(self.root)
            choice_win.grab_set()
            choice_win.resizable(False, False)
            ttk.Label(choice_win, text="Export spectrum data for:",
                      font=('TkDefaultFont', 10, 'bold')).pack(padx=10, pady=(10, 5))
            scope_var = tk.StringVar(value='current')
            for i, label in enumerate(choices):
                val = ['current', 'selected', 'all'][i] if len(sel_indices) > 1 \
                    else ['current', 'all'][i]
                ttk.Radiobutton(choice_win, text=label,
                                variable=scope_var, value=val
                                ).pack(anchor='w', padx=20, pady=2)
            result_holder = [None]

            def on_ok():
                result_holder[0] = scope_var.get()
                choice_win.destroy()

            def on_cancel():
                choice_win.destroy()

            btn_f = ttk.Frame(choice_win)
            btn_f.pack(padx=10, pady=10)
            ttk.Button(btn_f, text="OK", command=on_ok).pack(side=tk.LEFT, padx=4)
            ttk.Button(btn_f, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=4)
            choice_win.wait_window()
            scope = result_holder[0]
            if scope is None:
                return

        # Determine which indices to export
        if scope == 'current':
            export_indices = [self.current_sample_index]
        elif scope == 'selected':
            export_indices = sel_indices
        else:
            export_indices = list(range(len(self.df)))

        if len(export_indices) == 1:
            # Single file: use save-as dialog
            idx = export_indices[0]
            sample_name = self.df.iloc[idx].get('sample_name', 'unknown')
            stored_sigma = self.df.iloc[idx].get('smoothing_sigma', 0.0)
            if pd.isna(stored_sigma):
                stored_sigma = 0.0
            sigma_str = f"_smoothed_sigma{float(stored_sigma):.1f}" \
                if stored_sigma > 0 else ""
            default_fn = f"{sample_name}{sigma_str}.csv"
            fp = filedialog.asksaveasfilename(
                defaultextension=".csv", initialfile=default_fn,
                initialdir=self._last_dir,
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Save Spectrum Data As")
            if not fp:
                return
            self._last_dir = os.path.dirname(fp)
            try:
                self._export_single_spectrum(idx, fp)
            except ValueError as e:
                messagebox.showwarning("Export Failed", str(e))
        else:
            # Multiple files: choose output directory
            out_dir = filedialog.askdirectory(
                title=f"Choose folder to save {len(export_indices)} spectrum files",
                initialdir=self._last_dir)
            if not out_dir:
                return
            self._last_dir = out_dir
            saved = 0
            for idx in export_indices:
                sample_name = self.df.iloc[idx].get('sample_name', 'unknown')
                # Sanitize filename
                safe_name = "".join(c if c.isalnum() or c in '-_. ' else '_'
                                    for c in sample_name)
                fp = os.path.join(out_dir, f"{safe_name}_spectrum.csv")
                try:
                    self._export_single_spectrum(idx, fp)
                    saved += 1
                except Exception:
                    pass
            self._set_status(
                f"Exported {saved}/{len(export_indices)} spectrum files to {out_dir}")
            messagebox.showinfo(
                "Export Complete",
                f"Exported {saved}/{len(export_indices)} spectrum files to:\n{out_dir}")

    def _export_single_spectrum(self, idx, filepath):
        """Export a single sample's spectrum data (energy + intensity) to CSV."""
        original_idx = self.current_sample_index
        row = self.df.iloc[idx]
        sample_name = row.get('sample_name', 'unknown')

        # Load spectrum data for this sample
        self.current_sample_index = idx
        self.load_spectrum_data(row)

        # Restore smoothing if needed
        stored_sigma = row.get('smoothing_sigma', 0.0)
        if pd.isna(stored_sigma):
            stored_sigma = 0.0
        stored_sigma = float(stored_sigma)

        if stored_sigma > 0 and self.original_spectrum_data is not None:
            from scipy.ndimage import gaussian_filter1d as _gf1d
            smoothed = _gf1d(
                self.original_spectrum_data['intensity'],
                sigma=stored_sigma)
            en = self.original_spectrum_data['energy']
            inten = smoothed
        elif self.current_spectrum_data is not None:
            en = self.current_spectrum_data['energy']
            inten = self.current_spectrum_data['intensity']
        else:
            self.current_sample_index = original_idx
            raise ValueError(f"No spectrum data for {sample_name}")

        with open(filepath, 'w') as f:
            f.write(f"# Saved by IronLPeaks\n")
            f.write(f"# Sample: {sample_name}\n")
            f.write(f"# Smoothing sigma: {stored_sigma:.2f}\n")
            f.write(f"# Data points: {len(en)}\n")
            f.write("# energy, intensity\n")
            for e_val, i_val in zip(en, inten):
                f.write(f" {e_val:.6f},  {i_val:.6f}\n")

        self.current_sample_index = original_idx

    def on_smoothing_change(self, event=None):
        try:
            self.smoothing_var.get()
            self.min_peak_height_var.get()
        except tk.TclError:
            self.smoothing_var.set(0.0)
            self.min_peak_height_var.set(0.1)
        self.detect_peaks()
        self.redraw_plots()

    # ------------------------------------------------------------------
    # Peak actions
    # ------------------------------------------------------------------
    def _get_selected_indices(self):
        """Return listbox selection indices, or [current_sample_index] if only one."""
        indices = list(self.sample_listbox.curselection())
        if not indices:
            indices = [self.current_sample_index]
        return indices

    def detect_peaks(self):
        """Detect peaks for the current sample only (called internally)."""
        if self.current_spectrum_data is not None:
            self.find_peak_candidates()
            self.redraw_plots()

    def detect_peaks_selected(self):
        """Detect peaks for all selected samples in the listbox."""
        indices = self._get_selected_indices()
        if len(indices) <= 1:
            self.detect_peaks()
            return
        original_idx = self.current_sample_index
        processed = 0
        for idx in indices:
            try:
                self.current_sample_index = idx
                row = self.df.iloc[idx]
                self.load_spectrum_data(row)
                if self.current_spectrum_data is None:
                    continue  # spectrum unavailable — skip
                # Analyze this sample's own smoothing state, not the
                # previously displayed sample's
                self._restore_row_smoothing(row)
                self.current_baseline = None
                self.current_baseline_fitter = None
                self._baseline_params_are_manual = False
                self.offset_var.set("auto")
                self.slope_var.set("auto")
                self.step_height_var.set("auto")
                self.edge_width_var.set("auto")
                self.edge_position_var.set("auto")
                self.edge_l2_position_var.set("auto")
                self.fit_baseline()
                self.find_peak_candidates()
                processed += 1
            except Exception:
                pass
        self.current_sample_index = original_idx
        self.load_current_sample()
        self._set_status(f"Detected peaks for {processed}/{len(indices)} samples")

    def clear_peaks(self):
        """Clear peaks for the current sample only."""
        self.selected_peaks = []
        if self.df is not None and len(self.df) > 0:
            idx = self.current_sample_index
            for col in ('peak1_energy', 'peak2_energy', 'peak1_intensity',
                        'peak2_intensity', 'delta_ev', 'intensity_ratio'):
                self.df.at[idx, col] = np.nan
        self.update_peak_analysis_display()
        if hasattr(self, 'peak_tree'):
            self.refresh_peak_analysis()
        if hasattr(self, 'analysis_ax'):
            self.update_analysis_plot()
        self.redraw_plots()

    def clear_peaks_selected(self):
        """Clear peaks for all selected samples in the listbox."""
        indices = self._get_selected_indices()
        if len(indices) <= 1:
            self.clear_peaks()
            return
        for idx in indices:
            for col in ('peak1_energy', 'peak2_energy', 'peak1_intensity',
                        'peak2_intensity', 'delta_ev', 'intensity_ratio'):
                self.df.at[idx, col] = np.nan
        # Reset the current sample's visual state
        self.selected_peaks = []
        self.update_peak_analysis_display()
        if hasattr(self, 'peak_tree'):
            self.refresh_peak_analysis()
        if hasattr(self, 'analysis_ax'):
            self.update_analysis_plot()
        self.redraw_plots()
        self._set_status(f"Cleared peaks for {len(indices)} samples")

    def auto_select_peaks_selected(self):
        """Auto-select top 2 peaks for selected samples (or all if only 1 selected)."""
        if self.df is None or len(self.df) == 0:
            messagebox.showwarning("Warning", "No samples to process")
            return

        indices = self._get_selected_indices()
        # If only 1 selected, offer to do all
        if len(indices) <= 1:
            result = messagebox.askyesno(
                "Auto-Select Peaks",
                f"Auto-select top 2 peaks in the detection window "
                f"({self.window_start:.0f}\u2013{self.window_end:.0f} eV) "
                f"for all {len(self.df)} samples?\n\n"
                f"Tip: Multi-select samples first to process only those.\n\n"
                f"Existing selections will be overwritten.")
            if not result:
                return
            indices = list(range(len(self.df)))
        else:
            result = messagebox.askyesno(
                "Auto-Select Peaks",
                f"Auto-select top 2 peaks for {len(indices)} selected samples "
                f"in window ({self.window_start:.0f}\u2013"
                f"{self.window_end:.0f} eV)?\n\n"
                f"Existing selections will be overwritten.")
            if not result:
                return

        original_idx = self.current_sample_index
        success = 0
        missing = []
        skipped = []
        baseline_failed = []
        for idx in indices:
            try:
                self.current_sample_index = idx
                row = self.df.iloc[idx]
                self.load_spectrum_data(row)
                if self.current_spectrum_data is None:
                    missing.append(str(row.get('sample_name', idx)))
                    continue
                # Analyze this sample's own smoothing state, not the
                # previously displayed sample's
                self._restore_row_smoothing(row)
                # Honor stored (possibly manually tuned) baseline params;
                # auto-fit only when none are stored
                bl_offset = row.get('baseline_offset')
                self._baseline_params_are_manual = not pd.isna(bl_offset)
                if pd.isna(bl_offset):
                    self.offset_var.set("auto")
                    self.slope_var.set("auto")
                    self.step_height_var.set("auto")
                    self.edge_width_var.set("auto")
                    self.edge_position_var.set("auto")
                    self.edge_l2_position_var.set("auto")
                else:
                    self.offset_var.set(f"{bl_offset:.6f}")
                    self.slope_var.set(f"{row.get('baseline_slope', 0):.6f}")
                    self.step_height_var.set(f"{row.get('baseline_step', 0):.6f}")
                    self.edge_width_var.set(f"{row.get('baseline_width', 0):.6f}")
                    self.edge_position_var.set(f"{row.get('baseline_edge', 710):.3f}")
                    bl_l2 = row.get('baseline_edge_l2', np.nan)
                    self.edge_l2_position_var.set(
                        f"{bl_l2:.3f}" if not pd.isna(bl_l2) else "auto")
                self.current_baseline = None
                self.current_baseline_fitter = None
                self.fit_baseline()
                if self.current_baseline is None:
                    # Baseline failed: peak intensities/ratios from the
                    # un-subtracted spectrum would not be comparable to the
                    # rest of the batch
                    baseline_failed.append(str(row.get('sample_name', idx)))
                    continue
                self.find_peak_candidates()

                if len(self.peak_candidates) >= 2:
                    energy, inten = self._get_active_spectrum()
                    if self.current_baseline is not None:
                        inten = inten - self.current_baseline
                    pi = [(pe, np.interp(pe, energy, inten))
                          for pe in self.peak_candidates]
                    pi.sort(key=lambda x: x[1], reverse=True)

                    # Validate separation: try top pair first, then
                    # alternative pairs if separation is unphysical
                    best_pair = None
                    for i in range(len(pi)):
                        for j in range(i + 1, len(pi)):
                            sep = abs(pi[i][0] - pi[j][0])
                            if (_FE_L3_MIN_SEPARATION <= sep
                                    <= _FE_L3_MAX_SEPARATION):
                                best_pair = (pi[i][0], pi[j][0])
                                break
                        if best_pair is not None:
                            break

                    if best_pair is not None:
                        self.selected_peaks = list(best_pair)
                        self.auto_calculate_peaks()
                        success += 1
                    else:
                        # No pair within the physical 0.8–3.5 eV L3
                        # splitting — leave unselected rather than store
                        # an unphysical pair
                        skipped.append(str(row.get('sample_name', idx)))
            except Exception:
                pass

        self.current_sample_index = original_idx
        self.load_current_sample()
        if hasattr(self, 'peak_tree'):
            self.refresh_peak_analysis()
        if hasattr(self, 'analysis_ax'):
            self.update_analysis_plot()
        self._set_status(f"Auto-selected peaks for {success}/{len(indices)} samples")
        msg = f"Auto-selected peaks for {success}/{len(indices)} samples."
        if baseline_failed:
            msg += (f"\n\n{len(baseline_failed)} sample(s) failed baseline "
                    f"fitting and were skipped (raw-spectrum intensities "
                    f"would not be comparable):\n" + "\n".join(baseline_failed))
        if skipped:
            msg += (f"\n\n{len(skipped)} sample(s) had no peak pair within "
                    f"{_FE_L3_MIN_SEPARATION}–{_FE_L3_MAX_SEPARATION} eV and "
                    f"were left unselected for manual review:\n"
                    + "\n".join(skipped))
        if missing:
            msg += (f"\n\n{len(missing)} sample(s) skipped — spectrum file "
                    f"not found (re-add via File → Open Files):\n"
                    + "\n".join(missing))
        messagebox.showinfo("Done", msg)

    # Keep old name as alias
    auto_select_all_peaks = auto_select_peaks_selected

    def auto_calculate_peaks(self):
        energy, inten = self._get_active_spectrum()
        if energy is None or len(self.selected_peaks) < 2:
            return
        sorted_peaks = sorted(self.selected_peaks[:2])
        p1e, p2e = sorted_peaks
        if self.current_baseline is not None:
            inten = inten - self.current_baseline
        p1i = np.interp(p1e, energy, inten)
        p2i = np.interp(p2e, energy, inten)
        delta = p2e - p1e
        ratio = p1i / p2i if p2i != 0 else np.nan

        idx = self.current_sample_index
        self.df.at[idx, 'peak1_energy'] = p1e
        self.df.at[idx, 'peak2_energy'] = p2e
        self.df.at[idx, 'peak1_intensity'] = p1i
        self.df.at[idx, 'peak2_intensity'] = p2i
        self.df.at[idx, 'delta_ev'] = delta
        self.df.at[idx, 'intensity_ratio'] = ratio

        if hasattr(self, 'peak_tree'):
            self.refresh_peak_analysis()
        if hasattr(self, 'analysis_ax'):
            self.update_analysis_plot()

    def update_peak_analysis_display(self):
        if not hasattr(self, 'peak_analysis_label'):
            return
        if len(self.selected_peaks) == 0:
            self.peak_analysis_label.config(
                text="Select 2 peaks to see analysis", foreground="gray")
        elif len(self.selected_peaks) == 1:
            self.peak_analysis_label.config(
                text=f"Peak 1: {self.selected_peaks[0]:.3f} eV\nSelect second peak",
                foreground=_ACCENT)
        else:
            p1 = min(self.selected_peaks[0], self.selected_peaks[1])
            p2 = max(self.selected_peaks[0], self.selected_peaks[1])
            delta = p2 - p1
            en, inten = self._get_active_spectrum()
            if en is not None:
                if self.current_baseline is not None:
                    inten = inten - self.current_baseline
                i1 = np.interp(p1, en, inten)
                i2 = np.interp(p2, en, inten)
                # Same quantity as stored/exported intensity_ratio (p1i/p2i)
                ratio_txt = (f"Ratio (Low E / High E): {i1/i2:.3f}"
                             if i2 != 0 else "Ratio: N/A")
                txt = (f"Peak 1: {p1:.3f} eV\n"
                       f"Peak 2: {p2:.3f} eV\n"
                       f"\u0394 eV: {delta:.3f}\n{ratio_txt}")
                self.peak_analysis_label.config(text=txt, foreground=_SUCCESS)

    # ------------------------------------------------------------------
    # van Aken Fe³⁺/ΣFe quantification
    # ------------------------------------------------------------------
    def _save_vanaken_to_df(self, idx, results):
        """Save van Aken results to DataFrame row at idx."""
        self.df.at[idx, 'vanaken_fe3'] = results['fe3']
        self.df.at[idx, 'vanaken_ratio'] = results['ratio']
        self.df.at[idx, 'vanaken_p'] = results['p_value']
        self.df.at[idx, 'vanaken_il3'] = results['il3']
        self.df.at[idx, 'vanaken_il2'] = results['il2']
        self.df.at[idx, 'vanaken_l3_lo'] = results['l3_lo']
        self.df.at[idx, 'vanaken_l3_hi'] = results['l3_hi']
        self.df.at[idx, 'vanaken_l2_lo'] = results['l2_lo']
        self.df.at[idx, 'vanaken_l2_hi'] = results['l2_hi']
        self.df.at[idx, 'vanaken_l3_centroid'] = results['l3_centroid']

    def compute_vanaken(self):
        """Run van Aken & Liebscher (2002) quantification on the current sample."""
        if VanAkenFeQuantifier is None:
            messagebox.showerror("Error",
                                 "van_aken_fe_quantification module not found.")
            return
        if self.current_spectrum_data is None:
            messagebox.showwarning("No Data", "Load a spectrum first.")
            return

        energy, intensity = self._get_active_spectrum()
        if energy is None:
            return

        # Baseline correction — required. Never quantify the un-subtracted
        # spectrum: the continuum corrupts the L3'/L2' integrals.
        if self.current_baseline is None:
            messagebox.showwarning(
                "No Baseline",
                "Fit a baseline first — van Aken quantification requires "
                "the continuum-subtracted spectrum.")
            return
        intensity = intensity - self.current_baseline

        # Read GUI parameters
        try:
            l3_lo = self.vanaken_l3_lo_var.get()
        except (tk.TclError, ValueError):
            l3_lo = VanAkenFeQuantifier.DEFAULT_L3_WINDOW[0]
        try:
            l3_hi = self.vanaken_l3_hi_var.get()
        except (tk.TclError, ValueError):
            l3_hi = VanAkenFeQuantifier.DEFAULT_L3_WINDOW[1]
        try:
            l2_lo = self.vanaken_l2_lo_var.get()
        except (tk.TclError, ValueError):
            l2_lo = VanAkenFeQuantifier.DEFAULT_L2_WINDOW[0]
        try:
            l2_hi = self.vanaken_l2_hi_var.get()
        except (tk.TclError, ValueError):
            l2_hi = VanAkenFeQuantifier.DEFAULT_L2_WINDOW[1]

        quantifier = VanAkenFeQuantifier(
            energy, intensity,
            l3_window=(l3_lo, l3_hi),
            l2_window=(l2_lo, l2_hi))
        results = quantifier.compute_all()

        self.current_vanaken = quantifier

        # Save to DataFrame
        if self.df is not None and len(self.df) > 0:
            self._save_vanaken_to_df(self.current_sample_index, results)

        self._update_vanaken_display()
        self._refresh_vanaken_sample_list()
        self._update_vanaken_info_panel()
        self.redraw_vanaken_plot()
        self.redraw_plots()

        fe3_str = f"{results['fe3']:.3f}" if results['fe3'] is not None and not np.isnan(results['fe3']) else "N/A"
        cen_str = f"{results['l3_centroid']:.2f}" if results['l3_centroid'] is not None and not np.isnan(results['l3_centroid']) else "N/A"
        self._set_status(
            f"van Aken: Fe\u00b3\u207a/\u03a3Fe = {fe3_str}, "
            f"L3 centroid = {cen_str} eV")

    def compute_vanaken_and_next(self):
        """Compute van Aken for current sample, then advance to next."""
        self.compute_vanaken()
        self.next_sample()

    def clear_vanaken_results(self):
        """Clear van Aken analysis for the current sample."""
        self.current_vanaken = None
        if self.df is not None and len(self.df) > 0:
            idx = self.current_sample_index
            for col in ('vanaken_fe3', 'vanaken_ratio',
                        'vanaken_p', 'vanaken_il3',
                        'vanaken_il2', 'vanaken_l3_lo',
                        'vanaken_l3_hi', 'vanaken_l2_lo',
                        'vanaken_l2_hi', 'vanaken_l3_centroid'):
                self.df.at[idx, col] = np.nan
        self._update_vanaken_display()
        self._refresh_vanaken_sample_list()
        self._update_vanaken_info_panel()
        self.redraw_vanaken_plot()
        self.redraw_plots()
        self._set_status("van Aken results cleared")

    def _update_vanaken_display(self):
        """Update the van Aken results label in the tab."""
        if not hasattr(self, 'vanaken_result_label'):
            return

        q = self.current_vanaken
        if q is not None and q.fe3 is not None:
            results = q.get_results()
            lines = []
            fe3 = results['fe3']
            ratio = results['ratio']
            p_val = results['p_value']
            centroid = results['l3_centroid']

            fe3_str = f"Fe\u00b3\u207a/\u03a3Fe = {fe3:.3f}" if fe3 is not None and not np.isnan(fe3) else "Fe\u00b3\u207a/\u03a3Fe = N/A"
            ratio_str = f"I(L3')/I(L2') = {ratio:.3f}" if ratio is not None and not np.isnan(ratio) else ""
            p_str = f"p = {p_val:.4f}" if p_val is not None and not np.isnan(p_val) else ""
            lines.append(f"Modified integral: {fe3_str}  {ratio_str}  {p_str}")
            lines.append(f"  I(L3') = {results['il3']:.4f},  I(L2') = {results['il2']:.4f}")

            cen_str = f"L3 centroid = {centroid:.2f} eV" if centroid is not None and not np.isnan(centroid) else "L3 centroid = N/A"
            lines.append(cen_str)

            self.vanaken_result_label.config(
                text="\n".join(lines), foreground=_SUCCESS)
            return

        # Check DataFrame for stored results
        if self.df is not None and len(self.df) > 0:
            row = self.df.iloc[self.current_sample_index]
            fe3 = row.get('vanaken_fe3')
            if not pd.isna(fe3):
                lines = []
                ratio = row.get('vanaken_ratio', np.nan)
                ratio_str = f"I(L3')/I(L2') = {ratio:.3f}" if not pd.isna(ratio) else ""
                lines.append(f"Modified integral: Fe\u00b3\u207a/\u03a3Fe = {fe3:.3f}  {ratio_str}")

                centroid = row.get('vanaken_l3_centroid', np.nan)
                if not pd.isna(centroid):
                    lines.append(f"L3 centroid = {centroid:.2f} eV")

                self.vanaken_result_label.config(
                    text="\n".join(lines), foreground=_SUCCESS)
                return

        self.vanaken_result_label.config(
            text="No van Aken computation. Select a sample and click "
            "'Compute Fe\u00b3\u207a/\u03a3Fe' to begin.",
            foreground="gray")

    def redraw_vanaken_plot(self):
        """Redraw the van Aken tab plot with integration windows."""
        if not hasattr(self, 'vanaken_ax'):
            return
        self.vanaken_ax.clear()

        quantifier = self.current_vanaken

        # Get baseline-corrected spectrum for display
        has_spectrum = False
        if self.current_spectrum_data is not None:
            energy, intensity = self._get_active_spectrum()
            if energy is not None:
                if self.current_baseline is not None:
                    intensity = intensity - self.current_baseline
                has_spectrum = True

        if not has_spectrum:
            self.vanaken_ax.text(
                0.5, 0.5, "No spectrum loaded",
                ha='center', va='center',
                transform=self.vanaken_ax.transAxes,
                fontsize=12, color='gray')
            self.vanaken_canvas.draw()
            return

        # Plot the baseline-corrected spectrum
        self.vanaken_ax.plot(energy, intensity, 'k-', linewidth=1.2,
                               label='Baseline-corrected', alpha=0.8)

        if quantifier is not None and quantifier.fe3 is not None:
            # L3' integration window (red shading)
            l3_lo, l3_hi = quantifier.l3_window
            l3_mask = (energy >= l3_lo) & (energy <= l3_hi)
            if np.any(l3_mask):
                self.vanaken_ax.fill_between(
                    energy[l3_mask], 0, intensity[l3_mask],
                    alpha=0.30, color='#e65100',
                    label=f"L3' window ({l3_lo:.1f}\u2013{l3_hi:.1f} eV)")

            # L2' integration window (blue shading)
            l2_lo, l2_hi = quantifier.l2_window
            l2_mask = (energy >= l2_lo) & (energy <= l2_hi)
            if np.any(l2_mask):
                self.vanaken_ax.fill_between(
                    energy[l2_mask], 0, intensity[l2_mask],
                    alpha=0.30, color='#1565c0',
                    label=f"L2' window ({l2_lo:.1f}\u2013{l2_hi:.1f} eV)")

            # L3 centroid marker (green dashed line)
            if quantifier.l3_centroid is not None and not np.isnan(quantifier.l3_centroid):
                self.vanaken_ax.axvline(
                    x=quantifier.l3_centroid, color='#2e7d32',
                    linestyle='--', linewidth=1.2, alpha=0.8,
                    label=f'L3 centroid ({quantifier.l3_centroid:.2f} eV)')

            # Text annotation
            results = quantifier.get_results()
            fe3 = results['fe3']
            fe3_s = f"{fe3:.3f}" if fe3 is not None and not np.isnan(fe3) else "N/A"
            cen = results['l3_centroid']
            cen_s = f"{cen:.2f}" if cen is not None and not np.isnan(cen) else "N/A"
            ann_text = (f"Fe\u00b3\u207a/\u03a3Fe = {fe3_s}\n"
                        f"L3 centroid = {cen_s} eV")
            self.vanaken_ax.text(
                0.02, 0.95, ann_text,
                transform=self.vanaken_ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='lightyellow', alpha=0.9,
                          edgecolor='gray'))
        else:
            self.vanaken_ax.text(
                0.5, 0.95, "No van Aken computation for this sample",
                ha='center', va='top',
                transform=self.vanaken_ax.transAxes,
                fontsize=10, color='gray')

        self.vanaken_ax.set_xlabel('Energy (eV)')
        self.vanaken_ax.set_ylabel('Intensity (a.u.)')
        sample_name = ""
        if self.df is not None and len(self.df) > 0:
            sample_name = self.df.iloc[self.current_sample_index].get(
                'sample_name', '')
        self.vanaken_ax.set_title(
            f'van Aken Fe\u00b3\u207a/\u03a3Fe \u2014 {sample_name}')
        self.vanaken_ax.legend(loc='upper right', fontsize=7, framealpha=0.7)
        self.vanaken_ax.grid(True, alpha=0.15, which='major')
        self.vanaken_canvas.draw()

    def batch_vanaken(self):
        """Run van Aken quantification on all selected samples (or all if <=1 selected)."""
        if VanAkenFeQuantifier is None:
            messagebox.showerror("Error",
                                 "van_aken_fe_quantification module not found.")
            return
        if self.df is None or len(self.df) == 0:
            messagebox.showwarning("Warning", "No samples to process")
            return

        indices = self._get_selected_indices()
        if len(indices) <= 1:
            result = messagebox.askyesno(
                "Batch van Aken",
                f"Run van Aken Fe\u00b3\u207a/\u03a3Fe for all "
                f"{len(self.df)} samples?\n\n"
                f"Requires baseline to be fitted for each sample.\n"
                f"Existing van Aken results will be overwritten.")
            if not result:
                return
            indices = list(range(len(self.df)))
        else:
            result = messagebox.askyesno(
                "Batch van Aken",
                f"Run van Aken Fe\u00b3\u207a/\u03a3Fe for "
                f"{len(indices)} selected samples?\n\n"
                f"Existing van Aken results will be overwritten.")
            if not result:
                return

        # Read GUI parameters
        try:
            l3_lo = self.vanaken_l3_lo_var.get()
        except (tk.TclError, ValueError):
            l3_lo = VanAkenFeQuantifier.DEFAULT_L3_WINDOW[0]
        try:
            l3_hi = self.vanaken_l3_hi_var.get()
        except (tk.TclError, ValueError):
            l3_hi = VanAkenFeQuantifier.DEFAULT_L3_WINDOW[1]
        try:
            l2_lo = self.vanaken_l2_lo_var.get()
        except (tk.TclError, ValueError):
            l2_lo = VanAkenFeQuantifier.DEFAULT_L2_WINDOW[0]
        try:
            l2_hi = self.vanaken_l2_hi_var.get()
        except (tk.TclError, ValueError):
            l2_hi = VanAkenFeQuantifier.DEFAULT_L2_WINDOW[1]

        original_idx = self.current_sample_index
        success = 0
        missing = []
        baseline_failed = []
        for idx in indices:
            try:
                self.current_sample_index = idx
                row = self.df.iloc[idx]
                self.load_spectrum_data(row)
                if self.current_spectrum_data is None:
                    missing.append(str(row.get('sample_name', idx)))
                    continue

                # Restore smoothing for the sample being processed
                self._restore_row_smoothing(row)

                # Use stored (possibly manually tuned) baseline params when
                # present; only auto-fit when none are stored
                bl_offset = row.get('baseline_offset')
                self._baseline_params_are_manual = not pd.isna(bl_offset)
                if pd.isna(bl_offset):
                    self.offset_var.set("auto")
                    self.slope_var.set("auto")
                    self.step_height_var.set("auto")
                    self.edge_width_var.set("auto")
                    self.edge_position_var.set("auto")
                    self.edge_l2_position_var.set("auto")
                else:
                    self.offset_var.set(f"{bl_offset:.6f}")
                    self.slope_var.set(f"{row.get('baseline_slope', 0):.6f}")
                    self.step_height_var.set(f"{row.get('baseline_step', 0):.6f}")
                    self.edge_width_var.set(f"{row.get('baseline_width', 0):.6f}")
                    self.edge_position_var.set(f"{row.get('baseline_edge', 710):.3f}")
                    bl_l2 = row.get('baseline_edge_l2', np.nan)
                    self.edge_l2_position_var.set(
                        f"{bl_l2:.3f}" if not pd.isna(bl_l2) else "auto")
                self.current_baseline = None
                self.current_baseline_fitter = None
                self.fit_baseline()

                if self.current_baseline is None:
                    # Baseline fit failed — never quantify the
                    # un-subtracted spectrum; mark the result as failed
                    for col in ('vanaken_fe3', 'vanaken_ratio', 'vanaken_p',
                                'vanaken_il3', 'vanaken_il2',
                                'vanaken_l3_centroid'):
                        self.df.at[idx, col] = np.nan
                    baseline_failed.append(str(row.get('sample_name', idx)))
                    continue

                energy, intensity = self._get_active_spectrum()
                if energy is None:
                    continue
                intensity = intensity - self.current_baseline

                quantifier = VanAkenFeQuantifier(
                    energy, intensity,
                    l3_window=(l3_lo, l3_hi),
                    l2_window=(l2_lo, l2_hi))
                results = quantifier.compute_all()
                self._save_vanaken_to_df(idx, results)
                success += 1
            except Exception:
                pass

        self.current_sample_index = original_idx
        self.load_current_sample()
        self._refresh_vanaken_sample_list()
        self._update_vanaken_info_panel()
        status = f"van Aken completed for {success}/{len(indices)} samples"
        if baseline_failed:
            status += f", {len(baseline_failed)} baseline failure(s)"
        if missing:
            status += f", {len(missing)} missing spectrum file(s)"
        self._set_status(status)
        msg = (f"van Aken Fe\u00b3\u207a/\u03a3Fe completed for "
               f"{success}/{len(indices)} samples.")
        if baseline_failed:
            msg += (f"\n\n{len(baseline_failed)} sample(s) failed baseline "
                    f"fitting (results set to N/A, not quantified):\n"
                    + "\n".join(baseline_failed))
        if missing:
            msg += (f"\n\n{len(missing)} sample(s) skipped \u2014 spectrum file "
                    f"not found (re-add via File \u2192 Open Files):\n"
                    + "\n".join(missing))
        messagebox.showinfo("Done", msg)

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------
    def on_include_change(self):
        if self.df is not None and len(self.df) > 0:
            self.df.at[self.current_sample_index, 'include_in_report'] = self.include_var.get()
            self.populate_sample_list()
            self.sample_listbox.selection_set(self.current_sample_index)

    def on_window_change(self, event=None):
        self.window_start = self.window_start_var.get()
        self.window_end = self.window_end_var.get()
        self.find_peak_candidates()
        self.redraw_plots()

    def on_classification_change(self, event=None):
        if self.df is None or len(self.df) == 0:
            return
        group = self.sample_group_var.get().strip()
        idx = self.current_sample_index
        self.df.at[idx, 'sample_group'] = group
        self.populate_sample_list(maintain_selection=True)
        if hasattr(self, 'peak_tree'):
            self.refresh_peak_analysis()
        if hasattr(self, 'analysis_ax'):
            self.update_analysis_plot()
            self.analysis_canvas.draw_idle()
        self._refresh_group_choices()

    def apply_classification_to_selected(self):
        """Apply the current Sample Group value to every selected sample."""
        if self.df is None or len(self.df) == 0:
            return
        indices = list(self.sample_listbox.curselection())
        if not indices:
            messagebox.showwarning("No Selection",
                                   "Select one or more samples to apply group.")
            return
        group = self.sample_group_var.get().strip()
        if len(indices) > 1:
            if not messagebox.askyesno(
                "Apply to Multiple",
                f"Set Sample Group = '{group or '(none)'}' "
                f"for {len(indices)} samples?"):
                return
        for i in indices:
            self.df.at[i, 'sample_group'] = group
        self.populate_sample_list(maintain_selection=True)
        self.sample_listbox.selection_clear(0, tk.END)
        for i in indices:
            self.sample_listbox.selection_set(i)
        if hasattr(self, 'peak_tree'):
            self.refresh_peak_analysis()
        if hasattr(self, 'analysis_ax'):
            self.update_analysis_plot()
            self.analysis_canvas.draw_idle()
        self._refresh_group_choices()
        self._set_status(
            f"Applied group '{group or '(none)'}' to {len(indices)} sample(s)")

    def clear_classification(self):
        """Clear the Sample Group for all selected samples."""
        if self.df is None or len(self.df) == 0:
            return
        indices = self._get_selected_indices()
        self.sample_group_var.set("")
        for i in indices:
            self.df.at[i, 'sample_group'] = ""
        self.populate_sample_list(maintain_selection=True)
        self.sample_listbox.selection_clear(0, tk.END)
        for i in indices:
            self.sample_listbox.selection_set(i)
        if hasattr(self, 'peak_tree'):
            self.refresh_peak_analysis()
        if hasattr(self, 'analysis_ax'):
            self.update_analysis_plot()
            self.analysis_canvas.draw_idle()
        self._refresh_group_choices()
        if len(indices) > 1:
            self._set_status(f"Cleared group for {len(indices)} samples")

    def _refresh_group_choices(self):
        """Update the Sample Group combobox dropdown with current unique groups."""
        if not hasattr(self, 'sample_group_combo') or self.df is None:
            return
        try:
            groups = sorted({str(g).strip() for g in self.df['sample_group'].dropna()
                             if str(g).strip()})
        except KeyError:
            groups = []
        self.sample_group_combo.configure(values=groups)

    def rename_selected_samples(self):
        """Rename all selected samples using a base name + optional numbering."""
        if self.df is None or len(self.df) == 0:
            return
        indices = self._get_selected_indices()
        if not indices:
            messagebox.showwarning("No Selection",
                                   "Select one or more samples to rename.")
            return

        # Popup dialog asking for the base name
        dialog = tk.Toplevel(self.root)
        dialog.title("Rename Selected Samples")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)

        # Center on parent
        dialog.geometry("+%d+%d" % (
            self.root.winfo_rootx() + 200,
            self.root.winfo_rooty() + 200))

        ttk.Label(dialog, text=f"Rename {len(indices)} selected sample(s)",
                  font=('TkDefaultFont', 10, 'bold')).pack(padx=16, pady=(12, 4))

        ttk.Label(dialog, text="Base name:").pack(anchor='w', padx=16)
        name_var = tk.StringVar(value="")
        name_entry = ttk.Entry(dialog, textvariable=name_var, width=30)
        name_entry.pack(padx=16, pady=(2, 8))
        name_entry.focus_set()

        number_var = tk.BooleanVar(value=len(indices) > 1)
        ttk.Checkbutton(dialog, text="Add sequential numbers (e.g. Sample_01, Sample_02)",
                        variable=number_var).pack(anchor='w', padx=16, pady=(0, 4))

        start_var = tk.IntVar(value=1)
        num_frame = ttk.Frame(dialog)
        num_frame.pack(fill=tk.X, padx=16, pady=(0, 8))
        ttk.Label(num_frame, text="Start numbering at:").pack(side=tk.LEFT)
        ttk.Entry(num_frame, textvariable=start_var, width=5).pack(side=tk.LEFT, padx=(4, 0))

        sep_var = tk.StringVar(value="_")
        sep_frame = ttk.Frame(dialog)
        sep_frame.pack(fill=tk.X, padx=16, pady=(0, 8))
        ttk.Label(sep_frame, text="Separator:").pack(side=tk.LEFT)
        for sep_label, sep_val in [("_", "_"), ("-", "-"), (" ", " "), ("none", "")]:
            ttk.Radiobutton(sep_frame, text=sep_label, variable=sep_var,
                            value=sep_val).pack(side=tk.LEFT, padx=(4, 0))

        preview_var = tk.StringVar(value="")
        preview_label = ttk.Label(dialog, textvariable=preview_var,
                                  foreground='gray', wraplength=300)
        preview_label.pack(padx=16, pady=(0, 8))

        def update_preview(*_):
            base = name_var.get().strip()
            if not base:
                preview_var.set("")
                return
            if number_var.get() and len(indices) > 1:
                sep = sep_var.get()
                start = start_var.get()
                n = len(indices)
                width = len(str(start + n - 1))
                examples = [f"{base}{sep}{str(start + i).zfill(width)}"
                            for i in range(min(3, n))]
                if n > 3:
                    examples.append(f"\u2026 {base}{sep}{str(start + n - 1).zfill(width)}")
                preview_var.set("Preview: " + ", ".join(examples))
            else:
                preview_var.set(f"Preview: {base}")

        name_var.trace_add('write', update_preview)
        number_var.trace_add('write', update_preview)
        sep_var.trace_add('write', update_preview)
        start_var.trace_add('write', update_preview)

        result = {'confirmed': False}

        def on_ok(event=None):
            result['confirmed'] = True
            dialog.destroy()

        def on_cancel(event=None):
            dialog.destroy()

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill=tk.X, padx=16, pady=(0, 12))
        ttk.Button(btn_frame, text="Rename", command=on_ok).pack(
            side=tk.RIGHT, padx=(4, 0))
        ttk.Button(btn_frame, text="Cancel", command=on_cancel).pack(
            side=tk.RIGHT)

        name_entry.bind('<Return>', on_ok)
        dialog.bind('<Escape>', on_cancel)
        dialog.wait_window()

        if not result['confirmed']:
            return

        base = name_var.get().strip()
        if not base:
            return

        # Build new names and apply
        new_names = []
        if number_var.get() and len(indices) > 1:
            sep = sep_var.get()
            start = start_var.get()
            width = len(str(start + len(indices) - 1))
            for i in range(len(indices)):
                new_names.append(f"{base}{sep}{str(start + i).zfill(width)}")
        else:
            for i in range(len(indices)):
                new_names.append(base if len(indices) == 1
                                 else f"{base}_{i + 1}")

        for list_idx, new_name in zip(indices, new_names):
            old_name = self.df.at[list_idx, 'sample_name']
            self.df.at[list_idx, 'sample_name'] = new_name
            # Update file path mapping if it exists
            if hasattr(self, 'sample_file_paths') and old_name in self.sample_file_paths:
                self.sample_file_paths[new_name] = self.sample_file_paths.pop(old_name)
            # Re-key the in-memory Athena spectrum so it isn't orphaned
            if hasattr(self, '_athena_spectra') and old_name in self._athena_spectra:
                self._athena_spectra[new_name] = self._athena_spectra.pop(old_name)

        self.populate_sample_list(maintain_selection=True)
        for i in indices:
            self.sample_listbox.selection_set(i)
        self._refresh_analysis_tab()
        self._set_status(f"Renamed {len(indices)} sample(s)")

    # ------------------------------------------------------------------
    # Analysis tab
    # ------------------------------------------------------------------
    def refresh_peak_analysis(self):
        if not hasattr(self, 'peak_tree'):
            return
        for item in self.peak_tree.get_children():
            self.peak_tree.delete(item)
        if self.df is None:
            return
        for idx, row in self.df.iterrows():
            sn = row.get('sample_name', f'Sample_{idx}')
            gr = row.get('sample_group', '')
            p1 = row.get('peak1_energy')
            p2 = row.get('peak2_energy')
            de = row.get('delta_ev')
            ir = row.get('intensity_ratio')
            self.peak_tree.insert('', 'end', values=(
                sn, gr,
                f"{p1:.3f}" if not pd.isna(p1) else "N/A",
                f"{p2:.3f}" if not pd.isna(p2) else "N/A",
                f"{de:.3f}" if not pd.isna(de) else "N/A",
                f"{ir:.3f}" if not pd.isna(ir) else "N/A"))
        self.update_analysis_plot()

    def update_analysis_plot(self):
        """Publication-quality scatter plot for analysis tab (Von der Heyden only)."""
        if not hasattr(self, 'analysis_fig'):
            return

        self.analysis_fig.clear()
        self.analysis_ax = self.analysis_fig.add_subplot(111)

        # Re-create annotation on the new axes
        self.analysis_annot = self.analysis_ax.annotate(
            "", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
            bbox=dict(boxstyle="round", fc="lightyellow", alpha=0.9,
                      edgecolor='gray'),
            arrowprops=dict(arrowstyle="->"))
        self.analysis_annot.set_visible(False)

        valid = self.df.dropna(subset=['delta_ev', 'intensity_ratio'])
        if len(valid) == 0:
            self.analysis_ax.text(
                0.5, 0.5,
                'No peak data available\nSelect 2 peaks for each sample',
                ha='center', va='center',
                transform=self.analysis_ax.transAxes,
                fontsize=12, color='gray')
            self.analysis_ax.set_xlabel('Intensity Ratio (Low E / High E)')
            self.analysis_ax.set_ylabel('\u0394 eV (Peak Separation)')
            self.analysis_ax.set_title('Fe L\u2083 Peak Analysis')
            self.analysis_canvas.draw()
            return

        unique_groups = [g for g in valid['sample_group'].unique() if g]
        cmap = plt.cm.tab10
        color_map = {g: cmap(i) for i, g in enumerate(unique_groups)}
        color_map[''] = '#9e9e9e'

        self.scatter_data = []
        for sg in valid['sample_group'].unique():
            sub = valid[valid['sample_group'] == sg]
            if len(sub) == 0:
                continue
            color = color_map.get(sg, '#9e9e9e')
            label = sg if sg else 'No group'
            first = True
            for _, r in sub.iterrows():
                sc = self.analysis_ax.scatter(
                    r['intensity_ratio'], r['delta_ev'],
                    c=[color], marker='o',
                    label=label if first else "",
                    alpha=0.8, s=120, edgecolors='black', linewidths=0.8)
                self.scatter_data.append((sc, [r['sample_name']]))
                first = False

        self.analysis_ax.set_xlabel('Intensity Ratio (Low E / High E)')
        self.analysis_ax.set_ylabel('\u0394 eV (Peak Separation)')
        self.analysis_ax.set_title('Fe L\u2083 Peak Analysis (Von der Heyden)')
        self.analysis_ax.grid(True, alpha=0.15, which='major')

        handles, labels = self.analysis_ax.get_legend_handles_labels()
        if labels:
            self.analysis_ax.legend(handles, labels, loc='best', fontsize=7,
                                    framealpha=0.8)

        self.analysis_canvas.draw()

    def on_analysis_hover(self, event):
        if event.inaxes != self.analysis_ax:
            if self.analysis_annot.get_visible():
                self.analysis_annot.set_visible(False)
                self.analysis_canvas.draw_idle()
            return
        if not hasattr(self, 'scatter_data') or not self.scatter_data:
            return
        for scatter, names in self.scatter_data:
            cont, ind = scatter.contains(event)
            if cont:
                i = ind["ind"][0]
                if i < len(names):
                    pos = scatter.get_offsets()[i]
                    self.analysis_annot.xy = pos
                    self.analysis_annot.set_text(names[i])
                    self.analysis_annot.set_visible(True)
                    self.analysis_canvas.draw_idle()
                    return
        if self.analysis_annot.get_visible():
            self.analysis_annot.set_visible(False)
            self.analysis_canvas.draw_idle()

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------
    def _add_files_to_session(self, filepaths):
        """Core logic: add a list of file paths to the current session (no duplicates)."""
        if not hasattr(self, 'sample_file_paths'):
            self.sample_file_paths = {}
        if self.selected_files is None:
            self.selected_files = []
        if self.df is None or not isinstance(self.df, pd.DataFrame):
            self.df = pd.DataFrame(columns=[
                'sample_name', 'sample_group',
                'peak1_energy', 'peak2_energy', 'peak1_intensity',
                'peak2_intensity', 'delta_ev', 'intensity_ratio',
                'include_in_report', 'baseline_offset', 'baseline_slope',
                'baseline_step', 'baseline_width', 'baseline_edge',
                'baseline_edge_l2',
                'vanaken_fe3', 'vanaken_ratio',
                'vanaken_p', 'vanaken_il3',
                'vanaken_il2', 'vanaken_l3_lo',
                'vanaken_l3_hi', 'vanaken_l2_lo',
                'vanaken_l2_hi', 'vanaken_l3_centroid'])

        added = 0
        first_new_idx = len(self.df)
        for filepath in filepaths:
            basename = os.path.basename(filepath)
            sample_name = basename
            while True:
                name, ext = os.path.splitext(sample_name)
                if ext.lower() in _SPECTRUM_EXTENSIONS:
                    sample_name = name
                else:
                    break
            if sample_name in self.sample_file_paths:
                # True re-add of the same file: skip. A DIFFERENT file with
                # the same stem (sampleA.csv + sampleA.txt) gets a
                # disambiguated name instead of being silently dropped.
                existing = self.sample_file_paths[sample_name]
                if os.path.normcase(os.path.abspath(str(existing))) == \
                        os.path.normcase(os.path.abspath(str(filepath))):
                    continue
                n = 2
                unique_name = f"{sample_name}_{n}"
                while unique_name in self.sample_file_paths:
                    n += 1
                    unique_name = f"{sample_name}_{n}"
                sample_name = unique_name
            self.selected_files.append(filepath)
            self.sample_file_paths[sample_name] = filepath
            new_row = pd.DataFrame([{
                'sample_name': sample_name,
                'sample_group': "",
                'peak1_energy': np.nan,
                'peak2_energy': np.nan,
                'peak1_intensity': np.nan,
                'peak2_intensity': np.nan,
                'delta_ev': np.nan,
                'intensity_ratio': np.nan,
                'include_in_report': True,
                'smoothing_sigma': 0.0,
                'baseline_offset': np.nan,
                'baseline_slope': np.nan,
                'baseline_step': np.nan,
                'baseline_width': np.nan,
                'baseline_edge': np.nan,
                'baseline_edge_l2': np.nan,
                'vanaken_fe3': np.nan,
                'vanaken_ratio': np.nan,
                'vanaken_p': np.nan,
                'vanaken_il3': np.nan,
                'vanaken_il2': np.nan,
                'vanaken_l3_lo': np.nan,
                'vanaken_l3_hi': np.nan,
                'vanaken_l2_lo': np.nan,
                'vanaken_l2_hi': np.nan,
                'vanaken_l3_centroid': np.nan,
            }])
            self.df = pd.concat([self.df, new_row], ignore_index=True)
            added += 1

        if added > 0:
            was_empty = first_new_idx == 0
            self.populate_sample_list(maintain_selection=not was_empty)
            if was_empty:
                self.current_sample_index = 0
                self.sample_listbox.selection_set(0)
                self.load_current_sample()
            self._set_status(f"Added {added} file(s) ({len(self.df)} total)")
            self._update_title()
            self._refresh_analysis_tab()
        return added

    def open_spectrum_files(self, event=None):
        """Add spectrum files to the current session (never replaces)."""
        fps = filedialog.askopenfilenames(
            title="Add Spectrum Files",
            initialdir=self._last_dir,
            filetypes=_SPECTRUM_FILETYPES)
        if not fps:
            return
        self._last_dir = os.path.dirname(fps[0])
        # Route .prj files through Athena importer
        prj_files = [f for f in fps if f.lower().endswith('.prj')]
        regular_files = [f for f in fps if not f.lower().endswith('.prj')]
        if regular_files:
            self._add_files_to_session(list(regular_files))
        for prj in prj_files:
            self.open_athena_project_path(prj)

    def open_athena_project_path(self, prj_path):
        """Import spectra from an Athena .prj file path (no dialog)."""
        if load_prj_spectra is None:
            self._set_status("Athena import unavailable (extract_athena_spectra.py not found)")
            return
        try:
            spectra = load_prj_spectra(prj_path, energy_range=(695, 740))
        except Exception as e:
            self._set_status(f"Failed to parse Athena project: {e}")
            return
        if not spectra:
            return

        if not hasattr(self, '_athena_spectra'):
            self._athena_spectra = {}
        if not hasattr(self, 'sample_file_paths'):
            self.sample_file_paths = {}
        if self.selected_files is None:
            self.selected_files = []
        if self.df is None or not isinstance(self.df, pd.DataFrame):
            self.df = pd.DataFrame(columns=[
                'sample_name', 'sample_group',
                'peak1_energy', 'peak2_energy', 'peak1_intensity',
                'peak2_intensity', 'delta_ev', 'intensity_ratio',
                'include_in_report', 'smoothing_sigma',
                'baseline_offset', 'baseline_slope',
                'baseline_step', 'baseline_width', 'baseline_edge',
                'baseline_edge_l2',
                'vanaken_fe3', 'vanaken_ratio',
                'vanaken_p', 'vanaken_il3',
                'vanaken_il2', 'vanaken_l3_lo',
                'vanaken_l3_hi', 'vanaken_l2_lo',
                'vanaken_l2_hi', 'vanaken_l3_centroid'])

        added = 0
        first_new_idx = len(self.df)
        for spec in spectra:
            sample_name = spec['label']
            if sample_name in self.sample_file_paths or sample_name in self._athena_spectra:
                continue
            self._athena_spectra[sample_name] = {
                'energy': spec['energy'],
                'intensity': spec['intensity'],
            }
            self.sample_file_paths[sample_name] = f"athena://{prj_path}#{sample_name}"
            new_row = pd.DataFrame([{
                'sample_name': sample_name,
                'sample_group': "",
                'peak1_energy': np.nan, 'peak2_energy': np.nan,
                'peak1_intensity': np.nan, 'peak2_intensity': np.nan,
                'delta_ev': np.nan, 'intensity_ratio': np.nan,
                'include_in_report': True, 'smoothing_sigma': 0.0,
                'baseline_offset': np.nan, 'baseline_slope': np.nan,
                'baseline_step': np.nan, 'baseline_width': np.nan,
                'baseline_edge': np.nan, 'baseline_edge_l2': np.nan,
                'vanaken_fe3': np.nan, 'vanaken_ratio': np.nan,
                'vanaken_p': np.nan, 'vanaken_il3': np.nan,
                'vanaken_il2': np.nan, 'vanaken_l3_lo': np.nan,
                'vanaken_l3_hi': np.nan, 'vanaken_l2_lo': np.nan,
                'vanaken_l2_hi': np.nan, 'vanaken_l3_centroid': np.nan,
            }])
            self.df = pd.concat([self.df, new_row], ignore_index=True)
            added += 1

        if added > 0:
            was_empty = first_new_idx == 0
            self.populate_sample_list(maintain_selection=not was_empty)
            if was_empty:
                self.current_sample_index = 0
                self.sample_listbox.selection_set(0)
                self.load_current_sample()
            self._set_status(f"Imported {added} from Athena project ({len(self.df)} total)")
            self._update_title()
            self._refresh_analysis_tab()

    # Aliases for backwards compatibility
    open_csv_files = open_spectrum_files
    add_spectrum_files = open_spectrum_files

    def open_spectrum_folder(self):
        """Add all spectrum files in a folder to the current session."""
        folder = filedialog.askdirectory(
            title="Select Folder Containing Spectrum Files",
            initialdir=self._last_dir)
        if not folder:
            return
        self._last_dir = folder
        self.spectra_dir = folder
        files = []
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith(_SPECTRUM_EXTENSIONS):
                files.append(os.path.join(folder, fname))
        if not files:
            messagebox.showinfo("No Files Found",
                                f"No spectrum files found in:\n{folder}\n\n"
                                f"Supported extensions: "
                                f"{', '.join(_SPECTRUM_EXTENSIONS)}")
            return
        added = self._add_files_to_session(files)
        if added == 0:
            messagebox.showinfo("No New Files",
                                "All files in that folder are already loaded.")

    def open_athena_project(self):
        """Import spectra from an Athena .prj file."""
        if load_prj_spectra is None:
            messagebox.showerror(
                "Import Error",
                "Athena project import is not available.\n\n"
                "Make sure extract_athena_spectra.py is in the same directory.")
            return

        fp = filedialog.askopenfilename(
            title="Open Athena Project",
            initialdir=self._last_dir,
            filetypes=[("Athena Project", "*.prj"), ("All files", "*.*")])
        if not fp:
            return
        self._last_dir = os.path.dirname(fp)

        try:
            spectra = load_prj_spectra(fp, energy_range=(695, 740))
        except Exception as e:
            messagebox.showerror("Parse Error",
                                 f"Failed to parse Athena project:\n{e}")
            return

        if not spectra:
            messagebox.showinfo("No Spectra",
                                "No Fe L-edge spectra found in this project file.\n"
                                "(Looking for data in 695-740 eV range)")
            return

        # Initialize session structures if needed
        if not hasattr(self, 'sample_file_paths'):
            self.sample_file_paths = {}
        if not hasattr(self, '_athena_spectra'):
            self._athena_spectra = {}
        if self.selected_files is None:
            self.selected_files = []
        if self.df is None or not isinstance(self.df, pd.DataFrame):
            self.df = pd.DataFrame(columns=[
                'sample_name', 'sample_group',
                'peak1_energy', 'peak2_energy', 'peak1_intensity',
                'peak2_intensity', 'delta_ev', 'intensity_ratio',
                'include_in_report', 'smoothing_sigma',
                'baseline_offset', 'baseline_slope',
                'baseline_step', 'baseline_width', 'baseline_edge',
                'baseline_edge_l2',
                'vanaken_fe3', 'vanaken_ratio',
                'vanaken_p', 'vanaken_il3',
                'vanaken_il2', 'vanaken_l3_lo',
                'vanaken_l3_hi', 'vanaken_l2_lo',
                'vanaken_l2_hi', 'vanaken_l3_centroid'])

        added = 0
        first_new_idx = len(self.df)
        for spec in spectra:
            sample_name = spec['label']
            if sample_name in self.sample_file_paths or sample_name in self._athena_spectra:
                continue

            self._athena_spectra[sample_name] = {
                'energy': spec['energy'],
                'intensity': spec['intensity'],
            }
            self.sample_file_paths[sample_name] = f"athena://{fp}#{sample_name}"
            new_row = pd.DataFrame([{
                'sample_name': sample_name,
                'sample_group': "",
                'peak1_energy': np.nan,
                'peak2_energy': np.nan,
                'peak1_intensity': np.nan,
                'peak2_intensity': np.nan,
                'delta_ev': np.nan,
                'intensity_ratio': np.nan,
                'include_in_report': True,
                'smoothing_sigma': 0.0,
                'baseline_offset': np.nan,
                'baseline_slope': np.nan,
                'baseline_step': np.nan,
                'baseline_width': np.nan,
                'baseline_edge': np.nan,
                'baseline_edge_l2': np.nan,
                'vanaken_fe3': np.nan,
                'vanaken_ratio': np.nan,
                'vanaken_p': np.nan,
                'vanaken_il3': np.nan,
                'vanaken_il2': np.nan,
                'vanaken_l3_lo': np.nan,
                'vanaken_l3_hi': np.nan,
                'vanaken_l2_lo': np.nan,
                'vanaken_l2_hi': np.nan,
                'vanaken_l3_centroid': np.nan,
            }])
            self.df = pd.concat([self.df, new_row], ignore_index=True)
            added += 1

        if added > 0:
            was_empty = first_new_idx == 0
            self.populate_sample_list(maintain_selection=not was_empty)
            if was_empty:
                self.current_sample_index = 0
                self.sample_listbox.selection_set(0)
                self.load_current_sample()
            self._set_status(f"Imported {added} spectrum/spectra from Athena project ({len(self.df)} total)")
            self._update_title()
            self._refresh_analysis_tab()
        else:
            messagebox.showinfo("No New Spectra",
                                "All spectra from this project are already loaded.")

    def new_session(self):
        """Clear all samples and start fresh."""
        if self.df is not None and len(self.df) > 0:
            if not messagebox.askyesno(
                    "Clear All Samples",
                    f"Remove all {len(self.df)} samples and start fresh?\n\n"
                    f"Unsaved work will be lost."):
                return
        self.df = pd.DataFrame(columns=[
            'sample_name', 'sample_group',
            'peak1_energy', 'peak2_energy', 'peak1_intensity',
            'peak2_intensity', 'delta_ev', 'intensity_ratio',
            'include_in_report', 'smoothing_sigma',
            'baseline_offset', 'baseline_slope',
            'baseline_step', 'baseline_width', 'baseline_edge',
            'baseline_edge_l2',
            'vanaken_fe3', 'vanaken_ratio',
            'vanaken_p', 'vanaken_il3',
            'vanaken_il2', 'vanaken_l3_lo',
            'vanaken_l3_hi', 'vanaken_l2_lo',
            'vanaken_l2_hi', 'vanaken_l3_centroid'])
        self.selected_files = []
        self.sample_file_paths = {}
        self._athena_spectra = {}
        self.current_sample_index = 0
        self.current_spectrum_data = None
        self.original_spectrum_data = None
        self.current_smoothed_data = None
        self.current_baseline = None
        self.current_baseline_fitter = None
        self.current_vanaken = None
        self.selected_peaks = []
        self.peak_candidates = np.array([])
        self.populate_sample_list(maintain_selection=False)
        self._show_empty_state()
        self._update_title()
        self._refresh_analysis_tab()
        self._update_vanaken_display()
        if hasattr(self, 'vanaken_ax'):
            self.vanaken_ax.clear()
            self.vanaken_canvas.draw()
        self._set_status("Session cleared")

    def load_session_csv(self):
        """Load a previously saved session CSV (from Export All Data)."""
        fp = filedialog.askopenfilename(
            title="Load Saved Session CSV",
            initialdir=self._last_dir,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not fp:
            return
        self._last_dir = os.path.dirname(fp)
        self._session_save_path = fp
        self.csv_path = fp
        self.load_csv()
        self.populate_sample_list(maintain_selection=False)
        if self.df is not None and len(self.df) > 0:
            self.current_sample_index = 0
            self.load_current_sample()
        self._update_title()
        self._refresh_analysis_tab()
        self._set_status(f"Loaded session: {os.path.basename(fp)}")

    def _update_title(self):
        """Update the window title with file count."""
        n = len(self.df) if self.df is not None else 0
        if n > 0:
            self.root.title(f"IronLPeaks \u2014 {n} samples loaded")
        else:
            self.root.title("IronLPeaks \u2014 Fe L-edge Peak Selector")

    def _refresh_analysis_tab(self):
        """Refresh the analysis tab tree + scatter plot after data changes."""
        if hasattr(self, 'peak_tree'):
            self.refresh_peak_analysis()
        if hasattr(self, 'analysis_ax'):
            self.update_analysis_plot()
        self._refresh_group_choices()

    def remove_selected_files(self):
        if self.df is None or len(self.df) == 0:
            messagebox.showwarning("No Files", "No files loaded to remove.")
            return
        indices = list(self.sample_listbox.curselection())
        if not indices:
            messagebox.showwarning("No Selection",
                                   "Select one or more samples to remove.")
            return
        names = [self.df.iloc[i]['sample_name'] for i in indices]
        preview = "\n".join(names[:5])
        if len(names) > 5:
            preview += f"\n\u2026 and {len(names) - 5} more"
        if not messagebox.askyesno("Remove Files",
                                   f"Remove {len(indices)} file(s)?\n\n{preview}"):
            return
        for i in sorted(indices, reverse=True):
            sn = self.df.iloc[i]['sample_name']
            self.df = self.df.drop(self.df.index[i]).reset_index(drop=True)
            if hasattr(self, 'sample_file_paths') and sn in self.sample_file_paths:
                fp = self.sample_file_paths.pop(sn)
                if hasattr(self, 'selected_files') and self.selected_files and fp in self.selected_files:
                    self.selected_files.remove(fp)
            # Drop the in-memory Athena entry so the spectrum can be
            # re-imported from the same .prj later
            if hasattr(self, '_athena_spectra'):
                self._athena_spectra.pop(sn, None)
        self.populate_sample_list(maintain_selection=False)
        if len(self.df) > 0:
            self.current_sample_index = 0
            self.sample_listbox.selection_set(0)
            self.load_current_sample()
        else:
            self._show_empty_state()
        self._update_title()
        self._refresh_analysis_tab()
        self._set_status(f"Removed {len(indices)} sample(s)")

    def save_session_csv(self, event=None):
        """Save the current session to CSV. Re-uses last save path if available."""
        if self.df is None or len(self.df) == 0:
            messagebox.showwarning("No Data", "No data to save.")
            return
        # If we already have a session path, save directly; otherwise prompt
        if hasattr(self, '_session_save_path') and self._session_save_path:
            fp = self._session_save_path
        else:
            fp = filedialog.asksaveasfilename(
                defaultextension=".csv",
                initialdir=self._last_dir,
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Save Session As")
            if not fp:
                return
        self._last_dir = os.path.dirname(fp)
        self._session_save_path = fp
        try:
            self.df.to_csv(fp, index=False)
            self._set_status(f"Session saved: {os.path.basename(fp)}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save: {e}")

    def export_all_data(self, event=None):
        if self.df is None or len(self.df) == 0:
            messagebox.showwarning("No Data", "No data to export.")
            return
        fp = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialdir=self._last_dir,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export All Data to CSV")
        if not fp:
            return
        self._last_dir = os.path.dirname(fp)
        try:
            self.df.to_csv(fp, index=False)
            self._set_status(f"Exported: {os.path.basename(fp)}")
            messagebox.showinfo("Export Complete", f"Exported to:\n{fp}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export: {e}")

    def export_peak_data(self, event=None):
        """Bug 9 fix: always use file dialog instead of relying on csv_path."""
        if self.df is None or len(self.df) == 0:
            messagebox.showwarning("No Data", "No data to export.")
            return
        fp = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialdir=self._last_dir,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export Peak Data to CSV")
        if not fp:
            return
        self._last_dir = os.path.dirname(fp)
        try:
            cols = ['sample_name', 'sample_group',
                    'peak1_energy', 'peak2_energy', 'peak1_intensity',
                    'peak2_intensity', 'delta_ev', 'intensity_ratio',
                    'vanaken_fe3', 'vanaken_ratio',
                    'vanaken_p', 'vanaken_il3',
                    'vanaken_il2', 'vanaken_l3_lo',
                    'vanaken_l3_hi', 'vanaken_l2_lo',
                    'vanaken_l2_hi', 'vanaken_l3_centroid']
            export_df = self.df[[c for c in cols if c in self.df.columns]].copy()
            export_df.to_csv(fp, index=False)
            self._set_status(f"Peak data exported: {os.path.basename(fp)}")
            messagebox.showinfo("Export Complete", f"Peak data exported to:\n{fp}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export: {e}")

    def save_and_exit(self):
        """Bug 10 fix: always prompt for save location via file dialog."""
        fp = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialdir=self._last_dir,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save Data As")
        if not fp:
            return  # user cancelled, don't exit
        try:
            self.df.to_csv(fp, index=False)
            messagebox.showinfo("Saved", f"Data saved to:\n{fp}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")
            return
        self.root.quit()

    def on_closing(self):
        if hasattr(self, 'fig'):
            plt.close(self.fig)
        if hasattr(self, 'analysis_fig'):
            plt.close(self.analysis_fig)
        if hasattr(self, 'vanaken_fig'):
            plt.close(self.vanaken_fig)
        self.root.quit()


# ======================================================================
# Standalone file selector
# ======================================================================
def select_spectrum_files(spectra_dir=None):
    temp_root = tk.Tk()
    temp_root.withdraw()
    fps = filedialog.askopenfilenames(
        title="Select Spectrum Files to Analyze",
        initialdir=spectra_dir or os.getcwd(),
        filetypes=_SPECTRUM_FILETYPES)
    temp_root.destroy()
    return list(fps) if fps else None


def _get_base_path():
    """Get base path for assets (handles both dev and PyInstaller)."""
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    return Path(__file__).parent


def _create_splash(root):
    """Create a splash screen showing the app icon during loading."""
    splash = tk.Toplevel(root)
    splash.overrideredirect(True)  # No title bar or borders

    base_path = _get_base_path()

    # Load the largest available icon for the splash
    icon_image = None
    for size in [256, 128, 64]:
        png_path = base_path / f"ironlpeaks_{size}.png"
        if png_path.exists():
            try:
                icon_image = tk.PhotoImage(file=str(png_path))
                break
            except tk.TclError:
                continue

    if icon_image is None:
        splash.destroy()
        return None, None

    # Build splash content
    splash.configure(bg='#2b2b2b')
    img_label = tk.Label(splash, image=icon_image, bg='#2b2b2b')
    img_label.pack(padx=30, pady=(25, 10))
    text_label = tk.Label(splash, text="IronLPeaks", font=("Segoe UI", 16, "bold"),
                          fg='white', bg='#2b2b2b')
    text_label.pack(pady=(0, 5))
    sub_label = tk.Label(splash, text="Loading...", font=("Segoe UI", 10),
                         fg='#aaaaaa', bg='#2b2b2b')
    sub_label.pack(pady=(0, 20))

    # Center the splash on screen
    splash.update()
    w = splash.winfo_width()
    h = splash.winfo_height()
    screen_w = splash.winfo_screenwidth()
    screen_h = splash.winfo_screenheight()
    x = (screen_w - w) // 2
    y = (screen_h - h) // 2
    splash.geometry(f"{w}x{h}+{x}+{y}")
    splash.lift()
    splash.update()

    return splash, icon_image


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="IronLPeaks \u2014 Fe L-edge Peak Selector GUI")
    parser.add_argument('--csv', type=str,
                        help='Path to CSV file with sample list')
    parser.add_argument('--spectra-dir', type=str,
                        help='Directory containing spectrum files')
    parser.add_argument('--report-id', type=str, default='test',
                        help='Report ID for output files')
    parser.add_argument('--select-files', action='store_true',
                        help='Show file selection dialog')
    parser.add_argument('--files', nargs='+',
                        help='Specific files to analyze')

    args = parser.parse_args()

    selected_files = None
    if args.files:
        selected_files = args.files
    elif args.select_files:
        selected_files = select_spectrum_files(args.spectra_dir)

    # Close PyInstaller bootloader splash if present (shows during extraction)
    try:
        import pyi_splash  # noqa: F401 - only exists in PyInstaller builds
        pyi_splash.close()
    except ImportError:
        pass

    root = tk.Tk()
    root.withdraw()  # Hide main window while loading

    # Show splash screen with icon
    splash, _splash_icon = _create_splash(root)

    # Build the application (this is the slow part)
    PeakSelectorGUI(root, csv_path=args.csv, report_id=args.report_id,
                     spectra_dir=args.spectra_dir,
                     selected_files=selected_files)

    # Close splash and show main window
    if splash is not None:
        splash.destroy()
    root.deiconify()

    root.mainloop()


if __name__ == '__main__':
    main()
