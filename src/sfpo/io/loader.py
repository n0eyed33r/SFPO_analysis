# src/sfpo/io/loader.py
from tkinter import filedialog as fd
import tkinter as tk
from tkinter import ttk
import glob
import os


class FileLoader:
    """
    Shows a window for file and folder picking - searching for *.txt files (measurements)
    
    Now includes option for ANOVA analysis with Bootstrap (only available in Mode 3)
    """

    def __init__(self) -> None:
        self.filetypes = (
            ("Text files", "*.txt"),
            ("All files", "*.*")
        )
        self.title = "SFPO Analyzer"
        self._selected_mode = None
        self._anova_enabled = False  # NEW: ANOVA checkbox state

    def run(self):
        """
        Main method: opens GUI, waits for choice and loads string/s of *.txt-files
        
        Returns: 
            - mode 1: str(path)
            - mode 2: list(paths)  
            - mode 3: tuple(dictionary, anova_enabled)
        """
        mode, anova_enabled = self._show_selection_gui()
        
        if not mode:
            print("No option was selected.")
            return None
        
        # Store ANOVA setting
        self._anova_enabled = anova_enabled
        
        # Every mode uses different approach
        if mode == "1":
            return self._load_single_file()
        elif mode == "2":
            return self._load_measurement_series()
        elif mode == "3":
            result = self._load_all_series_nested()
            # Return tuple with ANOVA flag for Mode 3
            return (result, self._anova_enabled) if result else None
        
        return None

    def _show_selection_gui(self):
        """
        Shows the selection GUI with mode options and ANOVA checkbox.
        
        Returns:
            Tuple of (selected_mode, anova_enabled)
        """
        # Create window
        root = tk.Tk()
        root.title(self.title)
        root.lift()
        root.attributes('-topmost', True)
        
        # Variables
        choice = tk.StringVar(value="1")
        anova_var = tk.BooleanVar(value=False)
        
        # Main frame with padding
        main_frame = tk.Frame(root, padx=20, pady=15)
        main_frame.pack(fill='both', expand=True)
        
        # Title label
        title_label = tk.Label(
            main_frame, 
            text="SFPO Analyzer",
            font=('Helvetica', 14, 'bold')
        )
        title_label.pack(pady=(0, 15))
        
        # Analysis mode section
        mode_label = tk.Label(
            main_frame, 
            text="Select analysis mode:",
            font=('Helvetica', 11)
        )
        mode_label.pack(anchor='w')
        
        # Radio buttons for modes
        mode_frame = tk.Frame(main_frame)
        mode_frame.pack(fill='x', pady=10)
        
        rb1 = tk.Radiobutton(
            mode_frame, 
            text="Single measurement", 
            variable=choice, 
            value="1",
            font=('Helvetica', 10)
        )
        rb1.pack(anchor='w', padx=20)
        
        rb2 = tk.Radiobutton(
            mode_frame, 
            text="One measurement series", 
            variable=choice, 
            value="2",
            font=('Helvetica', 10)
        )
        rb2.pack(anchor='w', padx=20)
        
        rb3 = tk.Radiobutton(
            mode_frame, 
            text="All measurement series at once (comparison)", 
            variable=choice, 
            value="3",
            font=('Helvetica', 10)
        )
        rb3.pack(anchor='w', padx=20)
        
        # Separator
        separator = ttk.Separator(main_frame, orient='horizontal')
        separator.pack(fill='x', pady=15)
        
        # ANOVA section
        anova_label = tk.Label(
            main_frame, 
            text="Statistical Analysis (only for Mode 3):",
            font=('Helvetica', 11)
        )
        anova_label.pack(anchor='w')
        
        # ANOVA checkbox
        anova_frame = tk.Frame(main_frame)
        anova_frame.pack(fill='x', pady=5)
        
        anova_checkbox = tk.Checkbutton(
            anova_frame,
            text="ANOVA + Bootstrap (BCa) for IFSS comparison",
            variable=anova_var,
            font=('Helvetica', 10),
            state='disabled'  # Initially disabled
        )
        anova_checkbox.pack(anchor='w', padx=20)
        
        # Info label for ANOVA
        anova_info = tk.Label(
            anova_frame,
            text="• Performs one-way ANOVA\n• BCa Bootstrap confidence intervals\n• Games-Howell post-hoc test",
            font=('Helvetica', 9),
            fg='gray',
            justify='left'
        )
        anova_info.pack(anchor='w', padx=40, pady=(0, 5))
        
        # Function to enable/disable ANOVA checkbox based on mode selection
        def on_mode_change(*args):
            if choice.get() == "3":
                anova_checkbox.config(state='normal')
            else:
                anova_checkbox.config(state='disabled')
                anova_var.set(False)
        
        # Bind mode change
        choice.trace_add('write', on_mode_change)
        
        # Button frame
        button_frame = tk.Frame(main_frame)
        button_frame.pack(fill='x', pady=(15, 0))
        
        # Select button
        def on_button_click():
            self._selected_mode = choice.get()
            self._anova_enabled = anova_var.get()
            root.destroy()
        
        select_button = tk.Button(
            button_frame, 
            text="Select", 
            command=on_button_click,
            font=('Helvetica', 11),
            width=15,
            height=1
        )
        select_button.pack(pady=5)
        
        # Window centering
        root.update_idletasks()
        window_width = root.winfo_reqwidth()
        window_height = root.winfo_reqheight()
        
        # Set minimum size
        min_width = 380
        min_height = 350
        window_width = max(window_width, min_width)
        window_height = max(window_height, min_height)
        
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        root.geometry(f'{window_width}x{window_height}+{int(x)}+{int(y)}')
        
        root.mainloop()
        
        return self._selected_mode, self._anova_enabled
    
    def _center_window(self, window):
        """Helper method to center a window on screen"""
        window.update_idletasks()
        width = window.winfo_reqwidth()
        height = window.winfo_reqheight()
        x = (window.winfo_screenwidth() // 2) - (width // 2)
        y = (window.winfo_screenheight() // 2) - (height // 2)
        window.geometry(f'{width}x{height}+{int(x)}+{int(y)}')
      
    def _load_single_file(self):
        """
        Mode 1: Load a single measurement file
        
        Returns:
            str: Path to selected file, or None
        """
        filename = fd.askopenfilename(
            title="Open a measurement file",
            initialdir="/",
            filetypes=self.filetypes
        )
        if filename:
            print(f"Single file selected: {filename}")
            return filename
        return None

    def _load_measurement_series(self):
        """
        Mode 2: Load all measurement files from one folder
        
        Returns:
            list: List of file paths, or empty list
        """
        folder = fd.askdirectory(title="Select folder with measurement series")
        if folder:
            search_pattern = os.path.join(folder, "*.txt")
            filenames = glob.glob(search_pattern)
            filenames.sort()
            
            print(f"Series found: {len(filenames)} files.")
            return filenames
        return []

    def _load_all_series_nested(self):
        """
        Mode 3: Load all measurement files from nested folder structure
        
        Returns:
            dict: Dictionary with {folder_name: [file_paths]}, or empty dict
        """
        main_folder = fd.askdirectory(title="Select main folder containing all measurement series")
        structure = {}

        if main_folder:
            print(f"Scanning structure in: {main_folder}...")
            
            for current_folder, _, files in os.walk(main_folder):
                txt_files = [f for f in files if f.endswith(".txt")]
                
                if txt_files:
                    folder_name = os.path.basename(current_folder)
                    full_paths = [os.path.join(current_folder, f) for f in txt_files]
                    full_paths.sort()
                    structure[folder_name] = full_paths
            
            print(f"Structure found: {len(structure)} series.")
            
            if self._anova_enabled:
                print("ANOVA + Bootstrap analysis enabled for IFSS comparison")
            
            return structure
        return {}
    
    @property
    def anova_enabled(self) -> bool:
        """Returns whether ANOVA analysis is enabled"""
        return self._anova_enabled