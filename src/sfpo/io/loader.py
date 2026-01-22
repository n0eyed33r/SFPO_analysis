# src/sfpo/io/loader.py
from tkinter import filedialog as fd
import tkinter as tk
import glob
import os


class file_loader:
    """
    shows a window for file and folder picking - searching for *.txt files (measurements)
    """

    def __init__(self) -> None: # small config
        self.filetypes = (
                ("ext files", "*.txt"),
                ("All files", "*.")
            )
        self.title = "SFPO analyzer"

    def run(self):
        """
        Main method: opens GUI, waits for choise and load string/s of *.txt-files
        return: mode-1: str(path), mode-2: list(paths), mode-3: dictionary (folder: [paths])
        """
        mode = self._show_selection_gui()
        if not mode:
            print("No options were selected.")
            return None
        
        # every mode uses different approach
        if mode == "1":
            return self._load_single_file()
        elif mode == "2":
            return self._load_measurement_series()
        elif mode == "3":
            return self._load_all_series_nested()
        
        return None

    def _show_selection_gui(self):
        result = []  # list of choosen results

        # create window
        root = tk.Tk()  
        root.title("SFPO analyzer")
        root.lift()  # window on front
        root.attributes('-topmost', True)
        # window content
        tk.Label(root, text="Choose analyse type:").pack(pady=25)
        choice = tk.StringVar(value="1")  # you have a choice - more than morpheus
        tk.Radiobutton(root, text="Single measurement", variable=choice, value="1").pack()
        tk.Radiobutton(root, text="One measurement series", variable=choice, value="2").pack()
        tk.Radiobutton(root, text="All measurement series at once", variable=choice, value="3",padx=50).pack()

        def on_button_click():
            result.append(choice.get())  # collect the choice
            root.destroy()    
        tk.Button(root, text="Select", command=on_button_click).pack(pady=25)

        # window centering
        root.update_idletasks()  # calculate window size
        window_width = root.winfo_reqwidth()  # window width
        window_height = root.winfo_reqheight()  # window height
        screen_width = root.winfo_screenwidth()  # screen width
        screen_height = root.winfo_screenheight() # screen height
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        root.geometry(f'{window_width}x{window_height}+{int(x)}+{int(y)}')

        root.mainloop()
        return 
    
    def _center_window(self, window):
        """Hilfsmethode zum Zentrieren des Fensters"""
        window.update_idletasks()
        width = window.winfo_reqwidth()
        height = window.winfo_reqheight()
        x = (window.winfo_screenwidth() // 2) - (width // 2)
        y = (window.winfo_screenheight() // 2) - (height // 2)
        window.geometry(f'{width}x{height}+{int(x)}+{int(y)}')
      
    def _load_single_file(self):
        """
        mode 1: single file
        """
        filename = fd.askopenfilename(
            title="Open a file",
            initialdir="/",
            filetypes=self.filetypes
        )
        if filename:
            print(f"Single file selected: {filename}")
            return filename  # Rückgabe: String
        return None

    def _load_measurement_series(self):
        """
        mode 2: everey measruement file of one measurement series
        """
        
        folder = fd.askdirectory(title="Open all files of the measurement series")
        if folder:
            search_pattern = os.path.join(folder, "*.txt")
            filenames = glob.glob(search_pattern)
            filenames.sort() # Immer gut zu sortieren
            
            print(f"Series found: {len(filenames)} files.")
            return filenames  # Rückgabe: Liste
        return []

    def _load_all_series_nested(self):
        """
        mode 3: everey measruement file of every measurement series nested in choosen folder
        """

        main_folder = fd.askdirectory(title="Main folder of all measurement series")
        structure = {}

        if main_folder:
            print(f"Scanning structure in: {main_folder}...")
            for current_folder, _, files in os.walk(main_folder):
                txt_files = [f for f in files if f.endswith(".txt")]
                
                if txt_files:
                    folder_name = os.path.basename(current_folder)
                    full_paths = []
                    for f in txt_files:
                        full_paths.append(os.path.join(current_folder, f))
                    
                    full_paths.sort()
                    structure[folder_name] = full_paths
            
            print(f"Structure found: {len(structure)} groups.")
            return structure  # Rückgabe: Dictionary
        return {}