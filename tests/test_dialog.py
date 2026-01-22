from pathlib import Path
from tkinter import filedialog as fd
import tkinter as tk
import glob
import os

"""
shows a window for file picking
"""

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

if result:  # the choise was made
    x = result[0]  # list position 0

    filetypes = (
        ("ext files", "*.txt"),
        ("All files", "*.")
    )

    if x == "1":  # file was choosen
        filename = fd.askopenfilename(
            title="Open a file",
            initialdir="/",
            filetypes=filetypes
            )
                # --- controlling ---
        print("Name of .txt file" + filename)
        

    elif x == "2":  # folder was choosen
        filenames = []
        folder_directory = fd.askdirectory(title="Open all files of the measurement series")
        if folder_directory:
            txt_file_search = os.path.join(folder_directory, "*.txt")
            filenames = glob.glob(txt_file_search)
        else:
            print("Nothing")

        # --- controlling ---
        print("\n".join(filenames))
        print(f"Number of list-entries: {len(filenames)}")

    elif x == "3":  # nested folders were choosen
        main_folder = fd.askdirectory(title="main folder of all measurement series")
        all_measurements_structure = {}

        if main_folder:
            for current_folder, nested_folder, data in os.walk(main_folder):
                txt_files = [f for f in data if f.endswith(".txt")]  # but what if there is a readme ... nobody does that :D

                if txt_files:
                    folder_name = os.path.basename(current_folder)

                    full_path = []
                    for data in txt_files:
                        path = os.path.join(current_folder, data)
                        full_path.append(path)
                    full_path.sort()

                    all_measurements_structure[folder_name] = full_path
        # --- controlling ---
        print("Gefundene Struktur:")
        for groupe, data_list in all_measurements_structure.items():
            print(f"Group '{groupe}': {len(data_list)} files were found.")