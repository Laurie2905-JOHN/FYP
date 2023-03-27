from tkinter import *
from tkinter import ttk
import tkinter.filedialog as fd
from pathlib import Path, PureWindowsPath



# Create an instance of tkinter frame or window
def file_chooser():

    win = Tk()
    # Set the geometry of tkinter frame
    win.geometry("800x350")
    # Add a Label widget
    label = Label(win, text="Select the Button to Open the File", font=('Aerial 11'))
    label.pack(pady=30)
    # Add a Treeview widget to display the selected file names
    tree = ttk.Treeview(win, columns=('Filename'))
    tree.heading('#0', text='Index')
    tree.heading('Filename', text='Filename')
    tree.pack(side=LEFT, padx=30, pady=30)

    def open_file():
        files = fd.askopenfilenames(parent=win, title='Choose a File')
        global file_paths
        file_paths = list(win.splitlist(files))
        for file_path in file_paths:
            file_paths.append(Path(file_path))
        # Clear the Treeview widget before inserting new file names
        tree.delete(*tree.get_children())
        # Update the table with the selected file names
        global file_names
        file_names = []
        for i, file_path in enumerate(file_paths):
            file_name = file_path.split("/")[-1]
            file_names.append(file_name)
            tree.insert('', 'end', text=str(i + 1), values=(file_name,))
        return file_paths, file_names

    # Add a Button Widget
    ttk.Button(win, text="Select a File", command=open_file).pack()
    # Add a Close Button Widget

    def close_window():
        win.destroy()

    # Add a Label widget for close button
    label = Label(win, text="Close window once files are added", font=('Aerial 11'))
    label.pack(pady=30)
    ttk.Button(win, text="Close", command=close_window).pack()

    win.mainloop()

def cal_velocity(BarnFilePath):
    import numpy as np
    import scipy.io as sio

    # Constants
    rho = 997
    fs = 16  # sample rate

    # File retrieving
    ZeroFolder = "C:/Users/lauri/OneDrive/Documents (1)/University/Year 3/Semester 2/BARNACLE/Example Data/"
    ZeroFile = 'Mon1527.txt'
    CalFolder = ZeroFolder
    CalFile = 'IanYawAndDynCalMk2.mat'
    Cal = sio.loadmat(CalFolder + CalFile)

    Cal = Cal["Cal"]
    Dynfit = Cal[0][0][0].flatten()
    Yawfit = Cal[0][0][1].flatten()
    LDyn = Cal[0][0][2].flatten()
    LYaw = Cal[0][0][3].flatten()
    LDyn_0 = Cal[0][0][4].flatten()

    # Evaluating yawcal for a polynomial Cal.Yawfit and dyncal
    yawcal = np.zeros((91, 2))
    yawcal[:, 0] = np.linspace(-45, 45, 91)
    yawcal[:, 1] = np.polyval(Yawfit, yawcal[:, 0])
    dyncal = np.polyval(Dynfit, yawcal[:, 0])
    dyncal = dyncal * LDyn_0

    # Importing Zeroes
    zeros = {}
    zeros['pr_raw'] = np.loadtxt(ZeroFolder + ZeroFile, delimiter=',')
    zeros['pr_mean'] = np.mean(zeros['pr_raw'][1300:1708, :], axis=0)

    # Loading actual Barnacle data
    prb = {}
    prb['raw'] = np.loadtxt(BarnFilePath, delimiter=',')
    prb['raw'] -= zeros['pr_mean']

    # Data analysis
    prb['denom'] = np.mean(prb['raw'][:, :4], axis=1)
    prb['Lyaw'] = (prb['raw'][:, 1] - prb['raw'][:, 3]) / prb['denom']
    prb['Lpitch'] = (prb['raw'][:, 0] - prb['raw'][:, 2]) / prb['denom']

    from scipy import interpolate

    ayaw_interp = interpolate.interp1d(yawcal[:, 1], yawcal[:, 0], kind='linear', fill_value='extrapolate')
    apitch_interp = interpolate.interp1d(yawcal[:, 1], yawcal[:, 0], kind='linear', fill_value='extrapolate')
    prb['ayaw'] = ayaw_interp(prb['Lyaw'])
    prb['apitch'] = apitch_interp(prb['Lpitch'])
    prb['pitchbigger'] = np.abs(prb['apitch']) > np.abs(prb['ayaw'])
    prb['amax'] = prb['pitchbigger'] * prb['apitch'] + (1 - prb['pitchbigger']) * prb['ayaw']
    ldyn_interp = interpolate.interp1d(yawcal[:, 0], dyncal, kind='linear', fill_value='extrapolate')
    prb['ldyn'] = ldyn_interp(prb['amax'])

    # Splitting into velocities
    prb['U1'] = np.sqrt(2 * -prb['ldyn'] * np.mean(prb['raw'][:, :4], axis=1) / rho)
    prb['U1'][np.imag(prb['U1']) > 0] = 0
    prb['Ux'] = prb['U1'] * np.cos(np.deg2rad(prb['apitch'])) * np.cos(np.deg2rad(prb['ayaw']))
    prb['Uy'] = prb['U1'] * np.cos(np.deg2rad(prb['apitch'])) * np.sin(np.deg2rad(prb['ayaw']))
    prb['Uz'] = prb['U1'] * np.sin(np.deg2rad(prb['apitch']))
    prb['t'] = np.linspace(0, prb['raw'].shape[0] / fs, prb['raw'].shape[0]);

    return prb




file_chooser()

print(file_names)
#print(file_paths)




print(file_paths)


BarnFilePath = file_paths

#=cal_velocity(file_paths)

prb = []
df = {}
# for file_path in selected_files:
#     prb1 = cal_velocity(file_path)
#     prb.append(prb1)
#     file_name = os.path.basename(file_path)
#     df[file_name] = prb




