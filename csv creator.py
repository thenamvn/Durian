import os
import csv
from pydub.utils import mediainfo
from tkinter import Tk
from tkinter.filedialog import askdirectory
# from pydub import AudioSegment
# AudioSegment.ffmpeg = "D:/environment/ffmpeg/ffmpeg.exe"
# AudioSegment.ffprobe = "D:/environment/ffmpeg/ffprobe.exe"
# Hide the root Tkinter window
root = Tk()
root.withdraw()

# Open a dialog to select the folder containing m4a files
audio_folder = askdirectory(title='Select Folder Containing m4a Files')
folder_name = os.path.basename(audio_folder)

# Check if a folder was selected
if not audio_folder:
    print("No folder selected. Exiting...")
    exit()

csv_file = 'output.csv'

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Ghi tiêu đề các cột
    writer.writerow(['slice_file_name', 'fsID', 'start', 'end', 'classID', 'class', 'relative_path'])
    
    # Get sorted list of files numerically
    sorted_files = sorted(os.listdir(audio_folder), key=lambda x: int(os.path.splitext(x)[0]))
    
    for idx, filename in enumerate(sorted_files):
        if filename.endswith('.m4a'):
            file_path = os.path.join(audio_folder, filename)
            
            # Lấy thông tin file
            info = mediainfo(file_path)
            duration = float(info['duration'])
            
            # Giả sử fsID là chỉ số của file và fold, classID, salience là các giá trị cố định (có thể tùy chỉnh sau)
            fsID = idx + 1  # fsID starts from 1
            start = 0
            end = duration          
            # Determine class_name based on fsID
            if 1 <= fsID <= 10:
                class_name = '7.5 to 8'
                classID = 2
            elif 11 <= fsID <= 20:
                class_name = 'Under 7.5'
                classID = 1
            elif 21 <= fsID <= 30:
                class_name = 'Over 8'
                classID = 3
            elif 31 <= fsID <= 35:
                class_name = '7.5 to 8'
                classID = 2
            elif 36 <= fsID <= 40:
                class_name = 'Under 7.5'
                classID = 1
            else:
                class_name = 'unknown'  # Default value if fsID is out of the specified ranges
                classID = 0
            relative_path = f'/{folder_name}/{filename}'
            
            # Ghi dòng vào CSV
            writer.writerow([filename, fsID, start, end, classID, class_name, relative_path])

print(f"CSV created: {csv_file}")