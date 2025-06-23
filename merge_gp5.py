import os
import guitarpro as gp

def merge_gp5_files(file1, file2, output_file):
    # Load first file (will be the base)
    song1 = gp.parse(file1)
    
    # Load second file (to extract its track)
    song2 = gp.parse(file2)
    
    # Copy the track from the second file into the first
    if len(song2.tracks) > 0:
        new_track = song2.tracks[0]  # Clone to avoid reference issues
        song1.tracks.append(new_track)
    
    # Save the merged file
    gp.write(song1, output_file)

# Example usage (run in bulk)
folder_bass = "/home/alexandre/PhD/bassGeneration/gen_21-04/gp5_bass"
folder_rg = "/home/alexandre/PhD/bassGeneration/gen_21-04/gp5"
output_folder = "merged_gp5"

for i in range(2301):  # Assuming files are named 1.gp5, 2.gp5, etc.
    file1 = os.path.join(folder_bass, f"{i}_generated_bass.tokens..gp5")
    file2 = os.path.join(folder_rg, f"{i}_rhythm_guitar.tokens.rhythm_guitar.gp5")
    output_file = os.path.join(output_folder, f"merged_{i}.gp5")
    if os.path.exists(output_file):
        continue
    
    if os.path.exists(file1) and os.path.exists(file2):
        try:
            merge_gp5_files(file1, file2, output_file)
            print(f"Merged {file1} and {file2} → {output_file}")
        except:
            print("Error while merging: ", file1, file2)
