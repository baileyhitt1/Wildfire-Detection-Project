#For personal use but tests to ensure image has EXIF metadata to be extracted
#Used to see if API key was working properly

import exifread

file_path = "/Users/bayhitt/Documents/DEEP LEARNING PROJECT/Images to use for gui/cloud_1062_with_gps.jpg" 
with open(file_path, 'rb') as f:
    tags = exifread.process_file(f)

# Print all metadata
for tag in tags.keys():
    print(f"{tag}: {tags[tag]}")

# Extract GPS data
gps_data = {tag: tags[tag] for tag in tags.keys() if "GPS" in tag}
print("\nGPS Data:", gps_data)
