import os
os.environ['TK_SILENCE_DEPRECATION'] = '1'   #suppresses Tkinter deprecation warnings for my Mac

import tkinter as tk                #standard Python GUI toolkit
from tkinter import filedialog, Button  #File selection dialog
from PIL import Image       #Used for opening and processing images
import torch                #Used to run the deep learning models
from torchvision import transforms  #Transforms images from CNN input
import exifread             #reads EXIF metadata,  specifcally the gps coordinates
import requests             #used to communite with the API key
import sys                  
#Added to the directory to the Python path to ensure we can import custom modules
sys.path.append("/Users/bayhitt/Documents/DEEP LEARNING PROJECT/")  # Add project directory to Python path
from cnn_model import EfficientNetB0  #Importing custom CNN model


# Load CNN Model
model_path = "efficientnetb0_trained.pth"  #file path
try:

    model = EfficientNetB0(num_classes=6)  # Ensure num_classes matches what was used during training
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Load on CPU incase it was trained elsewhere
    model.eval()  # Sets model to evaluation mode
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Model file not found. Proceeding with dummy classification.") #Handles missing model file
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Prevents crashing if model fails

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),   #Resize image to standard size
    transforms.ToTensor(),           #Converts image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  #Normalize using ImageNet values
])

# Image Classification Function
def classify_image(image_path):
    if model is None:
        return "Model not loaded"   #Handles case if model is missing

    try:
        image = Image.open(image_path).convert("RGB")  # Ensures 3 channel RGB format
        input_tensor = preprocess(image)        #Apply preprocessing transformations
        input_batch = input_tensor.unsqueeze(0) #Add batch dimentsion 

        with torch.no_grad():       #Disables gradient computations for efficiency 
            output = model(input_batch)     #Gets model prediction

        _, predicted_class = torch.max(output, 1)

        # Define the class labels to match what model was trained on
        class_labels = ['Smoke', 'Seaside', 'Land', 'Haze', 'Dust', 'Cloud']
        predicted_label = class_labels[predicted_class.item()]

        # If smoke is detected, return the prediction, else return a general no smoke message
        if predicted_label == 'Smoke':
            return f"Predicted class: {predicted_label}"
        else:
            return "No smoke detected"
    except Exception as e:
        return f"Error classifying image: {e}"      #Error handling if classification fails

# Get GPS Coordinates from Image EXIF Data
def get_gps_coordinates(image_path):
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f)     #Reads EXIF metadata

            #Ensures GPS data exists before continuing
            if all(k in tags for k in ['GPS GPSLatitude', 'GPS GPSLatitudeRef', 'GPS GPSLongitude', 'GPS GPSLongitudeRef']):
                lat, lat_ref = tags['GPS GPSLatitude'].values, tags['GPS GPSLatitudeRef'].values
                lon, lon_ref = tags['GPS GPSLongitude'].values, tags['GPS GPSLongitudeRef'].values

                # Convert GPS format to degrees
                lat_deg = float(lat[0].num)/float(lat[0].den) + float(lat[1].num)/(float(lat[1].den)*60) + float(lat[2].num)/(float(lat[2].den)*3600)
                if lat_ref == 'S':
                    lat_deg *= -1   #Negative sign if in Southern hemisphere
                lon_deg = float(lon[0].num)/float(lon[0].den) + float(lon[1].num)/(float(lon[1].den)*60) + float(lon[2].num)/(float(lon[2].den)*3600)
                if lon_ref == 'W':
                    lon_deg *= -1  #Negative sign if in Western hemisphere

                return lat_deg, lon_deg     
    except Exception as e:
        print(f"Error reading EXIF data: {e}")
    return None         #If no GPS data is found


#Get fire station phone number
def get_fire_station_phone_number(lat, lon, api_key):
    try:
        print(f"Searching for fire stations near: {lat}, {lon}")
        
        # specific endpoint structure for Google Places API (New)
        url = f"https://places.googleapis.com/v1/places:searchNearby"
        
        # The request body format for Places API (New)
        payload = {
            "locationRestriction": {
                "circle": {
                    "center": {
                        "latitude": lat,
                        "longitude": lon
                    },
                    "radius": 5000.0  # 5km radius
                }
            },
            "includedTypes": ["fire_station"]      #Ensures only fire stations are picked up on the map
        }
        
        # Headers required for Places API (New)
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.internationalPhoneNumber"
        }
        
        #POST request
        response = requests.post(url, json=payload, headers=headers)
        data = response.json()
        
        print(f"API Response Status: {response.status_code}")
        
        # Check if the request was successful
        if response.status_code != 200:
            print(f"API Error: Status code {response.status_code}")
            return f"API Error: {response.status_code}"
        
        # Check if there are results
        if "places" not in data or not data["places"]:
            print("No fire stations found in results")
            return "No fire stations found nearby"
        
        # Get the first result
        place = data["places"][0]
        
        # Extract information
        name = place.get("displayName", {}).get("text", "Unnamed Fire Station")
        phone = place.get("internationalPhoneNumber", "Phone number not available")
        address = place.get("formattedAddress", "Address unavailable")
        
        # Return dictionary with separate fields
        return {
            "name": name,
            "phone": phone,
            "address": address
        }
        
    except Exception as e:
        print(f"Exception in API retrieval: {e}")
        return f"Error retrieving phone number: {str(e)}"  #Error handling


# GUI Class
class WildfireDetection(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Wildfire Detection")   #Window title
        self.geometry("500x400")           #Window size
        self.configure(bg='white')         #Background  color

        # Colors
        self.normal_bg = 'white'
        self.smoke_bg = '#ffebee'  # Light red background
        self.safe_bg = '#e8f5e9'  # Light green background
        self.button_bg = '#5a5a5a'  # Gray button
        self.button_fg = 'white'
        self.title_fg = '#D84315'  # Orange title

        #Title label
        tk.Label(self, text="Wildfire Detection System", font=("Arial", 16, "bold"), 
                 bg=self.normal_bg, fg=self.title_fg).pack(pady=20)
        #Upload button
        self.upload_button = Button(self, text="Upload Image", command=self.upload_image, 
                font=("Arial", 12), bg=self.button_bg, fg=self.button_fg)
        self.upload_button.pack(pady=20)
        #Instructions label
        tk.Label(self, text="Click the button to select an image for analysis", 
                font=("Arial", 12), bg=self.normal_bg).pack(pady=10)
        #Status label
        self.status_label = tk.Label(self, text="Ready to analyze images", wraplength=450, 
                bg=self.normal_bg, font=("Arial", 12))
        self.status_label.pack(pady=20)
        #Results label
        self.result_label = tk.Label(self, text="", wraplength=450, 
                bg=self.normal_bg, font=("Arial", 12))
        self.result_label.pack(pady=10)

    #Opens file diaglog to select an image and process it
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("TIFF Files", "*.tif *.tiff"), ("JPEG Files", "*.jpg *.jpeg"), ("PNG Files", "*.png"), ("All Files", "*.*")]
        )
        
        if file_path:
            self.status_label.config(text=f"Processing {os.path.basename(file_path)}...")
            self.update()
            self.process_file(file_path)

    def process_file(self, file_path):
        classification = classify_image(file_path)
        if classification == 'Predicted class: Smoke':
            # Change background to light red if smoke is detected
            self.configure(bg=self.smoke_bg)
            self.status_label.config(bg=self.smoke_bg)
            self.result_label.config(bg=self.smoke_bg)
            
            # Update labels to match background
            for widget in self.winfo_children():
                if isinstance(widget, tk.Label):
                    widget.configure(bg=self.smoke_bg)
            
            location = get_gps_coordinates(file_path)
            if location:
                lat, lon = location
                api_key = "AIzaSyDjUxMxZPQO-wTmmNOvQ_I0t3cbDkPdRy0"  # My API key
                fire_station = get_fire_station_phone_number(lat, lon, api_key)
                
                # Check if the result is a dictionary
                if isinstance(fire_station, dict):
                    # Each item on new line
                    alert_message = (
                        f"Wildfire detected at:\n\n"
                        f"{lat}, {lon}.\n\n"
                        f"Nearest Fire Station:\n"
                        f"{fire_station['name']}\n"
                        f"{fire_station['phone']}\n"
                        f"{fire_station['address']}"
                    )
                else:
                    # Fallback for string responses
                    alert_message = f"Wildfire detected at:\n\n{lat}, {lon}.\n\nNearest Fire Station:\n{fire_station}"
            else:
                alert_message = "Smoke detected but no location data found."  #Error handling
        else:
            # Change background to light green if no smoke
            self.configure(bg=self.safe_bg)
            self.status_label.config(bg=self.safe_bg)
            self.result_label.config(bg=self.safe_bg)
            
            # Update labels to match background
            for widget in self.winfo_children():
                if isinstance(widget, tk.Label):
                    widget.configure(bg=self.safe_bg)
            
            alert_message = classification

        self.status_label.config(text="Analysis completed")
        self.result_label.config(text=alert_message)

# Run the GUI
if __name__ == "__main__":
    app = WildfireDetection()
    app.mainloop()      #Start event loop