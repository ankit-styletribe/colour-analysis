
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
import requests
import boto3
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
from dotenv import load_dotenv
from face_landmarker import analyze_image_colors

load_dotenv()

app = Flask(__name__)

# S3 Configuration
s3_client = boto3.client(
    's3',
    region_name=os.getenv('REGION'),
    aws_access_key_id=os.getenv('AWS_S3_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('AWS_S3_ACCESS_SECRET')
)

class FaceColorExtractor:
    def __init__(self):
        # Load the pre-trained Haar Cascade classifier for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # Load the pre-trained Haar Cascade classifier for eye detection
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.logging = 0 #1 = logs printed in terminal + intermediate images saved as file
        self.visualize = 0 #1 = plot visual output 

    def extract_colors(self, image_path):
        """
        Extract skin, eye, and hair colors from a face image.
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            dict: Colors for skin, eyes, and hair
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Failed to load image")

        # Convert to RGB for processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(image_rgb, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            raise ValueError("No face detected in image")

        # Extract colors for the first detected face
        x, y, w, h = faces[0]
        face_region = image_rgb[y:y+h, x:x+w]
        if(self.logging): 
            print(f"Face Region: {x, y, w, h}")
            cv2.imwrite('face_region.jpg', face_region)

        # Extract skin, eye and hair color
        skin_color = self._extract_skin_color(face_region)
        eye_color = self._extract_eye_color(image_rgb, faces[0])
        hair_color = self._extract_hair_color(image_rgb, y)

        # Update skin and eye color using mediapipe
        data = analyze_image_colors(image_rgb)['data']
        skin_color = data['skin_color_rgb']
        #eye_color = (data['left_iris_color_rgb'] + data['right_iris_color_rgb'])/2
        eye_color = data['left_iris_color_rgb'] 


        rgb_colors = {
            'skin_color': skin_color,
            'eye_color': eye_color,
            'hair_color': hair_color
        }

        # Convert RGB colors to CIELAB
        skin_color_lab = self.rgb_to_lab(skin_color)
        eye_color_lab = self.rgb_to_lab(eye_color)
        hair_color_lab = self.rgb_to_lab(hair_color)

        lab_colors = {
            'skin_color_lab': skin_color_lab,
            'eye_color_lab': eye_color_lab,
            'hair_color_lab': hair_color_lab
        }

        if(self.logging):
            print(f"Skin Color (RGB): {skin_color}, Lab: {skin_color_lab}")
            print(f"Eye Color (RGB): {eye_color}, Lab: {eye_color_lab}")
            print(f"Hair Color (RGB): {hair_color}, Lab: {hair_color_lab}")

        if(self.visualize): 
            # Visualize the detected regions
            self.visualize_regions(image_rgb, face_region, eye_color, x, y, w, h)
            # Visualize the output colors
            self.visualize_colors(rgb_colors)

        return rgb_colors, lab_colors

    def _extract_skin_color(self, face_region):
        """Extract dominant skin color from face region"""
        # Convert to HSV for better skin detection
        hsv = cv2.cvtColor(face_region, cv2.COLOR_RGB2HSV)

        # Define skin color range
        lower_skin = np.array([0, 20, 70])
        upper_skin = np.array([20, 255, 255])

        # Create mask and get average color
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_color = cv2.mean(face_region, mask=mask)[:3]

        return tuple(map(int, skin_color))

    def _extract_eye_color(self, image, face):
        """Extract dominant eye color using eye detection"""
        x, y, w, h = face
        roi_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        eyes = self.eye_cascade.detectMultiScale(roi_gray)

        eye_colors = []
        for (ex, ey, ew, eh) in eyes:
            ex = ex+int(ew/4)
            ey = ey+int(eh/4)
            ew = int(ew/2)
            eh = int(eh/2)
            eye_region = image[ey:ey + eh, ex:ex + ew]
            if(self.logging): 
                print(f"Eye Region: {ex, ey, ew, eh}")
                cv2.imwrite('eye_region.jpg', eye_region)
            eye_colors.append(cv2.mean(eye_region)[:3])

        if eye_colors:
            return tuple(map(int, np.mean(eye_colors, axis=0)))
        return (0, 0, 0)  # Default if no eyes detected

    def _extract_hair_color(self, image, face_y):
        """Extract dominant hair color from the upper region of the face"""
        hair_region = image[0:face_y, :]  # Upper part of the face
        if(self.logging): 
            cv2.imwrite('hair_region.jpg', hair_region)

        # Convert to HSV for better hair detection
        hsv = cv2.cvtColor(hair_region, cv2.COLOR_RGB2HSV)

        # Get average color
        hair_color = cv2.mean(hair_region)[:3]
        return tuple(map(int, hair_color))

    def visualize_regions(self, image, face_region, eye_color, x, y, w, h):
        """Visualize the detected skin, hair, and eye regions"""
        plt.figure(figsize=(10, 8))

        # Show the original image
        plt.subplot(2, 2, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')

        # Show the face region
        plt.subplot(2, 2, 2)
        plt.imshow(face_region)
        plt.title("Face Region")
        plt.axis('off')

        # Draw rectangles around detected eyes
        roi_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        eyes = self.eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            ex = ex+int(ew/4)
            ey = ey+int(eh/4)
            ew = int(ew/2)
            eh = int(eh/2)
            cv2.rectangle(image, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)
            cv2.rectangle(image, (ex+int(ew/4), ey+int(eh/4)), (ex + int(3/4*ew), ey + int(3/4*eh)), (255, 0, 0), 2)

        # Show the image with detected eyes
        plt.subplot(2, 2, 3)
        plt.imshow(image)
        plt.title("Detected Eyes")
        plt.axis('off')

        # Show the hair region (upper part of the face)
        hair_region = image[0:y, :]
        plt.subplot(2, 2, 4)
        plt.imshow(hair_region)
        plt.title("Hair Region")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    # Display dominant skin colors
    def visualize_colors(self,colors):
        plt.figure(figsize=(8, 2))

        plt.subplot(1, 3, 1)
        plt.axis('off')
        plt.imshow([[colors['skin_color']]])
        plt.title(f'Color - Skin {1}')

        plt.subplot(1, 3, 2)
        plt.axis('off')
        plt.imshow([[colors['eye_color']]])
        plt.title(f'Color - Eye {2}')

        plt.subplot(1, 3, 3)
        plt.axis('off')
        plt.imshow([[colors['hair_color']]])
        plt.title(f'Color - Hair {3}')

        plt.show()

    def rgb_to_lab(self, rgb):
        """Convert RGB to CIELAB"""
        # Normalize RGB values to [0, 1]
        rgb_normalized = np.array(rgb) / 255.0
        # Convert to CIELAB
        lab = color.rgb2lab(rgb_normalized.reshape(1, 1, 3))
        return tuple(map(int, lab[0][0]))

def upload_to_s3(file):
    """Upload file to S3 and return the URL"""
    try:
        # Generate unique filename
        timestamp = int(datetime.now().timestamp())
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        file_key = f"public/{timestamp}_img.{file_extension}"
        
        # Upload to S3
        s3_client.upload_fileobj(
            file,
            os.getenv('S3_BUCKET_NAME'),
            file_key,
            ExtraArgs={
                'ContentType': file.content_type
            }
        )
        
        # Return the S3 URL
        file_url = f"{os.getenv('S3_BUCKET_URL')}{file_key}"
        return file_url
        
    except Exception as e:
        print(f"S3 upload error: {str(e)}")
        raise e

def fetch_image_from_url(image_url):
    """Download image from URL and save locally"""
    response = requests.get(image_url)
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image

def extract_base_season(api_response):
    """
    Extracts the base season from API response based on the highest score.

    api_response: dict (parsed JSON from API)

    Returns: str (base season name: "winter", "summer", "spring", "autumn")
    """
    if "results" not in api_response:
        return None

    # Find the result with the maximum score
    results = api_response["results"]
    best_result = max(results, key=lambda x: x["score"])
    return best_result["season"]

def get_base_season(lab_colors):
    """Make a GET API call to the specified URL"""
    lab_values = ','.join(map(str, lab_colors['skin_color_lab'] + lab_colors['hair_color_lab'] + lab_colors['eye_color_lab']))
    url = f"https://api.colorwise.me/four-seasons/{lab_values}"
    
    response = requests.get(url)
    if response.status_code == 200:
        print("API Response:", response.json())
        return extract_base_season(response.json())
    else:
        print("Error:", response.status_code, response.text)
        return "error finding season"

def detect_sub_season(lab_colors):
    """
    base_season: str ("winter", "summer", "spring", "autumn")
    lab_skin: (L, a, b)
    lab_hair: (L, a, b)
    lab_eye: (L, a, b)
    
    Returns: str (sub-season)
    """
    # Get base_season
    base_season = get_base_season(lab_colors)
    print("Base Season:", base_season)

    # Unpack L*a*b* values
    lab_skin = lab_colors['skin_color_lab']
    lab_hair = lab_colors['hair_color_lab']
    lab_eye = lab_colors['eye_color_lab']

    L_skin, a_skin, b_skin = lab_skin
    L_hair, a_hair, b_hair = lab_hair
    L_eye, a_eye, b_eye = lab_eye

    # Calculate contrast
    contrast_skin_hair = abs(L_skin - L_hair)
    contrast_skin_eye = abs(L_skin - L_eye)
    avg_contrast = (contrast_skin_hair + contrast_skin_eye) / 2

    # Average warmth (a and b values)
    avg_a = (a_skin + a_hair + a_eye) / 3
    avg_b = (b_skin + b_hair + b_eye) / 3

    # Determine overall brightness
    avg_L = (L_skin + L_hair + L_eye) / 3

    # Main decision tree
    if base_season == "winter":
        if avg_L >= 60 and avg_contrast > 30:
            return "Bright Winter"
        elif avg_L <= 40:
            return "Dark Winter"
        else:
            return "True Winter"

    elif base_season == "summer":
        if avg_L >= 65:
            return "Light Summer"
        elif avg_contrast < 20:
            return "Soft Summer"
        else:
            return "True Summer"

    elif base_season == "spring":
        if avg_L >= 65 and avg_contrast > 30:
            return "Clear Spring"
        elif avg_L >= 65:
            return "Light Spring"
        else:
            return "Warm Spring"

    elif base_season == "autumn":
        if avg_L <= 40:
            return "Dark Autumn"
        elif avg_contrast < 20:
            return "Soft Autumn"
        else:
            return "True Autumn"

    else:
        return "Unknown Season"

@app.route('/upload-and-analyze', methods=['POST'])
def upload_and_analyze():
    """Main endpoint that handles file upload to S3 and performs color analysis"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "error": "No file provided"
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "No file selected"
            }), 400

        # Validate file type
        allowed_extensions = {'jpg', 'jpeg', 'png', 'jfif'}
        file_extension = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        if file_extension not in allowed_extensions:
            return jsonify({
                "success": False,
                "error": "Invalid file type. Only JPG, JPEG, PNG, and JFIF are allowed."
            }), 400

        # Upload to S3
        file_url = upload_to_s3(file)
        print(f"File uploaded to S3: {file_url}")

        # Download image for processing
        face_image_path = 'temp_face_image.jpg'
        face_image = fetch_image_from_url(file_url)
        cv2.imwrite(face_image_path, face_image)

        # Extract colors from face
        rgb_colors, lab_colors = extractor.extract_colors(face_image_path)
        
        # Detect sub-season
        detected_season = detect_sub_season(lab_colors)
        print(f"Detected Season: {detected_season}")

        # Clean up temporary file
        if os.path.exists(face_image_path):
            os.remove(face_image_path)

        return jsonify({
            "success": True,
            "file_url": file_url,
            "season": detected_season,
            "colors": {
                "rgb": rgb_colors,
                "lab": lab_colors
            }
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/colour-analysis', methods=['POST'])
def colour_analysis():
    """Endpoint for analyzing color from an existing image URL"""
    face_image_path = 'face_image.jpg'
    face_image = fetch_image_from_url(request.json.get("image_url"))
    cv2.imwrite(face_image_path, face_image)

    try:
        # Extract colors from face
        rgb_colors, lab_colors = extractor.extract_colors(face_image_path)
        
        # Detect Correct sub-season
        detected_season = detect_sub_season(lab_colors)
        print(f"Detected Season: {detected_season}")

        return jsonify({
            "success": True,
            "season": detected_season
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

# Initialize the face color extractor
extractor = FaceColorExtractor()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
