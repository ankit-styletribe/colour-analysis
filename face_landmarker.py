from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from flask import Flask, request, jsonify

# --------------------- Setup --------------

MODEL_PATH = 'face_landmarker_v2_with_blendshapes.task'
HAIR_MODEL_PATH = 'hair_segmenter.tflite'
image_path = 'face_samples/dangi.jpg'

visualize_color_analysis = 0 # set as 1 if need to visualise the color analysis
FaceLandmarkerVisualizer = 0 # set as 1 if need to visualise the landmarker

# Define the landmark indices for the irises. Skin tone will be calculated from these.
LANDMARK_INDICES = {
    'left_iris': 473,
    'right_iris': 468,
}

# Initialize MediaPipe solutions.
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a face_landmarker instance with the face_landmarker model:
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

try:
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        min_face_detection_confidence=0.3
    )
    landmarker = FaceLandmarker.create_from_options(options)
    print("FaceLandmarker initialized successfully with detection confidence at 0.3.")
except Exception as e:
    raise RuntimeError(f"Failed to initialize FaceLandmarker. Error: {e}")

# Create a image segmenter instance with the hair segmentation model:
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions

try:
    options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=HAIR_MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        output_category_mask=True)
    segmenter = ImageSegmenter.create_from_options(options)
except Exception as e:
    raise RuntimeError(f"Failed to initialize FaceLandmarker. Error: {e}")

#-------------------- Functions ------------------
def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image

def plot_face_blendshapes_bar_graph(face_blendshapes):
  # Extract the face blendshapes category names and scores.
  face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
  face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
  # The blendshapes are ordered in decreasing score value.
  face_blendshapes_ranks = range(len(face_blendshapes_names))

  fig, ax = plt.subplots(figsize=(12, 12))
  bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
  ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
  ax.invert_yaxis()

  # Label each bar with values
  for score, patch in zip(face_blendshapes_scores, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

  ax.set_xlabel('Score')
  ax.set_title("Face Blendshapes")
  plt.tight_layout()
  plt.show()

# ------ colour visualize -----------
def visualize_colors(colors):
        plt.figure(figsize=(8, 2))

        plt.subplot(1, 3, 1)
        plt.axis('off')
        plt.imshow([[colors['skin_color_rgb']]])
        plt.title(f'Color - Skin {1}')

        plt.subplot(1, 3, 2)
        plt.axis('off')
        plt.imshow([[colors['left_iris_color_rgb']]])
        plt.title(f'Color - Left Iris {2}')

        plt.subplot(1, 3, 3)
        plt.axis('off')
        plt.imshow([[colors['right_iris_color_rgb']]])
        plt.title(f'Color - Right Iris {3}')

        plt.show()

# --- Core Analysis Function ---
def analyze_image_colors(image_np):
    """
    Analyzes an image to find iris and skin colors. Skin color is found by
    sampling a patch 1/8 of the image height below the center of the eyes.
    """
    try:
        rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        detection_result = landmarker.detect(mp_image)

        if not detection_result.face_landmarks:
            return {'status': 'error', 'message': 'No face detected in the image.'}

        image_height, image_width, _ = image_np.shape
        landmarks = detection_result.face_landmarks[0]
        colors = {}
        patch_size = 5
        half_patch = patch_size // 2

        # --- Get Iris Colors and Coordinates ---
        iris_coords = {}
        for name, index in LANDMARK_INDICES.items():
            landmark = landmarks[index]
            center_x = int(landmark.x * image_width)
            center_y = int(landmark.y * image_height)
            iris_coords[name] = {'x': center_x, 'y': center_y}

            start_x = max(0, center_x - half_patch)
            start_y = max(0, center_y - half_patch)
            end_x = min(image_width, center_x + half_patch + 1)
            end_y = min(image_height, center_y + half_patch + 1)

            color_patch = image_np[start_y:end_y, start_x:end_x]
            avg_bgr = np.mean(color_patch, axis=(0, 1))
            avg_rgb = (int(avg_bgr[2]), int(avg_bgr[1]), int(avg_bgr[0]))
            colors[f'{name}_color_rgb'] = avg_rgb

        # --- USER LOGIC: Calculate dynamic skin sample point ---
        # 1. Find the center point between the two irises
        eye_center_x = int((iris_coords['left_iris']['x'] + iris_coords['right_iris']['x']) / 2)
        eye_center_y = int((iris_coords['left_iris']['y'] + iris_coords['right_iris']['y']) / 2)

        # 2. Go 1/8 of the image height below that point
        skin_sample_y = eye_center_y + int(image_height / 8)

        # Ensure the point is within the image bounds
        skin_sample_y = min(skin_sample_y, image_height - 1)

        # 3. Average the color from the new skin sample point
        start_x = max(0, eye_center_x - half_patch)
        start_y = max(0, skin_sample_y - half_patch)
        end_x = min(image_width, eye_center_x + half_patch + 1)
        end_y = min(image_height, skin_sample_y + half_patch + 1)

        skin_patch = image_np[start_y:end_y, start_x:end_x]
        avg_skin_bgr = np.mean(skin_patch, axis=(0, 1))
        avg_skin_rgb = (int(avg_skin_bgr[2]), int(avg_skin_bgr[1]), int(avg_skin_bgr[0]))
        colors['skin_color_rgb'] = avg_skin_rgb

        if (visualize_color_analysis):
            visualize_colors(colors)

        return {'status': 'success', 'data': colors}
    except Exception as e:
        return {'status': 'error', 'message': f'An unexpected error occurred: {str(e)}'}


def get_hair_color(image_np):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
    segmentation_result = segmenter.segment(mp_image)

    mask = segmentation_result.category_mask.numpy_view()  # Shape: [H, W]

    # Hair class ID — check your model’s documentation, often it's 1 or 2
    HAIR_CLASS_ID = 1

    hair_mask = (mask == HAIR_CLASS_ID).astype(np.uint8)  # Binary mask for hair
    hair_mask_3ch = np.stack([hair_mask]*3, axis=-1)

    hair_region = (image_np * hair_mask_3ch).astype(np.uint8)
    # Save the result
    #cv2.imwrite("hair_only.png", hair_region)

    non_black_pixels = hair_region[np.any(hair_region != [0, 0, 0], axis=-1)]
    if non_black_pixels.size == 0:
        return {'status': 'error', 'message': 'No hair pixels found'}

    avg_color = np.mean(non_black_pixels, axis=0)
    avg_color_rgb = tuple(int(c) for c in avg_color)

    return {'status': 'success', 'data': {'hair_color_rgb': avg_color_rgb}}

if __name__ == "__main__":
    #image = mp.Image.create_from_file(image_path)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (512, 512))
    
    # Run Analysis on Image
    skin_eye_result = analyze_image_colors(image)
    print(skin_eye_result)

    hair_result = get_hair_color(image)
    print(hair_result)
    


    #---------------------- Face Landmarker Visualization -------------------------
    if(FaceLandmarkerVisualizer):
        # # STEP 1: Preview Image.
        # img = cv2.imread(image_path)
        # cv2.imshow(img)

        # STEP 2: Create an FaceLandmarker object.
        base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                               output_face_blendshapes=True,
                                               output_facial_transformation_matrixes=True,
                                               num_faces=1)
        detector = vision.FaceLandmarker.create_from_options(options)

        # STEP 3: Load the input image.
        image = mp.Image.create_from_file(image_path)

        # STEP 4: Detect face landmarks from the input image.
        detection_result = detector.detect(image)

        # STEP 5: Process the detection result. In this case, visualize it.
        annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
        cv2.imshow("Annotated Image", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        plot_face_blendshapes_bar_graph(detection_result.face_blendshapes[0])

        print(detection_result.facial_transformation_matrixes)
