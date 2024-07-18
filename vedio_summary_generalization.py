# Import necessary libraries
from moviepy.editor import VideoFileClip
from transformers import BlipProcessor, BlipForConditionalGeneration
import cv2
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import openai


# Set up OpenAI API key
openai.api_key = "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"

# Load the image-to-text model with GPU support
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")

def calculate_frame_difference(frame1, frame2):
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between two frames
    ssim_index, _ = ssim(gray1, gray2, full=True)

    # Compute color histogram difference
    hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()
    hist_diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    return (1 - ssim_index) + (1 - hist_diff)

def generate_captions_batch(images):
    inputs = processor(images, return_tensors="pt").to("cuda", torch.float16)
    out = model.generate(**inputs)
    return [processor.decode(caption, skip_special_tokens=True) for caption in out]

def summarize_captions(captions, max_summary_length=200):
    # Create a single string with all captions in the batch
    batch_text = " ".join(captions)

    # Call OpenAI to summarize the batch of captions
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Summarize the following text in 200 words: {batch_text}"}
        ],
        max_tokens=150  # Adjust max tokens to ensure the summary is within 200 words
    )

    # Extract the summary from the response
    summary = response['choices'][0]['message']['content'].strip()

    return summary

def process_video_and_generate_captions(video_path, threshold=1.0):
    cap = cv2.VideoCapture(video_path)
    success, prev_frame = cap.read()

    if not success:
        print("Failed to read video file")
        return

    frame_count = 0
    prev_caption = None
    batch_captions = []
    batch_images = []

    while success:
        success, curr_frame = cap.read()
        if not success:
            break

        frame_diff = calculate_frame_difference(prev_frame, curr_frame)

        if frame_diff > threshold:
            frame_rgb = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)  # Convert to PIL image
            batch_images.append(pil_image)

            if len(batch_images) == 32:
                current_captions = generate_captions_batch(batch_images)
                for caption in current_captions:
                    if caption != prev_caption:
                        #print(f"Caption for key frame {frame_count}: {caption}")
                        batch_captions.append(caption)
                        prev_caption = caption
                        frame_count += 1
                batch_images = []

            prev_frame = curr_frame

        if frame_count >= 20:
            # Summarize the batch of captions
            summary = summarize_captions(batch_captions)
            print(f"Summary for batch: {summary}")

            # Reset the batch
            frame_count = 0
            batch_captions = []

    # Summarize any remaining captions
    if batch_captions:
        summary = summarize_captions(batch_captions)
        print(f"Summary for final batch: {summary}")

    cap.release()

# Define the video path from Google Drive
video_path = "/home/ygao/Multimodal-RAG-opensource/video/2016-12-31_0300_US_FOX-News_Hannity.mp4"

# Process the video and generate captions for key frames
process_video_and_generate_captions(video_path, threshold=0.5)  # Adjust threshold as needed