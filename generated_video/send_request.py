import requests

# Replace this with your ngrok-generated URL
NGROK_URL = "https://7a8e-34-125-153-14.ngrok-free.app/generate-video"

image_path = 'generated_images/cd0c13db-df3d-4ad0-b8b4-c4408d4168a7_with_logo.png'
files = {
    'image': ('cd0c13db_with_logo.png', open(image_path, 'rb'), 'image/png')
}

data = {'prompt': 'A beautiful sunset'}

# Send POST request to ngrok URL
response = requests.post(NGROK_URL, files=files, data=data)

# Check response status
if response.status_code == 200:
    with open("generated_video.mp4", "wb") as f:
        f.write(response.content)
    print("Video generated successfully!")
else:
    print(f"Error: {response.status_code}, {response.text}")
