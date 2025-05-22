from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from moviepy.editor import ImageClip
from pyngrok import ngrok
import os
import uuid

app = Flask(__name__)
CORS(app)

OUTPUT_DIR = "generated_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.route("/generate-video", methods=["POST"])
def generate_video():
    try:
        image = request.files.get("image")
        if not image:
            return jsonify({"error": "No image uploaded"}), 400

        prompt = request.form.get("prompt", "")  # Optional use
        image_id = str(uuid.uuid4())
        image_path = f"{image_id}.png"
        video_path = os.path.join(OUTPUT_DIR, f"{image_id}.mp4")

        image.save(image_path)

        # MoviePy video generation
        clip = ImageClip(image_path, duration=3).set_fps(16)
        clip.write_videofile(video_path, codec="libx264", fps=16, logger=None)

        os.remove(image_path)

        return send_file(video_path, mimetype="video/mp4")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Automatically use the saved ngrok authtoken without specifying it manually
    public_url = ngrok.connect(5000)
    print(f"🎯 Ngrok tunnel available at: {public_url}")
    app.run(debug=False, port=5000) 
