from flask import Flask, render_template, request, jsonify, send_from_directory
import requests,os
import logging
import matplotlib.pyplot as plt 
import shutil

LUXAND_API_KEY = '717c01f80d704d09a8a84f78e1cd4e0e'
LUXAND_API_URL = 'https://api.luxand.cloud/faces/register'
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/index')
def index1():
    return render_template("index.html")
@app.route('/login')
def login():
    return render_template("login.html")

@app.route('/register')
def register():
    return render_template("register.html")
@app.route('/video')
def video():
    return render_template("video.html")
@app.route('/free')
def free():
    return render_template("free.html")

@app.route('/premium')
def premium():
    return render_template("premium.html")
@app.route('/ultimate')
def ultimate():
    return render_template("ultimate.html")

@app.route('/face')
def face():
    return render_template("face.html")


@app.route('/text-generation', methods=["GET", "POST"])
def text_generation():
    if request.method == "POST":
        # Step 1: Upload the PDF file to FastAPI
        pdf_file = request.files.get("pdf_file")
        if not pdf_file:
            return render_template("text_generation.html", error_message="Please upload a PDF.")

        files = {"file": (pdf_file.filename, pdf_file.stream, pdf_file.mimetype)}
        upload_url = "http://127.0.0.1:5001/upload-pdf"
        upload_response = requests.post(upload_url, files=files)

        if upload_response.status_code != 200:
            return render_template("text_generation.html", error_message="PDF upload failed.")

        # Step 2: Collect other form inputs
        product_idea = request.form["product_idea"]
        motif = request.form["motif"]
        language = request.form["language"]
        platform = request.form["platform"]

        # Step 3: Call the generation endpoint
        payload = {
            "product_idea": product_idea,
            "motif": motif,
            "language": language,
            "platform": platform
        }

        fastapi_url = "http://127.0.0.1:5001/generate"
        response = requests.post(fastapi_url, json=payload)

        if response.status_code == 200:
            data = response.json()
            return render_template("text_generation.html", headline=data.get("headline"), ad_text=data.get("ad_text"))
        else:
            return render_template("text_generation.html", error_message=f"Ad generation failed: {response.text}")

    return render_template("text_generation.html")

GENERATED_IMAGES_FOLDER = 'generated_imagess'
# Ensure the folder exists
if not os.path.exists(GENERATED_IMAGES_FOLDER):
    os.makedirs(GENERATED_IMAGES_FOLDER)

# Route to serve images
@app.route('/generated_images/<filename>')
def serve_generated_image(filename):
    # Serve the image from the generated folder
    return send_from_directory(GENERATED_IMAGES_FOLDER, filename)

@app.route('/image-generation', methods=["GET", "POST"])
def image_generation():
    if request.method == "POST":
        product_description = request.form["product_description"]
        motif = request.form["motif"]

        payload = {
            "product_description": product_description,
            "motif": motif
        }

        # FastAPI URL for image generation
        fastapi_url = "http://127.0.0.1:5001/generate-image-from-description"
        response = requests.post(fastapi_url, data=payload)

        if response.status_code == 200:
            data = response.json()
            image_path = data.get("image_path", "")
            # Extract the image filename from the full path
            image_filename = os.path.basename(image_path)
            return render_template("image_generation.html", image_filename=image_filename)
        else:
            error_message = f"Error: {response.status_code} - {response.text}"
            return render_template("image_generation.html", error_message=error_message)

    return render_template("image_generation.html")

from fastapi.responses import FileResponse

@app.get("/download-audio")
def download_audio():
    audio_path = os.path.join(os.path.dirname(__file__), "..", "templatemo_562_space_dynamic", "generated_audio", "generated.wav")
    return FileResponse(audio_path, media_type='audio/wav', filename="headline_audio.wav")

@app.route('/audio_generation', methods=["GET", "POST"])
def audio_generation():
    if request.method == "POST":
        text = request.form.get("text")
        voice_gender = request.form.get("voice", "male")  # Default to male if not provided

        if not text:
            return render_template("audio_generation.html", error_message="Please enter text to generate audio.")

        # Appel à l'API FastAPI avec le style de voix
        fastapi_url = "http://127.0.0.1:5001/audio_generate"
        response = requests.post(fastapi_url, json={"text": text, "voice": voice_gender})

        if response.status_code == 200:
            data = response.json()
            audio_path = data.get("audio_path", "")
            audio_filename = os.path.basename(audio_path)
            return render_template("audio_generation.html", audio_filename=audio_filename)
        else:
            return render_template("audio_generation.html", error_message=f"Audio generation failed: {response.text}")

    return render_template("audio_generation.html")
 

GENERATED_AUDIO_FOLDER = 'generated_audio'
# Ensure the folder exists
if not os.path.exists(GENERATED_AUDIO_FOLDER):
    os.makedirs(GENERATED_AUDIO_FOLDER)


@app.route('/generated_audio/<filename>')
def serve_audio(filename):
    return send_from_directory(GENERATED_AUDIO_FOLDER, filename)

@app.route('/verify-face', methods=['POST'])
def verify_face():
    try:
        data = request.get_json()
        image_base64 = data.get('image')
        name = data.get('name')  # New: get name from the form

        if not image_base64:
            return jsonify(success=False, message='Missing image'), 400

        if image_base64.startswith('data:image'):
            image_base64 = image_base64.split(',', 1)[1]

        headers = {
            'token': LUXAND_API_KEY,
            'Content-Type': 'application/json'
        }
        payload = {'photo': image_base64}
        response = requests.post('https://api.luxand.cloud/photo/search', headers=headers, json=payload)

        if response.status_code == 200:
            matches = response.json()
            if matches:
                detected_name = matches[0].get('name')
                if detected_name.lower() == name.lower():
                    return jsonify(success=True, name=detected_name)
                else:
                    return jsonify(success=False, message='Face does not match the entered name.')
            else:
                return jsonify(success=False, message='No match found.')
        else:
            return jsonify(success=False, message=response.text), response.status_code
    except Exception as e:
        logging.exception("Error in /verify-face:")
        return jsonify(success=False, message=str(e)), 500


@app.route('/strategy_recommendation', methods=['GET', 'POST'])
def strategy_recommendation():
    recommendations = []
    plot_base64 = None  # Pour afficher le graphique s'il est généré
    if request.method == 'POST':
        FASTAPI_URL = "http://127.0.0.1:5001"  # Adresse de ton backend FastAPI

        comments = request.form.getlist('comments')  # Récupère la liste de commentaires depuis le formulaire
        payload = {"comments": comments}
        try:
            # 1. Appel à FastAPI pour obtenir les recommandations
            response = requests.post(f"{FASTAPI_URL}/recommend-strategies", json=payload)
            data = response.json()

            if 'recommendations' in data:
                recommendations = data['recommendations']

                # 2. Appel à FastAPI pour générer le graphique à partir des recommandations
                plot_response = requests.post(f"{FASTAPI_URL}/recommendation-plot", json={"recommendations": recommendations})
                plot_data = plot_response.json()

                if 'plot_base64' in plot_data:
                    plot_base64 = plot_data['plot_base64']
                else:
                    plot_base64 = None
            else:
                recommendations = [f"Erreur: {data.get('error', 'Inconnue')}"]
        except Exception as e:
            recommendations = [f"Erreur de requête: {str(e)}"]

    # 3. Envoi des résultats à la page HTML
    return render_template(
        'strategy_recommendation.html',
        recommendations=recommendations,
        plot_base64=plot_base64
    )





@app.route('/text_generation_with_engagement', methods=["GET", "POST"])
def text_generation_with_engagement():
    if request.method == "POST":
        # 1. Upload du PDF
        # Step 1: Upload the PDF file to FastAPI
        pdf_file = request.files.get("pdf_file")
        if not pdf_file:
            return render_template("text_generation_with_engagement.html", error_message="Please upload a PDF.")

        files = {"file": (pdf_file.filename, pdf_file.stream, pdf_file.mimetype)}
        upload_url = "http://127.0.0.1:5001/upload-pdf"
        upload_response = requests.post(upload_url, files=files)

        if upload_response.status_code != 200:
            return render_template("text_generation_with_engagement.html", error_message="PDF upload failed.")

        # 2. Données du formulaire
        product_idea = request.form["product_idea"]
        motif = request.form["motif"]
        language = request.form["language"]
        platform = request.form["platform"]

        # 3. Génération de texte
        payload = {
            "product_idea": product_idea,
            "motif": motif,
            "language": language,
            "platform": platform
        }

        generation_response = requests.post("http://127.0.0.1:5001/generate", json=payload)

        if generation_response.status_code != 200:
            return render_template("text_generation_with_engagement.html", error_message="Échec de la génération de texte.")

        generated_data = generation_response.json()
        ad_text = generated_data.get("ad_text", "")
        headline = generated_data.get("headline", "")

        # 4. Prédiction d'engagement
        engagement_payload = {"ad_text": ad_text}
        engagement_response = requests.post("http://127.0.0.1:5001/predict-engagement", json=engagement_payload)

        if engagement_response.status_code != 200:
            return render_template("text_generation_with_engagement.html", error_message="Échec de la prédiction d'engagement.")

        engagement_data = engagement_response.json()
        engagement_score = engagement_data.get("engagement_score", "N/A")

        return render_template(
            "text_generation_with_engagement.html",
            headline=headline,
            ad_text=ad_text,
            engagement_score=engagement_score
        )

    return render_template("text_generation_with_engagement.html")

@app.route("/generate_charte_bulk", methods=["GET", "POST"])
def generate_charte_bulk():
    if request.method == "POST":
        try:
            files = {"logo": request.files["logo"]} if "logo" in request.files else {}

            data = {
                "brand_name": request.form.get("brand_name"),
                "main_color": request.form.get("main_color"),
                "other_colors": request.form.get("other_colors"),
                "style_keywords": request.form.get("style_keywords"),
                "slogan": request.form.get("slogan"),
            }

            fastapi_url = "http://127.0.0.1:5001/generate_chartegraphique_bulk"
            response = requests.post(fastapi_url, data=data, files=files)

            if response.status_code == 200:
                results = response.json()["results"]
                os.makedirs(os.path.join("static", "generated"), exist_ok=True)

                image_outputs = []
                for res in results:
                    # Get absolute source path from FastAPI
                    src_path = res["filename"]  # already absolute
                    filename = os.path.basename(src_path)
                    out_path = os.path.join("static", "generated", filename)

                    # Copy to Flask's static folder
                    shutil.copyfile(src_path, out_path)
                    image_outputs.append({"title": res["title"], "filename": filename, "prompt": res["prompt"]})

                return render_template("generate_charte_bulk.html", images=image_outputs)
            else:
                return render_template("generate_charte_bulk.html", error_message="Failed: " + response.text)

        except Exception as e:
            return render_template("generate_charte_bulk.html", error_message=f"Error: {str(e)}")

    return render_template("generate_charte_bulk.html")


    
if __name__ == '__main__':
    app.run(debug=True, port=5000)
