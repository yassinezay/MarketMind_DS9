from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from services.text_generation.generate_ad import generate_ad_pipeline
from services.rag_memory.create_memory_for_llm import store_pdf_to_faiss
from services.image_generation.wallyscar_pipeline import WallysCarPipeline
from services.text_generation.copywriter import generate_visual_prompt_locally
from sentence_transformers import SentenceTransformer
from services.Charte_graphique_generation.charte_graphique import generate_sd_image
from services.recommandation_strategy.pipeline import (
    load_and_clean_data, load_model,
    compute_embeddings, recommander_strategie
)
from services.audio_generation.bark_audio_generator import (
    generate_audio,detect_language_and_select_voice
)
import requests
import os
import uuid
import joblib
import numpy as np
from scipy.io.wavfile import write
from transformers import AutoProcessor, BarkModel
from uuid import uuid4
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import pandas as pd
from fastapi.responses import StreamingResponse
from io import BytesIO
import shutil
app = FastAPI()

# -----------------------------
# 🔤 Ad Text Generation Endpoint
# -----------------------------
class AdRequest(BaseModel):
    product_idea: str
    motif: str  # Should be a choice like "elegance", "excitement", etc.
    language: str
    platform: str  # e.g., "Facebook", "Instagram", etc.

@app.post("/generate")
def generate_advertisement(request: AdRequest):
    ad_text, headline = generate_ad_pipeline(
        query=request.product_idea,
        motif=request.motif,
        language=request.language,
        platform=request.platform
    )
    return {
        "ad_text": ad_text,
        "headline": headline
    }

# -----------------------------
# 📄 PDF Upload and Ingestion
# -----------------------------
@app.post("/upload-pdf")
def upload_pdf(file: UploadFile = File(...)):
    try:
        file_location = f"temp/{file.filename}"
        with open(file_location, "wb") as f:
            f.write(file.file.read())

        chunk_count = store_pdf_to_faiss(file_location)
        return {"message": f"✅ {chunk_count} chunks indexed from {file.filename}"}
    except Exception as e:
        return {"error": str(e)}

# -----------------------------
# 🎥 Video Generation via Flask
# -----------------------------
# 🎥 Video Generation via Flask


# -----------------------------
# 🖼️ Image Generation with Logo Overlay
# -----------------------------
model_path = "models/wallyscar_session.ckpt"
image_generator = WallysCarPipeline(model_path)

@app.post("/generate-image")
def generate_image(prompt: str):
    output_dir = "generated_images"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{uuid.uuid4()}.png"
    output_path = os.path.join(output_dir, filename)

    generated_path = image_generator.generate(prompt, output_path)

    logo_path = "assets/wallylogo.png"
    final_path = WallysCarPipeline.overlay_logo(generated_path, logo_path, scale=0.15)

    return {
        "message": "✅ Image generated with logo",
        "path": final_path
    }
# -----------------------------
# 🖼️ Visual Prompt Generation
# -----------------------------
@app.post("/generate-visual-prompt")
def generate_visual_prompt(
    product_description: str = Form(...),
    motif: str = Form(...)
):
    visual_prompt = generate_visual_prompt_locally(product_description, motif)
    return {
        "visual_prompt": visual_prompt
    }
# -----------------------------
# 🖼️ Image Generation from Descriptio
# -----------------------------
@app.post("/generate-image-from-description")
def generate_image_from_description(
    product_description: str = Form(...),
    motif: str = Form(...)
):
    # Step 1: Generate the visual prompt
    visual_prompt = generate_visual_prompt_locally(product_description, motif)
    # Step 2: Generate image using the visual prompt
    output_dir = "templatemo_562_space_dynamic/generated_imagess"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{uuid.uuid4()}.png"
    output_path = os.path.join(output_dir, filename)
    generated_path = image_generator.generate(visual_prompt, output_path)
    

 
    # Step 3: Overlay logo
    logo_path = "assets/wallylogo.png"
    final_path = WallysCarPipeline.overlay_logo(generated_path, logo_path, scale=0.15)

    # Response
    return {
        "message": "✅ Image generated from product description",
        "visual_prompt": visual_prompt,
        "image_path": final_path
    }

# -----------------------------
# 📈 Load Engagement Prediction Model
# -----------------------------
engagement_model_path = "models/best_model_random_forest.pkl"
embedder_path = "models/embedder.pkl"
label_encoder_path = "models/label_encoder.pkl"

engagement_model = joblib.load(engagement_model_path)
embedder = joblib.load(embedder_path)
label_encoder = joblib.load(label_encoder_path)

class EngagementRequest(BaseModel):
    ad_text: str

@app.post("/predict-engagement")
def predict_engagement_endpoint(request: EngagementRequest):
    try:
        embedding = embedder.encode([request.ad_text], show_progress_bar=False)
        prediction = engagement_model.predict(embedding)
        label = label_encoder.inverse_transform(prediction)[0]
        return {
            "message": "✅ Engagement predicted successfully",
            "engagement_score": label
        }
    except Exception as e:
        return {
            "error": str(e)
        }
# -----------------------------
# 💡 Load Strategy Recommendation Components
# -----------------------------
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer, util


# Chargement des composants
strategies_path = "./marketing_strategies_v2.csv"
model_path = "models/model_marketing_reco"

# Fonctions nécessaires
def clean_text(text):
    if isinstance(text, str):
        return text.encode('latin1').decode('utf-8', errors='ignore')
    return text

def load_and_clean_data(csv_path: str):
    df = pd.read_csv(csv_path, encoding='latin1', sep=';')
    for col in df.columns:
        df[col] = df[col].apply(clean_text)
    return df

def load_model(model_path: str):
    return SentenceTransformer(model_path)

def compute_embeddings(model, texts):
    return model.encode(texts)

def recommander_strategie(commentaires, model, df_strat, strategy_embeddings):
    recommandations = []
    if isinstance(commentaires, str):
        commentaires = [commentaires]
    comment_embeddings = compute_embeddings(model, commentaires)
    for i, comment in enumerate(commentaires):
        similarities = util.cos_sim(comment_embeddings[i], strategy_embeddings)[0]
        best_idx = similarities.argmax().item()
        strat = df_strat.iloc[best_idx]
        recommandations.append({
            "Commentaire": comment,
            "Stratégie recommandée": strat["Stratégie"],
            "Objectif": strat["Objectif"],
            "Description": strat["Description"]
        })
    return recommandations

# Initialisation
df_strat = load_and_clean_data(strategies_path)
reco_model = load_model(model_path)
strategy_texts = df_strat['Description'].tolist()
strategy_embeddings = compute_embeddings(reco_model, strategy_texts)

# Pydantic models
class RecoRequest(BaseModel):
    comments: list[str]

class PlotRequest(BaseModel):
    recommendations: list[dict]

# Endpoint pour recommander les stratégies
@app.post("/recommend-strategies")
def recommend_strategies(request: RecoRequest):
    try:
        recommandations = recommander_strategie(
            request.comments, reco_model, df_strat, strategy_embeddings
        )
        return {
            "message": "✅ Recommandations générées avec succès",
            "recommendations": recommandations
        }
    except Exception as e:
        return {"error": str(e)}

# Endpoint pour générer le graphique à partir des recommandations
@app.post("/recommendation-plot")
def plot_recommendations(request: PlotRequest):
    try:
        df = pd.DataFrame(request.recommendations)

        strategie_counts = df['Stratégie recommandée'].value_counts().reset_index()
        strategie_counts.columns = ['Stratégie', 'Nombre']

        plt.figure(figsize=(10, 6))
        sns.barplot(data=strategie_counts, x='Nombre', y='Stratégie', palette='viridis')
        plt.title("Stratégies marketing recommandées en fonction des commentaires")
        plt.xlabel("Nombre de fois recommandée")
        plt.ylabel("Stratégie")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return {"plot_base64": image_base64}

    except Exception as e:
        return {"error": str(e)}


# -----------------------------
# 🎨 Generate Brand Charte Graphique
# -----------------------------
@app.post("/generate-charte")
async def generate_charte_graphique(logo: UploadFile = File(...)):
    try:
        # Save uploaded logo
        os.makedirs("temp", exist_ok=True)
        logo_path = f"temp/{uuid.uuid4()}_{logo.filename}"
        with open(logo_path, "wb") as f:
            f.write(await logo.read())

        # Extract primary palette from logo
        palette = extract_logo_colors(logo_path, n_colors=4)

        # Define a secondary (product) palette - can be static or from another source
        palette2 = ["#f4a261", "#e76f51", "#2a9d8f", "#264653"]

        # Initialize PDF
        pdf = BrandCharte()
        pdf.add_page()

        # --- Header ---
        pdf.set_font('Helvetica', 'B', 24)
        pdf.set_text_color(60, 60, 60)
        pdf.cell(0, 15, "CHARTE GRAPHIQUE ", 0, 1, 'C')

        pdf.set_draw_color(200, 200, 200)
        pdf.line(15, pdf.get_y(), pdf.w - 15, pdf.get_y())
        pdf.ln(10)

        # --- Logo Section ---
        pdf.set_font('Helvetica', 'B', 16)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(0, 10, "LOGO PRINCIPAL", 0, 1)

        logo_x = (pdf.w - 40) / 2
        logo_y = pdf.get_y()
        pdf.image(logo_path, x=logo_x, y=logo_y, w=40)
        pdf.rect(logo_x - 2.5, logo_y - 2, 45, 45)

        pdf.ln(50)

        # --- Color Palettes ---
        pdf.set_font('Helvetica', 'B', 16)
        pdf.cell(0, 10, "PALETTES COULEURS", 0, 1)

        pdf.set_font('Helvetica', '', 12)
        pdf.cell(0, 8, "Palette Logo:", 0, 1)
        pdf.draw_color_grid(palette, pdf.get_x(), pdf.get_y(), size=25)

        pdf.ln(30)

        pdf.cell(0, 8, "Palette Produits:", 0, 1)
        pdf.draw_color_grid(palette2, pdf.get_x(), pdf.get_y(), size=25)

        pdf.ln(30)

        # --- Typography ---
        pdf.set_font('Helvetica', 'B', 16)
        pdf.cell(0, 10, "TYPOGRAPHIE", 0, 1)

        pdf.set_font('Times', 'B', 14)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(0, 8, "Playfair Display Bold", 0, 1)
        pdf.set_font('Times', '', 12)
        pdf.cell(0, 6, "Pour les titres et accents", 0, 1)
        pdf.ln(5)

        pdf.set_font('Helvetica', '', 14)
        pdf.set_text_color(50, 50, 50)
        pdf.cell(0, 8, "Open Sans Regular", 0, 1)
        pdf.set_font('Helvetica', '', 12)
        pdf.cell(0, 6, "Pour le texte courant et paragraphes", 0, 1)

        pdf.ln(15)

        # --- Safe Zones ---
        pdf.set_font('Helvetica', 'B', 16)
        pdf.cell(0, 10, "ZONES DE PROTECTION", 0, 1)

        pdf.set_font('Helvetica', '', 10)
        pdf.multi_cell(0, 6, "Espace minimal autour du logo : 50% de sa hauteur\nNe jamais placer d'éléments à moins de cette distance.", 0, 'L')

        # Save PDF
        os.makedirs("charte_graphique", exist_ok=True)
        pdf_path = f"charte_graphique/charte_{uuid.uuid4().hex[:8]}.pdf"
        pdf.output(pdf_path)

        return {
            "message": "✅ Charte graphique générée avec succès.",
            "pdf_path": pdf_path,
            "palette_logo": palette,
            "palette_produits": palette2
        }
    except Exception as e:
        return {"error": str(e)}

# -----------------------------

@app.post("/generate-style-guide/")
async def generate_style_guide(image_paths: list[str]):
    try:
        # Set up the canvas, fonts, and brand details
        canvas = Image.new("RGB", (1800, 1400), "#f9f9f9")
        draw = ImageDraw.Draw(canvas)

        title_font = load_font(brand["typo_title"], 52)
        body_font = load_font(brand["typo_body"], 28)
        footer_font = load_font(brand["typo_body"], 24)

        left_x, right_x = 50, 950
        y_left, y_right = 60, 60

        # === Logo Variations Section ===
        y_left = section_title("Logo Variations", left_x, y_left)
        logo = Image.open(brand["logo_path"]).convert("RGBA").resize((110, 110))
        for i, bg in enumerate(brand["colors"]):
            bg_block = Image.new("RGB", (160, 160), bg)
            bg_block.paste(logo, (25, 25), logo)
            canvas.paste(bg_block, (left_x + i * 180, y_left))
        y_left += 190

        # === Color Palette Section ===
        y_left = section_title("Color Palette", left_x, y_left)
        for i, c in enumerate(brand["colors"]):
            draw_circle(draw, (left_x + 60 + i * 130, y_left + 40), 30, c)
            draw.text((left_x + 40 + i * 130, y_left + 90), c.upper(), fill="#222", font=body_font)
        y_left += 140

        # === Typography Section ===
        y_left = section_title("Typography", left_x, y_left)
        draw.text((left_x + 10, y_left + 20), f"Title Font: {Path(brand['typo_title']).stem}", fill="#333", font=body_font)
        draw.text((left_x + 10, y_left + 60), f"Body Font: {Path(brand['typo_body']).stem}", fill="#333", font=body_font)
        y_left += 120

        # === Business Card Section ===
        y_right = add_section("Business Card", "Business card for Wallys, metallic navy & silver, minimal layout", right_x, y_right, (700, 400))

        # === Product Showcase Section ===
        y_right = section_title("Product Showcase", right_x, y_right)
        img_showcase = generate_preview_with_chunks("Wallys SUV studio photoshoot, sleek Tunisian car", image_paths, size=(700, 400))
        y_right = paste_image_with_border(img_showcase, right_x, y_right)

        # === Showroom Ambiance Section ===
        y_left = add_section("Showroom Ambiance", "Tunisian car showroom, black matte floor, Wallys branding", left_x, y_left, (800, 300))

        # === Social Banner Section ===
        y_right = add_section("Social Banner", "Black-and-white icons banner for car brand Wallys: car, wheel, GPS, wrench", right_x, y_right, (700, 220))

        # === Footer ===
        draw.text((canvas.width // 2 - 100, canvas.height - 60),
                  f"© {brand['name']} · Designed with AI · {brand['style']}",
                  font=footer_font, fill="#777")

        # Apply a soft effect
        canvas = apply_soft_effect(canvas)

        # Save the style guide
        output_path = "charte_graphique/styleguide_wallys_FINAL.jpg"
        canvas.save(output_path, quality=95)

        # Return the saved style guide as a response
        return FileResponse(output_path, media_type="image/jpeg")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# -----------------------------
# 🎵 Audio Generatio
# -----------------------------
    
class AudioRequest(BaseModel):
    text: str
    voice: str  # "male" or "female"

@app.post("/audio_generate")
async def audio_generate(req: AudioRequest):
    try:
        text = req.text
        voice = req.voice.lower() if req.voice.lower() in ["male", "female"] else "male"

        processor = AutoProcessor.from_pretrained("suno/bark")
        model = BarkModel.from_pretrained("suno/bark")

        voice_preset = detect_language_and_select_voice(text, voice)
        inputs = processor(text, voice_preset=voice_preset)
        audio_array = model.generate(**inputs).cpu().numpy().squeeze()
        sample_rate = model.generation_config.sample_rate

        # Sauvegarder en .wav
        output_dir = "templatemo_562_space_dynamic/generated_audio"
        os.makedirs(output_dir, exist_ok=True)
        filename = f"generated_{uuid4().hex[:8]}.wav"
        audio_path = os.path.join(output_dir, filename)
        write(audio_path, sample_rate, audio_array)

        return JSONResponse(content={"audio_path": audio_path}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
# === FastAPI route ===
sections = [
    {
        "title": "Business Card",
        "prompt_template": "Luxury matte-finish business card for {brand}. {material} texture, {colors} palette, {style} typography. Includes: {custom_elements}.",
        "size": (750, 450),
        "custom_elements": ["QR code", "embossed logo", "signature foil accent"],
        "material": ["brushed metal", "carbon fiber", "matte parchment"],
        "extra_params": {"lighting": "studio softbox", "shadow_depth": 0.5}
    },
    {
        "title": "Product Showcase",
        "prompt_template": "Award-winning car advertisement shot for {brand} SUV. {environment} background, {colors} highlights, {style} cinematography. Focus on: {focus_point}.",
        "size": (1200, 675),
        "environment": ["mountain sunset", "urban loft", "futuristic tunnel"],
        "focus_point": ["grille details", "interior dashboard", "wheel design"],
        "extra_params": {"bokeh": "85mm f/1.2", "dynamic_range": "HDR"}
    },
    {
        "title": "Tunisian Showroom",
        "prompt_template": "Architectural digest-worthy showroom for {brand} in Tunis. {flooring} flooring, {lighting} lighting, {style} decor. Featuring: {showcase_items}.",
        "size": (1600, 900),
        "flooring": ["black marble", "polished concrete", "geometric tiles"],
        "lighting": ["track lighting", "neon accents", "skylight natural"],
        "showcase_items": ["2024 flagship model", "custom parts display", "VR configurator"],
        "extra_params": {"perspective": "wide-angle", "human_scale": True}
    },
    {
        "title": "Banner Pro",
        "prompt_template": "Next-gen monochrome banner for {brand}. {layout} layout with {icons} + {surprise_element}. {style} aesthetic. Hidden detail: {easter_egg}.",
        "size": (1500, 500),
        "icons": ["car silhouette", "circuit board pattern", "3D-wireframe wheel"],
        "surprise_element": ["AI hologram", "Tunisian motif", "dynamic gradient"],
        "layout": ["asymmetric", "modular grid", "fluid typography"],
        "easter_egg": ["micro-brand slogan", "pixel-art icon", "hidden date"],
        "extra_params": {"animation_ready": True, "CMYK_print_safe": False}
    }
]
@app.post("/generate_chartegraphique_bulk")
async def generate_chartegraphique_bulk(
    brand_name: str = Form(...),
    main_color: str = Form(...),
    other_colors: str = Form(...),  # comma-separated
    style_keywords: str = Form(...),
    slogan: str = Form(...),
    logo: UploadFile = File(None)
):
    try:
        logo_path = None
        if logo:
            os.makedirs("logos", exist_ok=True)
            logo_path = f"logos/{uuid.uuid4()}.png"
            with open(logo_path, "wb") as f:
                shutil.copyfileobj(logo.file, f)

        # Basic color parsing
        palette = [main_color] + [c.strip() for c in other_colors.split(",") if c.strip()]

        client_data = {
            "brand_name": brand_name,
            "main_color": main_color,
            "other_colors": palette[1:],
            "color_palette": palette,
            "style_keywords": style_keywords,
            "slogan": slogan,
            "logo_path": logo_path,
        }

        # REUSE your notebook logic here
        results = []
        for section in sections:  # <- import or define this
            prompt = section["prompt_template"].format(
                brand=client_data["brand_name"],
                colors=", ".join(client_data["color_palette"]),
                style=client_data["style_keywords"],
                material=section.get("material", ["matte"])[0],
                custom_elements=", ".join(section.get("custom_elements", [])),
                environment=section.get("environment", ["studio"])[0],
                focus_point=section.get("focus_point", ["car body"])[0],
                flooring=section.get("flooring", ["tile"])[0],
                lighting=section.get("lighting", ["daylight"])[0],
                showcase_items=", ".join(section.get("showcase_items", [])),
                layout=section.get("layout", ["grid"])[0],
                icons=", ".join(section.get("icons", [])),
                surprise_element=section.get("surprise_element", ["hologram"])[0],
                easter_egg=section.get("easter_egg", ["secret"])[0],
            )
            image = generate_sd_image(prompt, size=section["size"])
            output_dir = os.path.abspath("output")
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"{uuid.uuid4()}.png")
            image.save(filename)
            results.append({
                "title": section["title"],
                "filename": filename,  # this is an absolute path now
                "prompt": prompt
            })

        return JSONResponse(content={"results": results})
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
# -----------------------------
# 🌐 Root Health Check
# ----------------------------- 
@app.get("/")
def root():
    return {
        "message": "🚀 MarketMind API is running.",
        "endpoints": [
            "/generate",
            "/upload-pdf",
            "/generate-video",
            "/generate-visual-prompt",
            "/generate-image",
            "/predict-engagement",
            "/generate-charte",
            "/recommend-strategies"
        ]
    }
