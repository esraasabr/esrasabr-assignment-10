import os
import pandas as pd
from flask import Flask, request, render_template, jsonify
from open_clip import create_model_and_transforms, tokenizer
from PIL import Image
import torch.nn.functional as F
import torch
from scipy.spatial.distance import cosine
import open_clip

# Correct model name and pretrained weights
model_name = 'ViT-B-32'
pretrained = 'openai'

# Initialize the model and tokenizer
model, _, preprocess_val = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
tokenizer = open_clip.get_tokenizer(model_name)

# Move the model to the appropriate device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Load embeddings
df = pd.read_pickle('embeddings/image_embeddings.pickle')
# Preprocess the file_name column to ensure it only contains filenames
df['file_name'] = df['file_name'].apply(os.path.basename)



def get_most_similar_images(query_embedding, top_k=5):
    similarities = []
    query_vector = query_embedding.ravel()
    
    for i, embedding in enumerate(df['embedding']):
        embedding_vector = embedding.ravel()
        score = 1 - cosine(query_vector, embedding_vector)  # Compute similarity
        similarities.append({
            "file_name": df.iloc[i]['file_name'],
            "score": round(score, 4)
        })

    # Sort by similarity score in descending order
    similarities.sort(key=lambda x: x['score'], reverse=True)
    return similarities[:top_k]





@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query_type = request.form.get('query_type')
        text_query = request.form.get('text_query')
        image_query = request.files.get('image_query')
        lam = float(request.form.get('weight', 0.5)) if query_type == 'hybrid' else None
        print(f"Weight (λ): {lam}")

        query_embedding = None

        # Handle text queries
        if query_type in ['text', 'hybrid'] and text_query:
            text_tokens = tokenizer([text_query])
            with torch.no_grad():  # Disable gradient tracking
                text_emb = F.normalize(model.encode_text(text_tokens.to(device))).cpu().detach().numpy()
            query_embedding = text_emb if query_embedding is None else lam * text_emb + (1 - lam) * query_embedding

        # Handle image queries
        if query_type in ['image', 'hybrid'] and image_query:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_query.filename)
            image_query.save(image_path)
            image = preprocess_val(Image.open(image_path)).unsqueeze(0).to(device)
            with torch.no_grad():  # Disable gradient tracking
                img_emb = F.normalize(model.encode_image(image)).cpu().detach().numpy()
            query_embedding = img_emb if query_embedding is None else lam * query_embedding + (1 - lam) * img_emb

        # Process results if we have a query embedding
        if query_embedding is not None:
            query_embedding = F.normalize(torch.tensor(query_embedding)).numpy()
            results = get_most_similar_images(query_embedding)

            # Debug results
            print(f"Query Type: {query_type}, Results: {results}")

            # Render results page
            return render_template('results.html', results=results, query_type=query_type)

    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)
