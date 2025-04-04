import streamlit as st
import numpy as np
from PIL import Image
from skimage.transform import resize
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import lsq_linear
import time

st.title("Décomposition en 3 encres (soustractif + affichage corrigé)")

uploaded_file = st.file_uploader("Choisissez une image", type=["png", "jpg", "jpeg"])

# Convertisseur couleur HEX -> absorption (1 - RGB)
def hex_to_absorption(hex_color):
    hex_color = hex_color.lstrip('#')
    rgb = np.array([int(hex_color[i:i+2], 16) for i in (0, 2, 4)]) / 255.0
    return 1 - rgb

# Choix des couleurs d'encre
col1, col2, col3 = st.columns(3)
with col1:
    c1 = st.color_picker("Encre 1 (cuivré)", "#D97326")
with col2:
    c2 = st.color_picker("Encre 2 (sapin)", "#004D26")
with col3:
    c3 = st.color_picker("Encre 3 (blanc)", "#FFFFFF")

ink1 = hex_to_absorption(c1)
ink2 = hex_to_absorption(c2)
ink3 = hex_to_absorption(c3)

n_colors = st.slider("Nombre de couleurs (quantification)", 16, 1024, 64)
scale = st.slider("Réduction image pour clustering", 0.1, 1.0, 0.5)

def decompose_image_3inks_quantized_auto_resize(image_rgb, ink1, ink2, ink3, n_colors=64, scale=0.5):
    # Créer une zone pour afficher la progression
    progress_text = st.empty()
    progress_bar = st.progress(0)
    timings = {}
    
    progress_text.text("Début de la décomposition d'image...")
    
    # Étape 1: Redimensionnement
    start_time = time.time()
    H, W, _ = image_rgb.shape
    h_small = int(H * scale)
    w_small = int(W * scale)
    image_small = resize(image_rgb, (h_small, w_small), anti_aliasing=True)
    timings['redimensionnement'] = time.time() - start_time
    progress_text.text(f"1. Redimensionnement: {timings['redimensionnement']:.4f} secondes")
    progress_bar.progress(0.1)
    
    # Étape 2: Aplatissement
    start_time = time.time()
    flat_small = image_small.reshape(-1, 3)
    timings['aplatissement'] = time.time() - start_time
    progress_text.text(f"2. Aplatissement: {timings['aplatissement']:.4f} secondes")
    progress_bar.progress(0.2)
    
    # Étape 3: K-means clustering
    start_time = time.time()
    kmeans = KMeans(n_clusters=n_colors, n_init=3, random_state=0).fit(flat_small)
    palette = kmeans.cluster_centers_
    timings['kmeans'] = time.time() - start_time
    progress_text.text(f"3. K-means clustering: {timings['kmeans']:.4f} secondes")
    progress_bar.progress(0.4)
    
    # Étape 4: Aplatissement de l'image complète
    start_time = time.time()
    flat_full = image_rgb.reshape(-1, 3)
    timings['aplatissement_complet'] = time.time() - start_time
    progress_text.text(f"4. Aplatissement de l'image complète: {timings['aplatissement_complet']:.4f} secondes")
    progress_bar.progress(0.5)
    
    # Étape 5: Recherche des plus proches voisins
    start_time = time.time()
    nn = NearestNeighbors(n_neighbors=1).fit(palette)
    _, indices = nn.kneighbors(flat_full)
    labels = indices.flatten().reshape(H, W)
    timings['nearest_neighbors'] = time.time() - start_time
    progress_text.text(f"5. Recherche des plus proches voisins: {timings['nearest_neighbors']:.4f} secondes")
    progress_bar.progress(0.6)
    
    # Étape 6: Création de la matrice A
    start_time = time.time()
    A = np.stack([ink1, ink2, ink3], axis=1)
    timings['creation_matrice_A'] = time.time() - start_time
    progress_text.text(f"6. Création de la matrice A: {timings['creation_matrice_A']:.4f} secondes")
    progress_bar.progress(0.7)
    
    # Étape 7: Création du dictionnaire lookup (décomposition des couleurs)
    start_time = time.time()
    lookup = {}
    for idx, color in enumerate(palette):
        res = lsq_linear(A, 1 - color, bounds=(0, 1))
        lookup[idx] = res.x
    timings['decomposition_couleurs'] = time.time() - start_time
    progress_text.text(f"7. Décomposition des couleurs: {timings['decomposition_couleurs']:.4f} secondes")
    progress_bar.progress(0.8)
    
    # Étape 8: Reconstruction de l'image (version optimisée)
    start_time = time.time()
    alpha = np.zeros((H, W))
    beta = np.zeros((H, W))
    gamma = np.zeros((H, W))
    reconstructed = np.zeros((H, W, 3))
    
    # Créer une barre de progression spécifique pour la reconstruction
    reconstruction_progress = st.progress(0)
    reconstruction_status = st.empty()
    
    for i in range(H):
        # Mise à jour de la progression
        progress = (i + 1) / H
        reconstruction_progress.progress(progress)
        reconstruction_status.text(f"Reconstruction : {progress*100:.1f}% ({i+1}/{H} lignes)")
        
        for j in range(W):
            a, b, g = lookup[labels[i, j]]
            alpha[i, j] = a
            beta[i, j] = b
            gamma[i, j] = g
            reconstructed[i, j] = 1 - (a * ink1 + b * ink2 + g * ink3)
    
    # Nettoyage des éléments de progression de la reconstruction
    reconstruction_progress.empty()
    reconstruction_status.empty()
    
    timings['reconstruction'] = time.time() - start_time
    progress_text.text(f"8. Reconstruction de l'image: {timings['reconstruction']:.4f} secondes")
    progress_bar.progress(0.9)
    
    # Temps total et résumé
    total_time = sum(timings.values())
    
    # Afficher les statistiques détaillées
    st.write("### Statistiques de traitement")
    st.write(f"**Temps total**: {total_time:.4f} secondes")
    
    st.write("**Détail des étapes (par ordre décroissant):**")
    sorted_timings = sorted(timings.items(), key=lambda x: x[1], reverse=True)
    for step, t in sorted_timings:
        percentage = (t / total_time) * 100
        st.write(f"- {step}: {t:.4f} secondes ({percentage:.1f}%)")
    
    progress_bar.progress(1.0)
    progress_text.text("Traitement terminé !")
    
    return alpha, beta, gamma, reconstructed

def render_layer(alpha, ink_rgb):
    return ((1 - alpha[..., None] * ink_rgb) * 255).astype(np.uint8)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.asarray(image) / 255.0
    st.image(image, caption="Image originale", use_container_width=True)

    if st.button("Lancer la décomposition"):
        alpha, beta, gamma, recon = decompose_image_3inks_quantized_auto_resize(
            image_np, ink1, ink2, ink3, n_colors=n_colors, scale=scale
        )

        # Assurer que recon est entre 0 et 1 avant la conversion
        recon_safe = np.clip(recon, 0, 1)
        # Appliquer une correction gamma si nécessaire
        gamma_correction = 1.0  # Ajuster si nécessaire
        recon_gamma = np.power(recon_safe, 1/gamma_correction)
        
        st.image((recon_gamma * 255).astype(np.uint8), 
                caption="Image reconstruite", 
                use_container_width=True)

        st.image(render_layer(alpha, ink1), caption="Canal α (encre 1)", use_container_width=True)
        st.image(render_layer(beta,  ink2), caption="Canal β (encre 2)", use_container_width=True)
        st.image(render_layer(gamma, ink3), caption="Canal γ (encre 3)", use_container_width=True)
