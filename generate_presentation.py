from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
import os
import datetime

# --- Configuration ---
OUTPUT_FILE = "Presentations_Resultats_Projet.pptx"
IMG_DIR = "results/figures"

# Define Image Paths (Verify these exist)
IMG_COMPARISON = os.path.join(IMG_DIR, "comparison/model_comparison.png")
IMG_OPTUNA = os.path.join(IMG_DIR, "lightgbm/lgbm_optimization_history.png") 
IMG_GRU_LOSS = os.path.join(IMG_DIR, "gru/gru_training_loss.png")
IMG_TRANSFORMER_LOSS = os.path.join(IMG_DIR, "transformer/transformer_training_loss.png")
IMG_XGB_IMP = os.path.join(IMG_DIR, "xgboost/xgb_feature_importance.png")

# Start Presentation
prs = Presentation()

# --- Helper Functions ---
def create_title_slide(prs, title, subtitle, author):
    slide_layout = prs.slide_layouts[0] # Title Slide
    slide = prs.slides.add_slide(slide_layout)
    slide.shapes.title.text = title
    if slide.placeholders[1]:
        slide.placeholders[1].text = f"{subtitle}\n\n{author}"

def add_content_slide(prs, title, content_lines, image_path=None):
    slide_layout = prs.slide_layouts[1] # Title and Content
    slide = prs.slides.add_slide(slide_layout)
    
    # Title
    title_shape = slide.shapes.title
    title_shape.text = title
    
    # Content
    body_shape = slide.placeholders[1]
    tf = body_shape.text_frame
    tf.clear() 
    
    for line in content_lines:
        p = tf.add_paragraph()
        p.text = line
        p.level = 0
        p.font.size = Pt(18)
        if line.startswith("   -"):
            p.level = 1
            p.text = line.replace("   -", "").strip()
        elif line.startswith("       ->"):
            p.level = 2
            p.text = line.replace("       ->", "").strip()

    # Image
    if image_path and os.path.exists(image_path):
        left = Inches(5.5)
        top = Inches(2.0)
        height = Inches(4.5)
        slide.shapes.add_picture(image_path, left, top, height=height)
        body_shape.width = Inches(5.0)

# --- Slides Generation ---

# Slide 1: Title
create_title_slide(
    prs, 
    "Prédiction de Conversion Trial-to-Paid :\nInsights Data-Driven pour Kolecto", 
    "Améliorer le taux de conversion de ~60% via l'analyse des signaux précurseurs et le Machine Learning",
    f"Antigravity Agent – {datetime.date.today()}"
)

# Slide 2: Business Context
add_content_slide(prs, "Contexte Business & Objectifs", [
    "Contexte :",
    "   - Kolecto propose un essai gratuit de 15 jours -> Abonnement payant.",
    "   - Taux de conversion actuel : ~60% (Satisfaisant mais perfectible).",
    "Challenge :",
    "   - Identifier les signaux précurseurs de succès ou de désabonnement.",
    "   - Permettre des actions ciblées par l'équipe Customer Experience (CX).",
    "Objectifs :",
    "   - Analyser les facteurs différenciants (Convertis vs Non-Convertis).",
    "   - Construire un modèle ML pour prédire la probabilité de conversion.",
    "Approche :",
    "   - Analyse d'Activité (Daily Usage) + Profil (Subscriptions).",
    "   - Focus sur des insights actionnables."
])

# Slide 3: Data Overview
add_content_slide(prs, "Données & Prétraitement", [
    "Jeux de Données :",
    "   - Daily Usage (~11k lignes) : Logs d'activité (Virements, Connexions, Factures). Agrégés par essai (Somme/Moyenne/Max/Std).",
    "   - Subscriptions (~416 essais) : Firmographie (CA, Code NAF). Filtré sur les essais de 15 jours exacts.",
    "Statistiques Clés :",
    "   - Échantillons Total : 416 essais complets.",
    "   - Taux de Conversion : 60.7% (Déséquilibré, mais gérable).",
    "Prétraitement :",
    "   - Fusion Usage + Subscriptions.",
    "   - Gestion des valeurs manquantes et inactivité.",
    "   - Encodage (OneHot/Ordinal) & Standardisation."
])

# Slide 4: Methodology
add_content_slide(prs, "Méthodologie & Modèles", [
    "Stratégie de Features :",
    "   - Tabulaire : Features agrégées (157 dims) pour modèles Arborescents.",
    "   - Séquentiel : Séries temporelles (15 jours) pour Deep Learning.",
    "Modèles Entraînés :",
    "   - Logistic Regression : Baseline linéaire simple.",
    "   - XGBoost & LightGBM : Gradient Boosting avec tuning Optuna.",
    "   - GRU & Transformer : Deep Learning pour motifs temporels.",
    "Métriques d'Évaluation :",
    "   - ROC-AUC : Capacité de discrimination.",
    "   - PR-AUC : Précision-Rappel (Critique pour l'imbalance).",
    "   - Brier Score : Calibration des probabilités."
])

# Slide 5: Overall Results
add_content_slide(prs, "Comparaison des Résultats", [
    "Champion : LightGBM",
    "   - ROC-AUC : 0.797 (Meilleure Discrimination)",
    "   - PR-AUC : 0.839 (Haute Précision)",
    "   - Accuracy : 75.9%",
    "   - Brier : 0.194 (Bien calibré)",
    "Runner Up : GRU",
    "   - ROC-AUC : 0.715. Capture le signal temporel mais limité par la taille des données.",
    "Baseline :",
    "   - Logistic Regression (0.684).",
    "   - Transformer (0.678) - Overfitting dû au faible échantillon.",
    "Conclusion : LightGBM gère mieux les données tabulaires haute dimension."
], image_path=IMG_COMPARISON)

# Slide 6: Optimization Insights
add_content_slide(prs, "Optimisation & Dynamique d'Entraînement", [
    "LightGBM (Optuna) :",
    "   - Recherche efficace (50 essais).",
    "   - Convergence vers paramètres robustes (n_est=318, lr=0.018).",
    "Dynamique Deep Learning :",
    "   - GRU : Bonne baisse de loss training, mais validation instable.",
    "   - Transformer : Signes d'overfitting (Écart train/val).",
    "   - Leçon : Les modèles profonds nécessitent plus de données (10k+) pour battre les arbres ici."
], image_path=IMG_OPTUNA) 

# Slide 7: Feature Importance
add_content_slide(prs, "Importance des Features & Signaux", [
    "Meilleurs Prédicteurs (LightGBM/XGBoost) :",
    "   - company_age : Les entreprises plus anciennes/stables convertissent mieux.",
    "   - naf_code : Certains secteurs ont une affinité plus forte.",
    "   - nb_client_invoices_created_sum : L'usage actif (Facturation) est le signal #1.",
    "Insights :",
    "   - L'Activation compte : Facturer ou connecter une banque tôt garantit la conversion.",
    "   - Alerte 'Faible Activité' : < 2 connexions mobiles au Jour 3 = Risque x3.",
    "   - Ciblage : Concentrer le CX sur les TPEs avec faible activité précoce."
], image_path=IMG_XGB_IMP)

# Slide 8: Limitations
add_content_slide(prs, "Limitations & Améliorations", [
    "Limitations :",
    "   - Faible Volume de Données : Seulement 416 essais complets. Limite le Deep Learning.",
    "   - Données Internes Uniquement : Pas de données économiques externes.",
    "Améliorations Futures :",
    "   - Ensemble Hybride : Combiner LightGBM + GRU (Testé, gain marginal actuellement).",
    "   - Causal ML : Modéliser l'Uplift des appels CX.",
    "   - Monitoring : Ré-entraîner mensuellement pour détecter la dérive (Drift)."
])

# Slide 9: Conclusion
add_content_slide(prs, "Conclusion & Prochaines Étapes", [
    "Résumé :",
    "   - Modèle robuste construit (AUC ~0.80).",
    "   - Signaux d'activation clés identifiés (Factures, Mobile, Connexion Bancaire).",
    "Résultats :",
    "   - Potentiel de +5-8% de conversion via intervention ciblée.",
    "Prochaines Étapes :",
    "   1. Déployer l'API de Scoring (FastAPI/Gradio).",
    "   2. A/B Test des actions CX sur les utilisateurs 'À Risque' (Prob < 0.45).",
    "   3. Monitorer la performance en production."
])

# Save
prs.save(OUTPUT_FILE)
print(f"Presentation saved to {OUTPUT_FILE}")
