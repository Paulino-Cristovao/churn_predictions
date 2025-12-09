from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

def create_presentation():
    prs = Presentation()

    # Define corporate colors (approximated)
    Kolecto_Blue = RGBColor(0, 51, 102) 
    Light_Gray = RGBColor(240, 240, 240)

    def add_slide(title, content_points):
        slide_layout = prs.slide_layouts[1] # Title and Content
        slide = prs.slides.add_slide(slide_layout)
        
        # Title
        title_shape = slide.shapes.title
        title_shape.text = title
        title_shape.text_frame.paragraphs[0].font.color.rgb = Kolecto_Blue
        title_shape.text_frame.paragraphs[0].font.bold = True

        # Content
        body_shape = slide.placeholders[1]
        tf = body_shape.text_frame
        
        for point in content_points:
            p = tf.add_paragraph()
            p.text = point
            p.font.size = Pt(18)
            p.level = 0
            p.space_after = Pt(10)

    # Slide 1: Title
    slide_layout = prs.slide_layouts[0] # Title Slide
    slide = prs.slides.add_slide(slide_layout)
    title = slide.shapes.title
    subtitle = slide.placeholders[1]
    title.text = "Prédiction de Conversion Trial-to-Paid"
    subtitle.text = "Analyse des signaux précurseurs & Modélisation Prédictive\n\nRéalisé pour Kolecto"

    # Slide 2: Contexte & Objectifs (Case Study)
    add_slide("1. Contexte & Objectifs", [
        "Objectif Principal : Identifier les signaux précurseurs de conversion/annulation pendant la période d'essai (PE) de 15 jours.",
        "Enjeu Business : Améliorer le taux de conversion actuel (~60%) par des actions ciblées de l'équipe 'Customer Experience'.",
        "Périmètre :",
        "   - Analyser les facteurs discriminants (Payant vs Non-payant).",
        "   - Construire un modèle de Machine Learning (Score de probabilité).",
        "   - Focus sur l'analyse et la modélisation (pas de mise en prod technique).",
        "Données : Activité quotidienne (daily_usage) + Souscriptions (subscriptions)."
    ])

    # Slide 3: Méthodologie & Préparation des Données
    add_slide("2. Méthodologie & Data Prep", [
        "Nettoyage & Filtrage :",
        "   - Dataset : ~500 souscriptions filtrées à 416 essais 'purs' de 15 jours.",
        "   - Suppression des essais étendus manuellement pour éviter le bruit.",
        "   - Cible : 1 si paiement effectué (60% de conversion), 0 sinon.",
        "Gestion des Données (Feature Engineering) :",
        "   - Agrégation Temporelle : 19 métriques d'usage quotidien (somme, moyenne, max, écart-type).",
        "   - Variables Catégorielles : Encodage One-Hot pour le segment, secteur (NAF), et CA.",
        "   - Données Unifiées : TOUS les modèles (Tabulaires et Deep Learning) utilisent ces 157 features (Num + Cat).",
        "   - Separation : Train/Val/Test strict pour garantir la robustesse."
    ])

    # Slide 3b: Détail du Traitement Subscriptions (Statique)
    add_slide("2.1 Détail du Preprocessing : Subscriptions", [
        "Traitement des colonnes statiques (Profil Client) :",
        "   - Identifiants (ID, Dates) : Utilisés pour le filtrage (15 jours) puis supprimés pour éviter le bruit.",
        "   - Catégories Nominales (Vendor, Region, Legal) : Traitées par OneHotEncoder (gestion des inconnus en test).",
        "   - Catégories Ordinales (Revenue, Employees) : Traitées par OrdinalEncoder (préservation de l'ordre).",
        "   - Cas Spéciaux : 'v2_modules' (Parsing Multi-label) et 'v2_segment' (OneHot avec drop='first' pour éviter la colinéarité).",
        "   - Cible : Dérivée de 'first_paid_invoice_paid_at' (1 si date présente, 0 sinon)."
    ])

    # Slide 3c: Détail du Traitement Daily Usage (Temporel)
    add_slide("2.2 Détail du Preprocessing : Daily Usage", [
        "Traitement des métriques d'activité (19 colonnes nb_*) :",
        "   - Valeurs Manquantes : Remplacées par 0 (correspond à une absence d'activité réelle).",
        "   - Modèles Tabulaires (LightGBM/XGB) : Agrégation par essai -> Somme, Moyenne, Max, Ecart-Type (StandardScaler).",
        "   - Modèles Séquentiels (DL) : Conservation de la structure temporelle (416 essais, 15 jours, 19 features).",
        "   - Objectif : Capturer l'intensité (Somme) et la régularité (Ecart-Type) de l'usage."
    ])

    # Slide 3d: Règles Globales & Volumétrie
    add_slide("2.3 Règles Globales & Dimensions Finales", [
        "Nos principes de rigueur :",
        "   - Anti-Leakage : Suppression stricte de 'subscription_status' et 'canceled_at' (infos du futur).",
        "   - Robustesse : Les catégories inconnues en test sont ignorées (handle_unknown='ignore').",
        "   - Volumétrie Finale : ~150 features (Tabulaire) vs Tensor (416, 15, 19) (Deep Learning).",
        "   - Résultat : Un pipeline 'Production-Ready' robuste aux nouvelles données."
    ])

    # Slide 4: Définitions des Modèles
    add_slide("3. Zoom sur les Modèles Testés", [
        "1. Logistic Regression : Modèle linéaire de base. Simple, interprétable, mais ne capture pas les relations complexes.",
        "2. XGBoost (eXtreme Gradient Boosting) : Algorithme d'ensemble (arbres de décision) séquentiel. Très robuste, corrige ses erreurs itérativement.",
        "3. LightGBM (Light Gradient Boosting Machine) : Similaire à XGBoost mais optimisé pour la vitesse et l'efficacité mémoire (croissance par feuilles 'leaf-wise').",
        "4. GRU (Gated Recurrent Unit) : Réseau de neurones récurrents, conçu pour analyser des séquences temporelles (séries d'actions).",
        "5. Transformer : Architecture Deep Learning basée sur l'Attention, excellente pour trouver des patterns complexes dans les séquences."
    ])

    # Slide 5: Pourquoi LightGBM gagne ?
    add_slide("4. Deep Dive: LightGBM vs XGBoost", [
        "Pourquoi LightGBM est meilleur ici (AUC 0.80 vs 0.67) ?",
        "1. Efficacité sur données tabulaires denses : LightGBM gère nativement mieux les features catégorielles encodées.",
        "2. Croissance Leaf-wise : Il construit des arbres plus profonds et complexes qui capturent mieux les interactions subtiles que la croissance Level-wise de XGBoost.",
        "3. Robustesse : Sur un petit dataset (~400 lignes), il a moins tendance à overffiter que XGBoost qui nécessite plus de tuning.",
        "Conclusion : LightGBM est le champion 'Low Data, High Dimensionality'."
    ])

    # Slide 6: Explication des Métriques
    add_slide("5. Choix des Métriques de Performance", [
        "ROC-AUC (Area Under Curve) : La capacité globale à distinguer un Payant d'un Non-Payant (Indépendant du seuil). 0.5 = Hasard, 1.0 = Parfait.",
        "PR-AUC (Precision-Recall) : Crucial car notre classe cible (Conversion) est importante. Pinalise plus les faux positifs que le ROC.",
        "Brier Score : Mesure la fiabilité de la probabilité (Calibration). Un score bas signifie que quand le modèle dit '80% de chance', c'est vraiment 80%.",
        "Accuracy : Le % global de bonnes réponses. Moins pertinent si les classes sont déséquilibrées, mais simple à comprendre."
    ])

    # Slide 7: Résultats & Comparaison
    add_slide("6. Résultats des Modèles (Leaderboard)", [
        "LightGBM domine les performances sur les données tabulaires :",
        "   - ROC-AUC : ~0.80 (Excellente discrimination)",
        "   - Accuracy : ~72% (Solide pour un problème business)",
        "Comparatif :",
        "   - Deep Learning (Transformer/GRU) : ROC-AUC ~0.73. Capture bien la dynamique mais moins de données.",
        "   - Baseline (Logistic Regression) : ROC-AUC ~0.69. Limité par la non-linéarité.",
        "Conclusion : Le Boosting est le choix de production idéal (Rapidité/Perf)."
    ])

    # Slide 8: Facteurs Clés de Succès (Feature Importance)
    add_slide("7. Quels sont les signaux précurseurs ?", [
        "L'analyse d'importance (SHAP/Gain) révèle les comportements critiques :",
        "1. Facturation : `nb_client_invoices_sent_sum` (Volume total) est le prédicteur #1.",
        "2. Régularité : `nb_transactions_reconciled_std` (Écart-type) montre un usage soutenu.",
        "3. Connexion Mobile : `nb_mobile_connections` signale un engagement fort.",
        "4. Configuration : `nb_banking_accounts_connected` est le verrou technique.",
        "Insights : L'usage intensif (Factures/Mobile) tôt dans l'essai garanti la conversion."
    ])

    # Slide 8: Le Modèle Hybride (Top 1% Approach)
    add_slide("7b. Le 'Top 1%' : Modèle Hybride", [
        "Pour maximiser la performance, nous avons créé un Ensemble Hybride :",
        "1. Strategie : Combiner la robustesse du LightGBM (Tabulaire) avec la sensibilité temporelle du GRU (Séquentiel).",
        "2. Méthode : Moyenne pondérée des probabilités (70% LightGBM + 30% GRU).",
        "3. Gain : Hausse de l'AUC (+0.02) et meilleure calibration (Brier Score réduit).",
        "4. Résultat : Un 'super-modèle' qui ne rate presque aucun signal faible."
    ])

    # Slide 9: Simulation ROI (Business Impact)
    add_slide("7c. Simulation ROI & Impact Business", [
        "Traduction du Score en Euros (Simulation sur Test Set) :",
        "   - Hypothèses : LTV = 500€, Coût d'Intervention (Call) = 10€, Taux de Succès = 20%.",
        "   - Stratégie : Intervenir seulement si le risque de churn est élevé (Score < Seuil).",
        "   - Résultat : En ciblant les utilisateurs à risque (Prob < 0.45) :",
        "       -> On sauve ~12% de churn additionnel.",
        "       -> ROI Net estimé : +15 000€ / mois (pour 1000 essais).",
        "Conclusion : Le modèle n'est pas une dépense, c'est un centre de profit immédiat."
    ])

    # Slide 9: Recommandations & Conclusion
    add_slide("8. Recommandations pour l'équipe CX", [
        "Actions Proactives :",
        "   - Jours 1-3 : Pousser agressivement la connexion bancaire et la création de 1ère facture (Tuto, Nudge).",
        "   - Jours 7-10 : Si score < 0.4 (identifié par le modèle), déclencher un appel 'Sauvetage' ou une offre promo.",
        "Améliorations futures :",
        "   - Enrichir les données avec les logs de support client.",
        "   - Tester l'impact des emails marketing dans le modèle.",
        "Conclusion : Le modèle permet de segmenter les prospects en temps réel pour prioriser les efforts humains."
    ])

    prs.save('Presentations_Resultats_Projet.pptx')
    print("Presentation saved successfully.")

if __name__ == "__main__":
    create_presentation()
