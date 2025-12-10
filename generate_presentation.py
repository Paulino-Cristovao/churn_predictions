from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
import os
import datetime

# --- Configuration ---
OUTPUT_FILE = "Presentations_Resultats_Projet.pptx"
IMG_DIR = "results/figures"

# Define Image Paths
IMG_COMPARISON = os.path.join(IMG_DIR, "comparison/model_comparison.png")
IMG_OPTUNA = os.path.join(IMG_DIR, "lightgbm/lgbm_optimization_history.png") 
IMG_GRU_LOSS = os.path.join(IMG_DIR, "gru/gru_training_loss.png")
IMG_TRANSFORMER_LOSS = os.path.join(IMG_DIR, "transformer/transformer_training_loss.png")
IMG_XGB_IMP = os.path.join(IMG_DIR, "xgboost/xgb_feature_importance.png")
IMG_LOGO = os.path.join(IMG_DIR, "kolecto_logo.png")

# Start Presentation
prs = Presentation()

# --- Helper Functions ---
def set_title_format(title_shape):
    """Applies blue styling to titles."""
    if not title_shape or not title_shape.text_frame:
        return
    title_shape.text_frame.paragraphs[0].font.color.rgb = RGBColor(0, 51, 102) # Dark Blue
    title_shape.text_frame.paragraphs[0].font.name = 'Arial'

def create_title_slide(prs, title, subtitle, author, logo_path=None):
    slide_layout = prs.slide_layouts[0] # Title Slide
    slide = prs.slides.add_slide(slide_layout)
    
    # Title
    slide.shapes.title.text = title
    set_title_format(slide.shapes.title)
    
    # Subtitle
    if slide.placeholders[1]:
        slide.placeholders[1].text = f"{subtitle}\n\n{author}"
    
    # Add Logo if exists
    if logo_path and os.path.exists(logo_path):
        left = Inches(0.5)
        top = Inches(0.5)
        height = Inches(1.0) # Adjust size as needed
        slide.shapes.add_picture(logo_path, left, top, height=height)

def add_content_slide(prs, title, content_lines, image_path=None):
    slide_layout = prs.slide_layouts[1] # Title and Content
    slide = prs.slides.add_slide(slide_layout)
    
    # Title
    title_shape = slide.shapes.title
    title_shape.text = title
    set_title_format(title_shape)
    
    # Content
    body_shape = slide.placeholders[1]
    tf = body_shape.text_frame
    tf.clear() 
    
    for line in content_lines:
        p = tf.add_paragraph()
        p.text = line
        p.level = 0
        p.font.size = Pt(18)
        p.font.name = 'Arial'
        
        # Handle indentation
        if line.startswith("   -"):
            p.level = 1
            p.text = line.replace("   -", "").strip()
        elif line.startswith("       ->"):
            p.level = 2
            p.text = line.replace("       ->", "").strip()

    # Image
    if image_path and os.path.exists(image_path):
        # Adjust layout for image
        body_shape.width = Inches(5.0) 
        left = Inches(5.5)
        top = Inches(2.0)
        height = Inches(4.5)
        
        # Check if it's the comparison plot (needs to be wider)
        if "comparison" in image_path:
             height = Inches(3.5)
             top = Inches(2.5)
        
        slide.shapes.add_picture(image_path, left, top, height=height)

# --- Slides Generation ---

# Slide 1: Title
create_title_slide(
    prs, 
    "Predicting Trial-to-Paid Conversions:\nData-Driven Insights for Kolecto", 
    "Improving ~60% Conversion Rate Through Precursor Signal Analysis and ML Modeling",
    f"Antigravity Agent – {datetime.date.today()}",
    logo_path=IMG_LOGO
)

# Slide 2: Business Context & Objectives
add_content_slide(prs, "Business Context & Objectives", [
    "Context:",
    "   - Kolecto offers paid subscriptions with a 15-day trial.",
    "   - Current conversion rate ~60% (Satisfactory but improvable).",
    "Challenge:",
    "   - Identify precursor signals of cancellation/success.",
    "   - Enable targeted Customer Experience (CX) actions.",
    "Objectives:",
    "   - Analyze differentiating factors (Converters vs Non-Converters).",
    "   - Build ML model for conversion probability prediction.",
    "Approach:",
    "   - Usage Analysis (Activity signals) + Profile (Company info).",
    "   - Focus on actionable insights, not just black-box predictions."
])

# Slide 3: Data Overview
add_content_slide(prs, "Data Overview", [
    "Datasets:",
    "   - daily_usage.csv (~11k rows): Activity logs (transfers, connections). Aggregated to per-trial summaries (sum/mean/max/std).",
    "   - subscriptions.csv (~416 trials): Company info (Revenue, NAF). Filtered to exact 15-day trials.",
    "Key Stats:",
    "   - Total Samples: 416 complete trials.",
    "   - Conversion Rate: 60.7% (Imbalanced but manageable).",
    "Preprocessing:",
    "   - Merged Usage + Subscriptions.",
    "   - Handled Inactivity (NaN/Zero imputation).",
    "   - Encoding: OneHot (Categorical) & Robust Scaling (Numerical)."
])

# Slide 4: Methodology & Models
add_content_slide(prs, "Methodology & Models", [
    "Strategy:",
    "   - Tabular Features (157 dims) for Tree-based models.",
    "   - Sequential Data (15 days) for Deep Learning.",
    "Models Trained:",
    "   - Logistic Regression (Baseline).",
    "   - XGBoost & LightGBM (Gradient Boosting with Optuna tuning).",
    "   - GRU & Transformer (Sequential modeling for temporal signals).",
    "Evaluation Metrics:",
    "   - ROC-AUC: Discrimination capability (Distinguish churners).",
    "   - PR-AUC: Precision-Recall (Critical for imbalance).",
    "   - Brier Score: Probability calibration accuracy."
])

# Slide 5: Overall Results Comparison
add_content_slide(prs, "Overall Results Comparison", [
    "Winner: LightGBM",
    "   - ROC-AUC: 0.790 | PR-AUC: 0.835 | Accuracy: 72.3%",
    "   - Brier: 0.193 (Best Calibration)",
    "   - Why? Handles mixed features/sparsity best on small data.",
    "Runner Up: GRU",
    "   - ROC-AUC: 0.713. Captures temporal patterns well.",
    "Baselines:",
    "   - Transformer (0.711) - Comparable to GRU.",
    "   - Logistic Regression (0.684) - Limited linearity.",
    "   - XGBoost (0.671) - Underperformed LightGBM here.",
    "Conclusion: LightGBM is robust, fast, and most accurate."
], image_path=IMG_COMPARISON)

# Slide 6: Optimization & Training Insights
add_content_slide(prs, "Optimization & Training Insights", [
    "LightGBM (Optuna):",
    "   - Efficient search (50 trials).",
    "   - Converged to robust params (n_est=318, lr=0.018).",
    "Deep Learning Dynamics:",
    "   - GRU: Steady loss decrease, slight validation instability.",
    "   - Transformer: improved but still data-hungry.",
    "   - Takeaway: Deep Learning needs more than 400 samples to truly shine."
], image_path=IMG_OPTUNA) 

# Slide 7: Feature Importance & Insights
add_content_slide(prs, "Feature Importance & Insights", [
    "Top Predictors:",
    "   - company_age: Older/stable firms convert more.",
    "   - naf_code: Specific industries have higher affinity.",
    "   - nb_client_invoices_created_sum: Usage (Invoicing) is the #1 signal.",
    "Actionable Insights:",
    "   - Activation Matters: Early usage (Day 1-3) is critical.",
    "   - 'Low Activity' Alert: < 2 connections by Day 3 = 3x Churn Risk.",
    "   - Targeting: Focus CX on TPEs with low early activity."
], image_path=IMG_XGB_IMP)

# Slide 8: Business Impact & Recommendations
add_content_slide(prs, "Business Impact & Recommendations", [
    "Estimated Impact:",
    "   - Target: ~400 trials/month.",
    "   - Lift: +5-8% conversion via targeted intervention.",
    "   - Value: +€600k-€960k ARR (assuming €3k LTV).",
    "Recommendations:",
    "   - Day 1-3: Automated nudges for bank connection/invoicing.",
    "   - Day 7-10: Human CX call if Churn Prob > 60%.",
    "   - Deployment: A/B Test interventions to measure real uplift."
])

# Slide 9: Limitations & Improvements
add_content_slide(prs, "Limitations & Improvements", [
    "Limitations:",
    "   - Small Dataset: ~416 trials limits Deep Learning potential.",
    "   - External Factors: No data on economic context/seasonality.",
    "Future Improvements:",
    "   - Hybrid Ensemble: Combine LightGBM (Tabular) + GRU (Sequential).",
    "   - Causal ML: Model 'uplift' (Persuadables vs Do-not-disturb).",
    "   - Retraining: Monthly updates to handle concept drift."
])

# Slide 10: Conclusion & Next Steps
add_content_slide(prs, "Conclusion & Next Steps", [
    "Summary:",
    "   - Delivered robust model (LightGBM AUC 0.790).",
    "   - Identified key levers for +5-8% conversion lift.",
    "Next Steps:",
    "   1. Deploy Scoring API (Containerized).",
    "   2. Launch A/B Test for CX actions.",
    "   3. Monitor Performance (MLflow) & Collect more data."
])

# Save
prs.save(OUTPUT_FILE)
print(f"Presentation saved to {OUTPUT_FILE}")
