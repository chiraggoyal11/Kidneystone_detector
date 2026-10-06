"""
Kidney Stone Detection System
Advanced Medical Image Analysis using CNN & SVM Ensemble
"""

import gradio as gr
from ensemble_learning import ensemble

def predict_kidney_stone(image):
    """Main prediction function for Gradio interface"""
    if image is None:
        return "❌ Please upload an image", "Please upload a CT scan image to analyze."
    
    try:
        # Calls the ensemble pipeline
        result = ensemble.predict(image)
        
        if "error" in result:
             return f"❌ {result['error']}", "Analysis failed."
             
        prediction = result['prediction']
        ensemble_conf = result['confidence']
        details = result['details']
        
        # Format Output
        if prediction == "Stone Detected":
            status = "🔴 KIDNEY STONE DETECTED"
            advice = """
⚠️ **IMPORTANT MEDICAL ADVICE:**
- Consult a urologist immediately for proper diagnosis
- Stay well hydrated (drink plenty of water)
- Avoid foods high in oxalates (spinach, nuts, chocolate)
- Consider pain management options
- Follow up with imaging studies as recommended
            """
        else:
            status = "✅ NO KIDNEY STONES DETECTED"
            advice = """
✅ **PREVENTIVE RECOMMENDATIONS:**
- Maintain adequate hydration (2-3 liters of water daily)
- Follow a balanced, low-sodium diet
- Regular exercise and healthy lifestyle
- Monitor for symptoms and regular check-ups
- Continue preventive care measures
            """
            
        cnn_conf = details.get('cnn_confidence', 0.0)
        svm_conf = details.get('svm_confidence', 0.0)
        
        model_info = []
        if cnn_conf > 0 or result['confidence'] > 0: model_info.append("CNN")
        if svm_conf > 0 or result['confidence'] > 0: model_info.append("SVM")
        model_str = "+".join(model_info) if model_info else "Fallback"

        prediction_text = f"""
{status}

**Prediction Details:**
- Overall Confidence: {ensemble_conf:.1f}%
- CNN Model Confidence: {cnn_conf:.1f}%
- SVM Model Confidence: {svm_conf:.1f}%
- Status: Production Mode
- Model: {model_str} Ensemble
        """
        
        return prediction_text, advice
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        return error_msg, "An error occurred during analysis."

# Create Gradio interface
with gr.Blocks(title="Kidney Stone Detection System", theme=gr.themes.Soft()) as demo:

    gr.HTML("""
    <div style="text-align: center; padding: 20px;">
        <h1>🏥 Kidney Stone Detection System</h1>
        <p style="font-size: 1.2em;">Advanced Automated Diagnosis using Deep Learning</p>
        <p><strong>System Status:</strong> Active | <strong>Model:</strong> Hybrid CNN-SVM Ensemble</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                label="Upload CT Scan Image",
                type="pil"
            )
            predict_btn = gr.Button("🔍 Analyze Image", variant="primary")
            
        with gr.Column():
            prediction_output = gr.Textbox(
                label="Prediction Results",
                lines=8,
                interactive=False
            )
            advice_output = gr.Textbox(
                label="Medical Advice",
                lines=10,
                interactive=False
            )
    
    predict_btn.click(
        fn=predict_kidney_stone,
        inputs=[image_input],
        outputs=[prediction_output, advice_output]
    )
    
    gr.HTML("""
    <div style="text-align: center; padding: 20px; margin-top: 20px; border-top: 1px solid #ddd;">
        <p><strong>Disclaimer:</strong> This tool is designed to assist in preliminary screening.</p>
        <p>It does not replace professional medical diagnosis. Please consult a specialist for confirmation.</p>
        <p style="font-size: 0.8em; color: #888;">&copy; 2024 Kidney Stone Detection Project. All rights reserved.</p>
    </div>
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch()
