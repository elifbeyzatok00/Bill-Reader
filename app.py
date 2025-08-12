import gradio as gr
import torch
import re
import os
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import json
from huggingface_hub import login

# Hugging Face token ile giriş
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN", "your_token")
if HUGGING_FACE_TOKEN:
    login(token=HUGGING_FACE_TOKEN)

# Model ve processor'ı yükle
@gr.cache_examples
def load_model():
    processor = DonutProcessor.from_pretrained("elifbeyza/donut-base-invoices-donut-data-v1")
    model = VisionEncoderDecoderModel.from_pretrained("elifbeyza/donut-base-invoices-donut-data-v1")
    return processor, model

def process_invoice_image(image):
    """Fatura resmini işleyip JSON çıktısı döndürür"""
    try:
        # Model ve processor'ı yükle
        processor, model = load_model()
        
        # Cihaz seçimi
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        # Resmi işle
        pixel_values = processor(image, return_tensors="pt").pixel_values
        
        # Task prompt
        task_prompt = "<s_invoices-donut-data-v1>"
        decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
        
        # Model ile tahmin yap
        outputs = model.generate(
            pixel_values.to(device),
            decoder_input_ids=decoder_input_ids.to(device),
            max_length=model.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
            output_scores=True,
        )
        
        # Çıktıyı işle
        sequence = processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
        
        # JSON'a çevir
        try:
            result_json = processor.token2json(sequence)
            formatted_json = json.dumps(result_json, indent=2, ensure_ascii=False)
        except:
            # JSON çevirimi başarısız olursa ham metni döndür
            formatted_json = sequence
        
        return formatted_json, sequence
        
    except Exception as e:
        return f"Hata oluştu: {str(e)}", ""

# Gradio arayüzü
def create_interface():
    with gr.Blocks(title="Fatura Okuyucu - Donut Model", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 📄 Fatura Okuyucu Uygulaması")
        gr.Markdown("Bu uygulama Donut modelini kullanarak fatura resimlerinden bilgi çıkarır.")
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Fatura Resmi Yükleyin",
                    height=400
                )
                
                process_btn = gr.Button(
                    "Faturayı İşle", 
                    variant="primary",
                    size="lg"
                )
                
                gr.Markdown("### Desteklenen Format:")
                gr.Markdown("- PNG, JPG, JPEG dosyaları")
                gr.Markdown("- Fatura, invoice türü belgeler")
            
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("JSON Çıktısı"):
                        json_output = gr.Textbox(
                            label="Çıkarılan Bilgiler (JSON)",
                            lines=20,
                            max_lines=30,
                            show_copy_button=True,
                            interactive=False
                        )
                    
                    with gr.TabItem("Ham Çıktı"):
                        raw_output = gr.Textbox(
                            label="Ham Model Çıktısı",
                            lines=20,
                            max_lines=30,
                            show_copy_button=True,
                            interactive=False
                        )
        
        # Event handlers
        process_btn.click(
            fn=process_invoice_image,
            inputs=[image_input],
            outputs=[json_output, raw_output],
            show_progress=True
        )
        
        # Örnek resimler (eğer varsa)
        gr.Markdown("### 💡 Nasıl Kullanılır:")
        gr.Markdown("""
        1. Yukarıdaki alana bir fatura resmi yükleyin
        2. 'Faturayı İşle' butonuna tıklayın
        3. Model faturadan bilgileri çıkaracak ve JSON formatında gösterecek
        4. Çıktıları kopyalayabilir veya indirebilirsiniz
        """)
    
    return interface

if __name__ == "__main__":
    # Arayüzü oluştur ve başlat
    interface = create_interface()
    interface.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
