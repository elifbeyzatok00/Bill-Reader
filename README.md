# 📄 Fatura Okuyucu - Donut Model

Bu proje, Donut (Document Understanding Transformer) modelini kullanarak fatura ve invoice belgelerinden otomatik bilgi çıkarımı yapan bir AI uygulamasıdır. Fine-tune edilmiş Donut modeli sayesinde fatura resimlerinden JSON formatında yapılandırılmış veri elde edebilirsiniz.

![Gradio UI Image](https://raw.githubusercontent.com/elifbeyzatok00/Bill-Reader2/refs/heads/main/Gradio%20UI%20Image.png)
(Gradio UI Image)

## 🎯 Proje Amacı

Bu uygulama, fatura işleme süreçlerini otomatikleştirmek için geliştirilmiştir. Kullanıcılar fatura resimlerini yükleyerek şu bilgileri otomatik olarak çıkarabilir:
- Fatura numarası ve tarihi
- Satıcı ve müşteri bilgileri
- Ürün detayları (açıklama, miktar, fiyat)
- KDV ve toplam tutarlar
- IBAN ve vergi numaraları

## ✨ Özellikler

- 🤖 **Fine-tune edilmiş Donut modeli**: Faturalar için özel olarak eğitilmiş
- 🌐 **Gradio Web Arayüzü**: Kullanıcı dostu web tabanlı arayüz
- 📊 **JSON Çıktısı**: Yapılandırılmış veri formatında sonuçlar
- 🔄 **Ham Çıktı Görüntüleme**: Model çıktısının ham halini inceleme
- 🖼️ **Görsel Önizleme**: Yüklenen fatura resminin görüntülenmesi
- 💾 **Kopyalama Özelliği**: Sonuçları kolayca kopyalayabilme

## 🛠️ Teknoloji Stack

- **Python 3.8+**
- **Transformers**: Hugging Face transformers kütüphanesi
- **PyTorch**: Derin öğrenme framework'ü
- **Gradio**: Web arayüzü oluşturma
- **Pillow**: Görüntü işleme
- **Donut Model**: Document understanding için özel model

## 📋 Gereksinimler

```txt
gradio>=4.0.0
torch>=1.9.0
transformers>=4.20.0
datasets>=2.0.0
sentencepiece>=0.1.95
Pillow>=8.0.0
huggingface-hub>=0.10.0
accelerate>=0.20.0
python-dotenv>=0.19.0
```

## 🚀 Kurulum

### 1. Depoyu Klonlayın
```bash
git clone https://github.com/kullaniciadi/Bill-Reader2.git
cd Bill-Reader2
```

### 2. Sanal Ortam Oluşturun
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate     # Windows
```

### 3. Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### 4. Çevre Değişkenlerini Ayarlayın
Proje kök dizininde `.env` dosyası oluşturun:
```bash
touch .env  # Linux/Mac
# veya Windows'ta dosyayı manuel oluşturun
```

`.env` dosyasının içeriğini aşağıdaki gibi düzenleyin:
```env
HUGGING_FACE_TOKEN=your_huggingface_token_here
WANDB_API_KEY=your_wandb_api_key_here  # İsteğe bağlı (sadece eğitim için)
```

**Not**: `.env` dosyasını `.gitignore` dosyasına eklemeyi unutmayın!

Örnek `.gitignore` dosyası:
```gitignore
# Çevre değişkenleri
.env
.env.local

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/

# Model dosyaları (büyük dosyalar)
*.bin
*.safetensors

# IDE
.vscode/
.idea/
```

## 🎬 Demo Video

Uygulamanın nasıl çalıştığını görmek için demo videosunu izleyebilirsiniz:

📹 **[Fatura Okuyucu - Donut Model Demo](./Fatura%20Okuyucu%20-%20Donut%20Model.mp4)**

Video, uygulamanın temel özelliklerini ve kullanım adımlarını göstermektedir.

## 💻 Kullanım

### Web Uygulamasını Başlatma
```bash
python app.py
```

Uygulama başlatıldıktan sonra:
1. Tarayıcınızda `http://localhost:7860` adresine gidin
2. Fatura resmini yükleyin (PNG, JPG, JPEG formatları desteklenir)
3. "Faturayı İşle" butonuna tıklayın
4. Sonuçları JSON ve ham çıktı sekmelerinde görüntüleyin

### Programatik Kullanım
```python
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# Model ve processor'ı yükle
processor = DonutProcessor.from_pretrained("elifbeyza/donut-base-invoices-donut-data-v1")
model = VisionEncoderDecoderModel.from_pretrained("elifbeyza/donut-base-invoices-donut-data-v1")

# Resmi yükle ve işle
image = Image.open("fatura.jpg")
pixel_values = processor(image, return_tensors="pt").pixel_values

# Model ile tahmin yap
task_prompt = "<s_invoices-donut-data-v1>"
decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]

outputs = model.generate(
    pixel_values,
    decoder_input_ids=decoder_input_ids,
    max_length=model.decoder.config.max_position_embeddings,
    early_stopping=True,
    pad_token_id=processor.tokenizer.pad_token_id,
    eos_token_id=processor.tokenizer.eos_token_id,
    use_cache=True,
    num_beams=1,
    bad_words_ids=[[processor.tokenizer.unk_token_id]],
    return_dict_in_generate=True,
)

# Sonucu işle
sequence = processor.batch_decode(outputs.sequences)[0]
# ... sonuç işleme kodları
```

## 📁 Proje Yapısı

```
Bill-Reader2/
├── app.py                              # Ana uygulama dosyası
├── requirements.txt                    # Python bağımlılıkları
├── README.md                          # Proje dokümantasyonu
├── testImage.jpg                      # Test için örnek fatura
├── notebooks/                         # Jupyter notebook'ları
│   ├── Fine_Tune_Donut_on_invoices-donut-data-v1.ipynb
│   └── Quick_inference_with_Fine_tuned_Donut_for_Document_Parsing.ipynb
└── Fatura Okuyucu - Donut Model.mp4  # Uygulama demo videosu
```

## 🔬 Model Detayları

### Fine-tuning Süreci
Model, `katanaml-org/invoices-donut-data-v1` veri seti kullanılarak fine-tune edilmiştir:
- **Base Model**: `naver-clova-ix/donut-base`
- **Fine-tuned Model**: `elifbeyza/donut-base-invoices-donut-data-v1`
- **Görüntü Boyutu**: 640x480 piksel
- **Maksimum Uzunluk**: 384 token
- **Eğitim Verisi**: 425 fatura örneği
- **Doğrulama Verisi**: 50 fatura örneği

### Çıktı Formatı
Model aşağıdaki JSON yapısında çıktı üretir:

```json
{
  "header": {
    "invoice_no": "40378170",
    "invoice_date": "10/15/2012",
    "seller": "Patel, Thompson and Montgomery 356 Kyle Vista New James, MA 46228",
    "client": "Jackson, Odonnell and Jackson 267 John Track Suite 841 Jenniferville, PA 98601",
    "seller_tax_id": "958-74-3511",
    "client_tax_id": "998-87-7723",
    "iban": "GB77WRBQ31965128414006"
  },
  "items": [
    {
      "item_desc": "Ürün açıklaması",
      "item_qty": "1,00",
      "item_net_price": "7,50",
      "item_net_worth": "7,50",
      "item_vat": "10%",
      "item_gross_worth": "8,25"
    }
  ],
  "summary": {
    "total_net_worth": "$7,50",
    "total_vat": "$0,75",
    "total_gross_worth": "$8,25"
  }
}
```

## 🎓 Jupyter Notebook'lar

### 1. Fine-Tuning Notebook
`notebooks/Fine_Tune_Donut_on_invoices-donut-data-v1.ipynb`
- Model fine-tuning süreci
- Veri seti hazırlama
- Eğitim parametreleri
- Model performans değerlendirmesi

### 2. Hızlı Çıkarım Notebook
`notebooks/Quick_inference_with_Fine_tuned_Donut_for_Document_Parsing.ipynb`
- Fine-tune edilmiş modelin test edilmesi
- Örnek faturalar üzerinde çıkarım
- Sonuçların analizi

## ⚙️ Yapılandırma

### Ortam Değişkenleri (.env dosyası)
```env
HUGGING_FACE_TOKEN=your_token_here    # Hugging Face erişim token'ı
WANDB_API_KEY=your_wandb_key_here     # Weights & Biases API anahtarı (isteğe bağlı)
```

### Uygulama Ayarları
`app.py` dosyasında aşağıdaki ayarları değiştirebilirsiniz:
- `server_port`: Sunucu portu (varsayılan: 7860)
- `server_name`: Sunucu adresi (varsayılan: "0.0.0.0")
- `share`: Gradio share özelliği (varsayılan: True)

## 🔧 Sorun Giderme

### Yaygın Sorunlar

1. **CUDA/GPU Sorunları**
   ```bash
   # CPU modunda çalıştırma için
   export CUDA_VISIBLE_DEVICES=""
   ```

2. **Bellek Yetersizliği**
   - Batch size'ı azaltın
   - Görüntü boyutunu küçültün
   - CPU moduna geçin

3. **Model Yükleme Hataları**
   - İnternet bağlantınızı kontrol edin
   - Hugging Face token'ınızı doğrulayın
   - Modelin erişilebilir olduğundan emin olun

## 🤝 Katkıda Bulunma

1. Bu depoyu fork edin
2. Feature branch'i oluşturun (`git checkout -b feature/AmazingFeature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some AmazingFeature'`)
4. Branch'inizi push edin (`git push origin feature/AmazingFeature`)
5. Pull Request açın

## 🔗 Faydalı Bağlantılar

- [Hugging Face Fine Tuned Donut Model Sayfası](https://huggingface.co/elifbeyza/donut-base-invoices-donut-data-v1)
- [Hugging Face Donut Model Sayfası](https://huggingface.co/naver-clova-ix/donut-base)
- [Hugging Face Veri Seti Sayfası](https://huggingface.co/datasets/katanaml-org/invoices-donut-data-v1/viewer/default/train?row=0&views%5B%5D=train)
- [Donut Paper](https://arxiv.org/abs/2111.15664)
- [Transformers Dokümantasyonu](https://huggingface.co/docs/transformers/)
- [Gradio Dokümantasyonu](https://gradio.app/docs/)

