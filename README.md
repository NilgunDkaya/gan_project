# gan_project

PyTorch ile yazılmış GAN (Generative Adversarial Network) projesini TensorFlow'a dönüştürme.

Kurulum için gerekli paketler:
pip install tensorflow matplotlib numpy

Önemli Farklılıklar:

Model Tanımlama: PyTorch'ta nn.Module, TensorFlow'da keras.Model kullanılıyor
Activation Fonksiyonları: TensorFlow'da aktivasyon fonksiyonları doğrudan layer parametresi olarak verilebiliyor
Optimizer: TensorFlow'da @tf.function dekoratörü ile eğitim adımı optimize ediliyor
Gradient Hesaplama: PyTorch'ta backward() ve step(), TensorFlow'da GradientTape kullanılıyor
Veri Yükleme: PyTorch'ta DataLoader, TensorFlow'da tf.data.Dataset kullanılıyor

Yazdığım kod Fashion-MNIST veri setini otomatik indirecek ve 50 epoch boyunca eğitim yapacak.
Her epoch sonunda üretilen görüntüler generated_fake_imgs klasörüne kaydedilecek.
