import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
from generator import Generator
from discriminator import Discriminator

# GPU kullanımını kontrol et
print("GPU Kullanılabilir:", tf.config.list_physical_devices('GPU'))

# Fashion-MNIST veri setini yükle
(train_images, _), (_, _) = keras.datasets.fashion_mnist.load_data()

# Veriyi normalize et (-1, 1 aralığına)
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5

# Dataset oluştur
BATCH_SIZE = 128
dataset = tf.data.Dataset.from_tensor_slices(train_images)
dataset = dataset.shuffle(60000).batch(BATCH_SIZE)

# Modelleri oluştur
generator = Generator()
discriminator = Discriminator()

# Loss fonksiyonu
cross_entropy = keras.losses.BinaryCrossentropy()

# Optimizer'lar
generator_optimizer = keras.optimizers.Adam(learning_rate=0.0002)
discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.0002)

# Sabit gürültü vektörü (görselleştirme için)
fixed_noise = tf.random.normal([16, 100])

# Loss fonksiyonları
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Görüntüleri kaydetme fonksiyonu
def save_images(images, epoch):
    os.makedirs("generated_fake_imgs", exist_ok=True)
    
    # 4x4 grid oluştur
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < len(images):
            # Normalize et ve göster
            img = images[i].numpy().squeeze()
            img = (img + 1) / 2.0  # -1,1 aralığından 0,1'e
            ax.imshow(img, cmap='gray')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"generated_fake_imgs/epoch_{epoch+1}.png")
    plt.close()

# Eğitim adımı
@tf.function
def train_step(real_images):
    batch_size = tf.shape(real_images)[0]
    noise = tf.random.normal([batch_size, 100])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generator'dan sahte görüntüler üret
        fake_images = generator(noise, training=True)
        
        # Discriminator çıktıları
        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(fake_images, training=True)
        
        # Loss hesapla
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    
    # Gradient'leri hesapla
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    # Ağırlıkları güncelle
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss

# Eğitim döngüsü
EPOCHS = 50

for epoch in range(EPOCHS):
    total_gen_loss = 0
    total_disc_loss = 0
    num_batches = 0
    
    for batch in dataset:
        gen_loss, disc_loss = train_step(batch)
        total_gen_loss += gen_loss
        total_disc_loss += disc_loss
        num_batches += 1
    
    # Ortalama loss değerlerini hesapla
    avg_gen_loss = total_gen_loss / num_batches
    avg_disc_loss = total_disc_loss / num_batches
    
    # Her epoch sonunda örnek görüntüleri kaydet
    fake_images = generator(fixed_noise, training=False)
    save_images(fake_images, epoch)
    
    print(f"Epoch {epoch+1} D Loss: {avg_disc_loss:.4f} G Loss: {avg_gen_loss:.4f}")

print("Eğitim tamamlandı!")