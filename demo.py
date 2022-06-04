import random
import cv2
from imutils import paths
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import numpy as np
from keras.preprocessing.image import img_to_array


class GAN():
    def __init__(self):
        # 图片尺寸
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        # 输入的图片尺寸
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        # Adam优化器
        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    # 数据预处理
    def load_data(self, path):
        print("loading images...")
        data = []
        labels = []
        # grab the image paths and randomly shuffle them
        imagePaths = sorted(list(paths.list_images(path)))
        random.seed(42)
        random.shuffle(imagePaths)
        for imagePath in imagePaths:
            image = cv2.imread(imagePath)
            image = cv2.resize(image, (self.img_rows, self.img_cols))
            image = img_to_array(image)
            data.append(image)

            label = str(imagePath.split(os.path.sep)[-2])
            labels.append(label)

        # scale the raw pixel intensities to the range [0, 1]
        data = np.array(data, dtype="float") / 255.0
        # labels = np.array(labels)
        # return data, labels
        return data

    # 构建生成器
    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))  # 全连接层
        model.add(LeakyReLU(alpha=0.2))  # 带泄露修正线性单元
        model.add(BatchNormalization(momentum=0.8))  # 批归一化
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))  # np.prod()计算所有乘积，输入
        model.add(Reshape(self.img_shape))  # reshape成图片的尺寸

        # model.summary()  # 日志

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    # 构建判别器
    def build_discriminator(self):
        # 模型选用的是传统的线性模型，CNN中用的也是这个
        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))  # 展平层
        model.add(Dense(512))  # 全连接层
        model.add(LeakyReLU(alpha=0.2))  # 带泄露修正线性单元
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        # model.summary()

        img = Input(shape=self.img_shape)  # 输入尺寸
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50, file_path=None):

        # 加载数据
        X_train = self.load_data(file_path)

        # 标准化
        # X_train = np.expand_dims(X_train, axis=3)

        # 创建标签
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            if epoch % 200 == 0:
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

            if epoch % 2000 == 0:
                os.makedirs('weights', exist_ok=True)
                self.generator.save_weights("weights/gen_epoch%d.h5" % epoch)
                self.discriminator.save_weights("weights/dis_epoch%d.h5" % epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()



if __name__ == '__main__':
    gan = GAN()
    path = './data'
    gan.train(epochs=50001, batch_size=32, sample_interval=200, file_path=path)

