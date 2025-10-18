import numpy as np
import matplotlib.pyplot as plt

np.random.seed(12)
# Activation functions and their derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Xavier initialization function
def xavier_init(shape):
    fan_in, fan_out = shape
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(fan_in, fan_out) * std

def he_init(shape):
    fan_in = shape[0]
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(*shape) * std

# Neural Network implementation
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001):
        self.w1 = he_init((input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        print('W=', self.w1.shape)
        print('b=', self.b1.shape)
  
        self.w2 = he_init((hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))
        print('W=', self.w2.shape)
        print('b=', self.b2.shape)
        
        self.learning_rate = learning_rate

    def forward(self, x):
        self.z1 = np.dot(x, self.w1) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2
         
    def compute_loss(self, y_true, y_pred):
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def compute_loss_derivative(self, y_true, y_pred):
        return (y_pred - y_true) / (y_pred * (1 - y_pred))

    def backward(self, x, y):
        m = y.shape[0]
        y = y.reshape(-1, 1)

        dL_da2 = self.compute_loss_derivative(y, self.a2)
        da2_dz2 = sigmoid_derivative(self.z2)
        dz2_dw2 = np.dot(self.a1.T, dL_da2 * da2_dz2) / m
        dz2_db2 = np.sum(dL_da2 * da2_dz2, axis=0, keepdims=True) / m

        dL_da1 = np.dot(dL_da2 * da2_dz2, self.w2.T)
        da1_dz1 = relu_derivative(self.z1)
        dz1_dw1 = np.dot(x.T, dL_da1 * da1_dz1) / m
        dz1_db1 = np.sum(dL_da1 * da1_dz1, axis=0, keepdims=True) / m

        return dz1_dw1, dz1_db1, dz2_dw2, dz2_db2

    def update_params_sgd(self, dw1, db1, dw2, db2):
        self.w1 -= self.learning_rate * dw1
        self.b1 -= self.learning_rate * db1
        self.w2 -= self.learning_rate * dw2
        self.b2 -= self.learning_rate * db2

    def predict(self, x):
        return self.forward(x)

    def test_accuracy(self, x_test, y_test):
        predictions = self.predict(x_test)
        predictions = (predictions > 0.5).astype(int)
        accuracy = np.mean(predictions == y_test.reshape(-1, 1))
        return accuracy

    def train(self, x_train, y_train, epochs, batch_size=10):
        for epoch in range(epochs):
            for i in range(0, x_train.shape[0], batch_size):
                x_batch = x_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                y_pred = self.forward(x_batch)
                loss = self.compute_loss(y_batch, y_pred)
                gradients = self.backward(x_batch, y_batch)
                self.update_params_sgd(*gradients)

            print(f'Epoch {epoch + 1}, Loss: {loss:.4f}')
            accuracy = self.test_accuracy(test_data, test_labels)
            print(f'Accuracy: {accuracy:.2f}')

        # Clasificar imágenes según las predicciones (robusto ante faltantes)
    def sort_predicted_images(self, predictions):
        import os, shutil

        preds = np.round(predictions).astype(int).flatten()
        test_folder = "dataset/test"
        cat_folder = "dataset/predict/cat"
        nocat_folder = "dataset/predict/nocat"

        os.makedirs(cat_folder, exist_ok=True)
        os.makedirs(nocat_folder, exist_ok=True)

        files = sorted([f for f in os.listdir(test_folder) if not f.startswith('.')])
        total_files = len(files)
        total_preds = len(preds)

        if total_files != total_preds:
            print(f"Advertencia: número de imágenes distinto al de predicciones ({total_files} vs {total_preds}).")
            print("Clasificando solo las imágenes existentes...")

        limit = min(total_files, total_preds)

        for i in range(limit):
            value = preds[i]
            src = os.path.join(test_folder, files[i])
            dst = os.path.join(cat_folder if value == 1 else nocat_folder, files[i])

            try:
                shutil.copy(src, dst)
            except FileNotFoundError:
                print(f"No se encontró la imagen {files[i]}, se omite.")
                continue

        print("Clasificación completada. Las imágenes se copiaron a 'dataset/predict/cat' y 'dataset/predict/nocat'.")


# Load and preprocess the data
train_data_flattened = np.loadtxt('dataset/train/train_images.csv', delimiter=',')
train_labels = np.loadtxt('dataset/train/train_labels.csv', delimiter=',')

test_data_flattened = np.loadtxt('dataset/test/test_images.csv', delimiter=',')
test_labels = np.loadtxt('dataset/test/test_labels.csv', delimiter=',')

# Normalize the data
train_data = train_data_flattened / 255.0
test_data = test_data_flattened / 255.0

# Define the neural network parameters
input_size = train_data.shape[1]
print(input_size)

hidden_size = 128
output_size = 1

# Create the neural network
nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate=0.001)

# Train the network
nn.train(train_data, train_labels, epochs=150, batch_size=10)

# Predict
predictions = nn.predict(test_data)
print(predictions)

# Test the model
accuracy = nn.test_accuracy(test_data, test_labels)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Mostrar resumen de redondeo
rounded_preds = np.round(predictions).astype(int).flatten()
count_cat = np.sum(rounded_preds == 1)
count_nocat = np.sum(rounded_preds == 0)

print(f"Predicciones redondeadas: {rounded_preds}")
print(f"Total 'cat' = {count_cat}, total 'nocat' = {count_nocat}")

# Clasificar imágenes automáticamente
nn.sort_predicted_images(predictions)