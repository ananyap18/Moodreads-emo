from tensorflow import keras
from app.views import generate_model
model = generate_model()
model.load_weights("C:\\Users\\Ananya Prasad\\Moodreads\\BackEnd\\app\\model_weights.h5")
print(model)
