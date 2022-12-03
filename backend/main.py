

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("../saved_models/1")

CLASS_NAMES = ['Apple Braeburn', 'Apple Crimson Snow', 'Apple Golden 1', 
    'Apple Golden 2', 'Apple Golden 3', 'Apple Granny Smith', 'Apple Pink Lady',
    'Apple Red 1', 'Apple Red 2', 'Apple Red 3', 'Apple Red Delicious', 
    'Apple Red Yellow 1', 'Apple Red Yellow 2', 'Apricot', 'Avocado', 
    'Avocado ripe', 'Banana', 'Banana Lady Finger', 'Banana Red', 'Beetroot', 
    'Blueberry', 'Cactus fruit', 'Cantaloupe 1', 'Cantaloupe 2', 'Carambula', 
    'Cauliflower', 'Cherry 1', 'Cherry 2', 'Cherry Rainier', 'Cherry Wax Black', 
    'Cherry Wax Red', 'Cherry Wax Yellow', 'Chestnut', 'Clementine', 'Cocos',
    'Corn', 'Corn Husk', 'Cucumber Ripe', 'Cucumber Ripe 2', 'Dates', 'Eggplant',
    'Fig', 'Ginger Root', 'Granadilla', 'Grape Blue', 'Grape Pink', 'Grape White',
    'Grape White 2', 'Grape White 3', 'Grape White 4', 'Grapefruit Pink', 
    'Grapefruit White', 'Guava', 'Hazelnut', 'Huckleberry', 'Kaki', 'Kiwi',
    'Kohlrabi', 'Kumquats', 'Lemon', 'Lemon Meyer', 'Limes', 'Lychee', 
    'Mandarine', 'Mango', 'Mango Red', 'Mangostan', 'Maracuja', 
    'Melon Piel de Sapo', 'Mulberry', 'Nectarine', 'Nectarine Flat', 
    'Nut Forest', 'Nut Pecan', 'Onion Red', 'Onion Red Peeled', 'Onion White', 
    'Orange', 'Papaya', 'Passion Fruit', 'Peach', 'Peach 2', 'Peach Flat', 
    'Pear', 'Pear 2', 'Pear Abate', 'Pear Forelle', 'Pear Kaiser', 
    'Pear Monster', 'Pear Red', 'Pear Stone', 'Pear Williams', 'Pepino', 
    'Pepper Green', 'Pepper Orange', 'Pepper Red', 'Pepper Yellow', 'Physalis', 
    'Physalis with Husk', 'Pineapple', 'Pineapple Mini', 'Pitahaya Red', 'Plum', 
    'Plum 2', 'Plum 3', 'Pomegranate', 'Pomelo Sweetie', 'Potato Red', 
    'Potato Red Washed', 'Potato Sweet', 'Potato White', 'Quince', 'Rambutan', 
    'Raspberry', 'Redcurrant', 'Salak', 'Strawberry', 'Strawberry Wedge', 
    'Tamarillo', 'Tangelo', 'Tomato 1', 'Tomato 2', 'Tomato 3', 'Tomato 4', 
    'Tomato Cherry Red', 'Tomato Heart', 'Tomato Maroon', 'Tomato Yellow', 
    'Tomato not Ripened', 'Walnut', 'Watermelon']

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)