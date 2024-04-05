from .baseModel import BaseModel
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat
from utils import utils
import os
import cv2 as cv
from tqdm import tqdm

EMBEDDER_PATH = os.path.join(os.path.dirname(__file__), 'embedder.tflite')

options = vision.ImageEmbedderOptions(
    base_options = python.BaseOptions(model_asset_path=EMBEDDER_PATH),
    l2_normalize = True, quantize = True)

embedder = vision.ImageEmbedder.create_from_options(options)

def image_descriptor(image):
    mpimage = Image(image_format=ImageFormat.SRGB, data=cv.cvtColor(image, cv.COLOR_BGR2RGB))
    return embedder.embed(mpimage).embeddings[0]
class Embedder(BaseModel):
    """
        Classify the image based on the embedding of the image
    """

    def __init__(self, models_path, test_path=None):
        super().__init__(models_path, test_path)
    
    def loadModels(self, folder):
        print("Loading models")
        for subfolder in tqdm(os.listdir(folder)):
            images = utils.load_images(os.path.join(folder,subfolder))
            processed_images = [image_descriptor(image) for image in images]
            self.model[subfolder] = processed_images
    
    def loadTests(self, folder):
        print("Loading tests")
        for subfolder in tqdm(os.listdir(folder)):
            images = utils.load_images(os.path.join(folder,subfolder))
            processed_images = [image_descriptor(image) for image in images]
            self.test[subfolder] = processed_images
        
    
    def predict(self, descriptor, verbose=False):
        """
            Predict the letter of the image
            Parameters:
                image: image to classify
                verbose: if True, return the distances to the model
            Returns:
                letter: letter of the image
        """

        similarities = {

            key: [
                vision.ImageEmbedder.cosine_similarity(descriptor, model) 
                for model in self.model[key]] 

            for key in self.model
        }
        
        if verbose:
            import json
            print("Similarities: ", json.dumps(similarities, indent=4))
        return max(similarities, key=lambda x: max(similarities[x]))
    
    def accuracy(self):
        """
            Calculate the accuracy of the model
            Parameters:
                metric: metric to use
            Returns:
                accuracy: accuracy of the model
        """
        correct = 0
        total = 0
        print("Testing")
        for key in tqdm(self.test):
            for descriptor in self.test[key]:
                prediction = self.predict(descriptor)
                if prediction == key:
                    correct += 1
                total += 1
        return correct/total
    
    def classify(self, image: str, verbose=False):
        image = cv.imread(image)
        descriptor = image_descriptor(image)
        return self.predict(descriptor, verbose=verbose)
