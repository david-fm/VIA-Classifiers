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
        self.classying_img = None
    
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
        
    def show_comparition(self, similarities, result):
        # show the image and bars at the side with the similarities
        offset = 30
        # resize the image to add space at the side for the text
        # resize by adding black space
        self.classying_img = cv.copyMakeBorder(self.classying_img, 0, 0, 0, 500, cv.BORDER_CONSTANT, value=(0, 0, 0))
        for key in similarities:
            cv.putText(
                self.classying_img, 
                f"{key}: {float(similarities[key][0]):.2f}", 
                (self.classying_img.shape[1] - 470, offset),
                cv.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 0, 255) if key != result else (0, 255, 0),
                2)
            offset += 30
        cv.imshow("Image", self.classying_img)
        cv.waitKey(0)


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
        result = max(similarities, key=lambda x: max(similarities[x]))
        
        if verbose:
            import json
            print("Similarities: ", json.dumps(similarities, indent=4))
            self.show_comparition(similarities, result)
        return result
    
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
        self.classying_img = image
        descriptor = image_descriptor(image)
        return self.predict(descriptor, verbose=verbose)
