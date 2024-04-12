import cv2 as cv
from .baseModel import BaseModel
from utils import utils
import os
from tqdm import tqdm
from typing import NamedTuple, Sequence, Callable, List, Tuple

def good_matches(matches):
    good = []
    for mt in matches:
        if len(mt) == 2:
            best, second = mt
            if best.distance < 0.75*second.distance:
                good.append(best)    
    return good

class FrameInfo(NamedTuple):
    keypoints: list
    descriptors: list
    image: str

class Match(NamedTuple):
    punctuation: int
    imageInfo: FrameInfo

class SIFT(BaseModel):
    def __init__(self, models_path, test_path=None):
        self.sift = cv.SIFT_create(nfeatures=500,enable_precise_upscale = True, contrastThreshold = 0.09)
        self.matcher = cv.BFMatcher()
        self.x0 = None
        super().__init__(models_path, test_path)
        self.to_predict_image: FrameInfo = None
        
    def image_descriptor(self, image):
        keypoints , descriptors = self.sift.detectAndCompute(image, mask=None)
        return FrameInfo(keypoints, descriptors, image)
    
    def loadModels(self, folder):
        print("Loading models")
        for subfolder in tqdm(os.listdir(folder)):
            images = utils.load_images(os.path.join(folder,subfolder))
            processed_images = [self.image_descriptor(image) for image in images]
            self.model[subfolder] = processed_images

    def loadTests(self, folder):
        print("Loading tests")
        for subfolder in tqdm(os.listdir(folder)):
            images = utils.load_images(os.path.join(folder,subfolder))
            processed_images = [self.image_descriptor(image) for image in images]
            self.test[subfolder] = processed_images
        
    def show_comparition(self, to_predict_info: FrameInfo, model_info: FrameInfo, matches: List[cv.DMatch]):
        
        img_matches = cv.drawMatches(to_predict_info.image, to_predict_info.keypoints, model_info.image, model_info.keypoints, matches, None)
        cv.imshow("Matches", img_matches)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    def get_punctuation(self, descriptor, model):
        cleaned_matches = self.clean_match(good_matches(self.matcher.knnMatch(descriptor, model[1], k=2)))
        return len(cleaned_matches)
    
    def clean_match(self, match: Sequence[cv.DMatch]):
        """
            Remove the matches with repeated keypoints
            
            Parameters:
                match: match to clean
            Returns:
                match: cleaned match
        """
        def comparison(x: cv.DMatch, y: List[cv.DMatch]):
            for z in y:
                if x.trainIdx == z.trainIdx:
                    return True

        result = []
        for x in match:
            if not comparison(x, result):
                result.append(x)
        return result
        
        
        
    def predict(self, descriptor, verbose=False, *args, **kwargs):
        all_matches = {
            key: [
                Match(self.get_punctuation(descriptor, model), model)
                for model in self.model[key]
            ]
            for key in self.model
        }
        # for key in all_matches:
        #     print(f"{key}: {list(map(lambda x: x.punctuation, all_matches[key]))}")

        # match with the most punctuation
        result = ""
        max_match = Match(0, None)
        for key in all_matches:
            for match in all_matches[key]:
                if match.punctuation > max_match.punctuation:
                    max_match = match
                    result = key
        if verbose:
            cleaned_matches = self.clean_match(good_matches(self.matcher.knnMatch(descriptor, max_match.imageInfo.descriptors, k=2)))
            self.show_comparition(self.to_predict_image, max_match.imageInfo, cleaned_matches)

        return result

    def accuracy(self, *args, **kwargs):
        correct = 0
        total = 0
        print("Testing")
        for key in tqdm(self.test):
            for _, descriptor, _ in self.test[key]:
                prediction = self.predict(descriptor)
                if prediction == key:
                    correct += 1
                total += 1
        return correct/total
    
    def classify(self, image:str, verbose=False):
        image = cv.imread(image)
        self.to_predict_image = self.image_descriptor(image)
        predicted = self.predict(
            self.to_predict_image.descriptors, 
            verbose)
        return predicted