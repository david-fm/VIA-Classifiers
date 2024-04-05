"""
Min distance classifier, this classifier will classify the image based on the distance between the points of the image and the points of the model
"""
from utils import utils
import cv2
import numpy as np
import matplotlib.pyplot as plt
from .baseModel import BaseModel
import os

def getPointsFromImage(image):
    """
        Get the points from the image
        Parameters:
            image: image path
        Returns:
            points: list of points
    """
    image = cv2.imread(image)
    _, points = utils.get_results([image])
    return points[0]


# def dist(a,b):
#     mtx1, mtx2, disparity = procrustes(a, b)
#     return disparity




def scale( points):
    # scale the points based on the distance between the point 0 and the point 9
    distance09 = abs(points[9] - points[0])
    # change distance09 values that are 0 to 1 to avoid division by 0
    distance09 = np.where(distance09 == 0, 1, distance09)

    try:
        scaled_points = points/distance09
        return scaled_points
    except:
        return points




def transform( references, to_transform):
    """Transform the points of to_transform to the reference"""
    transformed_points = []
    for reference in references:
        reference = np.array(reference)
        transforming = np.array(to_transform)
        # Translate the points to the same position as the reference
        # Given that the model reference is has the point 0 in the origin
        # the translation will make the point 0 of the to_transform to be at the origin
        movement = reference[0] - transforming[0]
        
        transforming = transforming + movement


        # IMPORTANT: ROTATION IS NOT NECESSARY AFTER SCALING THE POINTS

        # angle = np.arctan2(reference[9][1], reference[9][0]) - np.arctan2(to_transform[9][1], to_transform[9][0]) # angle in radians arctan2(y,x)
        # cos = np.cos(angle)
        # sin = np.sin(angle)
        # transformation_matrix = np.array([[cos, -sin],[sin, cos]])
        # to_transform = to_transform @ transformation_matrix
        
        #fig, ax = HandDetector.plot(to_transform, "transformed", fig, ax, "g")
        transforming = scale(transforming)
        transformed_points.append(transforming)
    return transformed_points



def plot(transform, name="test", fig=None, ax=None, color=None):
    """Plot the transformed points, allow to plot multiple points in the same figure"""
    if fig is None:
        fig, ax = plt.subplots()
    # transform to int
    for i in range(21):
        x, y = transform[i]
        ax.plot(x, y)
        ax.text(x, y, str(i))

    conections = [
        [0,1,2,3,4],
        [0,5,6,7,8],
        [9,10,11,12],
        [13,14,15,16],
        [0,17,18,19,20],
        [5,9,13,17]
    ]
    # draw lines between points in each array of conections
    for conections in conections:
        for i in range(len(conections) - 1):
            x = [transform[conections[i]][0], transform[conections[i+1]][0]]
            y = [transform[conections[i]][1], transform[conections[i+1]][1]]
            if color is not None:
                ax.plot(x,y, color=color)
            else:
                ax.plot(x,y)
    
    reference = (1,0)
    if color is not None:
        ax.plot(reference[0], reference[1], "o", color=color)
    else:
        ax.plot(reference[0], reference[1], "o")

    ax.set_title(name)
    return fig, ax


def distance( a, b):
    return a-b


def distanceMean(distance: list[list[int]]):
    """Calculate the mean of the distances"""
    return np.linalg.norm( distance, axis=1).mean()

def distanceMedian(distance: list[list[int]]):
    return np.median(np.linalg.norm( distance, axis=1))

    

class MinDistance(BaseModel):
    def __init__(self, models_path, test_path=None):
        super().__init__(models_path, test_path)
        
    def loadModels(self, folder):
        """
            Load points from a folder with the models organized in subfoders
            The result of the predictions will be the best match to this models.
        """
        for subfolder in os.listdir(folder):
            images = utils.load_images(os.path.join(folder,subfolder))
            _, points = utils.get_results(images)
            for hand_points in points:
                if subfolder in self.model.keys():
                    self.model[subfolder].append(scale(hand_points - hand_points[0]))
                else:
                    self.model[subfolder] = [scale(hand_points - hand_points[0])]
    
    def loadPoints(self, folder):
            """
                Load points from a folder with the points organized in subfoders
                Use this method to test the model accuracy
            """

            for subfolder in os.listdir(folder):
                images = utils.load_images(os.path.join(folder,subfolder))
                _, points = utils.get_results(images)
                for hand_points in points:
                    if subfolder in self.test.keys():
                        self.test[subfolder].append(hand_points)
                    else:
                        self.test[subfolder] = [hand_points]

    def loadTests(self, folder):
        self.loadPoints(folder)
    
    def predict(self, points: list[list[int]], metric="mean", verbose=False):
        """
            Predict the letter of the points
            Parameters:
                points: list of points
                verbose: if True, return the distances to the model
            Returns:
                letter: letter of the points
        """
        # Search for the most similar after the transformation
        # Get the transformed points
        transformedA = transform(self.model["a"], points)
        transformedE = transform(self.model["e"], points)
        transformedI = transform(self.model["i"], points)
        transformedO = transform(self.model["o"], points)
        transformedU = transform(self.model["u"], points)

        # Calculate the distance between the transformed points and the model points

        distancesA = distance(np.array(transformedA), np.array(self.model["a"]))
        distancesE = distance(np.array(transformedE), np.array(self.model["e"]))
        distancesI = distance(np.array(transformedI), np.array(self.model["i"]))
        distancesO = distance(np.array(transformedO), np.array(self.model["o"]))
        distancesU = distance(np.array(transformedU), np.array(self.model["u"]))
        letters = ["a", "e", "i", "o", "u"] 
        distances = [distancesA, distancesE, distancesI, distancesO, distancesU]

        measures = []
        for d in distances:
            letter_measures = []
            for case in d:
                if metric == "mean":
                    letter_measures.append(distanceMean(case))
                else:
                    letter_measures.append(distanceMedian(case))
            # Append the best measure for each letter
            measures.append(min(letter_measures))

        
        if verbose:
            return letters[measures.index(min(measures))], measures

        return letters[measures.index(min(measures))]
        
    def accuracy(self, metric="mean"):
        """
            Measure how well the model is working
            Returns:
                accuracy: accuracy of the model
        """
        # Measure how well the model is working
        accuracy = 0
        num_examples = 0
        for letter in self.test.keys():
            num_examples += len(self.test[letter])
            for points_example in self.test[letter]:
                if self.predict(points_example, metric=metric) == letter:
                    accuracy += 1
        
        #print(f"Accuracy: {accuracy/num_examples} Num examples: {num_examples} Right: {accuracy}")

        return accuracy/num_examples
    
    def classify(self, image_path:str, verbose=False):
        """
            Classify the image
            Parameters:
                image: image path
            Returns:
                letter: letter of the image
        """
        points = getPointsFromImage(image_path)
        return self.predict(points, verbose=verbose)

