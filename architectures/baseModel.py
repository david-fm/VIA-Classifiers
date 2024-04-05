from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
        Base class for the models
    """
    def __init__(self, models_path, test_path=None):
        """
            Initialize the model"""

        self.model = {
        }

        self.test = {
        }

        self.models_path = models_path
        self.test_path = test_path
        self.loadModels(models_path)
        if test_path:
            self.loadTests(test_path)

    @abstractmethod
    def predict(self, verbose=False, *args, **kwargs):
        """
            Predict the class of the image
            Parameters:
                verbose: print the results
            Returns:
                letter: letter of the image
        """
        pass
    @abstractmethod
    def accuracy(self, *args, **kwargs):
        """
            Calculate the accuracy of the model
            Parameters:
                metric: metric to use
            Returns:
                accuracy: accuracy of the model
        """
        pass
    @abstractmethod
    def loadModels(self, folder):
        """
            Load the models from a folder
            Parameters:
                folder: folder with the models
            Returns:
                None
        """
        pass
    @abstractmethod
    def loadTests(self, folder):
        """
            Load the tests from a folder
            Parameters:
                folder: folder with the tests
            Returns:
                None
        """
        pass

    @abstractmethod
    def classify(self, image, verbose=False):
        """
            Classify the image
            Parameters:
                image: image to classify
            Returns:
                letter: letter of the image
        """
        pass