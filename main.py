import os
  

if __name__ == "__main__":
    import argparse

    MODELS_PATH = os.path.join(os.path.dirname(__file__), 'models')
    A_EXAMPLE_PATH = os.path.join(os.path.dirname(__file__), 'data','a','20240201-185821.png')
    TEST_FOLDER = os.path.join(os.path.dirname(__file__), 'data')

    parser = argparse.ArgumentParser(description='Create tickets')
    parser.add_argument('-m', '--models', type=str, help='Path to the models folder', required=True)
    parser.add_argument('-e', '--example', type=str, help='Path to the example to classify')
    parser.add_argument('-t', '--test', type=str, help='Path to the test folder')
    parser.add_argument('-a', '--architecture', type=str, required=True, help='Architecture to use as classifier', choices=['minDistance', 'embedder', 'sift'])
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose mode')


    args = vars(parser.parse_args())
    
    if args['architecture'] == 'minDistance':
        from architectures.linearHandModel import MinDistance
        
        hd = MinDistance(args['models'])
        if args['example']:
            print("Prediction: ", hd.classify(args['example'], args['verbose']))

        if args['test']:
            hd_2 = MinDistance(args['models'], args['test'])
            print("Accuracy",hd_2.accuracy())

    elif args['architecture'] == 'embedder':
        from architectures.embedder import Embedder

        hd = Embedder(args['models'])
        if args['example']:
            print("Prediction: ", hd.classify(args['example'], args['verbose']))

        if args['test']:
            hd_2 = Embedder(args['models'], args['test'])
            print("Accuracy",hd_2.accuracy())
    
    elif args['architecture'] == 'sift':
        from architectures.sift import SIFT

        hd = SIFT(args['models'])
        if args['example']:
            print("Prediction: ", hd.classify(args['example'], args['verbose']))

        if args['test']:
            hd_2 = SIFT(args['models'], args['test'])
            print("Accuracy",hd_2.accuracy())