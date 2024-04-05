import os
  

if __name__ == "__main__":
    import argparse

    MODELS_PATH = os.path.join(os.path.dirname(__file__), 'models')
    A_EXAMPLE_PATH = os.path.join(os.path.dirname(__file__), 'data','a','20240201-185821.png')
    TEST_FOLDER = os.path.join(os.path.dirname(__file__), 'data')

    parser = argparse.ArgumentParser(description='Create tickets')
    parser.add_argument('-m', '--models', type=str, default=MODELS_PATH, help='Path to the models folder')
    parser.add_argument('-e', '--example', type=str, default=A_EXAMPLE_PATH, help='Path to the example to classify')
    parser.add_argument('-t', '--test', type=str, default=TEST_FOLDER, help='Path to the test folder')
    parser.add_argument('-a', '--architecture', type=str, default='minDistance', help='Architecture to use as classifier', choices=['minDistance', 'embedder'])
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose mode')


    args = vars(parser.parse_args())
    
    if args['architecture'] == 'minDistance':
        from architectures.linearHandModel import MinDistance

        hd = MinDistance(args['models'])
        print("Prediction: ", hd.classify(args['example'], args['verbose']))

        hd_2 = MinDistance(args['models'], args['test'])
        print("Accuracy",hd_2.accuracy())

    elif args['architecture'] == 'embedder':
        from architectures.embedder import Embedder

        hd = Embedder(args['models'])
        print("Prediction: ", hd.classify(args['example'], args['verbose']))

        hd_2 = Embedder(args['models'], args['test'])
        print("Accuracy",hd_2.accuracy())