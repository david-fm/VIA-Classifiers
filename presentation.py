import marimo

__generated_with = "0.2.8"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    from architectures.linearHandModel import MinDistance
    return MinDistance, mo


@app.cell
def __(mo):
    mo.md(r'''
        # Syllabel classifier

        In this project, we will be using transformation and comparition between a base example (or set of examples) and a new example to classify the new example as one of the base examples.


    ''')
    return


@app.cell
def __(mo):
    mo.md(r'''
        ## Load examples

        First of all load the examples from your folder. For this indicate the absolute path of the folder containing the examples.

    ''')
    return


@app.cell
def __(mo):
    modelFolder = mo.ui.text(placeholder="Models Absolute Path")
    modelFolder
    return modelFolder,


@app.cell
def __(MinDistance, modelFolder):
    hd = MinDistance(modelFolder.value)
    return hd,


@app.cell
def __(mo):
    mo.md(r'''
        ## Load your new example

        Now load the new example to classify. For this indicate the absolute path of the file containing the new example.

    ''')
    return


@app.cell
def __(mo):
    imageFile = mo.ui.text(placeholder="Image Absolute Path")
    imageFile
    return imageFile,


@app.cell
def __(mo):
    verbose = mo.ui.checkbox(label="Verbose")
    verbose
    return verbose,


@app.cell
def __(imageFile, mo):
    mo.image(imageFile.value)
    return


@app.cell
def __(hd, imageFile, mo, verbose):
    result = hd.classify(imageFile.value)
    if verbose.value:
        toShow=mo.md(f'''
            The result of the prediction is: {result[0]}

            The best distances from the new example to the base examples are:

            - To a: {result[1][0]}
            - To e: {result[1][1]}
            - To i: {result[1][2]}
            - To o: {result[1][3]}
            - To u: {result[1][4]}
        ''')
    else:
        toShow=mo.md(f'''
            The result of the prediction is: {result}
        ''')
    toShow
    return result, toShow


@app.cell
def __(mo):
    mo.md(r'''
        ## Testing the model

        To test the accuracy of the model, load a folder containing the examples to test.

        The folder should contain a subfolder for each class, and each subfolder should contain the examples of that class. Hierarchy:
        ```
        folder
        |
        |---a
            |---example1
            |---example2
            |---...
        |---e
            |---example1
            |---example2
            |---...
        |---i
            |---example1
            |---example2
            |---...
        |---o
            |---example1
            |---example2
            |---...
        |---u
            |---example1
            |---example2
            |---...
        ```
    ''')
    return


@app.cell
def __(mo):
    pointsFolder = mo.ui.text(placeholder="Test Folder Absolute Path")
    pointsFolder
    return pointsFolder,


@app.cell
def __(hd, mo, pointsFolder):
    hd.loadTests(pointsFolder.value)
    accuracy = hd.accuracy()
    mo.md(f'''
        The accuracy of the model is: {accuracy}
    ''')
    return accuracy,


if __name__ == "__main__":
    app.run()
