# MediaEval Prediction

This is the project with the task of predicting the click-through rate of article-image pairs in the MediaEval 2020 dataset.

## Setup and Execution

You will need:

- python 3.8 or higher
- all python packages and additional data installed (run `setup.sh`on your machine)
- Run the main.py to predict the CTR and nReads of Maydata set:
The output file contains three columns representing predicted ctr, predicted nReads and weighted ctr. 
The predicted files are in the same order as the in the maydata file.
- Average runtime 20 min (measured on a device with 2,9 GHz 6-Core Intel Core i5, no graphics card accelerating)

## Default Setting
-  The original text data is not this source code due to the limitation of file size on isis.
The default file path of text data is "data/". All the used data is origin from gitlab MediaEval Data and Data2021
- The image files are not in this source code. The default path is "data/imgs" with 'imgFile' as file names.
- the default approach structure is the Bernoulli-regression model as described in the report.

## License

MIT
