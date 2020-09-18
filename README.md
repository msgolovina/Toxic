Bert Classifier for [Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) served with Flask API

To run the app:

 * clone the repo
 * run `pip install -r requirements.txt`
 * change `DEVICE`, `HOST`, `PORT` in config if necessary
 * download [pytorch_model.bin](https://drive.google.com/drive/folders/1RRDKurCrhFsDdOTE8h6xU7z7bpLY8Jd9?usp=sharing) to trained_models folder
 *  alternatively, reproduce training by running `python train.py`
 * run `python app.py`

