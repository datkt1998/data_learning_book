import os
import itertools
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from tensorflow import keras
from tensorflow.keras import layers
from tqdm.keras import TqdmCallback
from datpy.modeling.preprocessing import TextPreprocessing
from tensorflow.keras.callbacks import (ModelCheckpoint, TensorBoard,
                                        EarlyStopping, ReduceLROnPlateau)
from sklearn.metrics import (classification_report, accuracy_score,
                             precision_recall_fscore_support, confusion_matrix)


class DataGenerator:

    def __init__(self, ):
        pass

    def read_file_in_ML(filepath):
        data = pd.read_csv(filepath, sep="|", header=None)
        x_data = data.loc[:, 0].to_numpy()
        y_data = data.loc[:, 1].to_numpy()
        return (x_data, y_data)


class BaseModel:
    def __init__(self, model_name='model', savedir='outputs/models', is_neuralnet=False):
        self.savedir = savedir
        self.model_name = model_name
        self.is_neuralnet = is_neuralnet
        self.label_encoder, self.classnames = self.get_labelencoder()
        self.train, self.test = None, None

    def design_model(self):
        pass

    def get_data(**kwargs):
        pass

    def build_model(self, **kwargs):
        self.train, self.test = self.get_data()
        self.model = self.design_model()
        self.train_model(**kwargs)
        self.evaluate_model()
        return self

    def get_labelencoder(self):
        try:
            label_encoder = BaseModel().load_model('models/models_v1.0/label_encoder.pkl').model
            classnames = label_encoder.classes_
        except:
            label_encoder = classnames = None
        return label_encoder, classnames

    def train_model(self, **kwargs):
        if not self.is_neuralnet:
            self.model.fit(*self.train)
        else:
            cb = CallbackSetup(experiment_name=self.model_name)
            self.model.fit(*self.train,
                           # steps_per_epoch = self.train_len, 
                           batch_size=self.batch_size,
                           validation_data=self.test,
                           # validation_steps = self.test_len,
                           verbose=0, epochs=kwargs['epochs'],
                           callbacks=cb.setup(False, False, False, False),
                           )

    def evaluate_model(self):
        if self.test is None:
            self.train, self.test = self.get_data()
        if not self.is_neuralnet:
            self.test_score = self.model.score(*self.test)
            self.test_prob_preds = self.model.predict_proba(self.test[0])
            self.test_preds = self.model.predict(self.test[0])
        else:
            self.test_score = self.model.evaluate(*self.test)
            self.test_prob_preds = self.model.predict(self.test[0])
            self.test_preds = self.test_prob_preds.argmax(axis=-1)
        print("Test accuracy: ", self.test_score)

    def save_model(self, ):
        dt = datetime.now().strftime("_%Y%m%d_%H%M%S")
        if not os.path.exists(self.savedir):
            os.mkdir(self.savedir)
        if not self.is_neuralnet:
            modelpath = os.path.join(self.savedir, self.model_name + dt) + ".pkl"
            with open(modelpath, 'wb') as f:
                pickle.dump(self.model, f)
        else:
            modelpath = os.path.join(self.savedir, self.model_name + dt)  # + ".h5"
            self.model.save(modelpath)
        print("Model saved in: ", modelpath)

    def load_model(self, model_dir):
        if model_dir.endswith(".pkl"):
            with open(model_dir, 'rb') as f:
                self.model = pickle.load(f)
        else:
            self.model = keras.models.load_model(model_dir)
        return self

    def predict(self, x):
        x = [x] if not isinstance(x, list) else x
        res = self.label_encoder.inverse_transform(
            self.model.predict(x))
        return res[0] if len(res) == 1 else res

    def get_performance(self):
        print(classification_report(self.test[1], self.test_preds))
        precision, recall, fscore, support = precision_recall_fscore_support(self.test[1], self.test_preds)
        df = pd.DataFrame(data=[precision, recall, fscore],
                          index=['precision', 'recall', 'fscore'],
                          columns=self.classnames)
        df.loc['accuracy'] = accuracy_score(self.test[1], self.test_preds)
        df = df.reset_index().rename(columns={'index': 'metric'})
        df['model_name'] = self.model_name
        return df

    def make_confusion_matrix(self, figsize=(7, 7), text_size=10):
        classes = self.classnames
        y_true, y_pred = self.test[1], self.test_preds
        # Create the confustion matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # normalize it
        n_classes = cm.shape[0]

        fig, ax = plt.subplots(figsize=figsize)
        cax = ax.matshow(cm, cmap=plt.cm.Blues)
        fig.colorbar(cax)
        labels = np.arange(cm.shape[0]) if classes is None else classes

        # Label the axes
        ax.set(title="Confusion Matrix",
               xlabel="Predicted label",
               ylabel="True label",
               xticks=np.arange(n_classes),
               yticks=np.arange(n_classes),
               xticklabels=labels,
               yticklabels=labels)

        # Make x-axis labels appear on bottom
        ax.xaxis.set_label_position("bottom")
        ax.xaxis.tick_bottom()

        # Set the threshold for different colors
        threshold = (cm.max() + cm.min()) / 2.

        # Plot the text on each cell
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j] * 100:.1f}%)",
                     horizontalalignment="center",
                     color="white" if cm[i, j] > threshold else "black",
                     size=text_size)
        plt.show()

    def find_wrong(self, n=100):
        pred_df = pd.DataFrame({
            "text": self.test[0],
            "y_true": self.label_encoder.inverse_transform(self.test[1]),
            "y_pred": self.label_encoder.inverse_transform(self.test_preds),
            "pred_conf": self.test_prob_preds.max(axis=1)})
        wrong_pred = pred_df[pred_df['y_true'] != pred_df['y_pred']] \
                         .sort_values('pred_conf', ascending=False).iloc[:n].reset_index(drop=True)
        return wrong_pred


class MLClassifier(BaseModel):
    def __init__(self, classifier, cfg_params=None, model_name='model',
                 savedir='outputs/models', is_neuralnet=False):
        super().__init__(model_name, savedir, is_neuralnet)
        self.classifier = classifier
        self.cfg_params = cfg_params

    def get_data(self):
        train = DataGenerator.read_file_in_ML('datasets/preprocessed/train.txt')
        test = DataGenerator.read_file_in_ML('datasets/preprocessed/test.txt')
        return train, test

    def design_model(self):
        model = Pipeline([
            ('tfidf', TfidfVectorizer()),  # convert words to numbers using TF-IDF
            ('clf', self.classifier)  # model the text
        ])
        if self.cfg_params is not None:
            model = GridSearchCV(model, self.cfg_params, cv=3, scoring='accuracy')
        return model


class DLClassifier(BaseModel):
    def __init__(self, model_name='DL',
                 savedir='outputs/models', is_neuralnet=False):
        super().__init__(model_name, savedir, is_neuralnet)
        self.batch_size = 64
        self.MAX_SEQUENCE_LEN = 200

    def get_data(self, **kwargs):
        train = DataGenerator.read_file_in_ML('datasets/preprocessed/train.txt')
        test = DataGenerator.read_file_in_ML('datasets/preprocessed/test.txt')
        return train, test

    def design_model(self):
        max_vocab_length = 10000
        max_sequence_length = 200
        # setup TextVectorization
        tvect = layers.TextVectorization(
            # max_tokens=max_vocab_length,
            output_sequence_length=max_sequence_length,
            output_mode='int', name="text_vectorization_1"
        )
        tvect.adapt(self.train[0])

        # setup embedding layer
        embedding = layers.Embedding(input_dim=max_vocab_length,  # set input shape
                                     output_dim=128,  # set size of embedding vector
                                     embeddings_initializer="uniform",  # default, intialize randomly
                                     input_length=max_sequence_length,  # how long is each input
                                     name="embedding_1")

        # create model
        inputs = keras.layers.Input(shape=(1,), dtype='string')
        x = tvect(inputs)
        x = embedding(x)
        x = keras.layers.GlobalAveragePooling1D()(x)
        outputs = keras.layers.Dense(3, activation='softmax')(x)
        model = keras.Model(inputs, outputs, name=self.model_name)

        model.compile(loss="sparse_categorical_crossentropy",
                      optimizer=keras.optimizers.legacy.Adam(),
                      metrics=["accuracy"])
        return model


class CallbackSetup:
    def __init__(self, experiment_name, dir_name="outputs/models"):
        self.dir_name = dir_name
        self.experiment_name = experiment_name
        dt = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(dir_name, experiment_name, dt)

    def _create_checkpoint_save(self):
        self.checkpoint_path = os.path.join(self.path, "checkpoint.ckpt")
        print(f"Saving Modelcheckpoint to: {self.checkpoint_path}")
        checkpoint = ModelCheckpoint(self.checkpoint_path, save_weights_only=True)
        return checkpoint

    # setup log tensorboard to compare multi experiments
    def _create_tensorboard_log(self):
        tensorboard_callback = TensorBoard(log_dir=self.path)
        print(f"Saving TensorBoard to: {self.path}")
        return tensorboard_callback

    def _create_earlystop(self):
        earlystop = EarlyStopping(monitor='val_loss',
                                  min_delta=0.1,
                                  patience=4,
                                  )
        return earlystop

    def _create_reduce_lr(self):
        reduce_lr = ReduceLROnPlateau(monitor="val_loss",
                                      factor=0.2,
                                      patience=2,
                                      verbose=1,
                                      min_lr=1e-7)
        return reduce_lr

    def setup(self, earlystop=True, reduce_lr=True, checkpoint=True, log=True):
        cb = [
            TqdmCallback(),
        ]
        if checkpoint:
            cb.append(self._create_checkpoint_save())
        if log:
            cb.append(self._create_tensorboard_log())
        if earlystop:
            cb.append(self._create_earlystop())
        if reduce_lr:
            cb.append(self._create_reduce_lr())
        return cb


class Ensemble(BaseModel):

    def __init__(self, configs, model_name='ensemble'):
        self.version = configs.version
        self.model_name = model_name
        self.models_dir = 'models/models_v' + self.version
        assert os.path.exists(self.models_dir)
        self.list_modelpaths = [os.path.join(self.models_dir, i)
                                for i in os.listdir(self.models_dir)
                                if not i.startswith('label_encoder')]
        self.list_models = Ensemble.load_list_models(self.list_modelpaths)
        self.preprocessor = TextPreprocessing(configs.stopwords_path)
        self.label_encoder, self.classnames = self.get_labelencoder()

    def load_list_models(model_lspath):
        models = []
        for path in model_lspath:
            models.append(BaseModel().load_model(path).model)
        return models

    def ensemble_predict_proba(self, x):
        x = [x] if isinstance(x, str) else x
        x_pre = [self.preprocessor.transform(i) for i in x]
        predictions = []
        for model in self.list_models:
            if isinstance(model, Pipeline):
                predictions.append(model.predict_proba(x_pre))
            else:
                predictions.append(model.predict(x_pre, verbose=0))
        mean_pred = np.array(predictions).mean(axis=0)
        return mean_pred

    def ensemble_pred_classname(self, pred_prob):
        pred = pred_prob.argmax()
        pred_class = self.label_encoder.inverse_transform([pred])[0]
        return pred_class

    def ensemble_predict(self, x):
        pred_prob = self.ensemble_predict_proba(x)
        pred = pred_prob.argmax(axis=1)
        pred_class = self.label_encoder.inverse_transform(pred)
        res = map(lambda x: x.replace("_", " ").capitalize(), pred_class)
        return list(res)[0] if len(x) == 1 else list(res)

    def load_testset(self, x_test, y_test):
        self.test = (x_test, y_test)
        self.test_prob_preds = self.ensemble_predict_proba(x_test)
        self.test_preds = self.test_prob_preds.argmax(axis=1)
