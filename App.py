import datetime
import sys
import torch
import pandas as pd
import numpy as np

from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog, QPushButton, QLabel, \
    QSizePolicy
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.dates import DateFormatter
from matplotlib.figure import Figure
from palettable.matplotlib import matplotlib

from Predict import load_predictor, load_scaler, get_device, predict
from Training import init_hidden
from motec_preprocess import read_CSV
from matplotlib import pyplot as plt


class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'ACC Stint Estimator'
        self.device = 'cpu'
        self.predictor = load_predictor('./', self.device)
        self.scaler = load_scaler('./')
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 60
        self.initUI()


    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        #Load file button
        self.button = QPushButton('Open CSV', self)
        self.button.setToolTip('Load MoteC CSV File')
        self.button.move(20, 20)
        self.button.clicked.connect(self.on_click)

        # creating a label
        self.label = QLabel(self)
        self.label.move(110, 20)
        # setting geometry to the label
        self.label.setGeometry(110, 20,
                               510,
                               25)
        self.label.setStyleSheet("QLabel"
                                 "{"
                                 "border : 2px solid black;"
                                 "}")
        self.show()

    @pyqtSlot()
    def on_click(self):
        filename = self.openFileNameDialog()
        if filename:
            self.label.setText('Processing File: {}'.format(filename.split('/')[-1]))
            self.button.setEnabled(False)
            # Load DataFrame
            df = read_CSV(filename)
            # Scale DataFrame
            dfs_scaled = pd.DataFrame(self.scaler.transform(df.values), index=df.index, columns=df.columns)
            # init hidden and cell state
            hidden = init_hidden(1, 1024, self.device)
            #Predict 30 laps
            target = self.scaler.inverse_transform(dfs_scaled.values)

            input = torch.tensor(dfs_scaled.iloc[0:10].values, dtype=torch.float)
            input = torch.reshape(input, (1, input.shape[0], input.shape[1]))
            input.to(self.device)

            out1, hidden = predict(input, self.predictor, hidden)

            input = torch.reshape(out1, (1, out1.shape[0], out1.shape[1]))
            out2, hidden = predict(input, self.predictor, hidden)

            input = torch.reshape(out2, (1, out2.shape[0], out2.shape[1]))
            out3, hidden = predict(input, self.predictor, hidden)

            out = torch.cat((out1, out2, out3), dim=0)
            out_np = out.cpu().numpy()
            out_np = self.scaler.inverse_transform(out_np)
            out_np = np.concatenate((target[0:10], out_np), axis=0)

            # Plotting
            out_np_time = [datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=time) for time in out_np[:, -1]]
            plt.plot(out_np_time, label='prediction')
            target_time = [datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=time) for time in target[:, -1]]
            plt.plot(target_time, label='original')
            plt.title('Estimated Lap Times')
            plt.ylabel("Time")
            plt.xlabel("Lap")
            plt.gca().yaxis.set_major_formatter(DateFormatter('%M:%S'))
            plt.tight_layout()
            plt.legend()
            plt.show()

            self.button.setEnabled(True)

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "Comma Separated Value Files (*.csv)", options=options)
        if fileName:
            return fileName


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())