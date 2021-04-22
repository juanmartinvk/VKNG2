# -*- coding: utf-8 -*-
import sys
import soundfile as sf
import numpy as np
import acoustical_parameters as ap
import pyqtgraph as pg
from pyqtgraph import PlotWidget
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QDialog, QApplication

file_path = "IR_test/IR_test_mono.wav"
IR_raw, fs = sf.read(file_path)
b = 1
ETC_avg_dB, decay, EDT, T20, T30, C50, C80 = ap.parameters(IR_raw, fs, b=b, truncate='lundeby')


if b == 3:
    nominal_bands = ['25', '31.5', '40', '50', '63', '80', '100', '125', '160',
                     '200', '250', '315', '400', '500', '630', '800', '1k',
                     '1.3k', '1.6k', '2k', '2.5k', '3.2k', '4k', '5k', 
                     '6.3k', '8k', '10k', '12.5k', '16k', '20k']
elif b == 1:
    nominal_bands = ['31.5', '63', '125', '250', '500',
                     '1k', '2k', '4k', '8k', '16k']


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        loadUi("gui_main.ui",self)
        # self.paramTable.setColumnWidth(0,250)
        # self.paramTable.setColumnWidth(1,100)
        # self.paramTable.setColumnWidth(2,350)
        self.setWindowTitle("Acoustical Parameters")
        self.graphWidget.setBackground('w')
        
        
        self.loadData()
        self.plotData()

    def loadData(self):

        self.paramTable.setColumnCount(len(nominal_bands))
        self.paramTable.setHorizontalHeaderLabels(nominal_bands)
        
        for idx, band in enumerate(nominal_bands):
             self.paramTable.setItem(0, idx, QtWidgets.QTableWidgetItem(str(EDT[idx])))
             self.paramTable.setItem(1, idx, QtWidgets.QTableWidgetItem(str(T20[idx])))
             self.paramTable.setItem(2, idx, QtWidgets.QTableWidgetItem(str(T30[idx])))
             self.paramTable.setItem(3, idx, QtWidgets.QTableWidgetItem(str(C50[idx])))
             self.paramTable.setItem(4, idx, QtWidgets.QTableWidgetItem(str(C80[idx])))
             self.paramTable.setColumnWidth(idx,35)
    
    def plotData(self):
        self.graphWidget.plot(ETC_avg_dB[7], pen='b')
        self.graphWidget.plot(decay[7], pen='r')
             



# main
app = QApplication(sys.argv)
mainwindow = MainWindow()
mainwindow.show()
# widget = QtWidgets.QStackedWidget()
# widget.addWidget(mainwindow)
# widget.setFixedHeight(850)
# widget.setFixedWidth(1120)
# widget.show()
try:
    sys.exit(app.exec_())
except:
    print("Exiting")