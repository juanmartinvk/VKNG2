# -*- coding: utf-8 -*-
import sys
import soundfile as sf
import ntpath
import csv
import pandas as pd
import numpy as np
import ctypes
import acoustical_parameters as ap
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QMainWindow, QApplication, QAbstractItemView, QFileDialog, QListWidgetItem, QErrorMessage

myappid = u'VKNG'
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)


def analyzeFile(impulsePath, filterPath, b, truncate='lundeby', smoothing="schroeder", dataType="IR", median_window=20):

    # Read file
    IR_raw, fs = sf.read(impulsePath)
    IR_raw_L = IR_raw
    IR_raw_R = None
    
    # Check if stereo or mono. If stereo, split channels.
    if IR_raw.ndim == 2:
        IR_raw_L = IR_raw[0:, 0]
        IR_raw_R = IR_raw[0:, 1]
    
    # If sweep, convolve with inverse filter
    if filterPath is not None:
        inverse_filter, fs_filter = sf.read(filterPath)
        if inverse_filter.ndim == 2:
            inverse_filter = inverse_filter[0:, 0]
        if fs != fs_filter:
            print("Sampling rates of sweep and inverse filter do not match")
        IR_raw_L = ap.convolve_sweep(IR_raw_L, inverse_filter)
        if IR_raw_R is not None:
            IR_raw_R = ap.convolve_sweep(IR_raw_R, inverse_filter)
    
    
    # Calculate parameters
    acParamL = ap.parameters(IR_raw_L, fs, b, truncate=truncate, smoothing=smoothing, median_window=median_window)
    if IR_raw_R is not None:
        acParamR = ap.parameters(IR_raw_R, fs, b, truncate=truncate, smoothing=smoothing, median_window=median_window)
        acParamR.IACCe=np.round(ap.IACCe_from_IR(acParamL, acParamR), decimals=3)
        acParamL.IACCe=acParamR.IACCe
    else:
        acParamR = None
    
    # Define nominal bands
    if b == 3:
        nominalBands = ['25', '31.5', '40', '50', '63', '80', '100', '125', '160',
                         '200', '250', '315', '400', '500', '630', '800', '1k',
                         '1.3k', '1.6k', '2k', '2.5k', '3.2k', '4k', '5k', 
                         '6.3k', '8k', '10k', '12.5k', '16k', '20k']
    elif b == 1:
        nominalBands = ['31.5', '63', '125', '250', '500',
                         '1k', '2k', '4k', '8k', '16k']
        
    return acParamL, acParamR, nominalBands


class SetupWindow(QMainWindow):
    def __init__(self):
        super(SetupWindow, self).__init__()
        
        # Load designed UI
        loadUi("gui/gui_setup.ui",self)
        
        # Defaults
        self.octaveButton.setChecked(True)
        self.schroederButton.setChecked(True)
        self.lundebyButton.setChecked(True)
        self.impulseButton.setChecked(True)
        self.filterGroupBox.setEnabled(False)
        #self.filterGroupBox.hide()
        self.b = 1
        self.smoothing = "schroeder"
        self.truncate = 'lundeby'
        self.dataType= "IR"
        
        
        # Window settings
        self.setWindowTitle("VKNG - Setup")

        self.setWindowIcon(QtGui.QIcon("gui/icon.png"))
        
        # Button bindings
        self.browseImpulseButton.clicked.connect(self.browseImpulse)
        self.browseFilterButton.clicked.connect(self.browseFilter)
        self.analyzeButton.clicked.connect(self.analyzeData)
        self.cancelButton.clicked.connect(self.close)
        self.impulseButton.toggled.connect(self.toggleImpulse)
        self.schroederButton.toggled.connect(self.toggleSchroeder)
        self.medianButton.toggled.connect(self.toggleMedian)
        self.sweepButton.toggled.connect(self.toggleSweep)
        self.octaveButton.toggled.connect(self.toggleOctave)
        self.thirdOctaveButton.toggled.connect(self.toggleThirdOctave)
        self.lundebyButton.toggled.connect(self.toggleLundeby)
        self.noCompButton.toggled.connect(self.toggleNoComp)
    
    def browseImpulse(self):
        # Open file dialog and load path onto line edit widget
        self.filePath = QFileDialog.getOpenFileName()[0]
        self.impulsePathBox.setText(self.filePath)
    
    def browseFilter(self):
        filePath = QFileDialog.getOpenFileName()[0]
        self.filterPathBox.setText(filePath)
            
    def analyzeData(self):
        # Analyze file data, hide setup window and show data window
        impulsePath = self.impulsePathBox.text()
        filterPath = self.filterPathBox.text()
        
        try:
            # For Impulse Response data
            if self.dataType == "IR" and ntpath.exists(impulsePath):
                self.paramL, self.paramR, self.nominalBands = analyzeFile(impulsePath, None, self.b, self.truncate,
                                                                          self.smoothing, self.dataType, int(self.windowText.text()))
        
            # For Sweep data
            elif self.dataType == "sweep" and ntpath.exists(impulsePath) and ntpath.exists(filterPath):
                self.paramL, self.paramR, self.nominalBands = analyzeFile(impulsePath, filterPath, self.b, self.truncate,
                                                                          self.smoothing, self.dataType, int(self.windowText.text()))
            
            else:
                error_dialog = QErrorMessage()
                error_dialog.showMessage('Invalid file path')
        
        except:
            error_dialog = QErrorMessage()
            error_dialog.showMessage('Unknown error. Please try again')
        
        # Replace zeros (errors) with "-"
        def replaceZeros(myl):
            myl = list(myl)
            for idx, value in enumerate(myl):
                if value == 0:
                    myl[idx] = "--"
                
            return myl
        
        def replaceParam(param):
            param.EDT = replaceZeros(param.EDT)
            param.T20 = replaceZeros(param.T20)
            param.T30 = replaceZeros(param.T30)
            param.EDTTt = replaceZeros(param.EDTTt)
            return param
        
        self.paramL = replaceParam(self.paramL)
        if self.paramR is not None:
            self.paramR = replaceParam(self.paramR)
        
        
        self.dataWindow = DataWindow(self.paramL, self.paramR, self.nominalBands, self.b, self.filePath, self.smoothing)
        self.hide()
        self.dataWindow.show()
        
        
        
        
    def toggleImpulse(self):
        self.filterGroupBox.setEnabled(False)
        #self.filterGroupBox.hide()
        self.impulseGroupBox.setTitle("IR file")
        self.dataType = "IR"
        
    def toggleSweep(self):
        self.filterGroupBox.setEnabled(True)
        #self.filterGroupBox.show()
        self.impulseGroupBox.setTitle("Sweep file")
        self.dataType = "sweep"
        
    def toggleSchroeder(self):
        self.smoothing = "schroeder"
        self.windowLabel.setEnabled(False)
        self.windowText.setEnabled(False)
        
    def toggleMedian(self):
        self.smoothing = "median"
        self.windowLabel.setEnabled(True)
        self.windowText.setEnabled(True)
    
    def toggleOctave(self):
        self.b = 1
    
    def toggleThirdOctave(self):
        self.b = 3
        
    def toggleLundeby(self):
        self.truncate = 'lundeby'
    
    def toggleNoComp(self):
        self.truncate = None

        

class DataWindow(QMainWindow):
    def __init__(self, paramL, paramR, nominalBands, b, filePath, smoothing="schroeder"):
        super(DataWindow, self).__init__()
        self.paramL = paramL
        self.paramR = paramR
        self.b = b
        self.nominalBands = nominalBands
        self.filePath = filePath

        
        # Defaults
        self.currentParam = self.paramL
        self.currentBandIdx = self.nominalBands.index("1k")
        self.smoothing = smoothing
        
        #Load designed UI 
        loadUi("gui/gui_main.ui",self)
        
        #Load data to table
        self.loadData()
        
        # Window preferences
        self.setWindowTitle("VKNG - Acoustical Parameters")
        self.setWindowIcon(QtGui.QIcon("gui/icon.png"))
        
        # Graph settings
        self.graphWidget.setBackground('w')   
        self.graphWidget.addLegend(offset = (0, 10))
        self.graphWidget.showGrid(x=True, y=True)
        self.graphWidget.setYRange(-150, 0, padding=0)
        self.graphWidget.setLabel("left", "[dB]")
        self.graphWidget.setLabel("bottom", "Time [s]")
        self.graphWidget.setTitle(self.nominalBands[self.currentBandIdx] + "Hz band")

        # Plot data
        self.plotData()   
        
        # Channel list settings
        if paramR == None:      #For mono IR
            self.channelList.hide()
        else:                   #For stereo IR
            self.itemL = QListWidgetItem()
            self.itemR = QListWidgetItem()
            self.itemL.setText("Left Channel")
            self.itemR.setText("Right Channel")
            self.channelList.addItem(self.itemL)
            self.channelList.addItem(self.itemR)
            self.channelList.setCurrentItem(self.itemL)
            self.channelList.itemSelectionChanged.connect(self.switchChannel)
            
            
        # Update plot when a table cell of each band is clicked
        self.paramTable.cellClicked.connect(self.updateData)
        # Disable table edit mode
        self.paramTable.setEditTriggers(QAbstractItemView.NoEditTriggers)
        
        # Button bindings
        self.setupButton.clicked.connect(self.openSetup)
        self.exportButton.clicked.connect(self.exportData)
        
               

    def loadData(self):
        # Loads data from current param object onto table widget
        self.paramTable.setColumnCount(len(self.nominalBands))
        self.paramTable.setHorizontalHeaderLabels(self.nominalBands)
        
        # Set column width according to band filtering
        if self.b == 1:
            column_width = 60
        else:
            column_width = 40
            
        #Load data to table
        for idx, band in enumerate(self.nominalBands):
             self.paramTable.setItem(0, idx, QtWidgets.QTableWidgetItem(str(self.currentParam.EDT[idx])))
             self.paramTable.setItem(1, idx, QtWidgets.QTableWidgetItem(str(self.currentParam.T20[idx])))
             self.paramTable.setItem(2, idx, QtWidgets.QTableWidgetItem(str(self.currentParam.T30[idx])))
             self.paramTable.setItem(3, idx, QtWidgets.QTableWidgetItem(str(self.currentParam.C50[idx])))
             self.paramTable.setItem(4, idx, QtWidgets.QTableWidgetItem(str(self.currentParam.C80[idx])))
             self.paramTable.setItem(5, idx, QtWidgets.QTableWidgetItem(str(self.currentParam.Tt[idx])))
             self.paramTable.setItem(6, idx, QtWidgets.QTableWidgetItem(str(self.currentParam.EDTTt[idx])))
             if self.paramR is not None:
                 self.paramTable.setItem(7, idx, QtWidgets.QTableWidgetItem(str(self.currentParam.IACCe[idx])))
             self.paramTable.setColumnWidth(idx,column_width)
             
    def plotData(self):
        # Initializes plot and sets lines and labels
        if self.smoothing == "schroeder":
            self.decay_line = self.graphWidget.plot(self.currentParam.t, self.currentParam.decay[self.currentBandIdx], pen='r', name="Schroeder decay")
        elif self.smoothing == "median":
            self.decay_line = self.graphWidget.plot(self.currentParam.t, self.currentParam.decay[self.currentBandIdx], pen='r', name="Median filtered decay")
            
        self.ETC_line = self.graphWidget.plot(self.currentParam.t, self.currentParam.ETC_avg_dB[self.currentBandIdx], pen='b', name="Energy Time Curve")
        #self.ETC_line = self.graphWidget.plot(self.currentParam.t, self.currentParam.ETC_dB[self.currentBandIdx], pen='b', name="Energy Time Curve")

        
        
    def updateData(self, row, column):
        # Updates plot with data from current band and current param object
        self.decay_line.setData(self.currentParam.t, self.currentParam.decay[column])
        #self.ETC_line.setData(self.currentParam.t, self.currentParam.ETC_dB[column])
        self.ETC_line.setData(self.currentParam.t, self.currentParam.ETC_avg_dB[column])
        self.currentBandIdx = column
        self.graphWidget.setTitle(self.nominalBands[self.currentBandIdx] + "Hz band")
        
        
    def openSetup(self):
        # Opens setup window and closes data window
        self.setupWindow = SetupWindow()
        self.setupWindow.show()
        self.close()
    
    def switchChannel(self):
        # Switches current channel and loads the corresponding data
        if self.channelList.currentItem() == self.itemL:
            self.currentParam = self.paramL
            self.loadData()
            self.updateData(1, self.currentBandIdx)
        else:
            self.currentParam = self.paramR
            self.loadData()
            self.updateData(1, self.currentBandIdx)

    def exportData(self):
        file_types = "CSV (*.csv);; Excel Spreadsheet (*.xlsx)"
        options = QFileDialog.Options()
        
        default_name = ntpath.basename(self.filePath)
        ext_idx = default_name.find(".")
        default_name = default_name[:ext_idx]
        if self.paramR is not None:
            if self.channelList.currentItem() == self.itemL:
                default_name = default_name + "_L"
            else:
                default_name = default_name + "_R"
        
        filename, _ = QFileDialog.getSaveFileName(self, 'Save as... File', default_name, filter=file_types, options=options)
        
        
        
        if filename[-4:] != '.csv' and filename[-5:] != '.xlsx':
            filename = filename + '.csv'
        
        
        f = list(self.nominalBands)
        
        EDT = list(self.currentParam.EDT)
        T20 = list(self.currentParam.T20)
        T30 = list(self.currentParam.T30)
        C50 = list(self.currentParam.C50)
        C80 = list(self.currentParam.C80)
        Tt = list(self.currentParam.Tt)
        EDTTt = list(self.currentParam.EDTTt)
        IACCe = list(self.currentParam.IACCe)
        
        f.insert(0, "f [Hz]")
        EDT.insert(0, "EDT [s]")
        T20.insert(0, "T20 [s]")
        T30.insert(0, "T30 [s]")
        C50.insert(0, "C50 [dB]")
        C80.insert(0, "C80 [dB]")
        Tt.insert(0, "Tt [s]")
        EDTTt.insert(0, "EDTTt [s]")
        IACCe.insert(0, "IACCe")
        
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows([f, EDT, T20, T30, C50, C80, Tt, EDTTt])
            if self.paramR is not None:
                writer.writerow(IACCe)
        
        if filename[-5:] == '.xlsx':
            columns = ["EDT [s]", "T20 [s]", "T30 [s]", "C50 [dB]", "C80 [dB]", "Tt [s]", "EDTTt [s]"]
            if self.paramR is not None:
                columns.append("IACCe")
            xlsx = pd.read_csv(filename)
            xlsx.index = columns
            xlsx = xlsx.to_excel(filename, columns=self.nominalBands, index_label = "f [Hz]")
        



# main
app = QApplication(sys.argv)

setupwindow = SetupWindow()
setupwindow.show()
# 

try:
    sys.exit(app.exec_())
except:
    print("Exiting")