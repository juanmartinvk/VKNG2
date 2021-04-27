# -*- coding: utf-8 -*-
import sys
import soundfile as sf
import ntpath
import acoustical_parameters as ap
from PyQt5.uic import loadUi
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QAbstractItemView, QFileDialog, QListWidgetItem

def analyzeFile(impulsePath, filterPath, b, smoothing="schroeder", dataType="IR"):

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
    
    
    # Calculate parameters
    acParamL = ap.parameters(IR_raw_L, fs, b, truncate='lundeby', smoothing=smoothing)
    if IR_raw_R is not None:
        acParamR = ap.parameters(IR_raw_R, fs, b, truncate='lundeby', smoothing=smoothing)
        acParamR.IACCe=ap.IACCe_from_IR(acParamL, acParamR)
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
        loadUi("gui_setup.ui",self)
        
        # Defaults
        self.octaveButton.setChecked(True)
        self.schroederButton.setChecked(True)
        self.impulseButton.setChecked(True)
        self.filterGroupBox.setEnabled(False)
        #self.filterGroupBox.hide()
        self.b = 1
        self.smoothing = "schroeder"
        self.dataType= "IR"
        
        # Window settings
        self.setWindowTitle("Setup")
        
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
    
    def browseImpulse(self):
        # Open file dialog and load path onto line edit widget
        filePath = QFileDialog.getOpenFileName()[0]
        self.impulsePathBox.setText(filePath)
    
    def browseFilter(self):
        filePath = QFileDialog.getOpenFileName()[0]
        self.filterPathBox.setText(filePath)
            
    def analyzeData(self):
        # Analyze file data, hide setup window and show data window
        impulsePath = self.impulsePathBox.text()
        filterPath = self.filterPathBox.text()
        
        # For Impulse Response data
        if self.dataType == "IR" and ntpath.exists(impulsePath):
            self.paramL, self.paramR, self.nominalBands = analyzeFile(impulsePath, None, self.b, self.smoothing, self.dataType)
            self.dataWindow = DataWindow(self.paramL, self.paramR, self.nominalBands, self.b, self.smoothing)
            self.hide()
            self.dataWindow.show()
        
        # For Sweep data
        elif self.dataType == "sweep" and ntpath.exists(impulsePath) and ntpath.exists(filterPath):
            self.paramL, self.paramR, self.nominalBands = analyzeFile(impulsePath, filterPath, self.b, self.smoothing, self.dataType)
            self.dataWindow = DataWindow(self.paramL, self.paramR, self.nominalBands, self.b, self.smoothing)
            self.hide()
            self.dataWindow.show()
        
        else:
            print("Invalid file path")
        
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
        
    def toggleMedian(self):
        self.smoothing = "median"
    
    def toggleOctave(self):
        self.b = 1
    
    def toggleThirdOctave(self):
        self.b = 3

        

class DataWindow(QMainWindow):
    def __init__(self, paramL, paramR, nominalBands, b, smoothing="schroeder"):
        super(DataWindow, self).__init__()
        self.paramL = paramL
        self.paramR = paramR
        self.b = b
        self.nominalBands = nominalBands

        
        # Defaults
        self.currentParam = self.paramL
        self.currentBandIdx = self.nominalBands.index("1k")
        self.smoothing = smoothing
        
        #Load designed UI 
        loadUi("gui_main.ui",self)
        
        #Load data to table
        self.loadData()
        
        # Window preferences
        self.setWindowTitle("Acoustical Parameters")
        
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
        
               

    def loadData(self):
        # Loads data from current param object onto table widget
        self.paramTable.setColumnCount(len(self.nominalBands))
        self.paramTable.setHorizontalHeaderLabels(self.nominalBands)
        
        # Set column width according to band filtering
        if self.b == 1:
            column_width = 60
        else:
            column_width = 40
        
        for idx, band in enumerate(self.nominalBands):
             self.paramTable.setItem(0, idx, QtWidgets.QTableWidgetItem(str(self.currentParam.EDT[idx])))
             self.paramTable.setItem(1, idx, QtWidgets.QTableWidgetItem(str(self.currentParam.T20[idx])))
             self.paramTable.setItem(2, idx, QtWidgets.QTableWidgetItem(str(self.currentParam.T30[idx])))
             self.paramTable.setItem(3, idx, QtWidgets.QTableWidgetItem(str(self.currentParam.C50[idx])))
             self.paramTable.setItem(4, idx, QtWidgets.QTableWidgetItem(str(self.currentParam.C80[idx])))
             self.paramTable.setColumnWidth(idx,column_width)
             
    def plotData(self):
        # Initializes plot and sets lines and labels
        if self.smoothing == "schroeder":
            self.decay_line = self.graphWidget.plot(self.currentParam.t, self.currentParam.decay[self.currentBandIdx], pen='r', name="Schroeder decay")
        elif self.smoothing == "median":
            self.decay_line = self.graphWidget.plot(self.currentParam.t, self.currentParam.decay[self.currentBandIdx], pen='r', name="Median filtered decay")
            
        self.ETC_line = self.graphWidget.plot(self.currentParam.t, self.currentParam.ETC_avg_dB[self.currentBandIdx], pen='b', name="Energy Time Curve")
        

        
        
    def updateData(self, row, column):
        # Updates plot with data from current band and current param object
        self.decay_line.setData(self.currentParam.t, self.currentParam.decay[column])
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

             



# main
app = QApplication(sys.argv)

setupwindow = SetupWindow()
setupwindow.show()
# 

try:
    sys.exit(app.exec_())
except:
    print("Exiting")