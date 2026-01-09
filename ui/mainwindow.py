import PyQt6
from PyQt6 import QtCore, QtWidgets
from ui.toggle_ui import ToggleDisplay

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
         # set window to full screen
        screen = PyQt6.QtGui.QGuiApplication.primaryScreen().geometry()

        # self.canvas.setFixedSize(screen.width(), screen.height())
        MainWindow.setGeometry(0, 0, screen.width(), screen.height())

        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setSpacing(3)

        #==================================================================
        #===================== NAVIGATION BAR  ============================
        # Navigation Bar
        self.nav_widget = QtWidgets.QWidget(parent=self.centralwidget)
        self.nav_widget.setMaximumSize(QtCore.QSize(16777215, 60))
        self.nav_widget_layout = QtWidgets.QHBoxLayout(self.nav_widget)
        self.nav_widget_layout.setSpacing(2)

        # single_inference button
        self.single_inference = QtWidgets.QPushButton(parent=self.nav_widget)
        self.nav_widget_layout.addWidget(self.single_inference)
        # batch_inference button
        self.batch_inference = QtWidgets.QPushButton(parent=self.nav_widget)
        self.nav_widget_layout.addWidget(self.batch_inference)
        # inference history button
        self.inference_history = QtWidgets.QPushButton(parent=self.nav_widget)
        self.nav_widget_layout.addWidget(self.inference_history)
        # model_metadata button
        self.model_metadata = QtWidgets.QPushButton(parent=self.nav_widget)
        self.nav_widget_layout.addWidget(self.model_metadata)
        spacerItem = QtWidgets.QSpacerItem(352, 19, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.nav_widget_layout.addItem(spacerItem)
        self.verticalLayout.addWidget(self.nav_widget)
        #==================================================================

        #====================================================================
        self.main_display = QtWidgets.QWidget(parent=self.centralwidget)
        self.main_display_layout = QtWidgets.QVBoxLayout(self.main_display)

        self.verticalLayout.addWidget(self.main_display)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 761, 21))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.single_inference.setCheckable(True)
        self.batch_inference.setCheckable(True)
        self.inference_history.setCheckable(True)
        self.model_metadata.setCheckable(True)

        self.single_inference.setAutoExclusive(True)
        self.batch_inference.setAutoExclusive(True)
        self.inference_history.setAutoExclusive(True)
        self.model_metadata.setAutoExclusive(True)

        self.single_inference.setChecked(True)  # default active

        # give names so stylesheet rules apply
        self.nav_widget.setObjectName("nav_widget")

        for btn in (self.single_inference, self.batch_inference, self.model_metadata):
            btn.setCheckable(True)
            btn.setAutoExclusive(True)
        self.single_inference.setChecked(True)  # initial active

        dict_to_pass ={
            "main_display": self.main_display,
            "main_display_layout": self.main_display_layout,
            "single_inference_btn": self.single_inference,
            "batch_inference_btn": self.batch_inference,
            "history_btn": self.inference_history,
            "model_metadata_btn": self.model_metadata
        }

        self.toggle_display = ToggleDisplay(**dict_to_pass)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Cancer Variant Predictor"))
        self.single_inference.setText(_translate("MainWindow", "Single inference"))
        self.batch_inference.setText(_translate("MainWindow", "Batch inference"))
        self.inference_history.setText(_translate("MainWindow", "Saved History"))
        self.model_metadata.setText(_translate("MainWindow", "Model Metadata"))


