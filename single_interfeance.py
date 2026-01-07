import PyQt6
from PyQt6 import QtCore, QtGui, QtWidgets


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

        # single_interference button
        self.single_interference = QtWidgets.QPushButton(parent=self.nav_widget)
        self.nav_widget_layout.addWidget(self.single_interference)
        # batch_interference button
        self.batch_interference = QtWidgets.QPushButton(parent=self.nav_widget)
        self.nav_widget_layout.addWidget(self.batch_interference)
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

        #====================================================================
        #===================== SINGLE INTERFERENCE CARD ========================
        self.container_widget = QtWidgets.QWidget(parent=self.main_display)
        self.container_widget_layout = QtWidgets.QHBoxLayout(self.container_widget)

        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.container_widget_layout.addItem(spacerItem1)

        # card widget
        self.single_card = QtWidgets.QWidget(parent=self.container_widget)
        self.single_card_layout = QtWidgets.QVBoxLayout(self.single_card)

        self.label = QtWidgets.QLabel(parent=self.single_card)
        font = QtGui.QFont()
        font.setBold(True)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.single_card_layout.addWidget(self.label)

        # Gene input
        self.label_2 = QtWidgets.QLabel(parent=self.single_card)
        self.single_card_layout.addWidget(self.label_2)
        self.lineEdit = QtWidgets.QLineEdit(parent=self.single_card)
        self.single_card_layout.addWidget(self.lineEdit)

        # Variation input
        self.label_3 = QtWidgets.QLabel(parent=self.single_card)
        self.single_card_layout.addWidget(self.label_3)
        self.lineEdit_2 = QtWidgets.QLineEdit(parent=self.single_card)
        self.single_card_layout.addWidget(self.lineEdit_2)

        # Text input
        self.label_4 = QtWidgets.QLabel(parent=self.single_card)
        self.single_card_layout.addWidget(self.label_4)
        self.plainTextEdit = QtWidgets.QPlainTextEdit(parent=self.single_card)
        self.single_card_layout.addWidget(self.plainTextEdit)

        self.widget_5 = QtWidgets.QWidget(parent=self.single_card)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.widget_5)
        self.pushButton_4 = QtWidgets.QPushButton(parent=self.widget_5)
        self.horizontalLayout_3.addWidget(self.pushButton_4)
        self.single_card_layout.addWidget(self.widget_5)
        self.container_widget_layout.addWidget(self.single_card)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.container_widget_layout.addItem(spacerItem2)
        self.main_display_layout.addWidget(self.container_widget)

        self.verticalLayout.addWidget(self.main_display)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 761, 21))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.single_interference.setCheckable(True)
        self.batch_interference.setCheckable(True)
        self.model_metadata.setCheckable(True)

        self.single_interference.setAutoExclusive(True)
        self.batch_interference.setAutoExclusive(True)
        self.model_metadata.setAutoExclusive(True)

        self.single_interference.setChecked(True)  # default active

        # give names so stylesheet rules apply
        self.nav_widget.setObjectName("nav_widget")
        self.container_widget.setObjectName("container_widget")
        self.single_card.setObjectName("single_card")

        self.label.setObjectName("titleLabel")
        self.pushButton_4.setObjectName("predictButton")

        for btn in (self.single_interference, self.batch_interference, self.model_metadata):
            btn.setCheckable(True)
            btn.setAutoExclusive(True)
        self.single_interference.setChecked(True)  # initial active

        self.single_card.setMaximumWidth(560)
        self.container_widget_layout.setContentsMargins(40, 24, 40, 24)
        self.container_widget_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)


        shadow = QtWidgets.QGraphicsDropShadowEffect(self.single_card)
        shadow.setBlurRadius(28)
        shadow.setXOffset(0)
        shadow.setYOffset(8)
        shadow.setColor(QtGui.QColor(0, 0, 0, 50))
        self.single_card.setGraphicsEffect(shadow)




    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Cancer Variant Predictor"))
        self.single_interference.setText(_translate("MainWindow", "Single Interference"))
        self.batch_interference.setText(_translate("MainWindow", "Batch Interference"))
        self.model_metadata.setText(_translate("MainWindow", "Model Metadata"))
        self.label.setText(_translate("MainWindow", "Instance Prediction"))
        self.label_2.setText(_translate("MainWindow", "Gene:"))
        self.label_3.setText(_translate("MainWindow", "Variation:"))
        self.label_4.setText(_translate("MainWindow", "Text"))
        self.pushButton_4.setText(_translate("MainWindow", "Predict"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    def apply_styles(app):
        app.setStyleSheet("""
    /* GLOBAL */
    QWidget {
        font-family: "Segoe UI", "Inter", "Arial";
        font-size: 13px;
        color: #1f2937;
    }
    QMainWindow {
        background-color: #eef2f6; /* app background (light) */
    }

    /* NAV BAR container - dark */
    QWidget#nav_widget {
        background-color: #0b1220; /* dark navy */
        border-bottom: 1px solid #111827;
    }

    /* NAV BUTTONS - readable by default */
    QPushButton {
        background-color: transparent;
        border: none;
        color: #cbd5e1;         /* readable light-gray */
        padding: 10px 16px;
        font-size: 14px;
        font-weight: 700;
        min-height: 36px;
    }

    QPushButton:hover {
        background-color: rgba(255,255,255,0.03);
        color: #ffffff;
    }

    /* Active navigation: subtle left accent bar, not a wide highlight */
    QPushButton:checked {
        color: #ffffff;
        background-color: rgba(255,255,255,0.03);
        border-left: 4px solid #38bdf8; /* cyan-blue accent */
        padding-left: 12px; /* accommodate the left accent bar */
    }

    /* Make sure disabled-looking appearance is avoided */
    QPushButton:disabled {
        color: #94a3b8;
    }

    /* Container around the card - slightly different from app background */
    QWidget#container_widget {
        background-color: #f4f7fb; /* subtle panel color */
        padding-top: 28px;
        padding-bottom: 28px;
    }

    /* The actual card */
    QWidget#single_card {
        background-color: #ffffff;
        border-radius: 12px;
        border: 1px solid #d1d5db;
        padding: 20px;
        /* shadow is applied via QGraphicsDropShadowEffect in code */
    }

    /* Card title and labels */
    QLabel#titleLabel {
        font-size: 18px;
        font-weight: 800;
        color: #0f172a;
    }

    QLabel {
        font-weight: 600;
        color: #0f172a;
    }

    /* Inputs */
    QLineEdit, QPlainTextEdit {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        padding: 8px;
    }
    QLineEdit:focus, QPlainTextEdit:focus {
        border: 1px solid #2563eb;
    }

    /* Primary button */
    QPushButton#predictButton {
        background-color: #2563eb;
        color: #ffffff;
        border-radius: 6px;
        padding: 10px 18px;
        font-size: 14px;
        font-weight: 700;
    }
    QPushButton#predictButton:hover {
        background-color: #1d4ed8;
    }
    """)


    apply_styles(app)
    MainWindow.show()
    sys.exit(app.exec())
