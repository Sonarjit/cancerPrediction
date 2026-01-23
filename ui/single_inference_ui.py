from PyQt6 import QtCore, QtGui, QtWidgets
from ml.prediction import prediction

class Ui_Form(object):
    def setupUi(self, Form):
        #====================================================================
        #===================== SINGLE inference CARD ========================
        self.container_widget_layout = QtWidgets.QHBoxLayout(self)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.container_widget_layout.addItem(spacerItem1)

        # card widget
        self.single_card = QtWidgets.QWidget(parent=self)
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

        self.single_card.setMaximumWidth(560)
        self.container_widget_layout.setContentsMargins(40, 24, 40, 24)
        self.container_widget_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)

        # give names so stylesheet rules apply
        self.single_card.setObjectName("single_card")
        self.label.setObjectName("titleLabel")
        self.pushButton_4.setObjectName("predictButton")

        shadow = QtWidgets.QGraphicsDropShadowEffect(self.single_card)
        shadow.setBlurRadius(28)
        shadow.setXOffset(0)
        shadow.setYOffset(8)
        shadow.setColor(QtGui.QColor(0, 0, 0, 50))
        self.single_card.setGraphicsEffect(shadow)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        self.label.setText(_translate("Form", "Instance Prediction"))
        self.label_2.setText(_translate("Form", "Gene:"))
        self.label_3.setText(_translate("Form", "Variation:"))
        self.label_4.setText(_translate("Form", "Text"))
        self.pushButton_4.setText(_translate("Form", "Predict"))

    

class SingleInferenceWidget(QtWidgets.QWidget, Ui_Form):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setObjectName("single_inference_container")
        # setupUi expects a widget; provide self so this object becomes the UI widget
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setupUi(self)

        self.apply_styles()
        self.pushButton_4.clicked.connect(self.predict)

    def predict(self):
        
        gene = self.lineEdit.text().strip()
        variation = self.lineEdit_2.text().strip()
        text = self.plainTextEdit.toPlainText().strip()

        # warning if any field is empty
        if not gene or not variation or not text:
            QtWidgets.QMessageBox.warning(self, "Input Error", "Please fill in all fields: Gene, Variation, and Text.")
            return

        prediction_results = prediction(gene=[gene], variation=[variation], text=[text])
        # lists you asked for
        predicted_class = [r["pred_class"] for r in prediction_results]
        class1_probs = [r["prob_class_1"] for r in prediction_results]
        class2_probs = [r["prob_class_2"] for r in prediction_results]
        class3_probs = [r["prob_class_3"] for r in prediction_results]
        class4_probs = [r["prob_class_4"] for r in prediction_results]
        class5_probs = [r["prob_class_5"] for r in prediction_results]
        class6_probs = [r["prob_class_6"] for r in prediction_results]
        class7_probs = [r["prob_class_7"] for r in prediction_results]
        class8_probs = [r["prob_class_8"] for r in prediction_results]
        class9_probs = [r["prob_class_9"] for r in prediction_results]

        cls_predicted = predicted_class[0]
        prob_class_1 = class1_probs[0]
        prob_class_2 = class2_probs[0]
        prob_class_3 = class3_probs[0]
        prob_class_4 = class4_probs[0]
        prob_class_5 = class5_probs[0]
        prob_class_6 = class6_probs[0]
        prob_class_7 = class7_probs[0]
        prob_class_8 = class8_probs[0]
        prob_class_9 = class9_probs[0]

        self.main_display = self.parent
        self.main_display_layout = self.main_display.layout()
        layout = self.main_display_layout
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        
        from ui.result_single_ui import ResultSingleWidget
        self.result_window = ResultSingleWidget(parent=self.main_display)
        self.result_window.gene.setText(f"{gene}")
        self.result_window.variation.setText(f"{variation}")
        self.result_window.text.setPlainText(f"{text}")
        self.result_window.predicted_label.setText(f"{cls_predicted}")
        self.result_window.class1_prob.setText(f"{prob_class_1:.4f}")
        self.result_window.class2_prob.setText(f"{prob_class_2:.4f}")
        self.result_window.class3_prob.setText(f"{prob_class_3:.4f}")
        self.result_window.class4_prob.setText(f"{prob_class_4:.4f}")
        self.result_window.class5_prob.setText(f"{prob_class_5:.4f}")
        self.result_window.class6_prob.setText(f"{prob_class_6:.4f}")
        self.result_window.class7_prob.setText(f"{prob_class_7:.4f}")    
        self.result_window.class8_prob.setText(f"{prob_class_8:.4f}")
        self.result_window.class9_prob.setText(f"{prob_class_9:.4f}")
        self.main_display_layout.addWidget(self.result_window)

    def apply_styles(self):
        self.setStyleSheet("""
        /* GLOBAL */
        QWidget {
            font-family: "Segoe UI", "Inter", "Arial";
            font-size: 13px;
            color: #1f2937;
        }
                                         
        /* Container around the card - slightly different from app background */
        QWidget#single_inference_container {
            background-color: #f3f4f6; /* subtle panel color */
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

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    # instantiate the wrapper widget (can be added to any layout)
    widget = SingleInferenceWidget()
    widget.show()
    sys.exit(app.exec())