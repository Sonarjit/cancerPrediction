from PyQt6 import QtCore, QtGui, QtWidgets
import requests
from PyQt6.QtCore import QThread, pyqtSignal
# from ml.prediction_single import prediction

class PredictThread(QThread):
    finished_signal = pyqtSignal(object)  # emits parsed JSON result OR {'error': '...'}
    def __init__(self, payload: dict, parent=None):
        super().__init__(parent)
        self.payload = payload
        # FastAPI server runs on port 5001
        self._url = "http://127.0.0.1:5001/single_predict"

    def run(self):
        try:
            resp = requests.post(self._url, json=self.payload, timeout=30)
            resp.raise_for_status()
            # FastAPI returns JSON list-of-records; just pass it through
            self.finished_signal.emit(resp.json())
        except Exception as exc:
            self.finished_signal.emit({"error": str(exc)})

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
        
        self.gene = self.lineEdit.text().strip()
        self.variation = self.lineEdit_2.text().strip()
        self.text = self.plainTextEdit.toPlainText().strip()

        # warning if any field is empty
        if not self.gene or not self.variation or not self.text:
            QtWidgets.QMessageBox.warning(self, "Input Error", "Please fill in all fields: Gene, Variation, and Text.")
            return
        
        # Use dict-of-lists format so server can handle multiple rows easily
        payload = {
            "Gene": [self.gene],
            "Variation": [self.variation],
            "Text": [self.text]
        }

        # disable the button while waiting
        self.pushButton_4.setEnabled(False)

        self._predict_thread = PredictThread(payload)
        self._predict_thread.finished_signal.connect(self.on_prediction_result)
        self._predict_thread.start()
        
    def on_prediction_result(self, result):
        print("Received prediction result:", result)  # debug log
        # Re-enable button
        self.pushButton_4.setEnabled(True)

        if isinstance(result, dict) and result.get("error"):
            QtWidgets.QMessageBox.critical(self, "Prediction Error", f"Error: {result.get('error')}")
            return

        # result should be a list-of-records
        try:
            # Example: show predicted class of the first row in a label or message box
            first = result[0] if isinstance(result, list) and len(result) > 0 else None
            if first:
                gene = self.gene
                variation = self.variation
                text = self.text
                cls_predicted = first.get("pred_class")
                prob_class_1 = first.get("prob_class_1")
                prob_class_2 = first.get("prob_class_2")
                prob_class_3 = first.get("prob_class_3")
                prob_class_4 = first.get("prob_class_4")
                prob_class_5 = first.get("prob_class_5")
                prob_class_6 = first.get("prob_class_6")
                prob_class_7 = first.get("prob_class_7")
                prob_class_8 = first.get("prob_class_8")
                prob_class_9 = first.get("prob_class_9")
                

                dict_to_pass = {
                    "gene": gene,
                    "variation": variation,
                    "text": text,
                    "predicted_class": cls_predicted,
                    "class1_prob": prob_class_1,
                    "class2_prob": prob_class_2,
                    "class3_prob": prob_class_3,
                    "class4_prob": prob_class_4,
                    "class5_prob": prob_class_5,
                    "class6_prob": prob_class_6,
                    "class7_prob": prob_class_7,
                    "class8_prob": prob_class_8,
                    "class9_prob": prob_class_9,
                }

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
                self.result_window.populate_data(**dict_to_pass)
                self.main_display_layout.addWidget(self.result_window)
            else:
                QtWidgets.QMessageBox.information(self, "Prediction", "No prediction returned.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Prediction Parse Error", str(e))

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