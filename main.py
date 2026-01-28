import sys
from PyQt6 import  QtWidgets
from ml.pipeline_component import VariationResponseEncoder
from ui.mainwindow import Ui_MainWindow
from db_manager.create_app_table import create_history_table, create_single_inference_table

def create_app_databases():
    create_history_table()
    create_single_inference_table()
def main():
    create_app_databases()
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
    """)
    
    apply_styles(app)
    MainWindow.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()