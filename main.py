import os
import sys
import subprocess
import time
import threading
from PyQt6 import  QtWidgets
import signal

from server_flask import create_app

from ml.prediction_single import load_pipeline

from ui.mainwindow import Ui_MainWindow
from db_manager.create_app_table import create_history_table, create_single_inference_table

def create_app_databases():
    create_history_table()
    create_single_inference_table()

def start_flask_process():
    # run a python module or script that creates and runs flask app on port 5000
    return subprocess.Popen([sys.executable, "-u", "server_flask.py"])

def start_uvicorn_process():
    # run uvicorn pointing at your FastAPI app module
    # assume server_fast:app is available via import path
    subprocess.Popen([sys.executable, "-u", "server_fast.py"])

def start_fastapi_process():
    """
    Start server_fast.py using same Python interpreter.
    Returns subprocess.Popen instance.
    """
    cmd = [sys.executable, "-u", "server_fast.py"]

    if os.name == "posix":
        # Start in a new process group so we can kill the whole group later
        return subprocess.Popen(cmd, preexec_fn=os.setsid)
    else:
        # Windows: create new process group
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        return subprocess.Popen(cmd, creationflags=CREATE_NEW_PROCESS_GROUP)


def terminate_process(proc: subprocess.Popen, timeout: float = 5.0):
    """
    Try to terminate proc gracefully, wait `timeout` seconds, then forcibly kill.
    Works on POSIX and Windows. Safe if proc is None or already exited.
    """
    if proc is None:
        return

    # If already exited, nothing to do
    if proc.poll() is not None:
        return

    try:
        # graceful request
        proc.terminate()
    except Exception:
        pass

    # wait a bit
    try:
        proc.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass

    # still alive → escalate
    try:
        if os.name == "posix":
            # kill entire process group
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
            except Exception:
                # fallback to kill the single process
                proc.kill()
        else:
            # Windows: send CTRL_BREAK to group or kill
            try:
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            except Exception:
                proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass

    # final wait (best-effort)
    try:
        proc.wait(timeout=2)
    except Exception:
        pass


def main():
    create_app_databases()
    p1 = start_flask_process()
    p2 = start_fastapi_process()
    time.sleep(1)  # short pause to let servers start (avoid in-production waits)


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
    try:
        sys.exit(app.exec())
    finally:
        # ensure child processes are terminated on exit
        p1.terminate()
        terminate_process(p2, timeout=5.0)
        # p2.terminate()

if __name__ == "__main__":
    main()