from ui.single_inference_ui import SingleInferenceWidget
from ui.batch_inference_ui import BatchInferenceWidget

from ui.result_batch_ui import ResultBatchWidget
from ui.save_history_ui import HistoryFormWidget

class ToggleDisplay:
    
    def __init__(self, **kwargs):

        self.main_display = kwargs.get("main_display")
        self.main_display_layout = kwargs.get("main_display_layout")

        # buttons
        self.single_inference = kwargs.get("single_inference_btn")

        self.batch_inference = kwargs.get("batch_inference_btn")

        self.history_button = kwargs.get("history_btn")

        # self.model_metadata = kwargs.get("model_metadata_btn")

        # connect buttons to view switcher
        self.single_inference.clicked.connect(lambda: self._switch_ut_view(0))
        self.batch_inference.clicked.connect(lambda: self._switch_ut_view(1))
        self.history_button.clicked.connect(lambda: self._switch_ut_view(2))
        # self.model_metadata.clicked.connect(lambda: self._switch_ut_view(3))
        self._switch_ut_view(0)

    def _clear_main_display(self):
        """Remove and delete all widgets from main_display_layout."""
        layout = self.main_display_layout
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        print("cleared main display")

    def _switch_ut_view(self, index: int):
        """
        index 0 → 
        index 1 → 
        """
        
        # clear existing widget(s)
        self._clear_main_display()

        # create and add the appropriate widget
        if index == 0:
            self.single_inference_widget = SingleInferenceWidget(parent=self.main_display)
            self.main_display_layout.addWidget(self.single_inference_widget)
            
        elif index == 1:
            self.batch_inference_widget = BatchInferenceWidget(parent=self.main_display)
            self.main_display_layout.addWidget(self.batch_inference_widget)

        elif index == 2:
            self.history_widget = HistoryFormWidget(parent=self.main_display)
            self.main_display_layout.addWidget(self.history_widget)

        elif index == 3:
            pass