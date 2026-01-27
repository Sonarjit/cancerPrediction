TABLE_COLUMNS = {
    "history_table":
        [("Name", "TEXT"),
         ("Saved on", "TEXT"),
         ("Single/Batch", "TEXT"),
         ],

    "single_inference_table":
        [("Name", "TEXT"),
         ("Gene", "TEXT"),
         ("Variation", "TEXT"),
         ("Text", "TEXT"),
         ("Predicted Class", "INTEGER"),
         ("Class 1 probability", "REAL"),
         ("Class 2 probability", "REAL"),
         ("Class 3 probability", "REAL"),
         ("Class 4 probability", "REAL"),
         ("Class 5 probability", "REAL"),
         ("Class 6 probability", "REAL"),
         ("Class 7 probability", "REAL"),
         ("Class 8 probability", "REAL"),
         ("Class 9 probability", "REAL"),
         ],

        "batch_inference_table":
            [("ID", "TEXT"),
            ("Gene", "TEXT"),
            ("Variation", "TEXT"),
            ("Text", "TEXT"),
            ("Predicted Class", "INTEGER"),
            ("Class 1 probability", "REAL"),
            ("Class 2 probability", "REAL"),
            ("Class 3 probability", "REAL"),
            ("Class 4 probability", "REAL"),
            ("Class 5 probability", "REAL"),
            ("Class 6 probability", "REAL"),
            ("Class 7 probability", "REAL"),
            ("Class 8 probability", "REAL"),
            ("Class 9 probability", "REAL"),
            ],

}