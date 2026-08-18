"""Tkinter front-end for the trained classifier.

Written as a class so the model, encoders and widgets travel together instead
of relying on module-level globals the way the original script did.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox

import pandas as pd

from .modeling import Dataset, TrainedModel

# (field label, feature column) pairs for the manual-entry form.
MANUAL_FIELDS = [
    ("Age", "age"),
    ("Seniority (years)", "anciennete_client"),
    ("Montant local", "montant_local"),
    ("Days since last transaction", "days_since_last_transaction"),
    ("Average transaction amount", "avg_transaction_amount"),
]


class PredictionApp:
    """Two ways to exercise the model: a random test client, or typed input."""

    def __init__(self, trained: TrainedModel, dataset: Dataset) -> None:
        self.trained = trained
        self.dataset = dataset
        self.root = tk.Tk()
        self.root.title("Predictive Analytics Tool")
        self._build_widgets()

    def _build_widgets(self) -> None:
        frame = tk.Frame(self.root, padx=20, pady=20)
        frame.pack()

        tk.Label(frame, text="CIH Transaction Type Prediction", font=("Segoe UI", 12, "bold")).pack(pady=(0, 4))
        tk.Label(frame, text=f"Model accuracy: {self.trained.accuracy:.1%}", fg="#555").pack(pady=(0, 12))

        tk.Button(frame, text="Predict a random test client", width=30, command=self.show_random_prediction).pack(pady=4)
        # The original defined this window but never attached it to a button,
        # so manual entry was unreachable.
        tk.Button(frame, text="Enter client data manually", width=30, command=self.open_manual_input).pack(pady=4)

    def _decode(self, encoded) -> str:
        return self.dataset.label_encoder.inverse_transform([encoded])[0]

    def show_random_prediction(self) -> None:
        """Predict for a genuinely random held-out client.

        The original passed random_state=42 here, so the 'random' button
        returned the same client on every click.
        """
        sample = self.trained.X_test.sample(1)
        index = sample.index[0]

        predicted = self._decode(self.trained.model.predict(sample)[0])
        actual = self._decode(self.trained.y_test.loc[index])

        details = "\n".join(
            f"- {label}: {sample[column].values[0]:,.2f}"
            for label, column in MANUAL_FIELDS
            if column in sample.columns
        )
        messagebox.showinfo(
            "Client Prediction",
            f"Client information:\n{details}\n\nPredicted: {predicted}\nActual: {actual}",
        )

    def open_manual_input(self) -> None:
        """Collect the numeric features and predict from them."""
        window = tk.Toplevel(self.root)
        window.title("Input Client Data")
        entries: dict[str, tk.Entry] = {}

        for label, column in MANUAL_FIELDS:
            tk.Label(window, text=label).pack(pady=(6, 0))
            entry = tk.Entry(window)
            entry.pack(padx=12)
            entries[column] = entry

        def submit() -> None:
            try:
                values = {column: float(entry.get()) for column, entry in entries.items()}
            except ValueError:
                messagebox.showerror("Invalid input", "Please enter numeric values in every field.")
                return

            # Reindexing against the training columns fills every unseen
            # one-hot category with 0, which is what an unknown category means.
            row = pd.DataFrame([values]).reindex(columns=self.trained.X_train.columns, fill_value=0)
            prediction = self._decode(self.trained.model.predict(row)[0])

            messagebox.showinfo("Prediction Result", f"Predicted transaction type: {prediction}")
            window.destroy()

        tk.Button(window, text="Predict", command=submit).pack(pady=12)

    def run(self) -> None:
        self.root.mainloop()


def launch(trained: TrainedModel, dataset: Dataset) -> None:
    PredictionApp(trained, dataset).run()
