import tkinter as tk
from tkinter import filedialog, messagebox
import cornac
from cornac.eval_methods import RatioSplit
from cornac.models import BPR, MF, PMF
from cornac.metrics import MAE, RMSE, Precision, Recall, NDCG, AUC, MAP
from cornac.data import Dataset
import pandas as pd
import numpy as np

class RecommenderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Cornac BPR Batch Recommender (Vertical + Rank)")
        self.root.geometry("500x400")
        
        # State variables
        self.file_path = None
        self.is_trained = False
        self.dataset = None
        self.model = None

        self.setup_ui()

    def setup_ui(self):
        # --- File Loading Section ---
        self.lbl_file = tk.Label(self.root, text="No Excel file selected", fg="gray")
        self.lbl_file.pack(pady=(30, 5))

        self.btn_load = tk.Button(self.root, text="1. Load Input Excel Sheet", command=self.load_file, width=30)
        self.btn_load.pack(pady=5)

        # --- Training Section ---
        self.btn_train = tk.Button(self.root, text="2. Train BPR Model", command=self.train_model, state=tk.DISABLED, width=30)
        self.btn_train.pack(pady=10)

        # --- Batch Export Section ---
        self.btn_export = tk.Button(self.root, text="3. Export All Recommendations", command=self.export_all_recommendations, state=tk.DISABLED, width=30)
        self.btn_export.pack(pady=10)

        # --- Status/Results Section ---
        self.lbl_status_head = tk.Label(self.root, text="Status Console:", font=("Arial", 10, "bold"))
        self.lbl_status_head.pack(pady=(20, 0))
        
        self.listbox_results = tk.Listbox(self.root, width=55, height=8)
        self.listbox_results.pack(pady=5)

    def load_file(self):
        """Opens a file dialog to select the input Excel sheet."""
        filepath = filedialog.askopenfilename(
            title="Select Input Excel File",
            filetypes=(("Excel files", "*.xlsx;*.xls"), ("All files", "*.*"))
        )
        
        if filepath:
            self.file_path = filepath
            self.lbl_file.config(text=f"Loaded: {self.file_path.split('/')[-1]}", fg="green")
            self.btn_train.config(state=tk.NORMAL) 
            self.btn_export.config(state=tk.DISABLED) 
            self.is_trained = False
            
            self.listbox_results.delete(0, tk.END)
            self.listbox_results.insert(tk.END, "✅ File loaded successfully.")
            self.listbox_results.insert(tk.END, "Ready to train BPR model.")

    def train_model(self):
        """Reads the Excel file and trains the Cornac BPR model."""
        if not self.file_path:
            messagebox.showerror("Error", "Please load an Excel file first.")
            return

        self.listbox_results.delete(0, tk.END)
        self.listbox_results.insert(tk.END, "Reading data and training BPR model...")
        self.listbox_results.insert(tk.END, "Please wait (UI may temporarily freeze)...")
        self.root.update() 
        
        try:
            df = pd.read_excel(self.file_path)
            
            required_cols = ['user_id', 'item_id', 'rating']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Excel sheet must contain exact columns: {', '.join(required_cols)}")
            
            data_tuples = list(df[required_cols].itertuples(index=False, name=None))
            
            rs = RatioSplit(data=data_tuples, 
                            test_size=0.2, 
                            # rating_threshold=3.5, 
                            seed=123)

            self.dataset = cornac.data.Dataset.from_uir(data_tuples)
            # self.dataset.build()

            # 4. Instantiate the models you want to evaluate
            models = [
                # MF(k=10, max_iter=25, learning_rate=0.01, lambda_reg=0.02, use_bias=True, seed=123),
                # PMF(k=10, max_iter=100, learning_rate=0.001, lambda_reg=0.001, seed=123),
                BPR(k=20, max_iter=200, learning_rate=0.001, lambda_reg=0.01, seed=123),
            ]

            # 5. Define evaluation metrics
            metrics = [MAE(), RMSE(), Precision(k=10), Recall(k=10), NDCG(k=10), AUC(), MAP()]

            # 6. Put everything together into an experiment and run it
            cornac.Experiment(
                eval_method=rs,
                models=models,
                metrics=metrics,
                user_based=True
            ).run()
            
            self.is_trained = True
            self.btn_export.config(state=tk.NORMAL) 
            
            self.listbox_results.delete(0, tk.END)
            self.listbox_results.insert(tk.END, "✅ BPR Model trained successfully!")
            # self.listbox_results.insert(tk.END, f"Total Users: {self.dataset.num_users}")
            # self.listbox_results.insert(tk.END, f"Total Items: {self.dataset.num_items}")
            
        except Exception as e:
            messagebox.showerror("Training Error", f"An error occurred: {str(e)}")

    def export_all_recommendations(self):
        """Generates Top-10 predictions with ranks for ALL users and exports to a vertical Excel sheet."""
        if not self.is_trained:
            messagebox.showwarning("Warning", "Model is not trained yet.")
            return
            
        save_path = filedialog.asksaveasfilename(
            title="Save Recommendations As",
            defaultextension=".xlsx",
            filetypes=(("Excel files", "*.xlsx"), ("All files", "*.*"))
        )
        
        if not save_path:
            return 

        self.listbox_results.delete(0, tk.END)
        self.listbox_results.insert(tk.END, "Generating ranked batch recommendations...")
        self.root.update()

        try:
            # 1. Load and prepare data
            df = pd.read_excel(self.file_path)
            
            df['rating'] = np.log1p(df['rating'])
            data = list(df[['user_id', 'item_id', 'rating']].to_records(index=False))

            # 2. Build dataset and train model
            dataset = Dataset.from_uir(data)
            model = BPR(k=20, max_iter=200, learning_rate=0.01, lambda_reg=0.01, seed=123)
            model.fit(dataset)

            # ---------------------------------------------------------

            # dataset.save(".\\saved_datasets\\dataset.pkl")
            # model.save(save_dir="saved_models")

            # ---------------------------------------------------------
            # NEW: Batch Generation of Recommendations for All Users
            # ---------------------------------------------------------
            print("Generating recommendations for all users...")

            recommendations_list = []

            # Loop through every raw user ID present in the dataset
            for raw_user_id in dataset.user_ids:
                # Get recommendations (returns item IDs)
                # Note: model.recommend handles the string-to-index conversion automatically if you pass train_set
                recs = model.recommend(
                    user_id=raw_user_id, 
                    k=10, 
                    # remove_seen=True, 
                    train_set=dataset
                )
                
                # Store each recommended item alongside its rank position
                for rank, item_id in enumerate(recs, start=1):
                    recommendations_list.append({
                        'user_id': raw_user_id,
                        'item_id': item_id,
                        'rank': rank
                    })

            # 3. Convert the accumulated list into a pandas DataFrame
            recommendations_df = pd.DataFrame(recommendations_list)

            # 4. Save to your preferred format
            # Save to CSV
            # recommendations_df.to_csv('all_user_recommendations.csv', index=False)

            # Alternative: Save to Excel (requires 'openpyxl' library installed via pip)
            recommendations_df.to_excel(save_path, index=False)

            print("Recommendations saved successfully! First few rows:")
            print(recommendations_df.head(15))
            
            messagebox.showinfo("Success", f"Ranked top 10 recommendations for all users exported successfully!")

        except Exception as e:
            messagebox.showerror("Export Error", f"An error occurred during export: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = RecommenderGUI(root)
    root.mainloop()