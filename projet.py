import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
import time
import itertools
import heapq
from math import log, exp

# Default matrix and currency names 
default_matrix = np.array([
    [1, 1.19, 1.33, 1.62],
    [0.84, 1, 1.12, 1.37],
    [0.75, 0.89, 1, 1.22],
    [0.62, 0.73, 0.82, 1]
])

currency_names = ["Euro", "Dollar", "Pound", "Swiss Franc"]

# Algorithms
def best_forex_strategy_dynamic(p, matrix):
    p = p+1
    n = len(matrix)
    # Initialize the DP table
    dp = [[[0 for _ in range(n)] for _ in range(p+1)] for _ in range(n)]
    path = [[[None for _ in range(n)] for _ in range(p+1)] for _ in range(n)]

    # Base case: no exchanges
    for i in range(n):
        dp[i][0][0] = matrix[i][0]

    # Fill the DP table
    for k in range(1, p+1):
        for i in range(n):
            for j in range(n):
                max_profit = 0
                best_prev = None
                for prev in range(n):
                    profit = dp[prev][k-1][i] * matrix[i][j]
                    if profit > max_profit:
                        max_profit = profit
                        best_prev = prev
                dp[i][k][j] = max_profit
                path[i][k][j] = best_prev

    # Find the best profit and reconstruct the path
    best_profit = dp[0][p][0]
    best_path = [0]
    current = 0
    for k in range(p, 0, -1):
        current = path[current][k][best_path[-1]]
        best_path.append(current)
    best_path.reverse()
    best_path = best_path[1:]

    return best_profit, best_path


def brute_force_best_forex_strategy(p, matrix):
    # Time complexity: O(n^(p+1)), where n is the number of currencies and p is the number of exchanges
    n = len(matrix)
    all_paths = [path for path in itertools.product(range(n), repeat=p+1) if path[0] == 0 and path[-1] == 0]
    max_product = -1
    best_path = None
    for path in all_paths:
        product = 1
        for i in range(len(path) - 1):
            product *= matrix[path[i]][path[i+1]]
        if product > max_product:
            max_product = product
            best_path = path
    return float(max_product), list(best_path)


def dijkstra_forex_strategy(p, matrix):
   # Time complexity: O(n^2 * p), where n is the number of currencies and p is the number of exchanges
    dictionary = {
        0: "Euro",
        1: "Dollar",
        2: "Pound",
        3: "Swiss Franc"
    }
    chemin_euro = [dictionary[0]]
    chemin_dollar = [dictionary[0]]
    chemin_pound = [dictionary[0]]
    chemin_francs = [dictionary[0]]
    chemins = [chemin_euro, chemin_dollar, chemin_pound, chemin_francs]

    n = len(matrix)
    list_all_lambda = []
    list_lambda = []
    for j in range(n):
        list_lambda.append(matrix[0][j])
    list_all_lambda.append(list_lambda)

    for k in range(2, p+1):
        list_lambda_kmoins1 = list_all_lambda[-1].copy()
        list_lambda = []
        chemins_copy = []
        for j in range(n):
            tmp = -1
            for i in range(n):
                if tmp < matrix[i][j] * list_lambda_kmoins1[i]:
                    tmp = matrix[i][j] * list_lambda_kmoins1[i]
                    precedent = i  
            lambda_ = matrix[precedent][j] * list_lambda_kmoins1[precedent]
            list_lambda.append(lambda_)
            tmp = chemins[precedent].copy()
            tmp.append(dictionary[precedent])
            chemins_copy.append(tmp)
        list_all_lambda.append(list_lambda)
        chemins = chemins_copy.copy()
    
    tmp2 = chemins[0].copy()
    tmp2.append(dictionary[0])
    
    return float(list_all_lambda[-1][0]), list(tmp2)  # Return the best profit and the best path



def time_forex_strategy(p, matrix, function):
    start_time = time.time()
    result = function(p, matrix)
    end_time = time.time()
    return end_time - start_time


# function to predict execution time
def predict_execution_time(p, algorithm):
    n = 4  # Number of currencies
    base_time = 1e-6  # Base time for one operation (1 microsecond)
    
    if algorithm == "dijkstra":
        operations = n**2 * p
    elif algorithm == "brute_force":
        operations = p * n**(p+1)
    elif algorithm == "dynamic":
        operations = n**3 * p
    else:
        raise ValueError("Unknown algorithm")
    
    predicted_seconds = operations * base_time
    
    return predicted_seconds

def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.2f} hours"
    elif seconds < 2592000:  # 30 days
        days = seconds / 86400
        return f"{days:.2f} days"
    elif seconds < 31536000:  # 365 days
        months = seconds / 2592000
        return f"{months:.2f} months"
    else:
        years = seconds / 31536000
        return f"{years:.2f} years"

class ForexApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Forex Strategy Analyzer")
        self.geometry("800x500")
        self.configure(bg='#f0f0f0')
        self.matrix = default_matrix
        self.currency_names = currency_names
        self.create_widgets()

    def create_widgets(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TButton', background='#4CAF50', foreground='white', font=('Arial', 10, 'bold'))
        style.configure('TLabel', background='#f0f0f0', font=('Arial', 12))
        style.configure('Header.TLabel', font=('Arial', 16, 'bold'))

        self.main_frame = ttk.Frame(self, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(self.main_frame, text="Forex Strategy Analyzer", style='Header.TLabel').pack(pady=(0, 20))

        matrix_frame = ttk.Frame(self.main_frame)
        matrix_frame.pack(fill=tk.X, pady=10)
        ttk.Label(matrix_frame, text="Choose matrix:").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(matrix_frame, text="Use Default", command=self.use_default_matrix, width=15).pack(side=tk.LEFT, padx=5)
        ttk.Button(matrix_frame, text="Input Custom", command=self.input_custom_matrix, width=15).pack(side=tk.LEFT, padx=5)

        ttk.Separator(self.main_frame, orient='horizontal').pack(fill=tk.X, pady=20)

        operation_frame = ttk.Frame(self.main_frame)
        operation_frame.pack(fill=tk.X, pady=10)
        ttk.Label(operation_frame, text="Choose operation:").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(operation_frame, text="Calculate Strategy", command=self.calculate_strategy, width=20).pack(side=tk.LEFT, padx=5)
        ttk.Button(operation_frame, text="Compare Algorithms", command=self.compare_algorithms, width=20).pack(side=tk.LEFT, padx=5)
        ttk.Button(operation_frame, text="Predict Time", command=self.predict_time, width=20).pack(side=tk.LEFT, padx=5)

    
    def predict_time(self):
        predict_window = tk.Toplevel(self)
        predict_window.title("Predict Execution Time")
        predict_window.geometry("400x350")
        predict_window.configure(bg='#f0f0f0')

        ttk.Label(predict_window, text="Predict Execution Time", font=('Arial', 14, 'bold')).pack(pady=10)

        ttk.Label(predict_window, text="Choose Algorithm:").pack(pady=(10, 5))
        algo_var = tk.StringVar()
        ttk.Radiobutton(predict_window, text="Dynamic Programming (O(n^3 * p))", variable=algo_var, value="dynamic").pack(pady=2)
        ttk.Radiobutton(predict_window, text="Brute Force (O(p * n^(p+1)))", variable=algo_var, value="brute_force").pack(pady=2)
        ttk.Radiobutton(predict_window, text="Dijkstra (O(n^2 * p))", variable=algo_var, value="dijkstra").pack(pady=2)

        ttk.Label(predict_window, text="Number of exchanges:").pack(pady=(20, 5))
        p_entry = ttk.Entry(predict_window, width=10)
        p_entry.pack(pady=5)

        result_label = ttk.Label(predict_window, text="", font=('Arial', 12))
        result_label.pack(pady=(20, 5))

        def run_prediction():
            try:
                p = int(p_entry.get())
                algo = algo_var.get()
                if not algo:
                    raise ValueError("Please select an algorithm")
                
                predicted_time = predict_execution_time(p, algo)
                formatted_time = format_time(predicted_time)
                
                result_label.config(text=f"Predicted execution time:\n{formatted_time}")
            except ValueError as e:
                messagebox.showerror("Error", str(e), parent=predict_window)

        ttk.Button(predict_window, text="Predict", command=run_prediction, width=20).pack(pady=20)


    def use_default_matrix(self):
        self.matrix = default_matrix
        messagebox.showinfo("Matrix Selection", "Default matrix selected.", parent=self)

    def input_custom_matrix(self):
        matrix_window = tk.Toplevel(self)
        matrix_window.title("Input Custom Matrix")
        matrix_window.geometry("400x300")
        matrix_window.configure(bg='#f0f0f0')

        style = ttk.Style(matrix_window)
        style.configure('MatrixEntry.TEntry', padding=5)

        ttk.Label(matrix_window, text="Enter 4x4 Matrix Values:", font=('Arial', 14, 'bold')).pack(pady=10)

        entries_frame = ttk.Frame(matrix_window)
        entries_frame.pack(padx=20, pady=10)

        entries = []
        for i in range(4):
            row = []
            for j in range(4):
                entry = ttk.Entry(entries_frame, width=8, style='MatrixEntry.TEntry')
                entry.grid(row=i, column=j, padx=5, pady=5)
                row.append(entry)
            entries.append(row)

        def save_matrix():
            try:
                new_matrix = [[float(entry.get()) for entry in row] for row in entries]
                self.matrix = np.array(new_matrix)
                matrix_window.destroy()
                messagebox.showinfo("Matrix Input", "Custom matrix saved successfully.", parent=self)
            except ValueError:
                messagebox.showerror("Error", "Invalid input. Please enter numeric values.", parent=matrix_window)

        ttk.Button(matrix_window, text="Save Matrix", command=save_matrix, width=20).pack(pady=20)











    def calculate_strategy(self):
        algo_window = tk.Toplevel(self)
        algo_window.title("Calculate Best Strategy")
        algo_window.geometry("400x300")
        algo_window.configure(bg='#f0f0f0')

        ttk.Label(algo_window, text="Choose Algorithm", font=('Arial', 14, 'bold')).pack(pady=10)

        algo_var = tk.StringVar()
        ttk.Radiobutton(algo_window, text="Dynamic Programming (O(n^3 * p))", variable=algo_var, value="dynamic").pack(pady=5)
        ttk.Radiobutton(algo_window, text="Brute Force (O(p * n^(p+1)))", variable=algo_var, value="brute_force").pack(pady=5)
        ttk.Radiobutton(algo_window, text="Dijkstra (O(n^2 * p))", variable=algo_var, value="dijkstra").pack(pady=5)

        ttk.Label(algo_window, text="Number of exchanges:").pack(pady=(20, 5))
        p_entry = ttk.Entry(algo_window, width=10)
        p_entry.pack(pady=5)

        def run_algorithm():
            try:
                p = int(p_entry.get())
                algo = algo_var.get()
                
                if algo == "dynamic":
                    best_profit, best_path = best_forex_strategy_dynamic(p, self.matrix)
                elif algo == "brute_force":
                    best_profit, best_path = brute_force_best_forex_strategy(p, self.matrix)
                elif algo == "dijkstra":
                    best_profit, best_path = dijkstra_forex_strategy(p, self.matrix)
                else:
                    raise ValueError("Please select an algorithm")
                
                # Convert indices to currency names if necessary
                if all(isinstance(item, int) for item in best_path):
                    path_names = [self.currency_names[i] for i in best_path]
                else:
                    path_names = best_path  # If best_path already contains currency names

                messagebox.showinfo("Result", f"Best profit: {best_profit:.4f}\nBest path: {' -> '.join(path_names)}", parent=algo_window)
                algo_window.destroy()
            except ValueError as e:
                messagebox.showerror("Error", str(e), parent=algo_window)

        ttk.Button(algo_window, text="Calculate", command=run_algorithm, width=20).pack(pady=20)






    def compare_algorithms(self):
        compare_window = tk.Toplevel(self)
        compare_window.title("Compare Algorithms")
        compare_window.geometry("400x250")
        compare_window.configure(bg='#f0f0f0')

        ttk.Label(compare_window, text="Compare Algorithms", font=('Arial', 14, 'bold')).pack(pady=10)
        ttk.Label(compare_window, text="Maximum number of exchanges:").pack(pady=(20, 5))
        max_p_entry = ttk.Entry(compare_window, width=10)
        max_p_entry.pack(pady=5)

        status_label = ttk.Label(compare_window, text="", font=('Arial', 10))
        status_label.pack(pady=10)

        def run_comparison():
            try:
                max_p = max_p_entry.get().strip()
                if not max_p:
                    raise ValueError("Please enter a value for maximum exchanges.")
                
                max_p = int(max_p)
                if max_p <= 0:
                    raise ValueError("Maximum exchanges must be a positive integer.")
                
                status_label.config(text="Comparison in progress...")
                compare_window.update()

                p_values = range(1, max_p + 1)
                
                dynamic_times = []
                brute_force_times = []
                dijkstra_times = []

                for p in p_values:
                    dynamic_times.append(time_forex_strategy(p, self.matrix, best_forex_strategy_dynamic))
                    brute_force_times.append(time_forex_strategy(p, self.matrix, brute_force_best_forex_strategy))
                    dijkstra_times.append(time_forex_strategy(p, self.matrix, lambda p, matrix: dijkstra_forex_strategy(p, matrix)))

                plt.figure(figsize=(10, 6))
                plt.plot(p_values, dynamic_times, label='Dynamic Programming (O(n^3 * p))', marker='o')
                plt.plot(p_values, brute_force_times, label='Brute Force (O(n^(p+1)))', marker='s')
                plt.plot(p_values, dijkstra_times, label='Dijkstra (O(n^2 * p))', marker='^')
                plt.xlabel('Number of exchanges')
                plt.ylabel('Time (seconds)')
                plt.title('Algorithm Performance Comparison')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.yscale('log')  # Use log scale for y-axis
                plt.show()

                status_label.config(text="Comparison completed successfully.")
                compare_window.update()

            except ValueError as e:
                error_message = str(e)
                messagebox.showerror("Error", error_message, parent=compare_window)
                status_label.config(text=f"Error: {error_message}")
            except Exception as e:
                error_message = f"An unexpected error occurred: {str(e)}"
                messagebox.showerror("Error", error_message, parent=compare_window)
                status_label.config(text="An unexpected error occurred.")

        ttk.Button(compare_window, text="Compare", command=run_comparison, width=20).pack(pady=20)


if __name__ == "__main__":
    app = ForexApp()
    app.mainloop()
