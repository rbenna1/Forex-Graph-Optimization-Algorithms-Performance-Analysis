The problem addressed in this project focuses on optimizing currency exchanges in the Forex market. The objective is to identify the best sequence of currency exchanges to maximize profit, starting and returning to an initial currency within a specified number of transactions.
Given:

A set of currencies (e.g., Euro, Dollar, Pound, Swiss Franc)
An exchange rate matrix between these currencies
A maximum number of allowed exchanges
The challenge is to determine the sequence of exchanges that maximizes profit while starting and ending with the same currency (usually the Euro). This problem is modeled as a graph, where each currency is a node, exchange rates are weighted edges, and a sequence of exchanges represents a path. The solution involves finding the path that maximizes the product of edge weights. This graph-based modeling approach enables the use of graph theory algorithms for effective resolution of the optimization problem
