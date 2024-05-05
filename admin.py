import pandas as pd
import mysql.connector
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def connect_to_mysql(host, user, password, database):
    """
    Establishes a connection to MySQL database.

    Parameters:
        host (str): The host address of the MySQL server.
        user (str): The username for connecting to MySQL.
        password (str): The password for connecting to MySQL.
        database (str): The name of the MySQL database to connect to.

    Returns:
        connection (mysql.connector.connection_cext.CMySQLConnection): The connection object.
        cursor (mysql.connector.cursor_cext.CMySQLCursor): The cursor object.
    """
    try:
        connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        cursor = connection.cursor()
        print("Connected to MySQL database")
        return connection, cursor
    except mysql.connector.Error as e:
        print("Error connecting to MySQL database:", e)
        return None, None

def save_data_to_mysql(connection, cursor, data):
    """
    Saves data to MySQL database.

    Parameters:
        connection (mysql.connector.connection_cext.CMySQLConnection): The connection object.
        cursor (mysql.connector.cursor_cext.CMySQLCursor): The cursor object.
        data (dict): The data to be saved to the database.
    """
    try:
        # Construct the SQL query
        query = """
            INSERT INTO transactions (step, type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        # Execute the query
        cursor.execute(query, (
            data['step'],
            data['type'],
            data['amount'],
            data['oldbalanceOrg'],
            data['newbalanceOrig'],
            data['oldbalanceDest'],
            data['newbalanceDest']
        ))
        # Commit the transaction
        connection.commit()
        print("Data saved to MySQL database")
    except mysql.connector.Error as e:
        print("Error saving data to MySQL database:", e)

def close_mysql_connection(connection, cursor):
    """
    Closes the connection to MySQL database.

    Parameters:
        connection (mysql.connector.connection_cext.CMySQLConnection): The connection object.
        cursor (mysql.connector.cursor_cext.CMySQLCursor): The cursor object.
    """
    try:
        cursor.close()
        connection.close()
        print("Connection to MySQL database closed")
    except mysql.connector.Error as e:
        print("Error closing MySQL connection:", e)

def view_transaction_data(host, user, password, database):
    """
    Retrieves transaction data from MySQL database.

    Parameters:
        host (str): The host address of the MySQL server.
        user (str): The username for connecting to MySQL.
        password (str): The password for connecting to MySQL.
        database (str): The name of the MySQL database to connect to.

    Returns:
        transaction_data (pd.DataFrame): DataFrame containing transaction data.
    """
    connection, cursor = connect_to_mysql(host, user, password, database)
    if connection and cursor:
        try:
            # Execute SQL query to fetch transaction data
            query = "SELECT * FROM transactions"
            cursor.execute(query)
            # Fetch all rows of the result
            rows = cursor.fetchall()
            # Create DataFrame from fetched rows
            columns = [desc[0] for desc in cursor.description]
            transaction_data = pd.DataFrame(rows, columns=columns)
            return transaction_data
        except Exception as e:
            print("Error fetching transaction data:", e)
        finally:
            close_mysql_connection(connection, cursor)
    else:
        return pd.DataFrame()

def filter_transaction_data(transaction_data, min_amount, max_amount):
    """
    Filters transaction data based on amount range.

    Parameters:
        transaction_data (pd.DataFrame): DataFrame containing transaction data.
        min_amount (float): Minimum amount for filtering.
        max_amount (float): Maximum amount for filtering.

    Returns:
        filtered_data (pd.DataFrame): Filtered DataFrame.
    """
    filtered_data = transaction_data[(transaction_data['amount'] >= min_amount) & (transaction_data['amount'] <= max_amount)]
    return filtered_data

def admin_panel():
    st.title("Admin Panel")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "ab" and password == "cd":
            st.success("Logged in as admin")

            # Option to view transaction data
            if st.button("View Transaction Data"):
                host = "localhost"
                user = "root"
                password = "krishna"
                database = "fraud"
                transaction_data = view_transaction_data(host, user, password, database)

                if not transaction_data.empty:
                    st.subheader("Transaction Data")
                    st.write(transaction_data)
                else:
                    st.warning("No transaction data available.")

            # Option to visualize transaction data
            if st.button("Visualize Transaction Data"):
                host = "localhost"
                user = "root"
                password = "krishna"
                database = "fraud"
                transaction_data = view_transaction_data(host, user, password, database)

                if not transaction_data.empty:
                    st.subheader("Transaction Data Visualization")
                    visualize_data(transaction_data)
                else:
                    st.warning("No transaction data available.")

            # Option to add new transaction data
            if st.button("Add New Transaction Data"):
                st.subheader("Enter New Transaction Data")
                new_data = get_new_transaction_data()
                if new_data:
                    host = "localhost"
                    user = "root"
                    password = "krishna"
                    database = "fraud"
                    save_data_to_mysql(host, user, password, database, new_data)
                    st.success("New transaction data saved successfully.")

            # Other admin tasks
            if st.button("Other Admin Tasks"):
                st.subheader("Additional Admin Tasks")
                # Add any additional functionalities here
        else:
            st.error("Invalid username or password. Please try again.")

def visualize_data(data):
    """
    Visualize transaction data.

    Parameters:
        data (pd.DataFrame): DataFrame containing transaction data.
    """
    # Count plot for transaction types
    st.write("Transaction Types Distribution")
    sns.countplot(data=data, x='type', palette='viridis')
    st.pyplot()

    # Distribution plot for transaction amounts
    st.write("Transaction Amount Distribution")
    sns.histplot(data=data, x='amount', bins=20, kde=True, color='skyblue')
    st.pyplot()

    # Pairplot for transaction data
    st.write("Pairplot for Transaction Data")
    sns.pairplot(data=data, hue='type', palette='muted')
    st.pyplot()

def get_new_transaction_data():
    """
    Get input for new transaction data.

    Returns:
        new_data (dict): Dictionary containing new transaction data.
    """
    new_data = {}
    new_data['step'] = st.number_input("Step", value=0)
    new_data['type'] = st.selectbox("Type", ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
    new_data['amount'] = st.number_input("Amount", value=0.0)
    new_data['oldbalanceOrg'] = st.number_input("Old Balance Origin", value=0.0)
    new_data['newbalanceOrig'] = st.number_input("New Balance Origin", value=0.0)
    new_data['oldbalanceDest'] = st.number_input("Old Balance Destination", value=0.0)
    new_data['newbalanceDest'] = st.number_input("New Balance Destination", value=0.0)

    if st.button("Save"):
        return new_data
    else:
        return None

if __name__ == "__main__":
    admin_panel()


