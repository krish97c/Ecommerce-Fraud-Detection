import mysql.connector

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
            host=localhost,
            user=root,
            password=krishna,
            database=fraud
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
