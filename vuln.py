import os
import subprocess
import flask
from flask import request, jsonify
import sqlite3
import pickle
import hashlib

app = flask.Flask(__name__)
app.config["DEBUG"] = True

# Hardcoded secret
SECRET_KEY = '12345'

# Route with SQL Injection vulnerability
@app.route('/user/<username>', methods=['GET'])
def get_user(username):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE username = '" + username + "'"
    result = cursor.execute(query)
    user = result.fetchone()
    conn.close()
    if user:
        return jsonify({'username': user[0], 'email': user[1]})
    else:
        return jsonify({'error': 'User not found'}), 404

# Route with OS Command Injection vulnerability
@app.route('/ping', methods=['GET'])
def ping():
    host = request.args.get('host')
    response = os.system('ping -c 1 ' + host)
    return jsonify({'result': response})

# Route with Insecure Deserialization vulnerability
@app.route('/load', methods=['POST'])
def load_data():
    data = request.data
    obj = pickle.loads(data)
    return jsonify({'result': str(obj)})

# Route with Hardcoded Password vulnerability
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if username == 'admin' and password == 'admin123':
        return jsonify({'message': 'Login successful'})
    else:
        return jsonify({'message': 'Login failed'}), 401

# Route with XSS vulnerability
@app.route('/greet', methods=['GET'])
def greet():
    name = request.args.get('name')
    return f"<h1>Hello {name}</h1>"

# Route with Weak Hashing Algorithm vulnerability
@app.route('/hash', methods=['POST'])
def hash_password():
    data = request.get_json()
    password = data.get('password')
    hashed_password = hashlib.md5(password.encode()).hexdigest()
    return jsonify({'hashed_password': hashed_password})

if __name__ == '__main__':
    app.run()
