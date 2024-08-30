from flask import Flask, request, jsonify
import pandas as pd
from cct import get_consensus
import logging

app = Flask(__name__)

@app.route('/calculate', methods=['POST'])
def calculate():
    try:
        if request.method == 'POST':
            data = request.get_json()
            logging.info(f"Received data: {data}")
            
            df = pd.DataFrame(data['data'])
            result = get_consensus(df)
            
            # Convert NumPy array to Python list
            result = result.tolist()
            result = [i * 10 for i in result]
            result = [round(i, 1) for i in result]

            logging.info(f"Result: {result}")
            
            response = jsonify(result)
            response.headers['Content-Type'] = 'application/json'
            return response
        else:
            return 'Method not allowed', 405
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return f"Internal Server Error: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)