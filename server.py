from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
from datetime import datetime
import subprocess

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

DATA_DIR = os.getenv("DATA_DIR", "/app/data")
os.makedirs(DATA_DIR, exist_ok=True)

@app.route('/save-rankings', methods=['POST'])
def save_rankings():
    try:
        data = request.get_json(force=True)
        print("Received data:", data)

        if not data:
            print("Error: No data received or JSON format is wrong.")
            return jsonify({"error": "No data received"}), 400

        # Save timestamped version
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_filename = f"rankings_{timestamp}.json"
        timestamped_filepath = os.path.join(DATA_DIR, timestamped_filename)

        # Try writing files
        with open(timestamped_filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(" Saved:", timestamped_filename)

        latest_filepath = os.path.join(DATA_DIR, "latest_rankings.json")
        with open(latest_filepath, "w") as f:
            json.dump(data, f, indent=2)
        print("Updated: latest_rankings.json")

        # Run read.py and capture its output
        result = subprocess.run(['python', 'read.py'], capture_output=True, text=True)
        
        # Debug: Print raw output
        print("=== Raw read.py output ===")
        print(result.stdout)
        print("=== End raw output ===")
        
        # Parse the output to get final_table and best_transcription
        output_lines = result.stdout.split('\n')
        final_table = {}
        best_transcription = None
        headers = []
        
        # First, find the complete list of headers
        for line in output_lines:
            if "Columns in Operability Table Before AHP:" in line:
                # Extract headers from the list format
                headers_str = line.split(":", 1)[1].strip()
                headers = [h.strip("'[] ") for h in headers_str.split(",")]
                print(f"Found complete headers: {headers}")
                break
        
        # First pass: find the best transcription
        for line in output_lines:
            if "The preferred grammar pronunciation is" in line:
                best_transcription = line.split("is")[-1].strip()
                print(f"Found best transcription: {best_transcription}")
                break
        
        # Second pass: parse the final table
        table_started = False
        current_ipa = None
        current_scores = {}
        
        for line in output_lines:
            line = line.strip()
            if not line:
                continue
                
            # Look for the final table section
            if "Operability Table (Raw Scores):" in line:
                table_started = True
                print("Found Final Table section")
                continue
                
            # Stop parsing if we hit the next section
            if table_started and ("AHP Weights" in line or "[1 rows x" in line):
                # Save the last row if we have one
                if current_ipa and current_scores:
                    final_table[current_ipa] = current_scores
                break
                
            if table_started:
                # Skip empty lines and summary lines
                if not line or "rows x" in line:
                    continue
                    
                # Parse data rows
                if headers and not line.startswith(" "):
                    parts = line.split()
                    if len(parts) >= 2:  # At least IPA and one score
                        ipa = parts[0]
                        scores = {}
                        
                        # Handle the case where pandas truncates with "..."
                        if "..." in line:
                            # Get the first set of scores
                            for i, header in enumerate(headers):
                                if i + 1 < len(parts) and parts[i + 1] != "...":
                                    try:
                                        scores[header] = float(parts[i + 1])
                                    except (ValueError, IndexError):
                                        continue
                            
                            # Look for the second part of the row in the next line
                            next_line = next((l for l in output_lines[output_lines.index(line) + 1:] if l.strip() and not l.startswith(" ")), None)
                            if next_line:
                                next_parts = next_line.split()
                                if len(next_parts) >= 2:
                                    # Get the remaining scores
                                    remaining_headers = headers[len(scores):]
                                    for i, header in enumerate(remaining_headers):
                                        if i + 1 < len(next_parts):
                                            try:
                                                scores[header] = float(next_parts[i + 1])
                                            except (ValueError, IndexError):
                                                continue
                        else:
                            # Handle normal row without truncation
                            for i, header in enumerate(headers):
                                if i + 1 < len(parts):
                                    try:
                                        scores[header] = float(parts[i + 1])
                                    except (ValueError, IndexError):
                                        continue
                        
                        if scores:  # Only add if we have valid scores
                            final_table[ipa] = scores
                            print(f"Parsed row - IPA: {ipa}, Scores: {scores}")
        
        if not best_transcription:
            print("Error: Could not find best transcription in output")
            return jsonify({"error": "Could not find best transcription"}), 500
            
        if not final_table:
            print("Error: Could not parse final table from output")
            print("Current output_lines:", output_lines)
            return jsonify({"error": "Could not parse final table"}), 500
            
        # Get the target word from the request data
        target_word = data.get('word', '')
        
        response_data = {
            "targetWord": target_word,
            "bestTranscription": best_transcription,
            "finalTable": final_table
        }
        
        print(f"Response data: {response_data}")
        return jsonify(response_data)

    except Exception as e:
        print("Exception occurred:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)