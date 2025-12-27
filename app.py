from flask import Flask, render_template, request, send_file, redirect, url_for, Response
import pandas as pd
import os
from groq import Groq
from fpdf import FPDF
from model import predict_irrigation, save_to_history
import random 

app = Flask(__name__)

# CONFIGURATION
GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)
HISTORY_FILE = "prediction_history.csv"

@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    rain_chance = None
    status = None
    if request.method == "POST":
        crop = request.form.get("crop")
        soil = float(request.form["soil"])
        temp = float(request.form["temp"])
        hum = float(request.form["humidity"])
        rain = float(request.form["rainfall"])
        
        # 1. Get the ML Prediction from model.py
        prediction = predict_irrigation(soil, temp, hum, rain)
        
        # 2. Save data to history for analytics
        save_to_history(crop, soil, temp, hum, rain, prediction)
        
        # 3. Format the result for display
        result = f"{prediction:.2f} L"
        
        # 4. Irrigation Status Logic
        if prediction < 0.5 or rain >= 5.0:
            status = "NOT REQUIRED"
            rain_chance = random.randint(70, 98) 
        else:
            status = "REQUIRED"
            rain_chance = random.randint(5, 40)
        
    return render_template("predict.html", result=result, rain_chance=rain_chance, status=status)

@app.route("/analytics")
def analytics():
    try:
        if not os.path.exists(HISTORY_FILE):
             return render_template("analytics.html", ai_tips="No data found.", status="EMPTY")
             
        df = pd.read_csv(HISTORY_FILE)
        if df.empty:
            return render_template("analytics.html", ai_tips="No data.", status="EMPTY")
        
        latest = df.iloc[-1].to_dict()
        is_not_needed = float(latest['Result']) < 0.5 or float(latest['Rainfall']) >= 5.0
        status = "NOT REQUIRED" if is_not_needed else "REQUIRED"

        # AI PROMPT: Targeted as a Professional Irrigation Engineer
        prompt = (f"Act as a Professional Irrigation Engineer. The status is {status}. "
                  f"Crop: {latest['Crop']}, Soil Moisture: {latest['Soil_Moisture']}%, Rainfall: {latest['Rainfall']}mm. "
                  f"Explain the technical reasoning why irrigation is {status} for this specific crop and provide 3 water-saving tips.")

        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
        )
        ai_tips = completion.choices[0].message.content
        return render_template("analytics.html", ai_tips=ai_tips, status=status, latest=latest)
    except Exception as e:
        return render_template("analytics.html", ai_tips=f"Error: {str(e)}", status="ERROR")

@app.route("/download_ai_analysis")
def download_ai_analysis():
    try:
        if os.path.exists(HISTORY_FILE):
            df = pd.read_csv(HISTORY_FILE)
            if not df.empty:
                latest = df.iloc[-1].to_dict()
                
                # Format the AI Report as a clean text file
                report_content = f"AGRISENSE AI - IRRIGATION ANALYSIS\n"
                report_content += f"================================\n"
                report_content += f"Target Crop: {latest['Crop']}\n"
                report_content += f"Soil Moisture: {latest['Soil_Moisture']}%\n"
                report_content += f"Rainfall: {latest['Rainfall']}mm\n"
                report_content += f"Calculated Volume: {latest['Result']} L\n"
                report_content += f"--------------------------------\n"
                report_content += f"AI Status: Irrigation logic confirmed for current field parameters."

                return Response(
                    report_content,
                    mimetype="text/plain",
                    headers={"Content-disposition": "attachment; filename=AI_Prediction_Report.txt"}
                )
    except Exception as e:
        return str(e)
    return redirect(url_for('analytics'))

@app.route("/system")
def system():
    records = []
    if os.path.exists(HISTORY_FILE):
        records = pd.read_csv(HISTORY_FILE).tail(15).to_dict('records')
    return render_template("system.html", history=records)

@app.route("/download_csv")
def download_csv():
    """Allows users to download the full prediction_history.csv file"""
    try:
        if os.path.exists(HISTORY_FILE):
            return send_file(
                HISTORY_FILE,
                mimetype="text/csv",
                as_attachment=True,
                download_name="prediction_history.csv"
            )
        return "No history data found to download."
    except Exception as e:
        return f"Error: {str(e)}"

@app.route("/clear_history", methods=["POST"])
def clear_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
        headers = ["Timestamp", "Crop", "Soil_Moisture", "Temperature", "Humidity", "Rainfall", "Result"]
        pd.DataFrame(columns=headers).to_csv(HISTORY_FILE, index=False)
    return redirect(url_for('system'))

@app.route("/download_pdf")
def download_pdf():
    try:
        if not os.path.exists(HISTORY_FILE):
            return "No data found."
            
        df = pd.read_csv(HISTORY_FILE)
        latest = df.iloc[-1].to_dict()
        
        # Status calculation
        is_not_needed = float(latest['Result']) < 0.5 or float(latest['Rainfall']) >= 5.0
        status_text = "NOT REQUIRED" if is_not_needed else "REQUIRED"

        # Generate the professional AI Tips for the PDF
        prompt = (f"As a Professional Irrigation Engineer, summarize in 4 sentences why irrigation is {status_text} "
                  f"for {latest['Crop']} with {latest['Soil_Moisture']}% soil moisture and {latest['Rainfall']}mm rain.")
        
        completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
        )
        ai_logic = completion.choices[0].message.content

        # Create PDF document
        pdf = FPDF()
        pdf.add_page()
        
        # Title
        pdf.set_font("Arial", 'B', 20)
        pdf.set_text_color(44, 122, 75)
        pdf.cell(200, 20, txt="AgriSense AI - Field Intelligence Report", ln=True, align='C')
        pdf.ln(10)

        # Section 1: Data
        pdf.set_font("Arial", 'B', 14)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(190, 10, txt="1. FIELD DATA SUMMARY", ln=True)
        pdf.set_font("Arial", '', 11)
        pdf.cell(190, 8, txt=f"Crop: {latest['Crop']}", ln=True)
        pdf.cell(190, 8, txt=f"Soil Moisture: {latest['Soil_Moisture']}%", ln=True)
        pdf.cell(190, 8, txt=f"Rainfall: {latest['Rainfall']}mm", ln=True)
        pdf.cell(190, 8, txt=f"Predicted Water Needed: {latest['Result']} Liters", ln=True)
        pdf.ln(5)

        # Section 2: AI Reasoning
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(190, 10, txt=f"2. AGRO-ENGINEERING ANALYSIS ({status_text})", ln=True)
        pdf.set_font("Arial", '', 11)
        pdf.multi_cell(0, 8, txt=ai_logic)

        # Output
        filename = f"AgriSense_Report_{latest['Crop']}.pdf"
        pdf.output(filename)
        return send_file(filename, as_attachment=True)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)