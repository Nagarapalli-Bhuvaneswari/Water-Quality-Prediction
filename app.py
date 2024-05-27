from flask import Flask, render_template,request
from knn import k,predict,train_features,train_target,get_knn_accuracy
from decisiontree import dt,get_dt_accuracy
app = Flask(__name__)

@app.route('/')
def main():
    return render_template('main.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/result', methods=['POST'])
def make_prediction():
    user_input = []
    ph=float(request.form['ph'])
    Hardness=float(request.form['Hardness'])
    Solids=float(request.form['Solids'])
    Chloramines=float(request.form['Chloramines'])
    Sulfate=float(request.form['Sulfate'])
    Conductivity=float(request.form['Conductivity'])
    Organic_carbon=float(request.form['Organic_carbon'])
    Trihalomethanes=float(request.form['Trihalomethanes'])
    Turbidity=float(request.form['Turbidity'])
    algorithm=request.form['algorithm']

    user_input.append([ph,Hardness,Solids,Chloramines,Sulfate,Conductivity,Organic_carbon,Trihalomethanes,Turbidity])
                  
    
    if algorithm == 'knn':
      prediction = predict(train_features, train_target, user_input, k)[0]
      if prediction == 1:
        result = 'Potable'
      elif prediction == 0:
        result = 'Not Potable'
      return render_template('knnresult.html',result1=result)
    
    elif algorithm == 'decision-tree' :
       prediction1 = dt.predict(user_input)
       if prediction1[0] == 1:
        result = 'Potable'
       elif prediction1[0] ==0 :
        result = 'Not Potable'
       return render_template('decisiontreeresult.html', result2=result)
       
    
   

@app.route('/accurac')
def accurac():
    accuracy1=get_knn_accuracy()
    accuracy2=get_dt_accuracy()
    return render_template('accurac.html',accuracy1=accuracy1,accuracy2=accuracy2)

if __name__ == '__main__':
    app.run(debug=True)
