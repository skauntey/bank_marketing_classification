Abstract: The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe a term deposit (variable y).

I've build a simple Flask Microservice that loads a serialized model from disk, and then uses it to serve out predictions. 

I've created a machine learning model in order to predict how likely clients will subscribe to a bank term deposit. The model implemented was Gradient Boostuing Classifier with optimized hyperparameters. The model’s test performance (AUC) is 87.7%. The model was able to catch 60.7% of customers that will subscribe to a term deposit which is moderate to good. The model did not look at high importance features, which could increase the precision and recall rate to high level. Other models should be tried as well to compare the performance.

For deployment I have create four files like app.py, columns.pkl, best_classifier.joblib, std_scaler.pkl file is also created.


1. Open a terminal in Visual Studio Code.

2. cd into the projects directory where you want to create a model

3. make a folder: mkdir <'folder name of your choice'> e.g. bank marketing app 

4. cd into the newly created folder in step 3

4. If you don't have virtual environment installed, install virtualenv:  python3 -m pip install virtualenv

5. while in the folder create a virtualenv: virtualenv VENV

6. source the virtualenv (activate it):  source VENV/bin/activate

7. Move or Copy the files of the zip folder within the folder you created in step 3 (along with the virtual environment)

8. Install dependencies : python install -r requirements.txt

9. Open the shell/terminal and run 'app.py'

10. go to the link generated such as http://localhost:8080

11. you should be able to see something like ![app_page](/image/screen_capture1.jpg)