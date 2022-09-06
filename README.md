
# Hand Gesture Recognition(1st Project)

This is my first project based on Machine learning.My project is based on Hand Gesture Recognition. Gestures are any bodily motion or state, but commonly originate from the face or hand. My project detects four gestures- Victory, Thumb, Fist and Palm. I have used a CNN approach to construct it. 


## Requirements

### Python 3.8.6 Installation in Windows
- Check if Python is already installed by opening command prompt and running ```python --version```.
- If the above command returns ```Python <some-version-number>``` you're probably good - provided version number is above 3.6
- If Python's not installed, command would say something like: ```'python' is not recognized as an internal or external command, operable program or batch file.```
- If so, download the installer from: https://www.python.org/ftp/python/3.8.6/python-3.8.6.exe
- Run that. In the first screen of installer, there will be an option at the bottom to "Add Python 3.8.6 to Path". Make sure to select it.
- Open command prompt and run ```python --version```. If everything went well it should say ```Python 3.8.6```
- You're all set! 

### Install Required Libraries

```bash
   pip install tensorflow
```
```bash
   pip install keras
```
```bash
   pip install opencv-python
```
```bash
   pip install numpy 
```
```bash
   pip install pandas
```
```bash
   pip install matplotlib
```
```bash
   pip install scikit-learn
```
## Procedure
- First, by using Collecting_Data.py we have to collect the data and build a dataset.
- Second, by using TrainCNN.py we will train the CNN model on the developed dataset.
- Third, by using PredictCNN.py you can perform the real time analysis for Hand Gesture Recognition and can predict the four gestures(Victory, Thumb, Fist, Palm).   
