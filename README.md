#Alpha"Othello" Zero

please implement this scripts in Google Colaboratory

1. Select GPU.
<img src="./images/gpu.png", width=80%>

2. Mount your Google Drive
~~~python
#to access your google drive folder
from google.colab import auth, drive, files, output
drive.mount('/content/drive')
~~~

3. Import the folder from github & setup
~~~python
#to clone codes
!git clone https://github.com/takehiko-ohkawa/AlphaZero.git
import os
os.chdir('./AlphaZero')
!pwd
!chmod ugo+x ./setup_colab.sh #give a permission
! ./setup_colab.sh #install torch 
~~~

4. Train 
~~~python
!chmod ugo+x ./train_colab.sh
!./train_colab.sh
~~~

- make a folder or copy a folder
~~~python
#make
!mkdir <folder>
#copy
!cp -r <folderA> <folderB> #copy A->B
~~~

- Overwrite your code

~~~python
%%writefile oo.py
<contents of oo.py file>
~~~

- Check GPU status

~~~python
!df -h #desk status
!free -h #memory status
!ps aux #process status
!nvidia-smi #GPU status
!cat /proc/uptime |awk '{print $1/86400"days"}' #remaining time - upto 0.5days
~~~

- Download files

~~~python
files.download(<file>)
~~~