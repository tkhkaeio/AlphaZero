export PYTHONUNBUFFERED="True"
LOG="log/train-`date +'%Y-%m-%d-%H-%M-%S'`.log"
python main_colab.py > $LOG
