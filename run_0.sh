#!/usr/bin/env bash
WORK_PATH=$(pwd)
USER=`whoami`
HOST=`hostname`
<< COMMENTOUT
cd $WORK_PATH/Model/repeat1_attribute_prediction_new_utilize_new_attribute_link/Baseline
CURRENT_DIR=`pwd`
EXEC="python main.py"
NOW=`date`
echo "==============start: ${NOW}=============="
echo "${USER}@${HOST}:${CURRENT_DIR}$ ${EXEC}"
$EXEC
NOW=`date`
echo "==============end: ${NOW}=============="

cd $WORK_PATH/Model/repeat1_attribute_prediction_new_utilize_new_attribute_link/EGCNh
CURRENT_DIR=`pwd`
EXEC="python main.py"
NOW=`date`
echo "==============start: ${NOW}=============="
echo "${USER}@${HOST}:${CURRENT_DIR}$ ${EXEC}"
$EXEC
NOW=`date`
echo "==============end: ${NOW}=============="

cd $WORK_PATH/Model/repeat1_attribute_prediction_new_utilize_new_attribute_link/EGCNo
CURRENT_DIR=`pwd`
EXEC="python main.py"
NOW=`date`
echo "==============start: ${NOW}=============="
echo "${USER}@${HOST}:${CURRENT_DIR}$ ${EXEC}"
$EXEC
NOW=`date`
echo "==============end: ${NOW}=============="

cd $WORK_PATH/Model/repeat1_attribute_prediction_new_utilize_new_attribute_link/GCN
CURRENT_DIR=`pwd`
EXEC="python main.py"
NOW=`date`
echo "==============start: ${NOW}=============="
echo "${USER}@${HOST}:${CURRENT_DIR}$ ${EXEC}"
$EXEC
NOW=`date`
echo "==============end: ${NOW}=============="
cd $WORK_PATH/Model/repeat1_attribute_prediction_new_utilize_new_attribute_link/LSTM
CURRENT_DIR=`pwd`
EXEC="python main.py"
NOW=`date`
echo "==============start: ${NOW}=============="
echo "${USER}@${HOST}:${CURRENT_DIR}$ ${EXEC}"
$EXEC
NOW=`date`
echo "==============end: ${NOW}=============="
cd $WORK_PATH/Model/repeat1_attribute_prediction_new_utilize_new_attribute_link/STGCN
CURRENT_DIR=`pwd`
EXEC="python main.py"
NOW=`date`
echo "==============start: ${NOW}=============="
echo "${USER}@${HOST}:${CURRENT_DIR}$ ${EXEC}"
$EXEC
NOW=`date`
echo "==============end: ${NOW}=============="

cd $WORK_PATH/Model/repeat1_attribute_prediction_new_utilize_new_attribute_link/STGGNN
CURRENT_DIR=`pwd`
EXEC="python main.py"
NOW=`date`
echo "==============start: ${NOW}=============="
echo "${USER}@${HOST}:${CURRENT_DIR}$ ${EXEC}"
$EXEC
NOW=`date`
echo "==============end: ${NOW}=============="
COMMENTOUT
cd $WORK_PATH/Model/repeat1_attribute_prediction_new_utilize_new_attribute_link/FNN
CURRENT_DIR=`pwd`
EXEC="python main.py"
NOW=`date`
echo "==============start: ${NOW}=============="
echo "${USER}@${HOST}:${CURRENT_DIR}$ ${EXEC}"
$EXEC
NOW=`date`
echo "==============end: ${NOW}=============="

cd $WORK_PATH/Evaluation
CURRENT_DIR=`pwd`
EXEC="python repeat1_attribute_prediction_new_utilize_new_attribute_link.py"
NOW=`date`
echo "==============start: ${NOW}=============="
echo "${USER}@${HOST}:${CURRENT_DIR}$ ${EXEC}"
$EXEC
NOW=`date`
echo "==============end: ${NOW}=============="