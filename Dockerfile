FROM nvcr.io/nvidia/pytorch:23.10-py3

RUN pip install notebook statsmodels==0.14.1 seaborn==0.13.2 torchinfo==1.8.0
