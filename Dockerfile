FROM continuumio/miniconda3

WORKDIR /home/biolib

RUN conda install --yes pytorch tensorflow scikit-learn pandas numpy \
    && conda clean -afy
RUN pip install xgboost

COPY . .

ENTRYPOINT [ "python", "src/predict.py" ]
