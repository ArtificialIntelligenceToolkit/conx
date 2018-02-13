FROM jupyter/tensorflow-notebook

LABEL maintainer="Douglas Blank <doug.blank@gmail.com>"

RUN pip install conx==3.6.0 --upgrade --no-cache-dir
RUN pip install jyro --upgrade --no-cache-dir
RUN pip install jupyter notebook --upgrade --no-cache-dir

RUN sudo apt install --yes ffmpeg || true
RUN sudo apt install --yes libffi-dev libffi6 || true

RUN python -c "import conx as cx; cx.Dataset.get('mnist')"
RUN python -c "import conx as cx; cx.Dataset.get('cifar10')"
RUN python -c "import conx as cx; cx.Dataset.get('cmu_faces_full_size')"
RUN python -c "import conx as cx; cx.Dataset.get('cmu_faces_half_size')"
RUN python -c "import conx as cx; cx.Dataset.get('cmu_faces_quarter_size')"
