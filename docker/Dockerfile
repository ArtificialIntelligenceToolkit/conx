FROM jupyter/tensorflow-notebook

LABEL maintainer="Douglas Blank <doug.blank@gmail.com>"

RUN pip install conx --upgrade

RUN python -c "import conx as cx; cx.Dataset.get('mnist')"
RUN python -c "import conx as cx; cx.Dataset.get('cifar10')"
RUN python -c "import conx as cx; cx.Dataset.get('cmu_faces_full_size')"
RUN python -c "import conx as cx; cx.Dataset.get('cmu_faces_half_size')"
RUN python -c "import conx as cx; cx.Dataset.get('cmu_faces_quarter_size')"
