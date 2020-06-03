FROM python:3.7.7-buster AS compile-image

LABEL maintainer="Oleksis Fraga <oleksis.fraga at gmail.com>"

RUN useradd --create-home oleksis
WORKDIR /home/oleksis
RUN mkdir cubadebate
COPY . cubadebate
RUN chown oleksis:oleksis -R cubadebate
USER oleksis
WORKDIR /home/oleksis/cubadebate
RUN python -m venv .venv
# Ensure we use the virtualenv
ENV PATH=".venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt &&\
    python -m spacy download es_core_news_sm


FROM python:3.7.7-slim-buster AS runtime-image

RUN useradd --create-home oleksis
WORKDIR /home/oleksis
COPY --from=compile-image /home/oleksis/cubadebate cubadebate
RUN chown oleksis:oleksis cubadebate
USER oleksis
WORKDIR /home/oleksis/cubadebate
# Ensure we use the virtualenv
ENV PATH=".venv/bin:$PATH"
CMD ["python", "CUBADEBATE_SPACY.py"]
