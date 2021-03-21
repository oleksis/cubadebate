# cubadebate ![CI Status](https://github.com/oleksis/cubadebate/workflows/CD/CI/badge.svg?branch=master)
![WordCloud Cubadebate Comenta](https://oleksis.github.io/cubadebate/wordcloud_cubadebate.png)

## Intro
En este notebook crea una WordCloud o Nube de Palabras mediante el uso del Procesamiento del Lenguage Natural (nlp en inglés) sobre los comemtarios del sitio web [Cubadebate](http://www.cubadebate.cu/).

## Requerimientos
* [Dask](https://dask.org/)
* [Requests](https://requests.readthedocs.io/en/master/)
* [Spacy](https://spacy.io/)
* [Wordcloud](http://amueller.github.io/word_cloud/)
* [Numpy](https://numpy.org)
* [Pandas](https://pandas.pydata.org/)
* [Matplotlib](https://matplotlib.org/)


## GitHub Actions

**Run** -> **Build** -> **Deploy**

Utilizando [GitHub Actions](https://github.com/features/actions) para **ejecutar**, **construir** y **desplegar** el notebook, archivos y artefactos hacia [Github Pages](https://pages.github.com/) de la rama (branch) **gh-pages** del repositorio: [Cubadebate Comenta](https://oleksis.github.io/cubadebate/)

Las acciones utilizadas desde el [GitHub Marketplace para Actions](https://github.com/marketplace?type=actions) son:

Execute Notebook and Release Artifacts on: [release](https://help.github.com/es/actions/reference/events-that-trigger-workflows#)
* [release.yml](https://github.com/oleksis/cubadebate/blob/master/.github/workflows/release.yml)

Build and Deploy on: [schedule](https://help.github.com/es/actions/reference/events-that-trigger-workflows#)
* [deploy.yml](https://github.com/oleksis/cubadebate/blob/master/.github/workflows/deploy.yml)

Como resultado se obtiene cada cierta hora en el día la imagen **[wordcloud_cubadebate.png](https://github.com/oleksis/cubadebate/blob/gh-pages/wordcloud_cubadebate.png)** que contiene la nube de palabras sobre los comentarios en [Cubadebate](http://www.cubadebate.cu). También puede descargar los resultados en formato JSON en **[comments_tfidf.json](https://github.com/oleksis/cubadebate/raw/gh-pages/comments_tfidf.json)**



## Uso

### Open in Binder
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/oleksis/cubadebate/master?filepath=CUBADEBATE_SPACY.ipynb)


### Open in Colab
For use in Google Colab you should be install the next packages, add cell python code first and then go to **Restart the runtime**:
```python
!pip install dask[bag]
!pip install wordcloud
!pip install spacy
!python -m spacy download es_core_news_sm

from IPython.display import clear_output
clear_output()
print("Dask installed.")
print("WordCloud installed.")
print("Spacy es_core_news_model installed.\nRestart the runtime!")

```
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oleksis/cubadebate/blob/master/CUBADEBATE_SPACY.ipynb)



## Release
[Cubadebate WordsCloud v1.3.1](https://github.com/oleksis/cubadebate/releases/tag/v1.3.1)
