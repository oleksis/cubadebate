# cubadebate
![WordCloud Cubadebate Comenta](https://oleksis.github.io/cubadebate/wordcloud_cubadebate.png)

## Intro
En este notebook crea una WordCloud o Nube de Palabras mediante el uso del Procesamiento del Lenguage Natural (nlp en inglés) sobre los comemtarios del sitio web [Cubadebate](http://www.cubadebate.cu/).

## Requerimientos
* [Spacy](https://spacy.io/)
* [Wordcloud](http://amueller.github.io/word_cloud/)
* [Numpy](https://numpy.org)
* [Matplotlib](https://matplotlib.org/)
* [Requests](https://requests.readthedocs.io/en/master/)
* [Papermill](https://papermill.readthedocs.io/en/latest/)

## GitHub Actions

**Run** -> **Build** -> **Deploy**

Utilizando [GitHub Actions](https://github.com/features/actions) para **ejecutar**, **construir** y **desplegar** el notebook, archivos y artefactos hacia [Github Pages](https://pages.github.com/) de la rama (branch) **gh-pages** del repositorio: [Cubadebate Comenta](https://oleksis.github.io/cubadebate/)

Las acciones utilizadas desde el [GitHub Marketplace para Actions](https://github.com/marketplace?type=actions) son:

Execute Notebook and Release Artifacts on: [release](https://help.github.com/es/actions/reference/events-that-trigger-workflows#)
* [release.yml](https://github.com/oleksis/cubadebate/blob/master/.github/workflows/release.yml)

Build and Deploy on: [schedule](https://help.github.com/es/actions/reference/events-that-trigger-workflows#)
* [deploy.yml](https://github.com/oleksis/cubadebate/blob/master/.github/workflows/deploy.yml)

Como resultado se obtiene cada cierta hora en el día la imagen **[wordcloud_cubadebate.png](https://github.com/oleksis/cubadebate/blob/gh-pages/wordcloud_cubadebate.png)** que contiene la nube de palabras sobre los comentarios en [Cubadebate](http://www.cubadebate.cu). También puede descargar los resultados en formato JSON en [comments_tfidf.json](https://github.com/oleksis/cubadebate/raw/gh-pages/comments_tfidf.json)



## Uso
<a href="https://colab.research.google.com/github/oleksis/cubadebate/blob/master/CUBADEBATE_SPACY.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Release
[Cubadebate WordsCloud v1.1.4](https://github.com/oleksis/cubadebate/releases/tag/v1.1.4)
