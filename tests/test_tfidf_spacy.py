import os
import sys
import unittest
from unittest import TestCase


ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_PATH)

from CUBADEBATE_SPACY import get_comments_file, clean, comments_tfidf


class TestComments(TestCase):
    def test_get_comments(self, name="comments.dat"):
        """Test List of comments from file name"""
        file_name = os.path.join(ROOT_PATH, "tests", name)
        df_comments = get_comments_file(file_name)
        self.assertFalse(df_comments.empty)
        self.assertEqual(len(df_comments), 990)

    def test_clean_comment(self):
        """Test Clean process"""
        raw_comment = (
            "<p>Como se llama la aplicaci\u00f3n para"
            + " sacar los pasajes desde casa<\/p>\n"
        )
        clean_comment = clean(raw_comment)
        self.assertEqual(
            sorted(clean_comment),
            sorted(["llamar", "aplicacion", "sacar", "pasaje", "casar"]),
        )

    def test_comments_tfidf(self):
        """Test Calculate TF-IDF of the documents"""
        documents = [
            (
                "<p>Como se llama la aplicaci\u00f3n para"
                + " sacar los pasajes desde casa<\/p>\n"
            ),
            (
                "<p>Magnifico programa pero lamentablemente poco conocido,"
                + " ha tenido muy pobre divulgacion y lo debian repetir en"
                + " otros horarios y canales. No he podido descargar los videos"
                + " para conservarlos y debatir con mis estudiantes, estoy en"
                + " eso.<br \/>\nFelicitaciones<\/p>\n"
            ),
            (
                "<p>\u00bfPero est\u00e1n locos?... y sus padres tambi\u00e9n?"
                + " Acaso no saben en lo que andan sus hijos? Eso s\u00f3lo"
                + " ocurre en un pa\u00eds dirigido por un loco, apoyado por"
                + " otros m\u00e1s locos que \u00e9l."
                + " Se ver\u00e1n horrores.<\/p>\n"
            ),
            (
                "<p>Gracias Jaqueline por tu regreso y dejar ke podamos"
                + " disfrutar de tu presencia ,Dios te bendiga y deseo"
                + " triunfes  en tus proyectos<\/p>\n"
            ),
            (
                "<p>Actualmente est\u00e1 en desarrollo la versi\u00f3n"
                + " web de Viajando,  los que prefieran acceder"
                + " por esta via<\/p>\n"
            ),
        ]

        documents_normalized = [clean(doc) for doc in documents]
        tfidf_list = comments_tfidf(documents_normalized)
        self.assertEqual(round(tfidf_list[0]["llamar"], 2), 0.32)
        self.assertEqual(round(tfidf_list[1]["programar"], 2), 0.09)
        self.assertEqual(round(tfidf_list[2]["apoyar"], 2), 0.12)
        self.assertEqual(round(tfidf_list[3]["triunfar"], 2), 0.13)
        self.assertEqual(round(tfidf_list[4]["preferir"], 2), 0.23)


if __name__ == "__main__":
    unittest.main()
