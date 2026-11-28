"""Chunker jurídico jerárquico y agnóstico a la provincia para documentos Markdown de AiHoreca.

Principios de diseño:
1. Unidad lógica: fragmenta por Artículo, Disposición o Tabla Salarial (nunca corta una fila de tabla a la mitad).
2. Inyección de contexto: cada chunk comienza con una cabecera estructurada
   `[NORMA: ... | CAPA: ... | TERRITORIO: ... | SECCIÓN: ... | ARTÍCULO: ...]`
   para maximizar la precisión tanto en embeddings densos (`gemini-embedding-001`) como en BM25 (`FastEmbed`).
3. Preservación de cabeceras de tabla: si una tabla salarial muy extensa debe dividirse en varios sub-chunks,
   repite automáticamente la fila de cabeceras (`|...|`) y el separador (`|---|---|`) en cada sub-chunk.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
import uuid
from pathlib import Path

from src.config import PROCESSED_DATA_DIR
from src.retrieval.schemas import ContentType, LegalChunk, NormativeLayer

logger = logging.getLogger(__name__)

_RESIDUAL_NOISE_LINE = re.compile(
    r"(?i)^(?:"
    r"dl\s+[a-z]{1,2}\s+\d+-\d+|"
    r"issn\s+\d{4}-\d{3}[\dx]|"
    r"\d{1,3}/\d{1,3}|"
    r"legislaci[óo]n\s+consolidada|"
    r"p[áa]gina\s+\d+(?:\s+de\s+(?:\d+|un\s+total\s+de\s+\d+))?|"
    r"bolet[íi]n\s+oficial\s+de\s+la\s+provincia.*|"
    r"bolet[íi]n\s+oficial\s+del\s+principado\s+de\s+asturias.*|"
    r"n[úu]m\.?\s*\d+\s+de\s+\d{1,2}-[ivx]+-\d{4}.*|"
    r"c[óo]d\.?\s*20[12]\d-\d{4,6}.*|"
    r"verificable\s+en\s+https?://\S+|"
    r"sec\.\s+[ivx]+\.\s+p[áa]g\.\s*\d+|"
    r"n[úu]m(?:ero)?\.?\s*\d+\s*(?:•|-)?\s*p[áa]gina\s*\d+.*|"
    r"\[?n[úu]m(?:ero)?\.?\s*\d+\]?(?:\s*[-•]?\s*p[áa]gina\s*\d+)?|"
    r"n[úu]mero\s+de\s+anuncio.*|"
    r"p[áa]gina\s+\d+\s*/\s*\d+|"
    r"20[12]\d/\d{3,6}|"
    r"www\.dacoruna\.gal.*|"
    r"\**n[úu]m(?:ero)?\.?\s*\d+\b.*(?:fasc[íi]culo|sec\.\s+[ivx]+|p[áa]g\.?\s*\d+).*|"
    r"https?://\S+\s+d\.?\s*l\.?\s*:?\s*[a-z]{1,3}[\s-]*\d+.*|"
    r"fasc[íi]culo\s+\d+.*|"
    r"\d{1,2}\s+de\s+(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre|xaneiro|febreiro|maio|xu[ñn]o|xullo|setembro|outubro|novembro|decembro)\s+de\s+\d{4}|"
    r"boletin@dipuleon\.es|"
    r"bocm-\d+-\d+|"
    r"bocm|"
    r"b\.?\s*o\.?\s*p\.?(?:\s+d[ae]\b|\s+n[úu]m\.?\b).*|"
    r"bop|"
    r"n[º°]\.?\s*\d+\s*[-–—]\s*(?:lunes|martes|mi[ée]rcoles|jueves|viernes|s[áa]bado|domingo)\b.*|"
    r"#*\s*b\s*<mark>o</mark>\s*\*\*<mark>cm</mark>\*\*|"
    r"(?:lunes|luns|martes|mi[ée]rcoles|m[ée]rcores|jueves|xoves|viernes|venres|s[áa]bado|domingo),?\s+\d+\s+de\s+\w+\s+de\s+\d{4}|"
    r"===\s*document\s+parser\s+messages\s*===|"
    r"warning:\s*rapidocr\s+is\s+missing.*|"
    r"cve:\s*\d{8}-\d+-\d+.*|"
    r"cve:\s*bop[-a-z0-9]+.*|"
    r"n[úu]mero\s+\d{6,}|"
    r"centro\s+de\s+ediciones.*|"
    r"tel[ée]fonos?:\s*\d.*|"
    r"(?:e?k\\?cv\s+gao-[a-z0-9-]+.*)|"
    r"gipuzkoako\s+aldizkari\s+ofiziala.*|"
    r"bolet[íi]n\s+oficial\s+de\s+gipuzkoa.*|"
    r"edita:\s*deputaci[oó]n\s+de\s+pontevedra.*|"
    r"boppo|"
    r"n[úu]m\.|"
    r"pode\s+verificar\s+a\s+integridade.*|"
    r"https?://sede\.depo\.gal\S*|"
    r"c[óo]digo\s+seguro\s+de\s+verificaci[óo]n\s*:.*|"
    r"boc\.cantabria\.es.*|"
    r"cve-20[12]\d-\d{3,6}.*|"
    r"bopz|"
    r"firmado\s+de\s+conformidad\s+con\s+el\s+art\.\s*\d+\s+del\s+reglamento.*|"
    r"documento\s+firmado\s+electr[óo]nicamente\.?\s+verificable\s+en.*|"
    r"n[úu]m\.?\s*\d+|"
    r"\d{1,2}\s+(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre|xaneiro|febreiro|maio|xu[ñn]o|xullo|setembro|outubro|novembro|decembro)\s+\d{4}"
    r")$"
)

_STANDALONE_TABLE_SECTION_RE = re.compile(
    r"(?i)^#{1,3}\s+\**((?:SECCI[ÓO]N\s+(?:PRIMERA|SEGUNDA|TERCERA|CUARTA|QUINTA|SEXTA|S[ÉE]PTIMA|OCTAVA|NOVENA|\d+)|"
    r"ESTABLECIMIENTOS\s+DE\s+LA\s+SECCI[ÓO]N|"
    r"TABLAS?\s+SALARIAL(?:ES)?|"
    r"T[ÁA]BOAS?\s+SALARIA(?:IS|LES)|"
    r"CONVENIO\s+PROVINCIAL|"
    r"Conceptos\s+econ[óo]micos\s+no\s+incluidos\s+en\s+las\s+tablas\s+salariales|"
    r"Desde\s+\d{2}/\d{2}/202\d)[^\n]*?)\**$"
)

_CHAPTER_OR_TITLE_RE = re.compile(
    r"(?i)^(?:(?:#{1,4}\s*)?\**((?:(?:T[ÍI]TULO|CAP[ÍI]TULO|ANEXO|PROTOCOLO)\s+(?!d[ée]cimo\b|del?\s+presente\s+convenio\b)[IVXLCDM0-9]+(?!\s+del\s+convenio\s+colectivo,)|TABLAS?\s+DE\s+RETRIBUCIONES\b|T[ÁA]BOAS?\s+SALARIA(?:IS|LES)\.?\b)[^\n]*?)\**|"
    r"#{1,4}\s+\**((?:A\s*N\s*E\s*X\s*O(?:\s+(?:[ÚU]NICO|GRUPOS\s+PROFESIONALES))?|DISPOSICIONES\s+(?:ADICIONALES|TRANSITORIAS|DEROGATORIAS|FINALES)))\**)$"
)

_TABLE_YEAR_OR_PROVINCE_RE = re.compile(
    r"(?i)^(?:#{1,4}\s*)?\**(?:TABLAS?\s+SALARIAL(?:ES)?\s+(?:A[ÑN]OS?\s+(?:DEL\s+\d{2}/\d{2}/)?|PARA\s+EL\s+A[ÑN]O\s+)?2\.?02[3-9][^\n]*|T[ÁA]BOAS?\s+SALARIA(?:IS|LES)\s+2\.?02[3-9][^\n]*|SALARIO\s+BASE\s+(?:MENSUAL|MANSUAL|ANUAL)\s+2\.?02[3-9][^\n]*|HORAS?\s+EXTRAS?\s+2\.?02[3-9][^\n]*|INCREMENTO\s+2\.?02[3-9][^\n]*|[A-D]\)\s*(?:Barcelona|Tarragona|Girona|Lleida)[^\n]*)\**$"
)

_ARTICLE_START_RE = re.compile(
    r"(?i)^(?:#{1,4}\s*)?\**((?:Art[íi]culo|Artigo|Article|Art\.)(?:\.-|\.)?\s*\d+(?:\s*(?:º|\.º))?(?:\s*(?:bis|ter|quater))?(?:\s*$|(?:(?:\s*\.|\s*[-–—]|(?<=\bbis):|(?<=\bter):|(?<=\bquater):))\s*(?!(?:de\s+la\s+Ley|del\s+Estatuto|del\s+Real\s+Decreto))[^\n]*?))\**$"
)

_DISPOSITION_START_RE = re.compile(
    r"(?i)^(?:#{1,4}\s*)?\**(Disposici[óo]n\s+(?:adicional|transitoria|derogatoria|final)\s+(?![^\n]{0,40},\s+en\s+(?:el\s+que|la\s+que|los\s+que)\b)[^\n]+?)\**$"
)

_TOC_HEADER_RE = re.compile(
    r"(?i)^(?:#{1,4}\s*)?\**(?:[ÍI]NDICE(?:\s+SISTEM[ÁA]TICO|\s+GENERAL)?|SUMARIO|TABLA\s+DE\s+CONTENIDOS?)\**\s*$"
)

_MARKDOWN_SUBHEADING_RE = re.compile(r"^#{2,4}\s+(.+)$")


AUTONOMOUS_COMMUNITY_EXPANSION: dict[str, list[str]] = {
    "catalunya": ["barcelona", "girona", "tarragona", "lleida", "catalunya"],
}

_CANONICAL_TERRITORY_MAP: dict[str, str] = {
    "la coruna": "a_coruna",
    "a coruna": "a_coruna",
    "coruna": "a_coruna",
    "santiago": "a_coruna",
    "santiago de compostela": "a_coruna",
    "compostela": "a_coruna",
    "ferrol": "a_coruna",
    "naron": "a_coruna",
    "carballo": "a_coruna",
    "oleiros": "a_coruna",
    "arteixo": "a_coruna",
    "culleredo": "a_coruna",
    "ames": "a_coruna",
    "ribeira": "a_coruna",
    "betanzos": "a_coruna",
    "cambre": "a_coruna",
    "boiro": "a_coruna",
    "noia": "a_coruna",
    "araba": "alava",
    "araba/alava": "alava",
    "araba / alava": "alava",
    "alava/araba": "alava",
    "vitoria": "alava",
    "vitoria-gasteiz": "alava",
    "vitoria gasteiz": "alava",
    "gasteiz": "alava",
    "llodio": "alava",
    "laudio": "alava",
    "amurrio": "alava",
    "alacant": "alicante",
    "roquetas de mar": "almeria",
    "roquetas": "almeria",
    "el ejido": "almeria",
    "ejido": "almeria",
    "nijar": "almeria",
    "vicar": "almeria",
    "adra": "almeria",
    "huercal-overa": "almeria",
    "huercal overa": "almeria",
    "huercal de almeria": "almeria",
    "vera": "almeria",
    "cuevas del almanzora": "almeria",
    "berja": "almeria",
    "garrucha": "almeria",
    "mojacar": "almeria",
    "carboneras": "almeria",
    "albox": "almeria",
    "pulpi": "almeria",
    "olula del rio": "almeria",
    "macael": "almeria",
    "tabernas": "almeria",
    "sorbas": "almeria",
    "finana": "almeria",
    "laujar de andarax": "almeria",
    "alhama de almeria": "almeria",
    "pechina": "almeria",
    "benahadux": "almeria",
    "gador": "almeria",
    "viator": "almeria",
    "almerimar": "almeria",
    "cabo de gata": "almeria",
    "retamar": "almeria",
    "san juan de los terreros": "almeria",
    "velez-rubio": "almeria",
    "velez rubio": "almeria",
    "velez-blanco": "almeria",
    "velez blanco": "almeria",
    "tijola": "almeria",
    "cantoria": "almeria",
    "arboleas": "almeria",
    "turre": "almeria",
    "antas": "almeria",
    "los gallardos": "almeria",
    "seron": "almeria",
    "purchena": "almeria",
    "dalias": "almeria",
    "illes balears": "baleares",
    "islas baleares": "baleares",
    "balears": "baleares",
    "mallorca": "baleares",
    "menorca": "baleares",
    "ibiza": "baleares",
    "eivissa": "baleares",
    "formentera": "baleares",
    "bcn": "barcelona",
    "catalunya bcn": "catalunya",
    "comunidad autonoma de cantabria": "cantabria",
    "santander": "cantabria",
    "torrelavega": "cantabria",
    "castro urdiales": "cantabria",
    "castro-urdiales": "cantabria",
    "camargo": "cantabria",
    "maliano": "cantabria",
    "pielagos": "cantabria",
    "el astillero": "cantabria",
    "astillero": "cantabria",
    "laredo": "cantabria",
    "santona": "cantabria",
    "los corrales de buelna": "cantabria",
    "corrales de buelna": "cantabria",
    "santa cruz de bezana": "cantabria",
    "bezana": "cantabria",
    "reinosa": "cantabria",
    "suances": "cantabria",
    "colindres": "cantabria",
    "cabezon de la sal": "cantabria",
    "santillana del mar": "cantabria",
    "san vicente de la barquera": "cantabria",
    "comillas": "cantabria",
    "potes": "cantabria",
    "noja": "cantabria",
    "isla": "cantabria",
    "arnuero": "cantabria",
    "medio cudeyo": "cantabria",
    "solares": "cantabria",
    "marina de cudeyo": "cantabria",
    "pedrena": "cantabria",
    "somo": "cantabria",
    "ribamontan al mar": "cantabria",
    "ampuero": "cantabria",
    "ramales de la victoria": "cantabria",
    "castello": "castellon",
    "ciudad real": "ciudad_real",
    "lucena": "cordoba",
    "puente genil": "cordoba",
    "montilla": "cordoba",
    "priego de cordoba": "cordoba",
    "cabra": "cordoba",
    "baena": "cordoba",
    "palma del rio": "cordoba",
    "pozoblanco": "cordoba",
    "aguilar de la frontera": "cordoba",
    "penarroya-pueblonuevo": "cordoba",
    "penarroya pueblonuevo": "cordoba",
    "rute": "cordoba",
    "fernan nunez": "cordoba",
    "fernan-nunez": "cordoba",
    "montoro": "cordoba",
    "villanueva de cordoba": "cordoba",
    "almodovar del rio": "cordoba",
    "la carlota": "cordoba",
    "bujalance": "cordoba",
    "castro del rio": "cordoba",
    "posadas": "cordoba",
    "hornachuelos": "cordoba",
    "hinojosa del duque": "cordoba",
    "nueva carteya": "cordoba",
    "la rambla": "cordoba",
    "espejo": "cordoba",
    "zuheros": "cordoba",
    "iznajar": "cordoba",
    "fuente palmera": "cordoba",
    "el carpio": "cordoba",
    "villa del rio": "cordoba",
    "belmez": "cordoba",
    "espiel": "cordoba",
    "santaella": "cordoba",
    "montemayor": "cordoba",
    "benameji": "cordoba",
    "adamuz": "cordoba",
    "villafranca de cordoba": "cordoba",
    "gerona": "girona",
    "gipuzkoa": "guipuzcoa",
    "donostia": "guipuzcoa",
    "san sebastian": "guipuzcoa",
    "donostia-san sebastian": "guipuzcoa",
    "donostia san sebastian": "guipuzcoa",
    "la rioja": "la_rioja",
    "las palmas": "las_palmas",
    "las palmas de gran canaria": "las_palmas",
    "gran canaria": "las_palmas",
    "lanzarote": "las_palmas",
    "fuerteventura": "las_palmas",
    "lerida": "lleida",
    "comunidad de madrid": "madrid",
    "principado de asturias": "asturias",
    "oviedo": "asturias",
    "gijon": "asturias",
    "xixon": "asturias",
    "aviles": "asturias",
    "siero": "asturias",
    "pola de siero": "asturias",
    "langreo": "asturias",
    "la felguera": "asturias",
    "sama": "asturias",
    "mieres": "asturias",
    "castrillon": "asturias",
    "piedras blancas": "asturias",
    "san martin del rey aurelio": "asturias",
    "corvera": "asturias",
    "villaviciosa": "asturias",
    "llanes": "asturias",
    "cangas del narcea": "asturias",
    "cangas de onis": "asturias",
    "ribadesella": "asturias",
    "navia": "asturias",
    "luarca": "asturias",
    "valdes": "asturias",
    "grado": "asturias",
    "tineo": "asturias",
    "pravia": "asturias",
    "carreno": "asturias",
    "candas": "asturias",
    "gozon": "asturias",
    "luanco": "asturias",
    "laviana": "asturias",
    "lena": "asturias",
    "aller": "asturias",
    "cudillero": "asturias",
    "region de murcia": "murcia",
    "nafarroa": "navarra",
    "comunidad foral de navarra": "navarra",
    "pamplona": "navarra",
    "iruna": "navarra",
    "tudela": "navarra",
    "baranain": "navarra",
    "valle de egues": "navarra",
    "egues": "navarra",
    "burlada": "navarra",
    "burlata": "navarra",
    "estella": "navarra",
    "estella-lizarra": "navarra",
    "estella lizarra": "navarra",
    "lizarra": "navarra",
    "zizur mayor": "navarra",
    "tafalla": "navarra",
    "ansoain": "navarra",
    "antsoain": "navarra",
    "villava": "navarra",
    "atarrabia": "navarra",
    "berriozar": "navarra",
    "aranguren": "navarra",
    "mutilva": "navarra",
    "noain": "navarra",
    "cintruenigo": "navarra",
    "baztan": "navarra",
    "elizondo": "navarra",
    "corella": "navarra",
    "alsasua": "navarra",
    "altsasu": "navarra",
    "sanguesa": "navarra",
    "olite": "navarra",
    "erriberri": "navarra",
    "lodosa": "navarra",
    "san adrian": "navarra",
    "berrioplano": "navarra",
    "cascante": "navarra",
    "mendavia": "navarra",
    "huarte": "navarra",
    "uharte": "navarra",
    "orense": "ourense",
    "vigo": "pontevedra",
    "vilagarcia": "pontevedra",
    "vilagarcia de arousa": "pontevedra",
    "villagarcia": "pontevedra",
    "redondela": "pontevedra",
    "cangas": "pontevedra",
    "marin": "pontevedra",
    "ponteareas": "pontevedra",
    "a estrada": "pontevedra",
    "lalin": "pontevedra",
    "o porrino": "pontevedra",
    "porrino": "pontevedra",
    "sanxenxo": "pontevedra",
    "o grove": "pontevedra",
    "baiona": "pontevedra",
    "nigran": "pontevedra",
    "tui": "pontevedra",
    "cambados": "pontevedra",
    "poio": "pontevedra",
    "moana": "pontevedra",
    "bueu": "pontevedra",
    "caldas de reis": "pontevedra",
    "mos": "pontevedra",
    "gondomar": "pontevedra",
    "vilanova de arousa": "pontevedra",
    "a guarda": "pontevedra",
    "tomino": "pontevedra",
    "santa cruz de tenerife": "santa_cruz_tenerife",
    "tenerife": "santa_cruz_tenerife",
    "la palma": "santa_cruz_tenerife",
    "la gomera": "santa_cruz_tenerife",
    "el hierro": "santa_cruz_tenerife",
    "bizkaia": "vizcaya",
    "calatayud": "zaragoza",
    "utebo": "zaragoza",
    "ejea de los caballeros": "zaragoza",
    "ejea": "zaragoza",
    "tarazona": "zaragoza",
    "caspe": "zaragoza",
    "la almunia de dona godina": "zaragoza",
    "la almunia": "zaragoza",
    "tauste": "zaragoza",
    "cuarte de huerva": "zaragoza",
    "zuera": "zaragoza",
    "alagon": "zaragoza",
    "la puebla de alfinden": "zaragoza",
    "borja": "zaragoza",
    "epila": "zaragoza",
    "fuentes de ebro": "zaragoza",
    "maria de huerva": "zaragoza",
    "villanueva de gallego": "zaragoza",
    "mallen": "zaragoza",
    "carinena": "zaragoza",
    "daroca": "zaragoza",
    "pedrola": "zaragoza",
    "pinseque": "zaragoza",
    "cadrete": "zaragoza",
    "figueruelas": "zaragoza",
    "pina de ebro": "zaragoza",
    "belchite": "zaragoza",
    "sos del rey catolico": "zaragoza",
    "alhama de aragon": "zaragoza",
    "jaraba": "zaragoza",
    "nuevalos": "zaragoza",
    "cataluna": "catalunya",
}


_PRETTY_TERRITORY_TITLES: dict[str, str] = {
    "a_coruna": "A Coruña",
    "alava": "Álava",
    "almeria": "Almería",
    "avila": "Ávila",
    "caceres": "Cáceres",
    "cadiz": "Cádiz",
    "castellon": "Castellón",
    "ciudad_real": "Ciudad Real",
    "cordoba": "Córdoba",
    "guipuzcoa": "Guipúzcoa",
    "jaen": "Jaén",
    "la_rioja": "La Rioja",
    "las_palmas": "Las Palmas",
    "malaga": "Málaga",
    "santa_cruz_tenerife": "Santa Cruz de Tenerife",
}


def _canonicalize_territory(raw_terr: str) -> str:
    """Normaliza cualquier nombre territorial al identificador canónico usado en Qdrant y el router."""
    nfkd = unicodedata.normalize("NFKD", raw_terr.lower().replace("_", " "))
    cleaned = "".join(c for c in nfkd if not unicodedata.combining(c)).strip()
    cleaned = re.sub(r"^(?:provincia\s+de\s+|comunidad\s+de\s+)", "", cleaned).strip()
    if cleaned in _CANONICAL_TERRITORY_MAP:
        return _CANONICAL_TERRITORY_MAP[cleaned]
    return cleaned.replace(" ", "_")


def _infer_document_metadata(
    md_path: Path,
    boundaries_cache: dict[str, dict],
) -> tuple[str, NormativeLayer, str, list[str]]:
    """Infiere título, capa normativa, territorio y provincias aplicables de forma agnóstica."""
    pdf_name = f"{md_path.stem}.pdf"
    cached = boundaries_cache.get(pdf_name, {})
    doc_type = cached.get("doc_type", "")
    territorio = (cached.get("territorio") or "").lower().strip()

    stem_lower = md_path.stem.lower()
    if not doc_type:
        if "estatuto" in stem_lower:
            doc_type = "estatuto_trabajadores"
            territorio = "estatal"
        elif "aleh" in stem_lower:
            doc_type = "aleh_estatal"
            territorio = "estatal"
        else:
            doc_type = "convenio_hosteleria"
            territorio = stem_lower.replace("convenio_", "").replace("_texto", "")

    if doc_type == "estatuto_trabajadores":
        return (
            "Estatuto de los Trabajadores (RDL 2/2015)",
            "estatuto_trabajadores",
            "estatal",
            ["estatal"],
        )
    if doc_type == "aleh_estatal":
        return (
            "VI Acuerdo Laboral Estatal de Hostelería (ALEH VI)",
            "aleh_estatal",
            "estatal",
            ["estatal"],
        )

    if territorio and territorio != "estatal":
        territorio = _canonicalize_territory(territorio)

    # Convenio provincial o autonómico multiprovincial
    if territorio in {"catalunya", "cataluna", "barcelona", "catalunya_bcn"}:
        return (
            "Convenio Colectivo Interprovincial de Hostelería y Turismo de Cataluña",
            "convenio_provincial",
            "catalunya",
            ["barcelona", "girona", "tarragona", "lleida", "catalunya"],
        )
    if territorio == "madrid":
        return (
            "Convenio Colectivo de Hostelería y Actividades Turísticas de la Comunidad de Madrid",
            "convenio_provincial",
            "madrid",
            ["madrid"],
        )
    if territorio == "leon":
        return (
            "Convenio Colectivo Provincial de Hostelería y Turismo de León",
            "convenio_provincial",
            "leon",
            ["leon"],
        )

    pretty_terr = _PRETTY_TERRITORY_TITLES.get(territorio, territorio.replace("_", " ").title())
    provincias_aplicables = list(AUTONOMOUS_COMMUNITY_EXPANSION.get(territorio, [territorio]))
    subsector = (cached.get("subsector") or "general").lower().strip()
    if subsector == "alojamiento":
        norma_title = f"Convenio Colectivo de Alojamientos de {pretty_terr}"
    elif subsector == "restauracion":
        norma_title = f"Convenio Colectivo de Empresas de Restauración de {pretty_terr}"
    else:
        norma_title = f"Convenio Colectivo de Hostelería de {pretty_terr}"
    return (
        norma_title,
        "convenio_provincial",
        territorio,
        provincias_aplicables,
    )


def _strip_toc_index_block(lines: list[str]) -> list[str]:
    """Elimina un bloque inicial de Índice/Sumario para evitar stubs vacíos sin texto articulado."""
    result: list[str] = []
    in_toc = False
    seen_first_chapter_in_toc = False
    idx = 0

    while idx < len(lines):
        stripped = lines[idx].strip()
        if not in_toc and idx < 400 and _TOC_HEADER_RE.match(stripped):
            in_toc = True
            idx += 1
            continue

        if in_toc:
            if re.match(
                r"(?i)^(?:#{1,4}\s*)?\**(?:CAP[ÍI]TULO|T[ÍI]TULO)\s+(?:I\b|1\b|PRIMERO\b)",
                stripped,
            ):
                if not seen_first_chapter_in_toc:
                    seen_first_chapter_in_toc = True
                    idx += 1
                    continue
                in_toc = False
                result.append(lines[idx])
                idx += 1
                continue

            if (
                len(stripped) >= 130
                and not stripped.startswith("#")
                and ("." in stripped or "," in stripped)
                and not stripped.isupper()
            ):
                in_toc = False
                result.append(lines[idx])
                idx += 1
                continue

            idx += 1
            continue

        result.append(lines[idx])
        idx += 1

    return result


def _clean_markdown_lines(raw_text: str) -> list[str]:
    """Limpia líneas residuales de numeración o cabeceras de boletín oficial y normaliza espacios Unicode."""
    normalized = (
        raw_text.replace("\u2003", " ")
        .replace("\u2002", " ")
        .replace("\u00a0", " ")
    )
    normalized = re.sub(r"(?<!\d)(\d{1,2})(?:\.\s+|\s+\.)(\d{3},\d{2})\b", r"\1.\2", normalized)
    cleaned: list[str] = []
    for line in normalized.splitlines():
        stripped = line.strip()
        if _RESIDUAL_NOISE_LINE.match(stripped):
            continue
        cleaned.append(line)
    return _strip_toc_index_block(cleaned)


def _split_large_block(
    body_lines: list[str],
    max_chars: int = 2600,
    overlap_lines: int = 2,
) -> list[str]:
    """Divide un bloque extenso respetando tablas Markdown (repitiendo su cabecera) y párrafos."""
    full_text = "\n".join(body_lines).strip()
    if len(full_text) <= max_chars:
        return [full_text] if full_text else []

    chunks: list[str] = []
    current_lines: list[str] = []
    current_len = 0
    active_table_header: list[str] = []

    for line in body_lines:
        stripped = line.strip()
        is_table_line = stripped.startswith("|") and stripped.endswith("|")

        if is_table_line:
            if len(active_table_header) < 2:
                active_table_header.append(line)
        else:
            if stripped:
                active_table_header = []

        line_len = len(line) + 1
        if current_len + line_len > max_chars and current_lines:
            chunks.append("\n".join(current_lines).strip())
            if is_table_line and len(active_table_header) == 2:
                current_lines = list(active_table_header)
                current_len = sum(len(h) + 1 for h in current_lines)
            else:
                tail = current_lines[-overlap_lines:] if overlap_lines > 0 else []
                current_lines = list(tail)
                current_len = sum(len(t) + 1 for t in current_lines)

        current_lines.append(line)
        current_len += line_len

    if current_lines:
        chunks.append("\n".join(current_lines).strip())

    return [c for c in chunks if c]


def chunk_markdown_document(
    md_path: Path,
    boundaries_cache: dict[str, dict] | None = None,
    max_chunk_chars: int = 2600,
) -> list[LegalChunk]:
    """Fragmenta un documento Markdown normativo en objetos LegalChunk jerárquicos."""
    cache = boundaries_cache or {}
    norma_titulo, capa_normativa, territorio, provincias = _infer_document_metadata(
        md_path, cache
    )
    cached_meta = cache.get(f"{md_path.stem}.pdf", {})
    doc_subsector = (cached_meta.get("subsector") or "general").lower().strip()
    doc_role_meta = (cached_meta.get("doc_role") or "").lower().strip()
    expected_years_meta = cached_meta.get("expected_years") or []
    single_tablas_year = (
        int(expected_years_meta[0])
        if doc_role_meta == "tablas" and isinstance(expected_years_meta, list) and len(expected_years_meta) == 1
        else None
    )

    raw_text = md_path.read_text(encoding="utf-8")
    lines = _clean_markdown_lines(raw_text)
    is_table_doc = "tabla" in md_path.stem.lower()
    in_table_block = False

    current_base_chapter = "Disposiciones Generales / Preámbulo"
    current_chapter = current_base_chapter
    current_article = "Preámbulo / Encabezado"
    current_buffer: list[str] = []

    raw_sections: list[tuple[str, str, list[str]]] = []

    def flush_section() -> None:
        nonlocal current_buffer
        non_empty = [l for l in current_buffer if l.strip()]
        if non_empty:
            raw_sections.append((current_chapter, current_article, list(current_buffer)))
        current_buffer = []

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()

        chap_match = _CHAPTER_OR_TITLE_RE.match(stripped)
        if chap_match:
            in_table_block = False
            flush_section()
            chap_title = re.sub(r"^#{1,4}\s*", "", stripped).strip("* ")
            lookahead = idx + 1
            while lookahead < len(lines) and not lines[lookahead].strip():
                lookahead += 1
            if lookahead < len(lines):
                next_str = lines[lookahead].strip()
                clean_next = re.sub(r"^#{1,4}\s*", "", next_str).strip("* ")
                is_h1_bold_sub = next_str.startswith("# **")
                if (
                    clean_next
                    and len(clean_next) < 90
                    and (lookahead == idx + 1 or is_h1_bold_sub)
                    and (is_h1_bold_sub or not next_str.startswith("#"))
                    and not next_str.startswith("|")
                    and not _ARTICLE_START_RE.match(next_str)
                    and not _CHAPTER_OR_TITLE_RE.match(next_str)
                    and not _DISPOSITION_START_RE.match(next_str)
                    and not re.match(
                        r"(?i)^(?:Primera|Segunda|Tercera|Cuarta|Quinta|Sexta|S[ée]ptima|Octava|Novena|D[ée]cima|\d+[ªº])\.",
                        clean_next,
                    )
                ):
                    chap_title = f"{chap_title} - {clean_next}"
                    idx = lookahead
            current_base_chapter = chap_title
            current_chapter = chap_title
            current_article = chap_title
            idx += 1
            continue

        if _TABLE_YEAR_OR_PROVINCE_RE.match(stripped):
            in_table_block = True
            flush_section()
            sub_ctx = re.sub(r"^#{1,4}\s*", "", stripped).strip("* ")
            current_chapter = f"{current_base_chapter} | {sub_ctx}"
            current_article = sub_ctx
            current_buffer.append(f"## {sub_ctx}")
            idx += 1
            continue

        table_sec_match = (
            _STANDALONE_TABLE_SECTION_RE.match(stripped)
            if (is_table_doc or current_base_chapter == "Disposiciones Generales / Preámbulo")
            else None
        )
        if table_sec_match:
            in_table_block = True
            flush_section()
            sec_title = re.sub(r"^#{1,4}\s*", "", stripped).strip("* ")
            lookahead = idx + 1
            appended_lines = [f"## {sec_title}"]
            for _ in range(3):
                while lookahead < len(lines) and not lines[lookahead].strip():
                    lookahead += 1
                if lookahead >= len(lines):
                    break
                next_str = lines[lookahead].strip()
                clean_next = re.sub(r"^#{1,4}\s*", "", next_str).strip("* ")
                if (
                    not clean_next
                    or len(clean_next) >= 120
                    or next_str.startswith("|")
                    or _CHAPTER_OR_TITLE_RE.match(next_str)
                    or _DISPOSITION_START_RE.match(next_str)
                    or _ARTICLE_START_RE.match(clean_next)
                    or re.match(
                        r"(?i)^(?:SECCI[ÓO]N\s+(?:PRIMERA|SEGUNDA|TERCERA|CUARTA|QUINTA|SEXTA|S[ÉE]PTIMA|OCTAVA|NOVENA|\d+)|ESTABLECIMIENTOS\s+DE\s+LA\s+SECCI[ÓO]N|CONVENIO\s+PROVINCIAL)\b",
                        clean_next,
                    )
                ):
                    break
                if next_str.startswith("#") or re.match(
                    r"(?i)^(?:Desde\s+\d{2}/\d{2}/202\d|Del\s+\d{1,2}\s+de\s+|Salarios\s+|Personal\s+|Conceptos\s+econ[óo]micos|Tablas?\s+salarial)",
                    clean_next,
                ):
                    sec_title = f"{sec_title} | {clean_next}"
                    appended_lines.append(next_str if next_str.startswith("#") else f"### {clean_next}")
                    idx = lookahead
                    lookahead += 1
                else:
                    break
            current_chapter = (
                sec_title
                if "tabla" in sec_title.lower()
                else f"Tablas Salariales | {sec_title}"
            )
            current_article = sec_title
            current_buffer.extend(appended_lines)
            idx += 1
            continue

        art_match = None if (is_table_doc and in_table_block) else _ARTICLE_START_RE.match(stripped)
        if art_match:
            flush_section()
            art_title = re.sub(r"^#{1,4}\s*", "", stripped).strip("* ")
            if re.fullmatch(r"(?i)(?:Art[íi]culo|Artigo|Article|Art\.)\s*\d+(?:\.º|º)?\.?", art_title):
                lookahead = idx + 1
                while lookahead < len(lines) and not lines[lookahead].strip():
                    lookahead += 1
                if lookahead < len(lines):
                    rubrica = lines[lookahead].strip()
                    if (
                        len(rubrica) < 100
                        and not rubrica.startswith("#")
                        and not rubrica.startswith("|")
                    ):
                        art_title = f"{art_title}. {rubrica}"
                        idx = lookahead
            current_article = art_title
            current_buffer.append(f"### {art_title}")
            idx += 1
            continue

        disp_match = _DISPOSITION_START_RE.match(stripped)
        if disp_match:
            in_table_block = False
            flush_section()
            raw_disp = re.sub(r"^#{1,4}\s*", "", stripped).strip()
            inline_bold = re.match(
                r"^\*\*(Disposici[óo]n\s+[^*\n]+?)\*\*(?:\s*[-–—]?\s*(.+))?$",
                raw_disp,
                flags=re.IGNORECASE,
            )
            if inline_bold and inline_bold.group(2):
                disp_title = inline_bold.group(1).strip("* ")
                current_article = disp_title
                current_buffer.append(raw_disp)
            else:
                disp_title = raw_disp.strip("* ")
                current_article = disp_title
                current_buffer.append(f"### {disp_title}")
            idx += 1
            continue

        sub_match = _MARKDOWN_SUBHEADING_RE.match(stripped)
        if sub_match and (is_table_doc or "anexo" in current_base_chapter.lower() or "tabla" in current_chapter.lower()):
            sub_title = sub_match.group(1).strip("* ")
            if is_table_doc and in_table_block and _ARTICLE_START_RE.match(sub_title):
                current_buffer.append(line)
                idx += 1
                continue
            if re.match(r"(?i)^CLASIFICACI[ÓO]N\s+(?:[IVXLCDM]+|\d+[A-Z]?)\b", sub_title):
                subsector_label = re.sub(
                    r"(?i)\s*(?:[-–—.]\s*)?A\.?F\.?\s+1[ªa]?\b.*$",
                    "",
                    sub_title,
                ).strip(" .-–—")
                if subsector_label:
                    if sum(len(x) for x in current_buffer) >= 600:
                        flush_section()
                    current_chapter = f"{current_base_chapter} | {subsector_label}"
                    current_article = sub_title
                    current_buffer.append(line)
                    idx += 1
                    continue
            if (
                re.fullmatch(r"(?i)NIVEL\s+[IVX0-9]+", sub_title)
                and any(x.strip().startswith("|") for x in current_buffer)
                and sum(len(x) for x in current_buffer) < 2000
            ):
                current_buffer.append(line)
                idx += 1
                continue
            # Si el buffer actual ya tiene suficiente contenido (>600 chars), abrimos sub-sección nueva
            if sum(len(x) for x in current_buffer) >= 600:
                flush_section()
                current_article = sub_title
            elif current_article == current_base_chapter:
                current_article = sub_title
            current_buffer.append(line)
            idx += 1
            continue

        current_buffer.append(line)
        idx += 1

    flush_section()

    legal_chunks: list[LegalChunk] = []
    chunk_counter = 0

    for chapter, article, sec_lines in raw_sections:
        sub_blocks = _split_large_block(sec_lines, max_chars=max_chunk_chars)
        total_parts = len(sub_blocks)

        for part_idx, sub_text in enumerate(sub_blocks, start=1):
            pipe_count = sum(
                1 for l in sub_text.splitlines() if l.strip().startswith("|")
            )
            if pipe_count >= 3:
                tipo_contenido: ContentType = "tabla_salarial"
            elif (
                "anexo" in chapter.lower()
                or "disposición" in article.lower()
                or "disposicion" in article.lower()
                or "disposiciones" in article.lower()
            ):
                tipo_contenido = "anexo_o_disposicion"
            else:
                tipo_contenido = "articulo"

            part_suffix = f" (Parte {part_idx}/{total_parts})" if total_parts > 1 else ""
            art_label = f"{article}{part_suffix}"
            chapter_label = chapter
            if (
                single_tablas_year is not None
                and re.search(r"(?i)anexo|tabla", chapter_label)
                and not re.search(r"\b202\d\b", chapter_label)
            ):
                chapter_label = f"{chapter_label} | {single_tablas_year}"

            context_header = (
                f"[NORMA: {norma_titulo} | CAPA: {capa_normativa} | "
                f"TERRITORIO: {territorio.upper()} | SECCIÓN: {chapter_label} | "
                f"REFERENCIA: {art_label}]"
            )
            enriched_text = f"{context_header}\n\n{sub_text}"

            deterministic_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_OID,
                    f"{md_path.name}::{chapter}::{art_label}::{chunk_counter}",
                )
            )

            legal_chunks.append(
                LegalChunk(
                    chunk_id=deterministic_id,
                    source_file=md_path.name,
                    norma_titulo=norma_titulo,
                    capa_normativa=capa_normativa,
                    territorio=territorio,
                    subsector=doc_subsector,
                    provincias_aplicables=provincias,
                    titulo_o_capitulo=chapter_label,
                    articulo=art_label,
                    tipo_contenido=tipo_contenido,
                    chunk_index=chunk_counter,
                    text=enriched_text,
                )
            )
            chunk_counter += 1

    logger.info(
        "Documento %s fragmentado en %d chunks jurídicos (tablas=%d, artículos=%d, anexos=%d).",
        md_path.name,
        len(legal_chunks),
        sum(1 for c in legal_chunks if c.tipo_contenido == "tabla_salarial"),
        sum(1 for c in legal_chunks if c.tipo_contenido == "articulo"),
        sum(1 for c in legal_chunks if c.tipo_contenido == "anexo_o_disposicion"),
    )
    return legal_chunks


def chunk_all_processed_documents(
    processed_dir: Path = PROCESSED_DATA_DIR,
    max_chunk_chars: int = 2600,
) -> list[LegalChunk]:
    """Fragmenta todos los archivos .md en data/processed/ devolviendo la lista consolidada de LegalChunk."""
    cache_path = processed_dir / "boundaries_cache.json"
    boundaries_cache: dict[str, dict] = {}
    if cache_path.exists():
        try:
            boundaries_cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            boundaries_cache = {}

    all_chunks: list[LegalChunk] = []
    for md_path in sorted(processed_dir.glob("*.md")):
        doc_chunks = chunk_markdown_document(
            md_path,
            boundaries_cache=boundaries_cache,
            max_chunk_chars=max_chunk_chars,
        )
        all_chunks.extend(doc_chunks)

    logger.info("Total de chunks generados en el corpus: %d", len(all_chunks))
    return all_chunks
