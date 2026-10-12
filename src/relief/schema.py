"""
relief.schema
=============
Output schema for relief analysis.

Language models on Replicate offer no structured output, so this schema is not
sent as `response_format`. Instead:
  1. the schema is rendered to JSON Schema and inserted into the system prompt
     (see `build_system_prompt` in relief/prompts.py), and
  2. the model's text output is validated locally with
     `AnalisisRelief.model_validate`.

Conventions carried over from OpenAI strict mode, because they still help:
  - No field has a `default`. Every field must be present; optional fields are
    typed Optional[...] and may be null. Output that passes validation is
    guaranteed complete, just as in strict mode.
  - Controlled vocabularies (Literal) are kept. Without strict mode the model can
    step outside an enum; Pydantic rejects it and the call is retried with the
    error as feedback (`max_repairs` in relief.analyze).
  - The schema is kept compact: the whole JSON Schema counts as input tokens on
    every call.
"""

from typing import Any, List, Literal, Optional, get_args

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------
# vocabulary tolerance
# --------------------------------------------------------------------------
#
# Models often answer outside the controlled vocabulary with sensible values:
# "Hindu-Buddha" for tradition, "duduk_bersimpuh" for posture. Previously one
# such value rejected the WHOLE record, even though none of these fields is
# read by layer-1 scoring (`parse_predictions` only uses id, kategori, bbox,
# sel_grid, keyakinan, sumber_bukti, jumlah). That lost 10% of the grid
# without protecting any number.
#
# So unknown values are mapped to a fallback instead of failing the record.
# Two guarantees:
#   1. `mode="before"` does NOT change the JSON Schema, so the system prompt
#      stays byte-identical and dev/main results remain comparable.
#   2. The original value is not lost: it stays in attempt-*.parsed.json.

def _in_vocabulary(value: Any, enum_type: Any) -> bool:
    return value in get_args(enum_type)

# --------------------------------------------------------------------------
# controlled vocabularies
# --------------------------------------------------------------------------

Kategori = Literal[
    "figur_manusia", "figur_dewa", "figur_pendeta", "figur_bangsawan",
    "figur_hewan", "figur_mitologis", "benda_bawaan", "wadah", "senjata",
    "perabot", "arsitektur", "tumbuhan", "ornamen", "teks_aksara",
    "elemen_batu", "tidak_teridentifikasi",
]

Predikat = Literal[
    # actions
    "memegang", "menyentuh", "memberi_kepada", "menerima_dari", "menunjuk_ke",
    "memakai", "menaiki", "menyembah_kepada", "menghadap", "berbicara_kepada",
    "menuang_ke",
    # spatial
    "di_atas", "di_bawah", "di_kiri", "di_kanan", "di_depan", "di_belakang",
    "di_dalam", "berdampingan", "tumpang_tindih", "mengelilingi", "menaungi",
    "bagian_dari",
]

Postur = Literal[
    "duduk_bersila", "duduk_berlutut", "duduk_kaki_menjuntai", "berdiri",
    "berjalan", "berbaring", "membungkuk", "tidak_terbaca",
]

ArahHadap = Literal[
    "kiri", "kanan", "depan", "serong_kiri", "serong_kanan", "tidak_terbaca",
]

SumberBukti = Literal["RGB", "DEPTH", "FUSION", "CAVITY"]

Skenario = Literal["RGB_SAJA", "RGB_DEPTH", "RGB_FUSION_CAVITY"]


# --------------------------------------------------------------------------
# layer 1: object inventory
# --------------------------------------------------------------------------

class AtributObjek(BaseModel):
    postur: Postur
    arah_hadap: ArahHadap
    hiasan_kepala: Optional[str] = Field(description="null jika tidak terlihat")
    perhiasan: List[str] = Field(description="daftar kosong jika tidak terlihat")
    pakaian: Optional[str] = Field(description="null jika tidak terlihat")
    benda_dipegang: List[str] = Field(description="daftar kosong jika tidak ada")
    sikap_tangan: Optional[str] = Field(description="mudra atau gestur, null jika tidak terbaca")

    @field_validator("postur", mode="before")
    @classmethod
    def _lenient_posture(cls, v):
        return v if _in_vocabulary(v, Postur) else "tidak_terbaca"

    @field_validator("arah_hadap", mode="before")
    @classmethod
    def _lenient_facing(cls, v):
        return v if _in_vocabulary(v, ArahHadap) else "tidak_terbaca"


class Objek(BaseModel):
    id: str = Field(description="pengenal unik, misalnya O1")
    nama: str = Field(description="nama deskriptif singkat")
    kategori: Kategori
    jumlah: int = Field(description="1 kecuali ini kelompok objek identik")
    sel_grid: List[str] = Field(description="sel grid yang ditempati, misalnya ['C2','C3']")
    bbox_ternormalisasi: List[float] = Field(
        description="tepat 4 angka [x0,y0,x1,y1] rentang 0.0-1.0, asal di kiri atas"
    )
    atribut: AtributObjek
    ciri_visual_pendukung: str = Field(
        description="bentuk yang teramati, bukan kesimpulannya"
    )
    sumber_bukti: List[SumberBukti]
    keyakinan: float = Field(description="0.0 sampai 1.0")


# --------------------------------------------------------------------------
# layer 2: relations
# --------------------------------------------------------------------------

class Relasi(BaseModel):
    id: str
    subjek: str = Field(description="id objek")
    predikat: Predikat
    objek: str = Field(description="id objek")
    jenis: Literal["aksi", "spasial"]
    bukti_visual: str
    sumber_bukti: List[SumberBukti]
    keyakinan: float


# --------------------------------------------------------------------------
# layer 3: scene
# --------------------------------------------------------------------------

class Kelompok(BaseModel):
    id: str
    anggota: List[str] = Field(description="daftar id objek")
    sel_grid: List[str]
    alasan_dikelompokkan: str


class Adegan(BaseModel):
    pengelompokan: List[Kelompok]
    fokus_utama: Optional[str] = Field(description="id objek atau kelompok")
    aktivitas_teramati: str = Field(description="murni visual, tanpa menamai cerita")
    situasi_diperkirakan: str
    arah_alur_baca: Literal["kiri_ke_kanan", "kanan_ke_kiri", "tidak_jelas"]
    jumlah_panel_terlihat: int
    keyakinan: float


# --------------------------------------------------------------------------
# layer 4: historical interpretation
# --------------------------------------------------------------------------

class GayaPeriode(BaseModel):
    dugaan: str
    dasar: str
    keyakinan: float


class DugaanSitus(BaseModel):
    nama: str
    alasan: str
    keyakinan: float


class HipotesisNaratif(BaseModel):
    peringkat: int
    judul: str
    penjelasan: str
    bukti_pendukung: List[str]
    bukti_penyangkal: List[str] = Field(
        description="ciri pada citra yang justru TIDAK cocok dengan hipotesis ini"
    )
    keyakinan: float


class UnsurIkonografi(BaseModel):
    id_objek: str
    makna: str
    tradisi: Literal["Hindu", "Buddha", "lokal", "tidak_jelas"]

    @field_validator("tradisi", mode="before")
    @classmethod
    def _lenient_tradition(cls, v):
        return v if v in ("Hindu", "Buddha", "lokal", "tidak_jelas") else "tidak_jelas"


class InterpretasiHistoris(BaseModel):
    gaya_dan_periode: GayaPeriode
    dugaan_situs: List[DugaanSitus]
    hipotesis_naratif: List[HipotesisNaratif]
    unsur_ikonografi: List[UnsurIkonografi]
    informasi_yang_akan_menentukan: List[str]


# --------------------------------------------------------------------------
# meta, limitations, root
# --------------------------------------------------------------------------

class Meta(BaseModel):
    skenario_input: Skenario
    citra_digunakan: List[SumberBukti]
    keterbacaan_keseluruhan: float
    catatan_kondisi: str = Field(description="aus, lumut, retak, blok hilang")


class AreaTidakTerbaca(BaseModel):
    sel_grid: List[str]
    sebab: str


class ObjekMeragukan(BaseModel):
    id: str
    sebab: str


class Batasan(BaseModel):
    area_tidak_terbaca: List[AreaTidakTerbaca]
    objek_meragukan: List[ObjekMeragukan]
    dugaan_artefak_pengolahan: List[str] = Field(
        description="bentuk yang hanya muncul di FUSION atau CAVITY tanpa padanan di RGB"
    )
    pertentangan_antarcitra: List[str]
    asumsi_yang_dibuat: List[str]


class RingkasanJumlah(BaseModel):
    total_objek: int
    jumlah_figur_manusia: int
    jumlah_benda: int
    jumlah_arsitektur: int
    catatan: str = Field(description="objek yang terpotong tepi citra disebut di sini")


class AnalisisRelief(BaseModel):
    # NOTE: this docstring becomes the JSON Schema "description" that is pasted
    # into the system prompt, so it stays in Indonesian and must not change.
    """Hasil analisis relief batu dalam empat lapis terpisah."""

    meta: Meta
    objek: List[Objek]
    ringkasan_jumlah: RingkasanJumlah
    relasi: List[Relasi]
    adegan: Adegan
    interpretasi_historis: InterpretasiHistoris
    batasan: Batasan

    @field_validator("relasi", mode="before")
    @classmethod
    def _drop_unknown_predicates(cls, v):
        """
        Drop relations whose predicate is outside the vocabulary instead of
        failing the record.

        Unlike posture and tradition, `Predikat` has no neutral value to fall
        back to, and adding one would change the JSON Schema and hence the
        prompt. Relations are not used by layer-1 scoring, and the full list
        stays in attempt-*.parsed.json.
        """
        if not isinstance(v, list):
            return v
        return [
            r for r in v
            if not isinstance(r, dict) or _in_vocabulary(r.get("predikat"), Predikat)
        ]