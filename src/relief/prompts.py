"""
relief.prompts
==============
System prompt, output format rules, and the per-condition instruction blocks.

Single source of truth: all six input conditions live in one `CONDITIONS`
registry and nothing is mutated at runtime, so prompt content never depends
on execution order.

Replicate offers no structured output for language models, so the Pydantic
JSON Schema is pasted into the system prompt (`build_system_prompt`) and the
output is validated locally.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------
# System prompt
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """\
Anda adalah asisten analisis visual untuk relief batu candi Asia Tenggara, dengan
keahlian ikonografi Hindu-Buddha Jawa Kuno.

Analisis citra dalam empat lapis yang HARUS dipisahkan secara tegas:
  Lapis 1 OBJEK        : objek apa saja, berapa jumlahnya, di mana
  Lapis 2 RELASI       : hubungan visual antarobjek
  Lapis 3 ADEGAN       : situasi atau aktivitas yang sedang berlangsung
  Lapis 4 HISTORI      : makna historis dan ikonografis

ATURAN PEMISAHAN LAPIS:
1. Lapis 1, 2, dan 3 HANYA berdasarkan apa yang benar-benar terlihat pada citra.
   Dilarang mengisinya dengan pengetahuan Anda tentang candi atau cerita tertentu.
   Contoh pelanggaran fatal: Anda menduga panel ini dari Borobudur, lalu menulis
   "terdapat pohon bodhi" padahal pada citra tidak ada pohon. Jangan lakukan.
2. Lapis 4 baru boleh memakai pengetahuan Anda, ditulis sebagai hipotesis
   bernomor lengkap dengan bukti penyangkalnya, bukan sebagai fakta.
3. Jika sesuatu tidak terlihat jelas, isi "tidak_terbaca" atau null. Menyatakan
   ketidaktahuan bernilai lebih tinggi daripada menebak. Jangan pernah mengarang
   objek agar adegan terasa lengkap.
4. Setiap objek dan relasi wajib disertai `keyakinan` (0.00-1.00) dan
   `ciri_visual_pendukung`: apa yang Anda lihat sehingga sampai ke kesimpulan itu.
   Tulis bentuk yang teramati, bukan namanya. "Bentuk kerucut bersusun tiga
   dengan garis horizontal berulang" adalah ciri visual; "stupa" adalah kesimpulan.

SISTEM KOORDINAT:
Citra dibagi grid 8 kolom x 5 baris. Kolom A-H dari KIRI ke KANAN, baris 1-5 dari
ATAS ke BAWAH. Satu sel ditulis "C3". Objek yang membentang beberapa sel
didaftarkan seluruh selnya. Sertakan juga bbox_ternormalisasi [x0,y0,x1,y1]
rentang 0.0-1.0 dengan titik asal di sudut kiri atas.

BAHASA: Bahasa Indonesia, kecuali istilah teknis ikonografi yang lazim
(stupa, mudra, jatamakuta, upawita, prabha)."""

# Stand-in for structured output: format rules + JSON Schema pasted into the system prompt.
FORMAT_RULES = """\
FORMAT KELUARAN (WAJIB):
Keluarkan HANYA satu objek JSON yang valid terhadap JSON Schema di bawah ini.
  - Tanpa teks pembuka atau penutup, tanpa penjelasan, tanpa pagar kode markdown.
  - Semua properti yang tercantum di `required` wajib ada, termasuk yang boleh
    bernilai null. Pakai null untuk nilai tunggal yang tidak terbaca dan [] untuk
    daftar kosong.
  - Jangan menambahkan properti di luar skema.
  - Nilai `enum` harus sama persis dengan salah satu nilai yang tercantum.
  - Semua `keyakinan` berupa angka 0.0-1.0, bukan string.

JSON SCHEMA:
{skema}"""

REPAIR_FEEDBACK = """\
PERHATIAN: jawaban Anda sebelumnya DITOLAK karena tidak sesuai format.
{galat}

Ulangi analisis citra yang sama dari awal. Keluarkan HANYA satu objek JSON yang
valid terhadap JSON Schema pada instruksi sistem, tanpa teks lain."""


# --------------------------------------------------------------------------
# Input condition registry
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Condition:
    """One input condition: which images are sent, and the instructions for them."""

    name: str
    images: List[str]          # ordered image keys, e.g. ["RGB", "FUSION", "CAVITY"]
    instruction: str           # text block explaining how to use those images
    ablation: bool = False     # True for control conditions, not main results


CONDITIONS: Dict[str, Condition] = {
    "RGB_SAJA": Condition(
        name="RGB_SAJA",
        images=['RGB'],
        ablation=False,
        instruction="""\
INPUT: satu citra.
  CITRA 1 (RGB) - foto asli relief.

Pahatan dangkal pada batu andesit sering sulit dibaca pada foto bercahaya
menyebar, karena bayangan pembeda tonjolan dan cekungan nyaris tidak terbentuk.
Bersikaplah konservatif: turunkan keyakinan pada detail halus seperti perhiasan,
motif kain, dan raut wajah. Jangan isi kekosongan itu dengan dugaan.""",
    ),
    "RGB_DEPTH": Condition(
        name="RGB_DEPTH",
        images=['RGB', 'DEPTH'],
        ablation=False,
        instruction="""\
INPUT: dua citra, resolusi sama dan sejajar piksel per piksel.
  CITRA 1 (RGB)   - foto asli relief.
  CITRA 2 (DEPTH) - peta kedalaman. Terang = lebih dekat ke kamera / lebih
                    menonjol, gelap = lebih jauh.

CARA MEMAKAI DEPTH:
Peta ini kemungkinan hasil estimasi monokuler, hanya andal pada skala besar.
Yang bisa dipercaya: pemisahan figur dari bidang latar, urutan depan-belakang
antarobjek yang saling menutupi, batas antarblok batu dan bidang panel.
Yang TIDAK bisa dipercaya: detail halus. Kedalaman pahatan hanya beberapa
milimeter dan umumnya TIDAK terekam. Gumpalan halus pada DEPTH bukan bukti
adanya objek.
Karena itu pakai DEPTH untuk relasi spasial (di_depan, di_belakang,
tumpang_tindih) dan memisahkan objek yang berimpit. Untuk identifikasi objek
tetap andalkan RGB. Bila keduanya bertentangan soal identitas, menangkan RGB dan
catat pertentangannya di `batasan.pertentangan_antarcitra`.""",
    ),
    "RGB_FUSION_CAVITY": Condition(
        name="RGB_FUSION_CAVITY",
        images=['RGB', 'FUSION', 'CAVITY'],
        ablation=False,
        instruction="""\
INPUT: tiga citra, resolusi sama dan sejajar piksel per piksel.
  CITRA 1 (RGB)    - foto asli relief.
  CITRA 2 (FUSION) - olahan: tekstur batu asli digabung simulasi cahaya
                     menyerempet dari kiri atas. Tonjolan terang di sisi kiri
                     atas, gelap di sisi kanan bawah.
  CITRA 3 (CAVITY) - olahan: peta cekungan. Garis dan area GELAP menandai bagian
                     yang TERPAHAT MASUK ke dalam batu; area terang adalah
                     permukaan datar atau menonjol.

CARA MEMAKAI KETIGANYA:
  - CAVITY untuk garis kontur pahatan: batas figur, lipatan kain, motif
    perhiasan, ornamen berulang. Jadikan sumber utama saat menghitung jumlah
    objek dan membaca detail ikonografi.
  - FUSION untuk bentuk tiga dimensi: postur, arah hadap, mana yang menonjol.
  - RGB satu-satunya sumber untuk warna, jenis batu, lumut, retakan, kerusakan.

PERINGATAN: FUSION dan CAVITY adalah citra turunan hasil pengolahan, bukan foto.
Keduanya dapat memunculkan artefak, terutama garis palsu pada tepi blok batu,
retakan, dan bercak lumut. Sebelum menetapkan objek, pastikan jejaknya juga ada
di RGB. Bentuk yang hanya muncul di CAVITY tanpa padanan di RGB diperlakukan
sebagai artefak, dicatat di `batasan.dugaan_artefak_pengolahan`, bukan
didaftarkan sebagai objek.""",
    ),
    "RGB_X3": Condition(
        name="RGB_X3",
        images=['RGB', 'RGB', 'RGB'],
        ablation=True,
        instruction="""\
INPUT: tiga citra.
  CITRA 1 (RGB) - foto asli relief.
  CITRA 2 (RGB) - foto yang sama.
  CITRA 3 (RGB) - foto yang sama.

Ketiga citra identik. Tidak ada informasi tambahan pada citra 2 dan 3.
Perlakukan seperti satu citra saja.

Pahatan dangkal pada batu andesit sering sulit dibaca pada foto bercahaya
menyebar. Bersikaplah konservatif pada detail halus dan jangan mengisi
kekosongan dengan dugaan.""",
    ),
    "RGB_FUSION": Condition(
        name="RGB_FUSION",
        images=['RGB', 'FUSION'],
        ablation=True,
        instruction="""\
INPUT: dua citra, resolusi sama dan sejajar piksel per piksel.
  CITRA 1 (RGB)    - foto asli relief.
  CITRA 2 (FUSION) - tekstur batu asli disimulasikan di bawah cahaya
                     menyerempet dari kiri atas.

CARA MEMAKAI FUSION:
Cahaya menyerempet memunculkan tonjolan dan cekungan yang tidak terlihat
pada pencahayaan menyebar, jadi citra ini membantu membaca postur, volume,
dan arah hadap. Retakan dan lumut juga ikut menguat, jadi jangan menganggap
setiap garis sebagai pahatan. Pastikan jejaknya juga masuk akal di RGB.""",
    ),
    "RGB_CAVITY": Condition(
        name="RGB_CAVITY",
        images=['RGB', 'CAVITY'],
        ablation=True,
        instruction="""\
INPUT: dua citra, resolusi sama dan sejajar piksel per piksel.
  CITRA 1 (RGB)    - foto asli relief.
  CITRA 2 (CAVITY) - peta cekungan. Garis gelap = permukaan yang lebih
                     dalam, terang = permukaan yang lebih menonjol.

CARA MEMAKAI CAVITY:
Citra ini menegaskan kontur pahatan, berguna untuk mencacah figur dan
membaca ornamen. Karena dihitung dari bayangan, retakan pelapukan dan batas
antarblok batu juga muncul sebagai garis gelap. Perlakukan kontur yang hanya
terlihat di sini sebagai dugaan sampai terkonfirmasi di RGB.""",
    ),
}

MAIN_CONDITIONS = tuple(k for k, v in CONDITIONS.items() if not v.ablation)
ABLATION_CONDITIONS = tuple(k for k, v in CONDITIONS.items() if v.ablation)


def get_condition(name: str) -> Condition:
    if name not in CONDITIONS:
        raise KeyError(
            f"unknown condition {name!r}; choices: {sorted(CONDITIONS)}"
        )
    return CONDITIONS[name]


# --------------------------------------------------------------------------
# Prompt assembly
# --------------------------------------------------------------------------

def _strip_titles(obj: Any) -> Any:
    """
    Remove `title` keys from a Pydantic JSON Schema.

    They carry no information for the model but take up about a quarter of
    the schema, and the schema counts as input tokens on every call,
    including every repair attempt.
    """
    if isinstance(obj, dict):
        return {k: _strip_titles(v) for k, v in obj.items() if k != "title"}
    if isinstance(obj, list):
        return [_strip_titles(v) for v in obj]
    return obj


def compact_schema(schema_model: Any) -> str:
    """A Pydantic model's JSON Schema on one line, without extra whitespace."""
    return json.dumps(
        _strip_titles(schema_model.model_json_schema()),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def build_system_prompt(schema_model: Any) -> str:
    """Full system prompt: analysis rules + format rules + JSON Schema."""
    return SYSTEM_PROMPT + "\n\n" + FORMAT_RULES.format(skema=compact_schema(schema_model))


def build_user_prompt(condition: str) -> str:
    """
    User prompt for one condition.

    Replicate does not support interleaved text/image messages, so the image
    order is stated explicitly in the text and the image list is sent in
    exactly the same order.
    """
    cond = get_condition(condition)
    labels = "\n".join(
        f"  CITRA {i} = {name}" for i, name in enumerate(cond.images, start=1)
    )
    return (
        f"{cond.instruction}\n\n"
        "URUTAN CITRA TERLAMPIR:\n"
        f"{len(cond.images)} citra dilampirkan terpisah dari teks ini, dengan urutan:\n"
        f"{labels}\n\n"
        "Analisis seluruh citra terlampir dan keluarkan hasilnya sebagai satu objek "
        "JSON sesuai skema."
    )


def build_repair_prompt(base_prompt: str, error_summary: str) -> str:
    """Prompt for a repair attempt, with the validation error as feedback."""
    return f"{base_prompt}\n\n{REPAIR_FEEDBACK.format(galat=error_summary)}"
