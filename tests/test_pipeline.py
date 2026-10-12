"""
End-to-end pipeline tests that never call Replicate.

The client is replaced by a fake that returns the three kinds of output that
actually occur in practice:

    panel A  JSON valid against the schema                  -> status ok
    panel B  JSON that parses but violates a schema enum    -> schema_failed
    panel C  rambling text, not JSON at all                 -> json_failed

What is checked: all three still leave artefacts on disk, and the layers that
succeeded are stored separately from the ones that failed. This is why the
three-layer storage exists: panel B's output can still be analysed (and
partly scored) without calling the model again.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from relief import runner
from relief.artifacts import load_records
from relief.client import RawResponse
from relief.manifest import Cell
from relief.reporting import usage_table, write_usage_csv
from relief.usage import extract_usage

PANELS = ["Boro_01", "Boro_02", "Boro_03"]


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

def valid_record() -> dict:
    """One example output that passes AnalisisRelief validation."""
    return {
        "meta": {
            "skenario_input": "RGB_SAJA",
            "citra_digunakan": ["RGB"],
            "keterbacaan_keseluruhan": 0.7,
            "catatan_kondisi": "aus ringan di sudut kanan bawah",
        },
        "objek": [{
            "id": "O1",
            "nama": "figur duduk",
            "kategori": "figur_manusia",
            "jumlah": 1,
            "sel_grid": ["C3"],
            "bbox_ternormalisasi": [0.2, 0.3, 0.4, 0.7],
            "atribut": {
                "postur": "duduk_bersila",
                "arah_hadap": "kiri",
                "hiasan_kepala": None,
                "perhiasan": [],
                "pakaian": "kain berlipat",
                "benda_dipegang": [],
                "sikap_tangan": None,
            },
            "ciri_visual_pendukung": "bentuk oval dengan dua tonjolan simetris",
            "sumber_bukti": ["RGB"],
            "keyakinan": 0.6,
        }],
        "ringkasan_jumlah": {
            "total_objek": 1, "jumlah_figur_manusia": 1,
            "jumlah_benda": 0, "jumlah_arsitektur": 0,
            "catatan": "tidak ada objek terpotong tepi",
        },
        "relasi": [],
        "adegan": {
            "pengelompokan": [], "fokus_utama": "O1",
            "aktivitas_teramati": "satu figur duduk menghadap kiri",
            "situasi_diperkirakan": "adegan tunggal",
            "arah_alur_baca": "tidak_jelas",
            "jumlah_panel_terlihat": 1, "keyakinan": 0.5,
        },
        "interpretasi_historis": {
            "gaya_dan_periode": {"dugaan": "Jawa Tengah klasik",
                                 "dasar": "proporsi figur", "keyakinan": 0.3},
            "dugaan_situs": [], "hipotesis_naratif": [],
            "unsur_ikonografi": [],
            "informasi_yang_akan_menentukan": ["panel di sebelahnya"],
        },
        "batasan": {
            "area_tidak_terbaca": [], "objek_meragukan": [],
            "dugaan_artefak_pengolahan": [], "pertentangan_antarcitra": [],
            "asumsi_yang_dibuat": [],
        },
    }


def schema_violating_record() -> dict:
    """Well-formed JSON whose category is outside the controlled vocabulary."""
    rec = valid_record()
    rec["objek"][0]["kategori"] = "orang"       # not a member of the Kategori enum
    return rec


OUTPUTS = {
    "Boro_01": json.dumps(valid_record(), ensure_ascii=False),
    # deliberately wrapped in a code fence: models often add one even when told not to
    "Boro_02": "```json\n" + json.dumps(schema_violating_record(), ensure_ascii=False) + "\n```",
    "Boro_03": "Sorry, the image is too blurry to analyse in detail.",
}


class FakePrediction:
    def __init__(self, pid, text, metrics):
        self.id = pid
        self.output = [text]
        self.metrics = metrics
        self.logs = ""
        self.status = "succeeded"
        self.error = None


class FakeClient:
    """Stand-in for ReplicateClient: never touches the network."""

    def __init__(self, model, **kwargs):
        self.model = model
        self.max_tokens = 12000
        self.calls = 0
        self._panel = None

    def upload_images(self, files):
        # image labels would carry the panel name via file contents; the order is enough here
        meta = [{"label": label, "sha256": f"fake{i}", "bytes": len(data)}
                for i, (label, data) in enumerate(files)]
        return [f"https://example.invalid/{m['sha256']}.jpg" for m in meta], meta

    def invoke(self, system_prompt, user_prompt, images):
        self.calls += 1
        text = OUTPUTS[self._panel]
        prediction = FakePrediction(
            f"pred-{self.calls:03d}", text,
            {"input_token_count": 5380, "output_token_count": 4310,
             "predict_time": 1.25},
        )
        return RawResponse(
            text=text, prediction=prediction,
            usage=extract_usage(prediction),
            model_input={"prompt": user_prompt, "image_input": images},
        )

    def cleanup_uploads(self):
        pass


@pytest.fixture
def dataset(tmp_path):
    """Fake dataset: each panel has five small image files."""
    root = tmp_path / "dataset"
    for panel in PANELS:
        folder = root / panel
        folder.mkdir(parents=True)
        img = np.full((64, 96, 3), 128, np.uint8)
        for name in ("rgb.png", "depth.png", "fusion.png", "cavity.png", "seg2.png"):
            cv2.imwrite(str(folder / name), img)
    return root


@pytest.fixture
def patched_client(monkeypatch):
    """Inject FakeClient and let it know which panel is being processed."""
    holder = {}

    def factory(model, **kwargs):
        client = FakeClient(model, **kwargs)
        holder["client"] = client
        return client

    monkeypatch.setattr(runner, "ReplicateClient", factory)

    real_load = runner.load_condition_images

    def load_and_remember(root, panel_id, keys):
        holder["client"]._panel = panel_id
        return real_load(root, panel_id, keys)

    monkeypatch.setattr(runner, "load_condition_images", load_and_remember)
    return holder


def make_cells(model="openai/gpt-4.1", condition="RGB_SAJA"):
    return [
        Cell(cell_id=f"{model.replace('/', '-')}__{condition}__ms1536__{p}__r1",
             panel_id=p, site="Borobudur", group="Boro", split="test",
             condition=condition, model=model, rep=1, max_side=1536, plan="main")
        for p in PANELS
    ]


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_three_layers_are_written_per_attempt(dataset, patched_client, tmp_path):
    runs = tmp_path / "runs"
    stats = runner.run_manifest(make_cells(), dataset, runs, sleep_s=0, max_repairs=1)

    assert stats["done"] == 3
    assert stats["ok"] == 1
    assert stats["schema_failed"] == 1
    assert stats["json_failed"] == 1

    def files(panel):
        d = next(runs.rglob(f"*__{panel}__r1"))
        return {p.name for p in d.iterdir()}

    # Panel A: all three layers exist.
    a = files("Boro_01")
    assert {"record.json", "request.json", "attempt-01.raw.txt",
            "attempt-01.parsed.json", "attempt-01.validated.json"} <= a
    assert "attempt-01.error.txt" not in a
    # Raw output is saved on success too, not only on failure.
    assert "attempt-01.prediction.json" in a

    # Panel B: raw and parsed exist, validated does not; and a repair attempt was made.
    b = files("Boro_02")
    assert {"attempt-01.raw.txt", "attempt-01.parsed.json",
            "attempt-01.error.txt", "attempt-02.raw.txt"} <= b
    assert "attempt-01.validated.json" not in b

    # Panel C: raw only; there is no JSON to store.
    c = files("Boro_03")
    assert "attempt-01.raw.txt" in c
    assert "attempt-01.parsed.json" not in c
    assert "attempt-01.validated.json" not in c


def test_raw_text_round_trips_exactly(dataset, patched_client, tmp_path):
    runs = tmp_path / "runs"
    runner.run_manifest(make_cells(), dataset, runs, sleep_s=0, max_repairs=0)
    d = next(runs.rglob("*__Boro_02__r1"))
    assert (d / "attempt-01.raw.txt").read_text(encoding="utf-8") == OUTPUTS["Boro_02"]


def test_parsed_survives_schema_failure(dataset, patched_client, tmp_path):
    """JSON that violates the schema is still stored and still readable."""
    runs = tmp_path / "runs"
    runner.run_manifest(make_cells(), dataset, runs, sleep_s=0, max_repairs=0)
    d = next(runs.rglob("*__Boro_02__r1"))
    parsed = json.loads((d / "attempt-01.parsed.json").read_text(encoding="utf-8"))
    assert parsed["objek"][0]["kategori"] == "orang"

    record = json.loads((d / "record.json").read_text(encoding="utf-8"))
    assert record["status"] == "schema_failed"
    assert record["validated"] is None
    assert record["parsed"] is not None


def test_repair_attempt_is_billed_and_accumulated(dataset, patched_client, tmp_path):
    runs = tmp_path / "runs"
    runner.run_manifest(make_cells(), dataset, runs, sleep_s=0, max_repairs=1)
    d = next(runs.rglob("*__Boro_02__r1"))
    record = json.loads((d / "record.json").read_text(encoding="utf-8"))

    assert record["n_attempts"] == 2
    assert len(record["prediction_ids"]) == 2
    # Tokens from both attempts are summed: 2 x 5380 input.
    assert record["usage"]["input_tokens"] == 10760
    assert record["usage"]["n_predictions"] == 2
    assert len(record["attempts"]) == 2


def test_resume_skips_finished_cells(dataset, patched_client, tmp_path):
    runs = tmp_path / "runs"
    cells = make_cells()
    runner.run_manifest(cells, dataset, runs, sleep_s=0, max_repairs=0)
    again = runner.run_manifest(cells, dataset, runs, sleep_s=0, max_repairs=0)

    assert again["skipped"] == 3
    assert again["done"] == 0
    assert runner.progress(cells, runs) == {"total": 3, "done": 3, "remaining": 0}


def test_usage_report_and_csv(dataset, patched_client, tmp_path):
    runs = tmp_path / "runs"
    runner.run_manifest(make_cells(), dataset, runs, sleep_s=0, max_repairs=0)

    records = load_records(runs)
    assert len(records) == 3

    table = usage_table(records)
    row = table["rows"][0]
    assert row["calls"] == 3
    assert row["ok"] == 1
    assert row["tokens_in"] == 3 * 5380
    assert table["warning"] is None          # tokens were read for every record

    csv_path = write_usage_csv(records, tmp_path / "usage.csv")
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4                   # header + three calls
    assert "usage_source" in lines[0]


def test_request_json_records_what_was_sent(dataset, patched_client, tmp_path):
    runs = tmp_path / "runs"
    runner.run_manifest(make_cells(), dataset, runs, sleep_s=0, max_repairs=0)
    d = next(runs.rglob("*__Boro_01__r1"))
    request = json.loads((d / "request.json").read_text(encoding="utf-8"))

    assert "JSON SCHEMA" in request["system_prompt"]
    assert "CITRA 1 = RGB" in request["user_prompt"]
    assert len(request["images"]) == 1
    assert request["images"][0]["label"] == "RGB"


def test_condition_image_mismatch_is_caught_early(dataset, patched_client, tmp_path):
    """A three-image condition must not silently run with a single image."""
    from relief.analyze import analyze_panel

    client = FakeClient("openai/gpt-4.1")
    client._panel = "Boro_01"
    with pytest.raises(ValueError, match="expects images"):
        analyze_panel([("RGB", dataset / "Boro_01" / "rgb.png")],
                      "RGB_FUSION_CAVITY", client)
