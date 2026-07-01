import json

from harmonix.model import Overtone, Tone, ToneDatabase


def test_overtone_round_trip():
    ot = Overtone(frequency_factor=2.5, volume_factor=0.33)
    assert Overtone.from_dict(ot.to_dict()) == ot


def test_tone_round_trip(simple_tone):
    restored = Tone.from_dict(simple_tone.to_dict())
    assert restored == simple_tone


def test_tone_database_add_remove(simple_tone):
    db = ToneDatabase()
    db.add("a", simple_tone)
    assert "a" in db.tones
    db.remove("a")
    assert "a" not in db.tones
    db.remove("does-not-exist")  # must not raise


def test_tone_database_json_round_trip(simple_tone, tmp_path):
    db = ToneDatabase()
    db.add("alpha", simple_tone)
    db.add("beta", Tone(440.0, 1.0, []))

    path = tmp_path / "tones.json"
    db.save(str(path))

    # The file must be plain, human-readable JSON
    with open(path) as f:
        raw = json.load(f)
    assert set(raw.keys()) == {"alpha", "beta"}

    loaded = ToneDatabase.load(str(path))
    assert loaded.tones == db.tones


def test_tone_database_load_missing_file_returns_empty(tmp_path):
    db = ToneDatabase.load(str(tmp_path / "does-not-exist.json"))
    assert db.tones == {}


def test_tone_database_load_corrupt_file_returns_empty(tmp_path):
    path = tmp_path / "corrupt.json"
    path.write_text("{not valid json")
    db = ToneDatabase.load(str(path))
    assert db.tones == {}
